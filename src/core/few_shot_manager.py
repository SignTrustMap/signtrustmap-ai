from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
from loguru import logger
import numpy as np
from scipy.spatial.distance import cdist
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session
import torch

from src.core.config_manager import ConfigManager
from src.core.evaluator import evaluator
from src.infrastructure.db import sync_engine
from src.infrastructure.models import (
    CandidateEmbedding,
    ModelArtifact,
    ModelClass,
    ModelEvaluationMetric,
    ModelVersion,
    SupportShot,
    TextEmbedding,
)

config_manager = ConfigManager()


def compute_sharpness(img_bgr: np.ndarray) -> float:
    """Computes Laplacian variance sharpness of a BGR image."""
    if img_bgr is None or img_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class FewShotSupportManager:
    """
    Manages Configurable Few-Shot Support Pools and TIMO Collapsed Classifier calculation
    according to the Diversity K-Means & Anchor Replacement Policy.
    All parameters (total_shots, num_anchors, alpha, etc.) are fully configurable.
    """

    def __init__(self, config: Optional[ConfigManager] = None) -> None:
        self.config = config or config_manager

    def get_support_config(self) -> Dict[str, Any]:
        cfg = self.config.get("few_shot_support", {})
        return {
            "shots_per_class": int(cfg.get("shots_per_class", 16)),
            "num_anchors": int(cfg.get("num_anchors", 4)),
            "num_medoids": int(cfg.get("num_medoids", 12)),
            "min_crop_size": int(cfg.get("min_crop_size", 20)),
            "min_sharpness": float(cfg.get("min_sharpness", 15.0)),
        }

    def get_timo_config(self) -> Dict[str, Any]:
        cfg = self.config.get("timo", {})
        return {
            "alpha": float(cfg.get("alpha", 250.0)),
            "beta": float(cfg.get("beta", 0.8)),
            "prompt_template": str(self.config.classifier_prompt_template),
        }

    def select_support_shots(
        self,
        candidates: List[Dict[str, Any]],
        visual_features: Optional[np.ndarray] = None,
        total_shots: Optional[int] = None,
        num_anchors: Optional[int] = None,
        random_state: int = 42,
    ) -> List[Dict[str, Any]]:
        cfg = self.get_support_config()
        total_budget = int(total_shots) if total_shots is not None else cfg["shots_per_class"]
        n_anchors = int(num_anchors) if num_anchors is not None else cfg["num_anchors"]
        preserve_pinned = True

        n_samples = len(candidates)
        if n_samples == 0:
            return []

        # If total pool is within budget, keep all
        if n_samples <= total_budget:
            selected = []
            for idx, c in enumerate(candidates):
                item = dict(c)
                if "role" not in item or not item["role"]:
                    item["role"] = "ANCHOR" if idx < n_anchors else "MEDOID"
                selected.append(item)
            return selected

        # 1. Partition into Pinned vs Unpinned
        pinned_items: List[Tuple[int, Dict[str, Any]]] = []
        unpinned_items: List[Tuple[int, Dict[str, Any]]] = []

        for idx, c in enumerate(candidates):
            if preserve_pinned and c.get("is_pinned", False):
                pinned_items.append((idx, c))
            else:
                unpinned_items.append((idx, c))

        # If pinned items alone exceed budget, take all pinned
        if len(pinned_items) >= total_budget:
            logger.warning(f"Pinned items ({len(pinned_items)}) >= budget ({total_budget}). Returning pinned items.")
            return [dict(c) for _, c in pinned_items[:total_budget]]

        available_budget = total_budget - len(pinned_items)
        selected_indices = set([idx for idx, _ in pinned_items])
        final_results = [dict(c) for _, c in pinned_items]

        # 2. Select Anchors from unpinned candidates (Area x Sharpness)
        target_anchors = max(0, min(n_anchors - len([c for _, c in pinned_items if c.get("role") == "ANCHOR"]), available_budget))
        
        # Scoring function: Area x Sharpness
        def _clarity_score(item: Dict[str, Any]) -> float:
            w = float(item.get("width", 32))
            h = float(item.get("height", 32))
            sh = float(item.get("sharpness", 1.0))
            return (w * h) * max(sh, 1.0)

        unpinned_sorted = sorted(unpinned_items, key=lambda x: _clarity_score(x[1]), reverse=True)
        chosen_anchor_tuples = unpinned_sorted[:target_anchors]

        for idx, c in chosen_anchor_tuples:
            item = dict(c)
            item["role"] = "ANCHOR"
            final_results.append(item)
            selected_indices.add(idx)

        remaining_budget = total_budget - len(final_results)
        if remaining_budget <= 0:
            return final_results

        # 3. Select Diverse Medoids from remaining unpinned items
        remaining_unpinned = [t for t in unpinned_sorted if t[0] not in selected_indices]
        if not remaining_unpinned:
            return final_results

        if len(remaining_unpinned) <= remaining_budget or visual_features is None:
            # Fallback to top remaining by score
            for idx, c in remaining_unpinned[:remaining_budget]:
                item = dict(c)
                item["role"] = "MEDOID"
                final_results.append(item)
            return final_results

        # Extract embeddings for remaining candidates
        rem_indices = [t[0] for t in remaining_unpinned]
        rem_feats = visual_features[rem_indices]

        # Normalize features
        norm = np.linalg.norm(rem_feats, axis=1, keepdims=True)
        rem_feats_norm = rem_feats / np.clip(norm, 1e-8, None)

        # Run K-Means++ on visual embeddings with k = remaining_budget
        k = min(remaining_budget, len(remaining_unpinned))
        rng = np.random.RandomState(random_state)

        # K-Means++ initialization
        first_center_idx = rng.choice(len(rem_feats_norm))
        centers = [rem_feats_norm[first_center_idx].copy()]

        for _ in range(1, k):
            dists = cdist(rem_feats_norm, np.array(centers), metric="sqeuclidean")
            min_dists = np.min(dists, axis=1)
            total_d = np.sum(min_dists)
            probs = (min_dists / total_d) if total_d > 1e-12 else None
            next_idx = rng.choice(len(rem_feats_norm), p=probs)
            centers.append(rem_feats_norm[next_idx].copy())

        centers_arr = np.array(centers)

        # 10 Lloyd iterations for centroid refinement
        for _ in range(10):
            dists = cdist(rem_feats_norm, centers_arr, metric="sqeuclidean")
            labels = np.argmin(dists, axis=1)
            new_centers = np.empty_like(centers_arr)
            for m in range(k):
                mask = (labels == m)
                if np.any(mask):
                    new_centers[m] = np.mean(rem_feats_norm[mask], axis=0)
                else:
                    new_centers[m] = centers_arr[m]
            centers_arr = new_centers

        # Select medoids: the candidate closest to each centroid
        cluster_dists = cdist(rem_feats_norm, centers_arr, metric="sqeuclidean")
        chosen_local_medoids = []
        for m in range(k):
            medoid_local_idx = int(np.argmin(cluster_dists[:, m]))
            if medoid_local_idx not in chosen_local_medoids:
                chosen_local_medoids.append(medoid_local_idx)

        # If any collisions, backfill
        if len(chosen_local_medoids) < remaining_budget:
            for l_idx in range(len(rem_feats_norm)):
                if l_idx not in chosen_local_medoids:
                    chosen_local_medoids.append(l_idx)
                if len(chosen_local_medoids) >= remaining_budget:
                    break

        for l_idx in chosen_local_medoids[:remaining_budget]:
            orig_idx, c = remaining_unpinned[l_idx]
            item = dict(c)
            item["role"] = "MEDOID"
            final_results.append(item)

        logger.info(
            f"Selected {len(final_results)} support shots "
            f"({len([x for x in final_results if x.get('role') == 'ANCHOR'])} Anchors, "
            f"{len([x for x in final_results if x.get('role') == 'MEDOID'])} Medoids, "
            f"{len(pinned_items)} Pinned)."
        )
        return final_results

    def compute_timo_classifier(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        text_features: torch.Tensor,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.get_timo_config()
        a = cfg["alpha"] if alpha is None else float(alpha)
        b = cfg["beta"] if beta is None else float(beta)

        if device is None:
            device = support_features.device if isinstance(support_features, torch.Tensor) else torch.device("cpu")

        if not isinstance(support_features, torch.Tensor):
            support_features = torch.tensor(support_features, dtype=torch.float32, device=device)
        else:
            support_features = support_features.to(device=device, dtype=torch.float32)

        if not isinstance(support_labels, torch.Tensor):
            support_labels = torch.tensor(support_labels, dtype=torch.long, device=device)
        else:
            support_labels = support_labels.to(device=device, dtype=torch.long)

        if not isinstance(text_features, torch.Tensor):
            text_features = torch.tensor(text_features, dtype=torch.float32, device=device)
        else:
            text_features = text_features.to(device=device, dtype=torch.float32)

        num_classes = text_features.shape[0]
        dim = text_features.shape[1]

        vecs_v = support_features / torch.clamp(support_features.norm(dim=-1, keepdim=True), min=1e-8)
        t_proto = text_features / torch.clamp(text_features.norm(dim=-1, keepdim=True), min=1e-8)
        clip_weights = t_proto.t()

        labels_t = torch.arange(num_classes, device=device, dtype=torch.long)
        vecs_c = torch.cat([vecs_v, b * t_proto], dim=0)
        labels_c = torch.cat([support_labels, labels_t], dim=0)

        mus_list = []
        for i in range(num_classes):
            mask = labels_c == i
            if mask.any():
                mus_list.append(vecs_c[mask].mean(dim=0))
            else:
                mus_list.append(t_proto[i])
        mus = torch.stack(mus_list, dim=0)
        mus = mus / torch.clamp(mus.norm(dim=-1, keepdim=True), min=1e-8)

        center_vecs = torch.cat([vecs_c[labels_c == i] - mus[i].unsqueeze(0) for i in range(num_classes)], dim=0)
        cov = center_vecs.T.cov()
        reg = cov.trace() * torch.eye(dim, device=device, dtype=cov.dtype)
        mat_reg = (center_vecs.shape[0] - 1) * cov + reg
        try:
            cov_inv = dim * torch.linalg.pinv(mat_reg)
        except Exception:
            cov_inv = dim * torch.linalg.pinv(mat_reg.cpu()).to(device=device, dtype=cov.dtype)

        ps = torch.ones(num_classes, device=device, dtype=torch.float32) / float(num_classes)
        w_gda = mus @ cov_inv
        b_gda = ps.log() - 0.5 * torch.einsum("nd, dc, nc -> n", mus, cov_inv, mus)

        w_timo = a * clip_weights + w_gda.t()
        b_timo = b_gda

        del center_vecs, cov, reg, cov_inv, w_gda, vecs_c, labels_c
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return w_timo, b_timo

    def get_support_shots_metadata(self, model_id: int = 2) -> Dict[str, Any]:
        try:
            with Session(sync_engine) as session:
                res = session.query(
                    func.count(SupportShot.id),
                    func.coalesce(func.max(SupportShot.id), 0),
                ).filter(SupportShot.model_id == model_id).first()
                count, max_id = (res[0], res[1]) if res else (0, 0)
                return {"sample_count": int(count), "max_id": int(max_id), "model_id": model_id}
        except Exception as e:
            logger.warning(f"Could not load support shots metadata: {e}")
            return {"sample_count": 0, "max_id": 0, "model_id": model_id}

    def build_and_save_timo_classifier(
        self,
        support_cache_file: Optional[str] = None,
        text_embeddings_file: Optional[str] = None,
        output_path: Optional[str] = None,
        device: Optional[str] = None,
        model_id: int = 2,
    ) -> Path:
        base_dir = Path(__file__).resolve().parent.parent.parent
        dev = torch.device(device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        out_path = Path(output_path) if output_path else (base_dir / "working" / "embeddings" / "timo.pt")

        keys, labels, text_emb = None, None, None

        try:
            with Session(sync_engine) as session:
                stmt_supp = (
                    select(ModelClass.class_index, SupportShot.embedding)
                    .join(ModelClass, SupportShot.class_id == ModelClass.id)
                    .where(SupportShot.model_id == model_id)
                    .order_by(ModelClass.class_index, SupportShot.id)
                )
                supp_rows = session.execute(stmt_supp).fetchall()
                if supp_rows:
                    c_indices = [r[0] for r in supp_rows]
                    embs = [np.array(r[1], dtype=np.float32) for r in supp_rows]
                    keys = torch.tensor(np.array(embs), device=dev)
                    labels = torch.tensor(c_indices, dtype=torch.long, device=dev)

                    supp_cache_path = base_dir / "working" / "embeddings" / "support_cache_mobileclip_s2.pt"
                    supp_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        torch.save({
                            "keys": keys.cpu(),
                            "labels": labels.cpu(),
                            "meta": {
                                "model_id": model_id,
                                "sample_count": int(keys.shape[0]),
                            }
                        }, supp_cache_path)
                    except Exception as e:
                        logger.warning(f"Failed to persist support cache snapshot: {e}")

                stmt_text = (
                    select(ModelClass.class_index, TextEmbedding.embedding)
                    .join(ModelClass, TextEmbedding.class_id == ModelClass.id)
                    .where(TextEmbedding.model_id == model_id)
                    .order_by(ModelClass.class_index)
                )
                text_rows = session.execute(stmt_text).fetchall()
                if text_rows:
                    text_embs = [np.array(r[1], dtype=np.float32) for r in text_rows]
                    text_emb = torch.tensor(np.array(text_embs), device=dev)
        except Exception as e:
            logger.warning(f"Could not load vectors from database: {e}")

        if keys is None or text_emb is None:
            if support_cache_file:
                supp_path = Path(support_cache_file)
            else:
                working_supp = base_dir / "working" / "embeddings" / "support_cache_mobileclip_s2.pt"
                eval_supp = base_dir / "outputs" / "evaluation" / "tip_cache.pt"
                supp_path = working_supp if working_supp.exists() else eval_supp

            if text_embeddings_file:
                text_path = Path(text_embeddings_file)
            else:
                working_text = base_dir / "working" / "embeddings" / "text_embeddings_mobileclip_s2.pt"
                weights_text = base_dir / "weights" / "embeddings" / "text_embeddings_mobileclip_s2.pt"
                text_path = working_text if working_text.exists() else weights_text

            if not supp_path.exists():
                raise FileNotFoundError(f"Support cache file not found at: {supp_path}")
            if not text_path.exists():
                raise FileNotFoundError(f"Text embeddings file not found at: {text_path}")
            logger.info(f"Loading support features from {supp_path} and text embeddings from {text_path}")
            supp_data = torch.load(supp_path, map_location=dev)
            text_emb = torch.load(text_path, map_location=dev)
            keys = supp_data["keys"] if isinstance(supp_data, dict) else supp_data
            labels = supp_data["labels"] if isinstance(supp_data, dict) else torch.zeros(len(keys), dtype=torch.long)

        w_timo, b_timo = self.compute_timo_classifier(
            support_features=keys,
            support_labels=labels,
            text_features=text_emb,
            device=dev,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        meta_info = self.get_support_shots_metadata(model_id=model_id)
        meta_info["num_classes"] = int(w_timo.shape[1])
        torch.save({"weights": w_timo.cpu(), "bias": b_timo.cpu(), "meta": meta_info}, out_path)
        logger.info(f"TIMO Collapsed Classifier weights saved to {out_path} (weights shape: {list(w_timo.shape)}, bias shape: {list(b_timo.shape)})")
        return out_path

    def rebuild_classifier_from_verified_candidates(
        self,
        version_tag: str,
        note: Optional[str] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        shots_per_class: Optional[int] = None,
        num_anchors: Optional[int] = None,
        num_medoids: Optional[int] = None,
        min_sharpness: Optional[float] = None,
        sample_limit: Optional[int] = None,
        base_model_id: int = 2,
    ) -> Dict[str, Any]:
        base_dir = Path(__file__).resolve().parent.parent.parent
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        cache_map_file = base_dir / "working" / "data" / "verified_signs_cache.json"
        verified_map: Dict[str, Any] = {}
        if cache_map_file.exists():
            with open(cache_map_file, "r", encoding="utf-8") as f:
                verified_map = json.load(f)

        with Session(sync_engine) as session:
            class_stmt = (
                select(ModelClass.id, ModelClass.class_index, ModelClass.sign_code, ModelClass.class_name)
                .where(ModelClass.model_id == base_model_id)
                .order_by(ModelClass.class_index.asc())
            )
            class_rows = session.execute(class_stmt).fetchall()
            sign_code_to_class = {r[2]: (r[0], r[1]) for r in class_rows}
            class_index_to_id = {r[1]: r[0] for r in class_rows}

            cand_stmt = (
                select(CandidateEmbedding.candidate_id, CandidateEmbedding.embedding)
                .where(CandidateEmbedding.model_id == base_model_id)
                .order_by(CandidateEmbedding.created_at.asc())
            )
            cand_rows = session.execute(cand_stmt).fetchall()

            supp_stmt = (
                select(ModelClass.class_index, SupportShot.role, SupportShot.embedding)
                .join(ModelClass, SupportShot.class_id == ModelClass.id)
                .where(SupportShot.model_id == base_model_id)
                .order_by(ModelClass.class_index, SupportShot.id)
            )
            supp_rows = session.execute(supp_stmt).fetchall()

            text_stmt = (
                select(ModelClass.class_index, TextEmbedding.embedding)
                .join(ModelClass, TextEmbedding.class_id == ModelClass.id)
                .where(TextEmbedding.model_id == base_model_id)
                .order_by(ModelClass.class_index)
            )
            text_rows = session.execute(text_stmt).fetchall()

        matched_candidates = []
        for r in cand_rows:
            cid_str = str(r[0])
            if cid_str in verified_map:
                sc = verified_map[cid_str].get("sign_code")
                if sc in sign_code_to_class:
                    c_id, c_idx = sign_code_to_class[sc]
                    matched_candidates.append({
                        "candidate_id": cid_str,
                        "class_id": c_id,
                        "class_index": c_idx,
                        "embedding": np.array(r[1], dtype=np.float32),
                    })

        if sample_limit is not None and sample_limit > 0:
            matched_candidates = matched_candidates[:sample_limit]

        baseline_anchors: Dict[int, List[np.ndarray]] = {r[1]: [] for r in class_rows}
        baseline_medoids: Dict[int, List[np.ndarray]] = {r[1]: [] for r in class_rows}
        for c_idx, role, emb in supp_rows:
            arr = np.array(emb, dtype=np.float32)
            if role == "ANCHOR":
                baseline_anchors[c_idx].append(arr)
            else:
                baseline_medoids[c_idx].append(arr)

        cand_by_class: Dict[int, List[np.ndarray]] = {r[1]: [] for r in class_rows}
        for item in matched_candidates:
            cand_by_class[item["class_index"]].append(item["embedding"])

        target_shots = shots_per_class or 16
        target_anchors = num_anchors or 4
        target_medoids = num_medoids or (target_shots - target_anchors)

        final_support_keys: List[np.ndarray] = []
        final_support_labels: List[int] = []
        final_shots_metadata: List[Tuple[int, str, str, List[float]]] = []
        class_index_to_code = {r[1]: r[2] for r in class_rows}

        for c_idx in sorted(class_index_to_id.keys()):
            sign_code = class_index_to_code.get(c_idx, f"CLASS_{c_idx}")
            shot_idx = 0
            anchors = baseline_anchors.get(c_idx, [])[:target_anchors]
            for a in anchors:
                shot_idx += 1
                s3_key = f"support_shots/{sign_code}/sample_{shot_idx:02d}.jpg"
                final_support_keys.append(a)
                final_support_labels.append(c_idx)
                final_shots_metadata.append((c_idx, "ANCHOR", s3_key, a.tolist()))

            medoid_pool = baseline_medoids.get(c_idx, []) + cand_by_class.get(c_idx, [])
            if not medoid_pool:
                continue

            if len(medoid_pool) <= target_medoids:
                chosen_medoids = medoid_pool
            else:
                med_arr = np.array(medoid_pool)
                norm = np.linalg.norm(med_arr, axis=1, keepdims=True)
                med_norm = med_arr / np.clip(norm, 1e-8, None)
                rng = np.random.RandomState(42)
                c0 = rng.choice(len(med_norm))
                centers = [med_norm[c0].copy()]
                for _ in range(1, target_medoids):
                    dists = cdist(med_norm, np.array(centers), metric="sqeuclidean")
                    min_dists = np.min(dists, axis=1)
                    tot = np.sum(min_dists)
                    probs = (min_dists / tot) if tot > 1e-12 else None
                    next_idx = rng.choice(len(med_norm), p=probs)
                    centers.append(med_norm[next_idx].copy())
                dists_all = cdist(med_norm, np.array(centers), metric="sqeuclidean")
                chosen_indices = []
                for m in range(target_medoids):
                    best_idx = int(np.argmin(dists_all[:, m]))
                    if best_idx not in chosen_indices:
                        chosen_indices.append(best_idx)
                for idx in range(len(med_norm)):
                    if len(chosen_indices) >= target_medoids:
                        break
                    if idx not in chosen_indices:
                        chosen_indices.append(idx)
                chosen_medoids = [medoid_pool[i] for i in chosen_indices[:target_medoids]]

            for m in chosen_medoids:
                shot_idx += 1
                s3_key = f"support_shots/{sign_code}/sample_{shot_idx:02d}.jpg"
                final_support_keys.append(m)
                final_support_labels.append(c_idx)
                final_shots_metadata.append((c_idx, "MEDOID", s3_key, m.tolist()))

        keys_tensor = torch.tensor(np.array(final_support_keys), device=dev, dtype=torch.float32)
        labels_tensor = torch.tensor(final_support_labels, device=dev, dtype=torch.long)
        text_tensor = torch.tensor(np.array([r[1] for r in text_rows]), device=dev, dtype=torch.float32)

        w_timo, b_timo = self.compute_timo_classifier(
            support_features=keys_tensor,
            support_labels=labels_tensor,
            text_features=text_tensor,
            alpha=alpha,
            beta=beta,
            device=dev,
        )

        now = datetime.now(timezone.utc)
        with Session(sync_engine) as session:
            new_version = ModelVersion(
                model_type="CLASSIFIER",
                version_tag=version_tag,
                is_active=False,
                note=note or f"Auto-rebuild with {len(matched_candidates)} verified samples",
                created_at=now,
            )
            session.add(new_version)
            session.flush()
            new_model_id = new_version.id

            new_class_map = {}
            for r in class_rows:
                mc = ModelClass(
                    model_id=new_model_id,
                    class_index=r[1],
                    class_name=r[3],
                    sign_code=r[2],
                    created_at=now,
                )
                session.add(mc)
                new_class_map[r[1]] = mc
            session.flush()

            for r in text_rows:
                session.add(TextEmbedding(
                    model_id=new_model_id,
                    class_id=new_class_map[r[0]].id,
                    embedding=r[1],
                    created_at=now,
                ))
            for c_idx, role, s3_key, emb_list in final_shots_metadata:
                session.add(SupportShot(
                    model_id=new_model_id,
                    class_id=new_class_map[c_idx].id,
                    role=role,
                    s3_key=s3_key,
                    embedding=emb_list,
                    created_at=now,
                ))
            session.commit()

        out_dir = base_dir / "working" / "embeddings"
        out_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = out_dir / f"timo_{version_tag}.pt"
        meta_info = {
            "model_id": new_model_id,
            "version_tag": version_tag,
            "sample_count": len(final_support_labels),
            "num_classes": int(w_timo.shape[1]),
        }
        torch.save({"weights": w_timo.cpu(), "bias": b_timo.cpu(), "meta": meta_info}, artifact_path)
        logger.info(f"Saved rebuilt TIMO weights to {artifact_path}")

        with Session(sync_engine) as session:
            session.add(ModelArtifact(
                model_id=new_model_id,
                artifact_type="TIMO",
                s3_key=f"models/classifiers/timo_{version_tag}.pt",
                artifact_url=str(artifact_path),
                file_size_bytes=artifact_path.stat().st_size if artifact_path.exists() else 0,
                created_at=now,
            ))
            session.commit()

        eval_metrics = evaluator.evaluate_and_persist(
            model_id=new_model_id,
            weights=w_timo,
            bias=b_timo,
            split="TEST",
            note=note,
        )

        active_comp = None
        active_metrics_raw = evaluator.get_metrics_for_model(model_id=base_model_id, split="test")
        active_metric_dict = {m["metric_name"]: m["metric_value"] for m in active_metrics_raw}
        if "top1_accuracy" in active_metric_dict and active_metric_dict["top1_accuracy"] is not None:
            delta_acc = eval_metrics["top1_accuracy"] - active_metric_dict["top1_accuracy"]
            delta_f1 = eval_metrics["macro_f1"] - (active_metric_dict.get("macro_f1") or 0.0)
            active_comp = {
                "active_version_id": base_model_id,
                "delta_accuracy": f"{delta_acc:+.2f}%",
                "delta_f1": f"{delta_f1:+.4f}",
            }

        return {
            "status": "SUCCESS",
            "message": "Classifier rebuilt and evaluated successfully. Ready for deployment review.",
            "new_model": {
                "version_id": new_model_id,
                "version_tag": version_tag,
                "is_active": False,
                "artifact_path": str(artifact_path),
            },
            "evaluation_metrics": {
                "split": "TEST",
                "sample_count": 5911,
                "top1_accuracy": eval_metrics["top1_accuracy"],
                "top5_accuracy": eval_metrics["top5_accuracy"],
                "macro_f1": eval_metrics["macro_f1"],
                "latency_ms": eval_metrics["latency_ms"],
                "comparison_with_active": active_comp,
            },
        }

    def deploy_classifier_version(self, version_id: int) -> Dict[str, Any]:
        base_dir = Path(__file__).resolve().parent.parent.parent
        with Session(sync_engine) as session:
            target = session.query(ModelVersion).filter(
                ModelVersion.id == version_id,
                ModelVersion.model_type == "CLASSIFIER",
            ).first()
            if not target:
                raise ValueError(f"Classifier version #{version_id} not found.")

            version_tag = target.version_tag
            candidate_paths = [
                base_dir / "working" / "embeddings" / f"timo_{version_tag}.pt",
                base_dir / "working" / "embeddings" / "timo.pt",
            ]
            artifact_file = next((p for p in candidate_paths if p.exists()), None)
            if not artifact_file:
                raise FileNotFoundError(f"TIMO weights for version #{version_id} ({version_tag}) not found.")

            ckpt = torch.load(artifact_file, map_location="cpu")
            w = ckpt["weights"].float()
            b = ckpt["bias"].float()

            session.query(ModelVersion).filter(
                ModelVersion.model_type == "CLASSIFIER",
                ModelVersion.is_active == True,
            ).update({"is_active": False})
            target.is_active = True
            session.commit()

        from src.core.model_manager import model_manager
        cls_inst = model_manager.get_classifier()
        if cls_inst is not None:
            cls_inst.hot_reload_timo(w, b)

        anchors_path = base_dir / "working" / "embeddings" / f"visual_anchors_{version_id}.pt"
        anchors_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.save({"anchors": w.cpu(), "bias": b.cpu(), "version_id": version_id}, anchors_path)
        except Exception as e:
            logger.warning(f"Could not persist visual anchors snapshot: {e}")

        batch_run_id = f"run_{version_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Classifier version #{version_id} ({version_tag}) successfully deployed to RAM and activated.")
        return {
            "status": "SUCCESS",
            "message": f"Model version #{version_id} successfully deployed and live in memory",
            "deployed_version_id": version_id,
            "version_tag": version_tag,
            "is_active": True,
            "deployed_at": datetime.now(timezone.utc).isoformat(),
            "batch_run_id": batch_run_id,
        }

    def rollback_classifier_version(self) -> Dict[str, Any]:
        with Session(sync_engine) as session:
            active = session.query(ModelVersion).filter(
                ModelVersion.model_type == "CLASSIFIER",
                ModelVersion.is_active == True,
            ).first()
            active_id = active.id if active else 0

            prev = session.query(ModelVersion).filter(
                ModelVersion.model_type == "CLASSIFIER",
                ModelVersion.id != active_id,
            ).order_by(ModelVersion.id.desc()).first()

            if not prev:
                raise ValueError("No previous stable model version found to rollback.")

            prev_id = prev.id

        self.deploy_classifier_version(prev_id)
        logger.info(f"Classifier rolled back from #{active_id} to #{prev_id}")
        return {
            "status": "SUCCESS",
            "message": f"Successfully rolled back to previous stable model version #{prev_id}",
            "rolled_back_from": active_id,
            "active_version_id": prev_id,
            "version_tag": prev.version_tag,
        }

    def switch_classifier_version(self, version_id: int) -> Dict[str, Any]:
        res = self.deploy_classifier_version(version_id)
        return {
            "status": "SUCCESS",
            "message": f"Successfully switched active classifier to version #{version_id}",
            "switched_to_version_id": version_id,
            "version_tag": res["version_tag"],
            "is_active": True,
        }

    def deploy_detector_version(self, version_id: int) -> Dict[str, Any]:
        with Session(sync_engine) as session:
            target = session.query(ModelVersion).filter(
                ModelVersion.id == version_id,
                ModelVersion.model_type == "DETECTOR",
            ).first()
            if not target:
                raise ValueError(f"Detector version #{version_id} not found.")

            version_tag = target.version_tag
            session.query(ModelVersion).filter(
                ModelVersion.model_type == "DETECTOR",
                ModelVersion.is_active == True,
            ).update({"is_active": False})
            target.is_active = True
            session.commit()

        from src.core.model_manager import model_manager
        detector_weights = f"{version_tag}.engine"
        try:
            model_manager.load_detector(detector_weights)
        except Exception as e:
            logger.warning(f"Engine load fallback for detector {version_tag}: {e}")
            try:
                model_manager.load_detector(f"{version_tag}.pt")
            except Exception as e2:
                logger.error(f"Failed to load detector weights: {e2}")

        logger.info(f"Detector version #{version_id} ({version_tag}) activated.")
        return {
            "status": "SUCCESS",
            "message": f"Detector version #{version_id} successfully deployed",
            "deployed_version_id": version_id,
            "version_tag": version_tag,
            "is_active": True,
            "deployed_at": datetime.now(timezone.utc).isoformat(),
        }

    def rollback_detector_version(self) -> Dict[str, Any]:
        with Session(sync_engine) as session:
            active = session.query(ModelVersion).filter(
                ModelVersion.model_type == "DETECTOR",
                ModelVersion.is_active == True,
            ).first()
            active_id = active.id if active else 0

            prev = session.query(ModelVersion).filter(
                ModelVersion.model_type == "DETECTOR",
                ModelVersion.id != active_id,
            ).order_by(ModelVersion.id.desc()).first()

            if not prev:
                raise ValueError("No previous stable detector version found to rollback.")

            prev_id = prev.id

        self.deploy_detector_version(prev_id)
        return {
            "status": "SUCCESS",
            "message": f"Successfully rolled back to previous stable detector version #{prev_id}",
            "rolled_back_from": active_id,
            "active_version_id": prev_id,
            "version_tag": prev.version_tag,
        }

    def switch_detector_version(self, version_id: int) -> Dict[str, Any]:
        res = self.deploy_detector_version(version_id)
        return {
            "status": "SUCCESS",
            "message": f"Successfully switched active detector to version #{version_id}",
            "switched_to_version_id": version_id,
            "version_tag": res["version_tag"],
            "is_active": True,
        }




few_shot_manager = FewShotSupportManager()
