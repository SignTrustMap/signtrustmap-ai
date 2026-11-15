import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.core.active_learning.base import BaseActiveLearningScorer
from src.core.few_shot_manager import few_shot_manager
from src.infrastructure.db import sync_engine
from src.infrastructure.models import ModelClass, SupportShot


class FDALScorer(BaseActiveLearningScorer):
    name: str = "fdal"

    def compute_box_score(self, detection: Dict[str, Any]) -> float:
        if "fdal_priority" in detection:
            return float(detection["fdal_priority"])
        conf = float(detection.get("confidence", 0.0))
        return max(0.0, min(1.0, 1.0 - conf))


class FDALSelector:
    def __init__(
        self,
        eta: float = 0.03125,
        learning_rate: float = 0.1,
        optimization_steps: int = 10,
        lambda_hyp: float = 0.99,
        margin: float = 0.01,
        device: Optional[Union[str, torch.device]] = None,
        random_seed: int = 42,
    ) -> None:
        self.eta = float(eta)
        self.learning_rate = float(learning_rate)
        self.optimization_steps = int(optimization_steps)
        self.lambda_hyp = float(lambda_hyp)
        self.margin = float(margin)
        self.random_seed = int(random_seed)
        self.device = torch.device(device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self._anchors: Optional[torch.Tensor] = None

    def get_anchor_cache_path(self, model_id: int = 2) -> Path:
        base_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
        cache_dir = base_dir / "working" / "embeddings"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"visual_anchors_{model_id}.pt"

    def is_anchor_cache_valid(self, cache_path: Path, model_id: int = 2) -> bool:
        if not cache_path.exists():
            return False
        try:
            ckpt = torch.load(cache_path, map_location="cpu")
            if not isinstance(ckpt, dict) or "anchors" not in ckpt:
                return False
            meta = ckpt.get("meta", {})
            db_meta = few_shot_manager.get_support_shots_metadata(model_id=model_id)
            if db_meta["sample_count"] > 0:
                if meta.get("sample_count") != db_meta["sample_count"] or meta.get("max_id") != db_meta["max_id"]:
                    logger.info("Visual anchor cache differs from database support shots. Invalidation triggered.")
                    return False
            return True
        except Exception as e:
            logger.warning(f"Error checking visual anchor cache validity: {e}")
            return False

    def load_class_anchors(self, model_id: int = 2, force_rebuild: bool = False) -> torch.Tensor:
        if self._anchors is not None and not force_rebuild:
            return self._anchors.to(self.device)

        cache_path = self.get_anchor_cache_path(model_id=model_id)
        if not force_rebuild and self.is_anchor_cache_valid(cache_path, model_id=model_id):
            try:
                ckpt = torch.load(cache_path, map_location=self.device)
                self._anchors = ckpt["anchors"].to(self.device, dtype=torch.float32)
                logger.info(f"Loaded visual class anchors from cache: {cache_path} (shape={list(self._anchors.shape)})")
                return self._anchors
            except Exception as e:
                logger.warning(f"Failed to load visual anchors from cache {cache_path}: {e}")

        logger.info(f"Computing visual class anchors from database support shots (model_id={model_id})...")
        keys, labels = None, None
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
                    keys = torch.tensor(np.array(embs), device=self.device, dtype=torch.float32)
                    labels = torch.tensor(c_indices, dtype=torch.long, device=self.device)
        except Exception as e:
            logger.warning(f"Could not load support shots from database for anchors: {e}")

        if keys is None or labels is None:
            base_dir = Path(__file__).resolve().parent.parent.parent.parent
            working_supp = base_dir / "working" / "embeddings" / "support_cache_mobileclip_s2.pt"
            eval_supp = base_dir / "outputs" / "evaluation" / "tip_cache.pt"
            supp_path = working_supp if working_supp.exists() else eval_supp
            if not supp_path.exists():
                raise FileNotFoundError(f"No support data available to build anchors at: {supp_path}")
            supp_data = torch.load(supp_path, map_location=self.device)
            keys = supp_data["keys"] if isinstance(supp_data, dict) else supp_data
            labels = supp_data["labels"] if isinstance(supp_data, dict) else torch.zeros(len(keys), dtype=torch.long, device=self.device)

        unique_classes = torch.unique(labels).sort().values
        num_classes = int(unique_classes.max().item()) + 1 if len(unique_classes) > 0 else 100
        dim = int(keys.shape[1])
        anchors = torch.zeros((num_classes, dim), dtype=torch.float32, device=self.device)

        for c in range(num_classes):
            mask = labels == c
            if mask.any():
                mean_vec = keys[mask].mean(dim=0)
                norm = torch.clamp(mean_vec.norm(dim=-1, keepdim=True), min=1e-8)
                anchors[c] = mean_vec / norm
            else:
                logger.warning(f"Class {c} has no support shots for visual anchor computation.")

        meta_info = few_shot_manager.get_support_shots_metadata(model_id=model_id)
        meta_info["num_classes"] = num_classes
        torch.save({"anchors": anchors.cpu(), "meta": meta_info}, cache_path)
        logger.info(f"Saved visual class anchors to working cache: {cache_path} (shape={list(anchors.shape)})")

        self._anchors = anchors
        return self._anchors

    def optimize_perturbation_and_count_flips(
        self,
        visual_embedding: Union[np.ndarray, torch.Tensor],
        original_predicted_class: int,
        timo_weights: torch.Tensor,
        timo_bias: torch.Tensor,
        anchors: Optional[torch.Tensor] = None,
        eta: Optional[float] = None,
    ) -> Tuple[int, List[int]]:
        eff_eta = float(eta if eta is not None else self.eta)
        class_anchors = anchors if anchors is not None else self.load_class_anchors()
        class_anchors = class_anchors.to(device=self.device, dtype=torch.float32)

        w = timo_weights.to(device=self.device, dtype=torch.float32)
        b = timo_bias.to(device=self.device, dtype=torch.float32)

        if isinstance(visual_embedding, (np.ndarray, list)):
            z = torch.as_tensor(visual_embedding, dtype=torch.float32, device=self.device)
        else:
            z = visual_embedding.to(device=self.device, dtype=torch.float32)

        if z.ndim == 1:
            z = z.unsqueeze(0)
        z = z / torch.clamp(z.norm(dim=-1, keepdim=True), min=1e-8)

        num_classes = class_anchors.shape[0]
        dim = class_anchors.shape[1]

        rng = torch.Generator(device=self.device)
        rng.manual_seed(self.random_seed)
        alpha_init = torch.normal(
            mean=eff_eta / 2.0,
            std=eff_eta / 4.0,
            size=(num_classes, dim),
            generator=rng,
            device=self.device,
        ).clamp_(1e-8, eff_eta)

        alpha = torch.nn.Parameter(alpha_init)
        optimizer = torch.optim.Adam([alpha], lr=self.learning_rate)

        z_rep = z.expand(num_classes, dim)
        orig_cls = int(original_predicted_class)

        for _ in range(self.optimization_steps):
            optimizer.zero_grad()
            mixed = (1.0 - alpha) * z_rep + alpha * class_anchors
            mixed_norm = mixed / torch.clamp(mixed.norm(dim=-1, keepdim=True), min=1e-8)
            logits = mixed_norm @ w + b

            pred_logits = logits[:, orig_cls]
            other_logits = logits.clone()
            other_logits[:, orig_cls] = -float("inf")
            max_other, _ = other_logits.max(dim=-1)

            margin_loss = torch.clamp(pred_logits - max_other + self.margin, min=0.0).mean()
            reg_loss = (alpha ** 2).mean()
            loss = self.lambda_hyp * margin_loss + (1.0 - self.lambda_hyp) * reg_loss

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                alpha.clamp_(1e-8, eff_eta)

        with torch.no_grad():
            final_mixed = (1.0 - alpha) * z_rep + alpha * class_anchors
            final_norm = final_mixed / torch.clamp(final_mixed.norm(dim=-1, keepdim=True), min=1e-8)
            final_logits = final_norm @ w + b
            final_preds = final_logits.argmax(dim=-1)

            flip_mask = final_preds != orig_cls
            flip_count = int(flip_mask.sum().item())
            flipped_classes = flip_mask.nonzero(as_tuple=True)[0].cpu().tolist()

        del alpha, optimizer, z_rep, final_mixed, final_norm, final_logits, final_preds, flip_mask
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return flip_count, flipped_classes

    def compute_calibrated_entropy(
        self,
        probabilities: Union[np.ndarray, torch.Tensor],
        reference_prior: Optional[Union[np.ndarray, torch.Tensor]] = None,
        eps: float = 1e-8,
    ) -> np.ndarray:
        if isinstance(probabilities, torch.Tensor):
            probs = probabilities.detach().cpu().numpy()
        else:
            probs = np.array(probabilities, dtype=np.float64)

        if probs.ndim == 1:
            probs = probs[None, :]

        n_samples, n_classes = probs.shape
        if n_samples == 0:
            return np.array([], dtype=np.float64)

        if reference_prior is not None:
            p0 = np.array(reference_prior, dtype=np.float64)
            if isinstance(reference_prior, torch.Tensor):
                p0 = reference_prior.detach().cpu().numpy()
        else:
            p0 = np.mean(probs, axis=0) if n_samples > 1 else np.ones(n_classes, dtype=np.float64) / float(n_classes)

        p0 = np.clip(p0, eps, 1.0)
        calibrated = probs / p0[None, :]
        calibrated /= np.sum(calibrated, axis=-1, keepdims=True)
        entropy = -np.sum(calibrated * np.log(np.clip(calibrated, eps, 1.0)), axis=-1)
        return entropy

    @staticmethod
    def compute_scalar_priority(flip_count: int, fallback_entropy: float, num_classes: int = 100) -> float:
        c = max(2, int(num_classes))
        s = int(flip_count)
        e = float(fallback_entropy)
        if s > 0:
            return round(float(s) / float(c), 6)
        denom = 2.0 * float(c) * math.log(float(c))
        return round(float(e) / max(1e-8, denom), 6)

    def score_candidates_batch(
        self,
        visual_embeddings: List[Union[np.ndarray, torch.Tensor]],
        predicted_classes: List[int],
        class_probabilities: List[Union[np.ndarray, torch.Tensor]],
        timo_weights: torch.Tensor,
        timo_bias: torch.Tensor,
        anchors: Optional[torch.Tensor] = None,
        eta: Optional[float] = None,
        reference_prior: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> List[Dict[str, Any]]:
        n_candidates = len(visual_embeddings)
        if n_candidates == 0:
            return []

        eff_eta = float(eta if eta is not None else self.eta)
        class_anchors = anchors if anchors is not None else self.load_class_anchors()
        num_classes = int(class_anchors.shape[0])

        probs_arr = np.array([p if isinstance(p, np.ndarray) else p.detach().cpu().numpy() for p in class_probabilities])
        entropies = self.compute_calibrated_entropy(probs_arr, reference_prior=reference_prior)

        results: List[Dict[str, Any]] = []

        for idx in range(n_candidates):
            v_emb = visual_embeddings[idx]
            pred_cls = int(predicted_classes[idx])
            e_fallback = float(entropies[idx])

            if v_emb is None:
                results.append({
                    "candidate_index": idx,
                    "flip_count": 0,
                    "fallback_entropy": e_fallback,
                    "fdal_tuple": (0, e_fallback),
                    "fdal_priority": self.compute_scalar_priority(0, e_fallback, num_classes=num_classes),
                    "flipped_anchor_classes": [],
                })
                continue

            s_flip, flipped_anchors = self.optimize_perturbation_and_count_flips(
                visual_embedding=v_emb,
                original_predicted_class=pred_cls,
                timo_weights=timo_weights,
                timo_bias=timo_bias,
                anchors=class_anchors,
                eta=eff_eta,
            )

            scalar_p = self.compute_scalar_priority(s_flip, e_fallback, num_classes=num_classes)

            results.append({
                "candidate_index": idx,
                "flip_count": s_flip,
                "fallback_entropy": e_fallback,
                "fdal_tuple": (s_flip, e_fallback),
                "fdal_priority": scalar_p,
                "flipped_anchor_classes": flipped_anchors,
            })

        logger.debug(
            f"FDAL scored {n_candidates} candidate(s): "
            f"mean(flip_count)={np.mean([r['flip_count'] for r in results]):.2f}, "
            f"mean(fallback_entropy)={np.mean([r['fallback_entropy'] for r in results]):.4f}"
        )
        return results

    def rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        timo_weights: torch.Tensor,
        timo_bias: torch.Tensor,
        anchors: Optional[torch.Tensor] = None,
        eta: Optional[float] = None,
        model_version: str = "v1.0",
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        embeddings = []
        preds = []
        probs = []

        for c in candidates:
            emb = c.get("visual_embedding")
            embeddings.append(emb)

            pred_cid = c.get("class_id", 0)
            preds.append(int(pred_cid) if pred_cid is not None else 0)

            p = c.get("probabilities")
            if p is not None:
                probs.append(p)
            else:
                conf = float(c.get("confidence", 0.5))
                num_c = int(timo_bias.shape[0])
                syn_p = np.full(num_c, (1.0 - conf) / max(1, num_c - 1), dtype=np.float32)
                if 0 <= pred_cid < num_c:
                    syn_p[pred_cid] = conf
                probs.append(syn_p)

        scores = self.score_candidates_batch(
            visual_embeddings=embeddings,
            predicted_classes=preds,
            class_probabilities=probs,
            timo_weights=timo_weights,
            timo_bias=timo_bias,
            anchors=anchors,
            eta=eta,
        )

        ranked = []
        for idx, score_info in enumerate(scores):
            item = dict(candidates[idx])
            item["flip_count"] = score_info["flip_count"]
            item["fallback_entropy"] = score_info["fallback_entropy"]
            item["fdal_tuple"] = score_info["fdal_tuple"]
            item["fdal_priority"] = score_info["fdal_priority"]
            item["model_version"] = model_version
            item["score"] = score_info["fdal_priority"]
            ranked.append(item)

        ranked.sort(key=lambda x: (x["flip_count"], x["fallback_entropy"]), reverse=True)
        return ranked

    def select_query_batch(
        self,
        candidates: List[Dict[str, Any]],
        query_budget: int = 16,
        timo_weights: Optional[torch.Tensor] = None,
        timo_bias: Optional[torch.Tensor] = None,
        anchors: Optional[torch.Tensor] = None,
        eta: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        if timo_weights is None or timo_bias is None:
            raise ValueError("timo_weights and timo_bias must be provided to select_query_batch.")

        ranked = self.rank_candidates(
            candidates=candidates,
            timo_weights=timo_weights,
            timo_bias=timo_bias,
            anchors=anchors,
            eta=eta,
        )
        k = min(query_budget, len(ranked))
        logger.info(f"FDAL selected top {k} query candidates from pool of {len(candidates)}.")
        return ranked[:k]
