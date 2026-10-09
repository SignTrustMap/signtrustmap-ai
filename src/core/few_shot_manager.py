from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from loguru import logger
from scipy.spatial.distance import cdist

from src.core.config_manager import ConfigManager

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
            "total_shots_per_class": int(cfg.get("total_shots_per_class", 16)),
            "num_anchors": int(cfg.get("num_anchors", 4)),
            "num_medoids": int(cfg.get("num_medoids", 12)),
            "min_crop_resolution": int(cfg.get("min_crop_resolution", 20)),
            "min_sharpness_laplacian": float(cfg.get("min_sharpness_laplacian", 15.0)),
            "replacement_policy": str(cfg.get("replacement_policy", "diversity_kmeans")),
            "preserve_pinned_shots": bool(cfg.get("preserve_pinned_shots", True)),
        }

    def get_timo_config(self) -> Dict[str, Any]:
        cfg = self.config.get("timo", {})
        return {
            "alpha": float(cfg.get("alpha", 250.0)),
            "beta": float(cfg.get("beta", 0.8)),
            "prompt_template": str(cfg.get("prompt_template", "a photo of a {} traffic sign.")),
            "normalize_embeddings": bool(cfg.get("normalize_embeddings", True)),
            "classifier_cache_file": str(cfg.get("classifier_cache_file", "weights/classifiers/timo_classifier_100.pt")),
        }

    def select_support_shots(
        self,
        candidates: List[Dict[str, Any]],
        visual_features: Optional[np.ndarray] = None,
        total_shots: Optional[int] = None,
        num_anchors: Optional[int] = None,
        random_state: int = 42,
    ) -> List[Dict[str, Any]]:
        """
        Executes Configurable Support Shot Selection & Replacement Policy:
        1. Preserves pinned shots (is_pinned=True) guaranteed by Admin/Staff.
        2. Top-K Representative Anchors selected by Area x Sharpness.
        3. Diverse Medoids selected by K-Means++ clustering on visual embeddings.

        Args:
            candidates: List of candidate dictionaries.
            visual_features: Optional (N, D) array of visual embeddings corresponding to candidates.
            total_shots: Maximum number of shots to select per class (defaults to config: 16).
            num_anchors: Number of anchor slots to reserve (defaults to config: 4).
            random_state: Random seed for K-Means clustering reproducibility.

        Returns:
            List of selected candidate dicts (up to total_shots, with role 'ANCHOR' or 'MEDOID').
        """
        cfg = self.get_support_config()
        total_budget = int(total_shots) if total_shots is not None else cfg["total_shots_per_class"]
        n_anchors = int(num_anchors) if num_anchors is not None else cfg["num_anchors"]
        preserve_pinned = cfg["preserve_pinned_shots"]

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

    def build_and_save_timo_classifier(
        self,
        support_cache_file: Optional[str] = None,
        text_embeddings_file: Optional[str] = None,
        output_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> Path:
        base_dir = Path(__file__).resolve().parent.parent.parent
        cfg = self.get_timo_config()

        dev = torch.device(device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu"))

        supp_path = Path(support_cache_file) if support_cache_file else (base_dir / "outputs" / "evaluation" / "tip_cache.pt")
        text_path = Path(text_embeddings_file) if text_embeddings_file else (base_dir / "weights" / "embeddings" / "text_embeddings_mobileclip_s2.pt")
        out_path = Path(output_path) if output_path else (base_dir / cfg["classifier_cache_file"])

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
        torch.save({"weights": w_timo.cpu(), "bias": b_timo.cpu()}, out_path)
        logger.info(f"TIMO Collapsed Classifier weights saved to {out_path} (weights shape: {list(w_timo.shape)}, bias shape: {list(b_timo.shape)})")
        return out_path


few_shot_manager = FewShotSupportManager()
