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
    Manages Configurable Few-Shot Support Pools and ProtoFuse Prototype calculation
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

    def get_protofuse_config(self) -> Dict[str, Any]:
        cfg = self.config.get("protofuse", {})
        return {
            "fusion_weight_alpha": float(cfg.get("fusion_weight_alpha", 0.5)),
            "prompt_template": str(cfg.get("prompt_template", "a photo of a {} traffic sign.")),
            "normalize_embeddings": bool(cfg.get("normalize_embeddings", True)),
            "prototypes_cache_file": str(cfg.get("prototypes_cache_file", "weights/classifiers/protofuse_prototypes_100.pt")),
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

    def compute_fused_prototype(
        self,
        visual_features: np.ndarray,
        text_feature: np.ndarray,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """
        Computes ProtoFuse fused prototype vector for a single class:
        v_c = mean(phi(x_i)) / ||mean(phi(x_i))||
        t_c = t_c / ||t_c||
        p_c = alpha * v_c + (1 - alpha) * t_c
        p_c = p_c / ||p_c||

        Args:
            visual_features: (M, D) array of visual embeddings from support shots.
            text_feature: (D,) array representing prompt embedding.
            alpha: Fusion weight. Defaults to config value (0.5).

        Returns:
            (D,) L2-normalized fused prototype vector.
        """
        cfg = self.get_protofuse_config()
        a = cfg["fusion_weight_alpha"] if alpha is None else alpha

        # Text prototype
        t_norm = np.linalg.norm(text_feature)
        t_c = text_feature / max(t_norm, 1e-8)

        if visual_features is None or len(visual_features) == 0:
            return t_c

        # Visual prototype
        v_mean = np.mean(visual_features, axis=0)
        v_norm = np.linalg.norm(v_mean)
        v_c = v_mean / max(v_norm, 1e-8)

        # ProtoFuse weighted fusion
        p_c = a * v_c + (1.0 - a) * t_c
        p_c_norm = np.linalg.norm(p_c)
        p_c = p_c / max(p_c_norm, 1e-8)
        return p_c


few_shot_manager = FewShotSupportManager()
