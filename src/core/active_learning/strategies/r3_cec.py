from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from loguru import logger
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from src.core.active_learning.base import BaseActiveLearningScorer
from src.core.config_manager import config_manager


class R3CECScorer(BaseActiveLearningScorer):
    name: str = "r3_cec"

    def compute_box_score(self, detection: Dict[str, Any]) -> float:
        if "acquisition_score" in detection:
            return float(detection["acquisition_score"])
        conf = float(detection.get("confidence", 0.0))
        return max(0.0, min(1.0, 1.0 - conf))


class R3CECSelector:
    def __init__(
        self,
        k_neighbors: Optional[int] = None,
        temperature: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> None:
        cfg_k = int(config_manager.get("active_learning.r3_cec.k_neighbors", 5))
        cfg_tau = float(config_manager.get("active_learning.r3_cec.temperature", 0.5))
        cfg_seed = int(config_manager.get("active_learning.r3_cec.random_state", 42))

        self.k_neighbors = int(k_neighbors if k_neighbors is not None else cfg_k)
        self.temperature = float(temperature if temperature is not None else cfg_tau)
        self.random_state = int(random_state if random_state is not None else cfg_seed)

    def calibrate_probabilities(
        self,
        probabilities: np.ndarray,
        reference_prior: Optional[np.ndarray] = None,
        top_n: Optional[int] = None,
        eps: float = 1e-8,
    ) -> Tuple[np.ndarray, np.ndarray]:
        probs = np.clip(probabilities, eps, 1.0)
        n_samples, n_classes = probs.shape

        if reference_prior is not None:
            p0 = np.asarray(reference_prior, dtype=np.float64)
            if p0.shape[0] != n_classes:
                raise ValueError(f"reference_prior length ({p0.shape[0]}) does not match n_classes ({n_classes})")
        else:
            if n_samples == 0:
                return probs, np.ones(n_classes, dtype=np.float64) / max(1, n_classes)

            configured_top_n = top_n if top_n is not None else int(config_manager.get("active_learning.cec.top_n_prior", 5))
            eff_top_n = max(1, min(int(configured_top_n), n_samples))
            p0 = np.zeros(n_classes, dtype=np.float64)

            for c in range(n_classes):
                col = probs[:, c]
                top_indices = np.argpartition(col, -eff_top_n)[-eff_top_n:]
                p0[c] = np.mean(col[top_indices])

        p0 = np.clip(p0, eps, 1.0)
        calibrated = probs / p0[None, :]
        denom = np.sum(calibrated, axis=1, keepdims=True)
        calibrated /= np.clip(denom, eps, None)
        return calibrated, p0

    def compute_calibrated_uncertainty(
        self,
        probabilities: np.ndarray,
        visual_features: np.ndarray,
        k_neighbors: Optional[int] = None,
        temperature: Optional[float] = None,
        apply_calibration: bool = True,
        reference_prior: Optional[np.ndarray] = None,
        top_n: Optional[int] = None,
        eps: float = 1e-8,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_samples = probabilities.shape[0]
        if n_samples == 0:
            return np.array([]), np.array([]), np.array([])

        k = self.k_neighbors if k_neighbors is None else k_neighbors
        tau = self.temperature if temperature is None else temperature

        if apply_calibration and n_samples > 1:
            p_hat, _ = self.calibrate_probabilities(
                probabilities,
                reference_prior=reference_prior,
                top_n=top_n,
                eps=eps,
            )
        else:
            p_hat = np.clip(probabilities, eps, 1.0)
            p_hat /= np.sum(p_hat, axis=1, keepdims=True)

        h_hat = -np.sum(p_hat * np.log(np.clip(p_hat, eps, 1.0)), axis=1)

        if n_samples <= 1 or k <= 0:
            h_nn = np.copy(h_hat)
            u = h_hat + h_nn
            return h_hat, h_nn, u

        norm = np.linalg.norm(visual_features, axis=1, keepdims=True)
        z = visual_features / np.clip(norm, eps, None)

        k_eff = min(k, n_samples - 1)
        dist_sq = cdist(z, z, metric="sqeuclidean")
        np.fill_diagonal(dist_sq, np.inf)

        nn_indices = np.argpartition(dist_sq, kth=k_eff - 1, axis=1)[:, :k_eff]
        row_indices = np.arange(n_samples)[:, None]
        nn_dists = dist_sq[row_indices, nn_indices]
        nn_h = h_hat[nn_indices]

        logits = -nn_dists / max(tau, eps)
        logits -= np.max(logits, axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= np.sum(weights, axis=1, keepdims=True)

        h_nn = np.sum(weights * nn_h, axis=1)
        u = h_hat + h_nn

        logger.debug(
            f"Computed calibrated uncertainty for {n_samples} samples: "
            f"mean(H_hat)={float(np.mean(h_hat)):.4f}, "
            f"mean(H_NN)={float(np.mean(h_nn)):.4f}, "
            f"mean(U)={float(np.mean(u)):.4f}"
        )
        return h_hat, h_nn, u

    def compute_residual_priority(
        self,
        predicted_classes: np.ndarray,
        labeled_counts: Union[np.ndarray, List[int], Dict[int, int]],
        n_classes: Optional[int] = None,
    ) -> np.ndarray:
        n_samples = len(predicted_classes)
        if n_samples == 0:
            return np.array([])

        if labeled_counts is None:
            return np.ones(n_samples, dtype=np.float64)

        if isinstance(labeled_counts, dict):
            max_c = max(
                int(np.max(predicted_classes)) if len(predicted_classes) > 0 else 0,
                max(int(k) for k in labeled_counts.keys()) if labeled_counts else 0,
                (n_classes - 1) if n_classes else 0,
            )
            counts_arr = np.zeros(max_c + 1, dtype=np.float64)
            for c_idx, count in labeled_counts.items():
                if 0 <= int(c_idx) <= max_c:
                    counts_arr[int(c_idx)] = float(count)
        else:
            counts_arr = np.asarray(labeled_counts, dtype=np.float64)

        max_idx = len(counts_arr) - 1
        safe_classes = np.clip(predicted_classes, 0, max_idx)
        counts = counts_arr[safe_classes]
        rho = 1.0 / (counts + 1.0)
        return rho

    def weighted_kmeans_clustering(
        self,
        visual_features: np.ndarray,
        sample_weights: np.ndarray,
        n_clusters: int,
        max_iter: int = 100,
        tol: float = 1e-4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_samples, d = visual_features.shape
        k = min(n_clusters, n_samples)
        if k <= 0:
            return np.zeros(0, dtype=int), np.zeros((0, d))

        if n_samples <= k:
            labels = np.arange(n_samples, dtype=int)
            centers = visual_features.copy()
            return labels, centers

        rng = np.random.RandomState(self.random_state)
        w = np.asarray(sample_weights, dtype=np.float64)
        w = np.clip(w, 1e-8, None)
        w_norm = w / np.sum(w)

        first_center_idx = rng.choice(n_samples, p=w_norm)
        centers = [visual_features[first_center_idx].copy()]

        for _ in range(1, k):
            current_centers = np.array(centers)
            dists_to_centers = cdist(visual_features, current_centers, metric="sqeuclidean")
            min_dist_sq = np.min(dists_to_centers, axis=1)

            candidate_probs = w * min_dist_sq
            total_prob = np.sum(candidate_probs)
            if total_prob > 1e-12:
                candidate_probs /= total_prob
            else:
                candidate_probs = w_norm

            next_center_idx = rng.choice(n_samples, p=candidate_probs)
            centers.append(visual_features[next_center_idx].copy())

        centers = np.array(centers, dtype=np.float64)
        labels = np.zeros(n_samples, dtype=int)

        for iteration in range(max_iter):
            dists = cdist(visual_features, centers, metric="sqeuclidean")
            new_labels = np.argmin(dists, axis=1)

            new_centers = np.empty_like(centers)
            for m in range(k):
                mask = (new_labels == m)
                if not np.any(mask):
                    worst_idx = int(np.argmax(w * np.min(dists, axis=1)))
                    new_centers[m] = visual_features[worst_idx].copy()
                else:
                    cluster_w = w[mask]
                    cluster_z = visual_features[mask]
                    weight_sum = np.sum(cluster_w)
                    if weight_sum > 1e-12:
                        new_centers[m] = np.sum(cluster_w[:, None] * cluster_z, axis=0) / weight_sum
                    else:
                        new_centers[m] = np.mean(cluster_z, axis=0)

            shift = float(np.linalg.norm(new_centers - centers))
            centers = new_centers
            labels = new_labels

            if shift < tol:
                logger.debug(f"Weighted K-Means converged in {iteration + 1} iterations (shift={shift:.6f}).")
                break

        return labels, centers

    def tri_rank_aggregation(
        self,
        cluster_indices: np.ndarray,
        residual_priority: np.ndarray,
        uncertainty: np.ndarray,
        visual_features: np.ndarray,
        centroid: np.ndarray,
        return_details: bool = False,
    ) -> Union[int, Tuple[int, Dict[int, Dict[str, Any]]]]:
        n_members = len(cluster_indices)
        member_rank_details: Dict[int, Dict[str, Any]] = {}

        if n_members == 1:
            idx = int(cluster_indices[0])
            dist_val = float(np.sum((visual_features[idx] - centroid) ** 2))
            member_rank_details[idx] = {
                "tri_rank_res": 1,
                "tri_rank_info": 1,
                "tri_rank_rep": 1,
                "aggregate_rank": 3,
                "distance_to_centroid": dist_val,
            }
            if return_details:
                return idx, member_rank_details
            return idx

        sub_rho = residual_priority[cluster_indices]
        sub_u = uncertainty[cluster_indices]
        sub_z = visual_features[cluster_indices]

        dists = np.sum((sub_z - centroid[None, :]) ** 2, axis=1)

        r_res = rankdata(-sub_rho, method="min")
        r_info = rankdata(-sub_u, method="min")
        r_rep = rankdata(dists, method="min")
        a_m = r_res + r_info + r_rep

        for local_i, global_i in enumerate(cluster_indices):
            member_rank_details[int(global_i)] = {
                "tri_rank_res": int(r_res[local_i]),
                "tri_rank_info": int(r_info[local_i]),
                "tri_rank_rep": int(r_rep[local_i]),
                "aggregate_rank": int(a_m[local_i]),
                "distance_to_centroid": float(dists[local_i]),
            }

        min_rank = np.min(a_m)
        best_candidates = np.where(a_m == min_rank)[0]

        if len(best_candidates) == 1:
            best_local_idx = best_candidates[0]
        else:
            best_u = np.max(sub_u[best_candidates])
            u_winners = best_candidates[sub_u[best_candidates] == best_u]
            if len(u_winners) == 1:
                best_local_idx = u_winners[0]
            else:
                best_dist_idx = np.argmin(dists[u_winners])
                best_local_idx = u_winners[best_dist_idx]

        winner_idx = int(cluster_indices[best_local_idx])
        if return_details:
            return winner_idx, member_rank_details
        return winner_idx

    def select_query_batch(
        self,
        visual_features: np.ndarray,
        probabilities: np.ndarray,
        labeled_counts: Union[np.ndarray, List[int], Dict[int, int]],
        query_budget: int = 16,
        apply_calibration: bool = True,
        reference_prior: Optional[np.ndarray] = None,
        top_n: Optional[int] = None,
    ) -> List[int]:
        selected, _ = self.select_candidates_with_details(
            visual_features=visual_features,
            probabilities=probabilities,
            labeled_counts=labeled_counts,
            query_budget=query_budget,
            apply_calibration=apply_calibration,
            reference_prior=reference_prior,
            top_n=top_n,
        )
        return selected

    def select_candidates_with_details(
        self,
        visual_features: np.ndarray,
        probabilities: np.ndarray,
        labeled_counts: Union[np.ndarray, List[int], Dict[int, int]],
        query_budget: int = 16,
        apply_calibration: bool = True,
        reference_prior: Optional[np.ndarray] = None,
        top_n: Optional[int] = None,
    ) -> Tuple[List[int], List[Dict[str, Any]]]:
        n_samples = visual_features.shape[0]
        if n_samples == 0 or query_budget <= 0:
            return [], []

        pred_classes = np.argmax(probabilities, axis=1) if probabilities.size > 0 else np.zeros(n_samples, dtype=int)
        rho = self.compute_residual_priority(
            predicted_classes=pred_classes,
            labeled_counts=labeled_counts,
            n_classes=probabilities.shape[1] if probabilities.size > 0 else None,
        )

        if n_samples <= query_budget:
            logger.info(f"Candidate pool size ({n_samples}) <= budget ({query_budget}). Selecting all.")
            h_hat, h_nn, u = self.compute_calibrated_uncertainty(
                probabilities=probabilities,
                visual_features=visual_features,
                apply_calibration=apply_calibration,
                reference_prior=reference_prior,
                top_n=top_n,
            )
            details = []
            for i in range(n_samples):
                details.append({
                    "sample_index": i,
                    "uncertainty_utility": float(u[i]) if len(u) > i else 0.0,
                    "calibrated_entropy": float(h_hat[i]) if len(h_hat) > i else 0.0,
                    "neighbor_uncertainty": float(h_nn[i]) if len(h_nn) > i else 0.0,
                    "residual_priority": float(rho[i]) if len(rho) > i else 1.0,
                    "predicted_class_id": int(pred_classes[i]),
                    "cluster_id": i,
                    "distance_to_centroid": 0.0,
                    "tri_rank_res": 1,
                    "tri_rank_info": 1,
                    "tri_rank_rep": 1,
                    "aggregate_rank": 3,
                    "is_selected": True,
                    "selection_status": "ALL_SELECTED",
                })
            return list(range(n_samples)), details

        h_hat, h_nn, u = self.compute_calibrated_uncertainty(
            probabilities=probabilities,
            visual_features=visual_features,
            apply_calibration=apply_calibration,
            reference_prior=reference_prior,
            top_n=top_n,
        )

        labels, centers = self.weighted_kmeans_clustering(
            visual_features=visual_features,
            sample_weights=u,
            n_clusters=query_budget,
        )

        selected_indices: List[int] = []
        all_member_ranks: Dict[int, Dict[str, Any]] = {}
        winner_set = set()

        for m in range(len(centers)):
            cluster_members = np.where(labels == m)[0]
            if len(cluster_members) == 0:
                continue

            best_idx, m_ranks = self.tri_rank_aggregation(
                cluster_indices=cluster_members,
                residual_priority=rho,
                uncertainty=u,
                visual_features=visual_features,
                centroid=centers[m],
                return_details=True,
            )
            selected_indices.append(best_idx)
            winner_set.add(best_idx)
            all_member_ranks.update(m_ranks)

        backfill_set = set()
        if len(selected_indices) < query_budget:
            remaining_mask = np.ones(n_samples, dtype=bool)
            remaining_mask[selected_indices] = False
            remaining_indices = np.where(remaining_mask)[0]
            if len(remaining_indices) > 0:
                deficit = query_budget - len(selected_indices)
                ranked_remaining = remaining_indices[np.argsort(-u[remaining_indices])][:deficit]
                for b_idx in ranked_remaining:
                    selected_indices.append(int(b_idx))
                    backfill_set.add(int(b_idx))

        details = []
        for i in range(n_samples):
            m_rank = all_member_ranks.get(i, {})
            c_id = int(labels[i]) if len(labels) > i else 0
            is_sel = i in winner_set or i in backfill_set
            if i in winner_set:
                sel_status = "CLUSTER_WINNER"
            elif i in backfill_set:
                sel_status = "BACKFILL"
            else:
                sel_status = "NOT_SELECTED"

            details.append({
                "sample_index": i,
                "uncertainty_utility": float(u[i]),
                "calibrated_entropy": float(h_hat[i]),
                "neighbor_uncertainty": float(h_nn[i]),
                "residual_priority": float(rho[i]),
                "predicted_class_id": int(pred_classes[i]),
                "cluster_id": c_id,
                "distance_to_centroid": float(m_rank.get("distance_to_centroid", 0.0)),
                "tri_rank_res": m_rank.get("tri_rank_res"),
                "tri_rank_info": m_rank.get("tri_rank_info"),
                "tri_rank_rep": m_rank.get("tri_rank_rep"),
                "aggregate_rank": m_rank.get("aggregate_rank"),
                "is_selected": is_sel,
                "selection_status": sel_status,
            })

        logger.info(
            f"R3-CEC selected {len(selected_indices)} queries from {n_samples} candidate pool "
            f"(target budget={query_budget}, winners={len(winner_set)}, backfill={len(backfill_set)})."
        )
        return selected_indices, details
