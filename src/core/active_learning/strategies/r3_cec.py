from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from loguru import logger
from scipy.spatial.distance import cdist
from scipy.stats import rankdata


class R3CECSelector:
    """
    R3-CEC (Residual Class Priority, Informativeness, and Representativeness)
    Active Learning Query Selection Strategy for Training-Free Visual-Language Models (CLIP/ProtoFuse).
    Reference: ICIT 2026 Paper Sec 3.1 - 3.3.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        temperature: float = 0.5,
        random_state: int = 42,
    ) -> None:
        self.k_neighbors = k_neighbors
        self.temperature = temperature
        self.random_state = random_state

    def calibrate_probabilities(
        self,
        probabilities: np.ndarray,
        eps: float = 1e-8,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        CEC Step 1 Contextual Prior Calibration:
        Computes marginal class prior p_0(c) over the unlabeled pool and normalizes
        predicted probabilities P(y=c|x):

        p_0(c) = (1 / |D_U|) * sum_{x in D_U} P(y=c | x)
        \hat{P}(y=c | x) \propto P(y=c | x) / p_0(c)

        Args:
            probabilities: Array of shape (N, C) with raw probabilities.
            eps: Numerical stability constant.

        Returns:
            Tuple of (calibrated_probabilities of shape (N, C), p0_prior of shape (C,)).
        """
        probs = np.clip(probabilities, eps, 1.0)
        p0 = np.mean(probs, axis=0)
        p0 = np.clip(p0, eps, 1.0)

        calibrated = probs / p0[None, :]
        calibrated /= np.sum(calibrated, axis=1, keepdims=True)
        return calibrated, p0

    def compute_calibrated_uncertainty(
        self,
        probabilities: np.ndarray,
        visual_features: np.ndarray,
        k_neighbors: Optional[int] = None,
        temperature: Optional[float] = None,
        apply_calibration: bool = True,
        eps: float = 1e-8,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        CEC Step 1 & Step 2:
        1. Contextual Prior Calibration -> \hat{P}(y=c | x)
        2. Calibrated entropy: \hat{H}(x) = -sum_c \hat{P}(y=c|x) * log(\hat{P}(y=c|x))
        3. Distance-weighted neighbor uncertainty: H_NN(x) over k-nearest neighbors
        4. Total uncertainty utility: U(x) = \hat{H}(x) + H_NN(x) (Formula 2)

        Args:
            probabilities: Array of shape (N, C) representing predicted class probabilities.
            visual_features: Array of shape (N, D) representing visual embeddings.
            k_neighbors: Number of nearest neighbors to consider. Defaults to self.k_neighbors.
            temperature: Softmax temperature tau for distance weighting. Defaults to self.temperature.
            apply_calibration: Whether to apply class prior calibration.
            eps: Numerical stability constant.

        Returns:
            Tuple of (H_hat, H_NN, U) each of shape (N,).
        """
        n_samples = probabilities.shape[0]
        if n_samples == 0:
            return np.array([]), np.array([]), np.array([])

        k = self.k_neighbors if k_neighbors is None else k_neighbors
        tau = self.temperature if temperature is None else temperature

        # 1. Contextual Prior Calibration
        if apply_calibration and n_samples > 1:
            p_hat, _ = self.calibrate_probabilities(probabilities, eps=eps)
        else:
            p_hat = np.clip(probabilities, eps, 1.0)
            p_hat /= np.sum(p_hat, axis=1, keepdims=True)

        # 2. Calibrated Entropy \hat{H}(x)
        h_hat = -np.sum(p_hat * np.log(np.clip(p_hat, eps, 1.0)), axis=1)

        # Edge case: Single sample or k <= 0
        if n_samples <= 1 or k <= 0:
            h_nn = np.copy(h_hat)
            u = h_hat + h_nn
            return h_hat, h_nn, u

        # 3. k-NN Neighbor Uncertainty H_NN(x) with distance weighting
        # Normalize features to unit sphere if not already normalized
        norm = np.linalg.norm(visual_features, axis=1, keepdims=True)
        z = visual_features / np.clip(norm, eps, None)

        k_eff = min(k, n_samples - 1)

        # Pairwise squared Euclidean distance: ||z_i - z_j||_2^2
        dist_sq = cdist(z, z, metric="sqeuclidean")

        # Exclude self-distance by setting diagonal to infinity
        np.fill_diagonal(dist_sq, np.inf)

        # Find k nearest neighbor indices for all samples
        nn_indices = np.argpartition(dist_sq, kth=k_eff - 1, axis=1)[:, :k_eff]

        row_indices = np.arange(n_samples)[:, None]
        nn_dists = dist_sq[row_indices, nn_indices]  # (N, k_eff)
        nn_h = h_hat[nn_indices]  # (N, k_eff)

        # Distance weights via Softmax kernel: w_j \propto exp(-d(x, x_j) / tau)
        logits = -nn_dists / max(tau, eps)
        # Shift logits for numerical stability
        logits -= np.max(logits, axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= np.sum(weights, axis=1, keepdims=True)

        h_nn = np.sum(weights * nn_h, axis=1)

        # 4. Total Uncertainty Utility U(x)
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
        """
        R3-CEC Residual Class Priority (Formulas 3, 4, 5):
        rho^{(t)}(x) = 1 / (n_{c_hat(x)}^{(t)} + 1)
        Prioritizes samples whose predicted class has fewer labeled examples.

        Args:
            predicted_classes: Array of shape (N,) containing integer class indices.
            labeled_counts: Labeled count per class as array, list, or dict {class_idx: count}.
            n_classes: Total number of classes (optional).

        Returns:
            Array of shape (N,) with residual priority weights rho(x).
        """
        n_samples = len(predicted_classes)
        if n_samples == 0:
            return np.array([])

        if isinstance(labeled_counts, dict):
            max_c = max(
                int(np.max(predicted_classes)) if len(predicted_classes) > 0 else 0,
                max(labeled_counts.keys()) if labeled_counts else 0,
                (n_classes - 1) if n_classes else 0,
            )
            counts_arr = np.zeros(max_c + 1, dtype=np.float64)
            for c_idx, count in labeled_counts.items():
                if 0 <= c_idx <= max_c:
                    counts_arr[c_idx] = float(count)
        else:
            counts_arr = np.asarray(labeled_counts, dtype=np.float64)

        # Safely index counts for each predicted class
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
        """
        CEC Step 2: Partitions visual features into K clusters weighted by uncertainty utility U(x).
        Uses weighted k-means++ initialization and Lloyd's iterations.

        Args:
            visual_features: Array of shape (N, D) of visual embeddings.
            sample_weights: Array of shape (N,) representing sample weights U(x).
            n_clusters: Number of clusters (query budget K).
            max_iter: Maximum number of Lloyd iterations.
            tol: Tolerance for centroid convergence.

        Returns:
            Tuple of (cluster_labels, cluster_centers) with shapes (N,) and (K, D).
        """
        n_samples, d = visual_features.shape
        k = min(n_clusters, n_samples)
        if k <= 0:
            return np.zeros(0, dtype=int), np.zeros((0, d))

        if n_samples <= k:
            # Each sample is its own cluster
            labels = np.arange(n_samples, dtype=int)
            centers = visual_features.copy()
            return labels, centers

        rng = np.random.RandomState(self.random_state)
        w = np.asarray(sample_weights, dtype=np.float64)
        w = np.clip(w, 1e-8, None)
        w_norm = w / np.sum(w)

        # 1. Weighted K-Means++ Initialization
        first_center_idx = rng.choice(n_samples, p=w_norm)
        centers = [visual_features[first_center_idx].copy()]

        for _ in range(1, k):
            current_centers = np.array(centers)
            # Distance from each point to nearest existing center
            dists_to_centers = cdist(visual_features, current_centers, metric="sqeuclidean")
            min_dist_sq = np.min(dists_to_centers, axis=1)

            # Probabilities proportional to W_i * D(x_i)^2
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

        # 2. Lloyd's Iteration
        for iteration in range(max_iter):
            # Assignment step
            dists = cdist(visual_features, centers, metric="sqeuclidean")
            new_labels = np.argmin(dists, axis=1)

            # Update step: weighted centroids
            new_centers = np.empty_like(centers)
            for m in range(k):
                mask = (new_labels == m)
                if not np.any(mask):
                    # Empty cluster: re-assign to sample with largest weighted distance
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
    ) -> int:
        """
        R3-CEC Tri-Rank Aggregation within cluster C_m (Formulas 7, 8, 9):
        - r_res: Rank descending on residual class priority rho(x)
        - r_info: Rank descending on informativeness uncertainty U(x)
        - r_rep: Rank ascending on squared Euclidean distance d_m(x) = ||z_x - mu_m||_2^2
        Aggregated rank: A_m(x) = r_res(x) + r_info(x) + r_rep(x)
        Returns: argmin A_m(x) with tie-breaking.

        Args:
            cluster_indices: Array of global indices belonging to cluster C_m.
            residual_priority: Array of shape (N,) with rho(x).
            uncertainty: Array of shape (N,) with U(x).
            visual_features: Array of shape (N, D) with z_x.
            centroid: Array of shape (D,) representing cluster center mu_m.

        Returns:
            Global index of the selected candidate x_m^*.
        """
        n_members = len(cluster_indices)
        if n_members == 1:
            return int(cluster_indices[0])

        sub_rho = residual_priority[cluster_indices]
        sub_u = uncertainty[cluster_indices]
        sub_z = visual_features[cluster_indices]

        # Representativeness: d_m(x) = ||z_x - mu_m||_2^2 (Formula 6)
        dists = np.sum((sub_z - centroid[None, :]) ** 2, axis=1)

        # Ranking: rankdata with method='min' (ties get the smallest rank index, 1-indexed)
        # r_res: rank descending on rho (negate so largest rho gets rank 1)
        r_res = rankdata(-sub_rho, method="min")

        # r_info: rank descending on U (negate so largest U gets rank 1)
        r_info = rankdata(-sub_u, method="min")

        # r_rep: rank ascending on distance d_m (smallest distance gets rank 1)
        r_rep = rankdata(dists, method="min")

        # Combined Aggregate Rank: A_m(x) = r_res + r_info + r_rep (Formula 8)
        a_m = r_res + r_info + r_rep

        # Tie-breaking: if aggregate ranks are equal, break ties using highest U, then lowest d_m
        min_rank = np.min(a_m)
        best_candidates = np.where(a_m == min_rank)[0]

        if len(best_candidates) == 1:
            best_local_idx = best_candidates[0]
        else:
            # Secondary tie-breaker: highest informativeness U
            best_u = np.max(sub_u[best_candidates])
            u_winners = best_candidates[sub_u[best_candidates] == best_u]
            if len(u_winners) == 1:
                best_local_idx = u_winners[0]
            else:
                # Tertiary tie-breaker: closest to centroid
                best_dist_idx = np.argmin(dists[u_winners])
                best_local_idx = u_winners[best_dist_idx]

        return int(cluster_indices[best_local_idx])

    def select_query_batch(
        self,
        visual_features: np.ndarray,
        probabilities: np.ndarray,
        labeled_counts: Union[np.ndarray, List[int], Dict[int, int]],
        query_budget: int = 16,
        apply_calibration: bool = True,
    ) -> List[int]:
        """
        Executes complete R3-CEC selection workflow (Sec 3.1 - 3.3):
        1. Contextual prior calibration & Calibrated entropy: \hat{H}(x)
        2. Neighbor uncertainty: H_NN(x) & total informativeness: U(x) = \hat{H}(x) + H_NN(x)
        3. Predicted pseudo-classes: \hat{c}(x) = argmax P(y=c|x)
        4. Residual class priority: \rho(x) = 1 / (n_{\hat{c}(x)} + 1)
        5. Weighted K-Means on visual features with sample weights U(x) -> K clusters & centroids
        6. Tri-Rank aggregation per cluster -> 1 sample per cluster
        Returns list of K selected sample indices.

        Args:
            visual_features: Array of shape (N, D) of visual embeddings.
            probabilities: Array of shape (N, C) of predicted probabilities.
            labeled_counts: Existing labeled counts per class (array, list, or dict).
            query_budget: Target query batch budget K.
            apply_calibration: Whether to apply class prior calibration.

        Returns:
            List of selected global indices of length min(query_budget, N).
        """
        n_samples = visual_features.shape[0]
        if n_samples <= query_budget:
            logger.info(f"Candidate pool size ({n_samples}) <= budget ({query_budget}). Selecting all.")
            return list(range(n_samples))

        # 1 & 2: Calibrated uncertainty utility U(x)
        _, _, u = self.compute_calibrated_uncertainty(
            probabilities=probabilities,
            visual_features=visual_features,
            apply_calibration=apply_calibration,
        )

        # 3 & 4: Residual Class Priority rho(x)
        pred_classes = np.argmax(probabilities, axis=1)
        rho = self.compute_residual_priority(
            predicted_classes=pred_classes,
            labeled_counts=labeled_counts,
            n_classes=probabilities.shape[1],
        )

        # 5: Weighted K-Means on visual features
        labels, centers = self.weighted_kmeans_clustering(
            visual_features=visual_features,
            sample_weights=u,
            n_clusters=query_budget,
        )

        # 6: Tri-Rank Aggregation per cluster
        selected_indices: List[int] = []
        for m in range(len(centers)):
            cluster_members = np.where(labels == m)[0]
            if len(cluster_members) == 0:
                continue

            best_idx = self.tri_rank_aggregation(
                cluster_indices=cluster_members,
                residual_priority=rho,
                uncertainty=u,
                visual_features=visual_features,
                centroid=centers[m],
            )
            selected_indices.append(best_idx)

        # If any clusters were empty and selected_indices < query_budget, backfill from highest U
        if len(selected_indices) < query_budget:
            remaining_mask = np.ones(n_samples, dtype=bool)
            remaining_mask[selected_indices] = False
            remaining_indices = np.where(remaining_mask)[0]
            if len(remaining_indices) > 0:
                deficit = query_budget - len(selected_indices)
                ranked_remaining = remaining_indices[np.argsort(-u[remaining_indices])][:deficit]
                selected_indices.extend(ranked_remaining.tolist())

        logger.info(
            f"R3-CEC selected {len(selected_indices)} queries from {n_samples} candidate pool "
            f"(target budget={query_budget})."
        )
        return selected_indices
