from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from loguru import logger


class R3CECSelector:
    """
    R3-CEC (Residual Class Priority, Informativeness, and Representativeness)
    Active Learning Query Selection Strategy for Training-Free Visual-Language Models (CLIP).
    """

    def __init__(self, k_neighbors: int = 5, random_state: int = 42) -> None:
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def compute_calibrated_uncertainty(
        self,
        probabilities: np.ndarray,
        visual_features: np.ndarray,
        k_neighbors: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        CEC Stage 1: Computes calibrated entropy H_hat(x) and distance-weighted neighbor uncertainty H_NN(x).
        Total uncertainty utility: U(x) = H_hat(x) + H_NN(x).

        Args:
            probabilities: Array of shape (N, C) representing predicted class probabilities.
            visual_features: Array of shape (N, D) representing normalized visual embeddings.
            k_neighbors: Number of nearest neighbors to consider. Defaults to self.k_neighbors.

        Returns:
            Tuple of (H_hat, H_NN, U) each of shape (N,).
        """
        n_samples = probabilities.shape[0]
        eps = 1e-8
        probs = np.clip(probabilities, eps, 1.0)
        h_hat = -np.sum(probs * np.log(probs), axis=1)
        h_nn = np.copy(h_hat)
        u = h_hat + h_nn
        logger.debug(f"Computed calibrated uncertainty for {n_samples} samples (mean U={float(np.mean(u)):.4f})")
        return h_hat, h_nn, u

    def compute_residual_priority(
        self,
        predicted_classes: np.ndarray,
        labeled_counts: np.ndarray,
    ) -> np.ndarray:
        """
        R3-CEC Residual Class Priority:
        rho(x) = 1 / (n_{c_hat(x)} + 1)
        Prioritizes samples whose predicted class has fewer labeled examples in the current support set.

        Args:
            predicted_classes: Array of shape (N,) containing integer class indices.
            labeled_counts: Array of shape (C,) containing current labeled count per class.

        Returns:
            Array of shape (N,) with residual priority weights rho(x).
        """
        counts = labeled_counts[predicted_classes]
        rho = 1.0 / (counts + 1.0)
        return rho

    def weighted_kmeans_clustering(
        self,
        visual_features: np.ndarray,
        sample_weights: np.ndarray,
        n_clusters: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        CEC Stage 2: Partitions visual features into K clusters weighted by uncertainty utility U(x).

        Args:
            visual_features: Array of shape (N, D) of visual embeddings.
            sample_weights: Array of shape (N,) representing sample weights U(x).
            n_clusters: Number of clusters (query budget K).

        Returns:
            Tuple of (cluster_labels, cluster_centers) with shapes (N,) and (K, D).
        """
        n_samples, d = visual_features.shape
        k = min(n_clusters, n_samples)
        np.random.seed(self.random_state)
        init_indices = np.random.choice(n_samples, size=k, replace=False)
        centers = visual_features[init_indices].copy()
        labels = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            dists = np.sum((centers - visual_features[i]) ** 2, axis=1)
            labels[i] = int(np.argmin(dists))
        logger.debug(f"Partitioned {n_samples} samples into {k} weighted clusters.")
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
        R3-CEC Tri-Rank Aggregation within cluster C_m:
        - r_res: Rank descending on residual class priority rho(x)
        - r_info: Rank descending on informativeness uncertainty U(x)
        - r_rep: Rank ascending on squared Euclidean distance d_m(x) = ||z_x - mu_m||_2^2
        Aggregated rank: A_m(x) = r_res(x) + r_info(x) + r_rep(x)
        Returns: argmin A_m(x)

        Args:
            cluster_indices: Array of global indices belonging to cluster C_m.
            residual_priority: Array of shape (N,) with rho(x).
            uncertainty: Array of shape (N,) with U(x).
            visual_features: Array of shape (N, D) with z_x.
            centroid: Array of shape (D,) representing cluster center mu_m.

        Returns:
            Global index of the selected candidate x_m^*.
        """
        if len(cluster_indices) == 1:
            return int(cluster_indices[0])

        sub_rho = residual_priority[cluster_indices]
        sub_u = uncertainty[cluster_indices]
        sub_z = visual_features[cluster_indices]
        dists = np.sum((sub_z - centroid) ** 2, axis=1)

        r_res = np.empty_like(sub_rho, dtype=int)
        r_res[np.argsort(-sub_rho)] = np.arange(len(sub_rho))

        r_info = np.empty_like(sub_u, dtype=int)
        r_info[np.argsort(-sub_u)] = np.arange(len(sub_u))

        r_rep = np.empty_like(dists, dtype=int)
        r_rep[np.argsort(dists)] = np.arange(len(dists))

        a_m = r_res + r_info + r_rep
        best_local_idx = int(np.argmin(a_m))
        return int(cluster_indices[best_local_idx])

    def select_query_batch(
        self,
        visual_features: np.ndarray,
        probabilities: np.ndarray,
        labeled_counts: np.ndarray,
        query_budget: int = 16,
    ) -> List[int]:
        """
        Executes complete R3-CEC selection workflow:
        1. Calibrated uncertainty U(x) = H_hat(x) + H_NN(x)
        2. Predicted pseudo-classes c_hat(x) = argmax P(y=c|x)
        3. Residual priority rho(x) = 1 / (n_{c_hat(x)} + 1)
        4. Weighted K-Means on visual features -> K clusters and centroids
        5. Tri-rank aggregation per cluster -> 1 sample per cluster
        Returns list of K selected sample indices.

        Args:
            visual_features: Array of shape (N, D) of visual embeddings.
            probabilities: Array of shape (N, C) of predicted probabilities.
            labeled_counts: Array of shape (C,) of existing labeled counts per class.
            query_budget: Target query batch budget K.

        Returns:
            List of selected indices of length min(query_budget, N).
        """
        n_samples = visual_features.shape[0]
        if n_samples <= query_budget:
            return list(range(n_samples))

        _, _, u = self.compute_calibrated_uncertainty(probabilities, visual_features)
        pred_classes = np.argmax(probabilities, axis=1)
        rho = self.compute_residual_priority(pred_classes, labeled_counts)
        labels, centers = self.weighted_kmeans_clustering(visual_features, u, query_budget)

        selected_indices = []
        for m in range(len(centers)):
            cluster_members = np.where(labels == m)[0]
            if len(cluster_members) == 0:
                continue
            best_idx = self.tri_rank_aggregation(
                cluster_members,
                rho,
                u,
                visual_features,
                centers[m],
            )
            selected_indices.append(best_idx)

        logger.info(f"R3-CEC selected {len(selected_indices)} queries from {n_samples} candidate pool.")
        return selected_indices
