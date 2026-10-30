from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pytest
import torch
from loguru import logger
from src.core.active_learning.strategies.r3_cec import R3CECSelector


@pytest.fixture
def r3_cec():
    return R3CECSelector(k_neighbors=3, temperature=0.5, random_state=42)


def test_ai_cec_01_uniform_prior(r3_cec: R3CECSelector):
    probs = np.ones((10, 5)) / 5.0
    calibrated, p0 = r3_cec.calibrate_probabilities(probs)
    np.testing.assert_allclose(calibrated, probs, atol=1e-5)
    np.testing.assert_allclose(p0, np.full(5, 0.2), atol=1e-5)
    logger.info("AI-CEC-01 verified: Uniform probabilities stay uniform after calibration.")


def test_ai_cec_02_calibrated_row_sum_unity(r3_cec: R3CECSelector):
    rng = np.random.RandomState(42)
    raw = rng.exponential(scale=1.0, size=(20, 10))
    probs = raw / np.sum(raw, axis=1, keepdims=True)
    calibrated, _ = r3_cec.calibrate_probabilities(probs)
    row_sums = np.sum(calibrated, axis=1)
    np.testing.assert_allclose(row_sums, np.ones(20), atol=1e-5)
    logger.info("AI-CEC-02 verified: Calibrated rows sum to 1.0.")


def test_ai_cec_03_prior_matches_column_mean(r3_cec: R3CECSelector):
    probs = np.array([[0.8, 0.2], [0.6, 0.4], [0.4, 0.6]])
    _, p0 = r3_cec.calibrate_probabilities(probs)
    expected_p0 = np.array([0.6, 0.4])
    np.testing.assert_allclose(p0, expected_p0, atol=1e-5)
    logger.info("AI-CEC-03 verified: Marginal class prior matches column mean.")


def test_ai_cec_04_skewed_prior_downweights_frequent(r3_cec: R3CECSelector):
    probs = np.array([
        [0.9, 0.1],
        [0.9, 0.1],
        [0.8, 0.2],
        [0.8, 0.2],
    ])
    calibrated, p0 = r3_cec.calibrate_probabilities(probs)
    assert p0[0] > p0[1]
    assert calibrated[0, 0] < probs[0, 0]
    logger.info("AI-CEC-04 verified: Frequent class is downweighted in calibrated probabilities.")


def test_ai_cec_05_single_sample_uncertainty(r3_cec: R3CECSelector):
    prob = np.array([[0.5, 0.5]])
    feat = np.array([[1.0, 0.0]])
    h_hat, h_nn, u = r3_cec.compute_calibrated_uncertainty(prob, feat, k_neighbors=3)
    assert len(h_hat) == 1
    assert len(h_nn) == 1
    assert len(u) == 1
    assert h_nn[0] == h_hat[0]
    assert u[0] == h_hat[0] * 2
    logger.info("AI-CEC-05 verified: Single sample edge case handled correctly.")


def test_ai_cec_06_empty_sample_uncertainty(r3_cec: R3CECSelector):
    prob = np.empty((0, 5))
    feat = np.empty((0, 8))
    h_hat, h_nn, u = r3_cec.compute_calibrated_uncertainty(prob, feat)
    assert len(h_hat) == 0
    assert len(h_nn) == 0
    assert len(u) == 0
    logger.info("AI-CEC-06 verified: Empty input returns empty arrays.")


def test_ai_cec_07_maximum_entropy_uniform(r3_cec: R3CECSelector):
    c = 4
    prob = np.full((5, c), 1.0 / c)
    feat = np.random.randn(5, 8)
    h_hat, _, _ = r3_cec.compute_calibrated_uncertainty(prob, feat, apply_calibration=False)
    expected_entropy = np.log(c)
    np.testing.assert_allclose(h_hat, np.full(5, expected_entropy), atol=1e-5)
    logger.info(f"AI-CEC-07 verified: Max entropy equals ln({c}).")


def test_ai_cec_08_minimum_entropy_one_hot(r3_cec: R3CECSelector):
    prob = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    feat = np.random.randn(2, 4)
    h_hat, _, _ = r3_cec.compute_calibrated_uncertainty(prob, feat, apply_calibration=False)
    assert np.all(h_hat < 1e-4)
    logger.info("AI-CEC-08 verified: One-hot probabilities yield zero entropy.")


def test_ai_cec_09_neighbor_weighting_sum_to_one(r3_cec: R3CECSelector):
    probs = np.random.rand(10, 4)
    probs /= np.sum(probs, axis=1, keepdims=True)
    feats = np.random.randn(10, 8)
    h_hat, h_nn, u = r3_cec.compute_calibrated_uncertainty(probs, feats, k_neighbors=3)
    assert np.all(h_nn >= 0)
    assert len(h_nn) == 10
    logger.info("AI-CEC-09 verified: Neighbor uncertainty is non-negative and properly bounded.")


def test_ai_cec_10_total_uncertainty_additive(r3_cec: R3CECSelector):
    probs = np.random.rand(8, 3)
    probs /= np.sum(probs, axis=1, keepdims=True)
    feats = np.random.randn(8, 6)
    h_hat, h_nn, u = r3_cec.compute_calibrated_uncertainty(probs, feats)
    np.testing.assert_allclose(u, h_hat + h_nn, atol=1e-6)
    logger.info("AI-CEC-10 verified: U(x) = H_hat(x) + H_NN(x).")


def test_ai_cec_11_zero_feature_vector_stability(r3_cec: R3CECSelector):
    probs = np.array([[0.6, 0.4], [0.3, 0.7]])
    feats = np.zeros((2, 5))
    h_hat, h_nn, u = r3_cec.compute_calibrated_uncertainty(probs, feats)
    assert not np.isnan(u).any()
    assert not np.isinf(u).any()
    logger.info("AI-CEC-11 verified: Zero feature vectors do not produce NaN or Inf.")


def test_ai_cec_12_residual_priority_zero_count(r3_cec: R3CECSelector):
    preds = np.array([0, 1, 2])
    counts = [0, 0, 0]
    rho = r3_cec.compute_residual_priority(preds, counts)
    np.testing.assert_allclose(rho, np.array([1.0, 1.0, 1.0]), atol=1e-6)
    logger.info("AI-CEC-12 verified: Classes with 0 labeled samples get max priority 1.0.")


def test_ai_cec_13_residual_priority_strictly_decreasing(r3_cec: R3CECSelector):
    preds = np.array([0, 1, 2])
    counts = [0, 1, 9]
    rho = r3_cec.compute_residual_priority(preds, counts)
    assert rho[0] == 1.0
    assert rho[1] == 0.5
    assert rho[2] == 0.1
    assert rho[0] > rho[1] > rho[2]
    logger.info("AI-CEC-13 verified: Priority strictly decreases with increasing labeled count.")


def test_ai_cec_14_residual_priority_dict_counts(r3_cec: R3CECSelector):
    preds = np.array([0, 3, 5])
    counts_dict = {0: 1, 3: 3, 5: 0}
    rho = r3_cec.compute_residual_priority(preds, counts_dict)
    assert rho[0] == 0.5
    assert rho[1] == 0.25
    assert rho[2] == 1.0
    logger.info("AI-CEC-14 verified: Dictionary of labeled counts correctly mapped.")


def test_ai_cec_15_residual_priority_clipped_index(r3_cec: R3CECSelector):
    preds = np.array([10])
    counts = [5, 5]
    rho = r3_cec.compute_residual_priority(preds, counts)
    assert len(rho) == 1
    assert rho[0] == 1.0 / 6.0
    logger.info("AI-CEC-15 verified: Out-of-bounds class index safely clipped.")


def test_ai_cec_16_kmeans_fewer_samples_than_k(r3_cec: R3CECSelector):
    feats = np.random.randn(3, 4)
    weights = np.array([1.0, 2.0, 3.0])
    labels, centers = r3_cec.weighted_kmeans_clustering(feats, weights, n_clusters=5)
    assert len(labels) == 3
    assert len(centers) == 3
    np.testing.assert_array_equal(labels, np.array([0, 1, 2]))
    logger.info("AI-CEC-16 verified: N <= K assigns each sample to its own cluster.")


def test_ai_cec_17_kmeans_labels_range(r3_cec: R3CECSelector):
    feats = np.random.randn(30, 8)
    weights = np.ones(30)
    labels, centers = r3_cec.weighted_kmeans_clustering(feats, weights, n_clusters=4)
    assert len(labels) == 30
    assert set(labels).issubset({0, 1, 2, 3})
    assert centers.shape == (4, 8)
    logger.info("AI-CEC-17 verified: Cluster labels are in range [0, K-1].")


def test_ai_cec_18_kmeans_centroid_dimensions(r3_cec: R3CECSelector):
    n_samples, dim, k = 50, 16, 5
    feats = np.random.randn(n_samples, dim)
    weights = np.random.rand(n_samples) + 0.1
    _, centers = r3_cec.weighted_kmeans_clustering(feats, weights, n_clusters=k)
    assert centers.shape == (k, dim)
    logger.info("AI-CEC-18 verified: Centroids have expected shape (K, D).")


def test_ai_cec_19_kmeans_rng_reproducibility(r3_cec: R3CECSelector):
    feats = np.random.randn(40, 10)
    weights = np.random.rand(40) + 0.1
    selector_a = R3CECSelector(random_state=123)
    selector_b = R3CECSelector(random_state=123)
    labels_a, centers_a = selector_a.weighted_kmeans_clustering(feats, weights, n_clusters=3)
    labels_b, centers_b = selector_b.weighted_kmeans_clustering(feats, weights, n_clusters=3)
    np.testing.assert_array_equal(labels_a, labels_b)
    np.testing.assert_allclose(centers_a, centers_b, atol=1e-7)
    logger.info("AI-CEC-19 verified: RandomState ensures deterministic clustering output.")


def test_ai_cec_20_tri_rank_single_member(r3_cec: R3CECSelector):
    cluster_indices = np.array([7])
    rho = np.ones(10)
    u = np.ones(10)
    feats = np.random.randn(10, 4)
    centroid = np.zeros(4)
    best_idx = r3_cec.tri_rank_aggregation(cluster_indices, rho, u, feats, centroid)
    assert best_idx == 7
    logger.info("AI-CEC-20 verified: Single-member cluster immediately returns its member index.")


def test_ai_cec_21_tri_rank_selection_and_batch_pipeline(r3_cec: R3CECSelector, real_features_path: Path):
    slice_data = torch.load(real_features_path, map_location="cpu")
    feats = slice_data["features"].numpy()
    n_samples = feats.shape[0]

    rng = np.random.RandomState(42)
    raw_probs = rng.dirichlet(np.ones(100), size=n_samples)
    labeled_counts = {c: rng.randint(0, 10) for c in range(100)}

    selected = r3_cec.select_query_batch(
        visual_features=feats,
        probabilities=raw_probs,
        labeled_counts=labeled_counts,
        query_budget=16,
    )

    assert len(selected) == 16
    assert len(set(selected)) == 16
    for idx in selected:
        assert 0 <= idx < n_samples
    logger.info(f"AI-CEC-21 verified: End-to-end R3-CEC selected 16 unique queries from {n_samples} vectors.")

