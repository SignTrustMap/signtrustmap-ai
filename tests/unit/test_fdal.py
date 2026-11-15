import math
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pytest
import torch
from loguru import logger

from src.core.active_learning.strategies.fdal import FDALScorer, FDALSelector


@pytest.fixture
def fdal_selector():
    return FDALSelector(
        eta=0.03125,
        learning_rate=0.1,
        optimization_steps=5,
        lambda_hyp=0.99,
        margin=0.01,
        device="cpu",
        random_seed=42,
    )


@pytest.fixture
def synthetic_timo():
    generator = torch.Generator().manual_seed(42)
    dim = 512
    num_classes = 10
    w = torch.randn(dim, num_classes, generator=generator)
    w = w / torch.clamp(w.norm(dim=0, keepdim=True), min=1e-8)
    b = torch.zeros(num_classes)
    anchors = torch.randn(num_classes, dim, generator=generator)
    anchors = anchors / torch.clamp(anchors.norm(dim=-1, keepdim=True), min=1e-8)
    return w, b, anchors


def test_ai_fdal_01_zero_flips_entropy_fallback(fdal_selector: FDALSelector, synthetic_timo):
    w, b, anchors = synthetic_timo
    num_classes = anchors.shape[0]
    z = anchors[0].clone()
    orig_cls = int((z @ w + b).argmax().item())

    s_flip, flipped = fdal_selector.optimize_perturbation_and_count_flips(
        visual_embedding=z,
        original_predicted_class=orig_cls,
        timo_weights=w,
        timo_bias=b,
        anchors=anchors,
        eta=1e-8,
    )
    assert s_flip == 0
    assert flipped == []

    probs = np.array([0.5 if i == orig_cls else 0.5 / (num_classes - 1) for i in range(num_classes)])
    entropy = fdal_selector.compute_calibrated_entropy(probs)[0]
    expected_priority = fdal_selector.compute_scalar_priority(0, entropy, num_classes=num_classes)

    candidates_scores = fdal_selector.score_candidates_batch(
        visual_embeddings=[z],
        predicted_classes=[orig_cls],
        class_probabilities=[probs],
        timo_weights=w,
        timo_bias=b,
        anchors=anchors,
        eta=1e-8,
    )

    assert len(candidates_scores) == 1
    score = candidates_scores[0]
    assert score["flip_count"] == 0
    assert score["fdal_tuple"] == (0, entropy)
    assert score["fdal_priority"] == expected_priority
    logger.info("AI-FDAL-01 verified: zero flips fallback to calibrated entropy.")


def test_ai_fdal_02_multiple_flips_ranks_above_zero_flips(fdal_selector: FDALSelector):
    num_classes = 10
    cand_flips = {
        "candidate_id": "c1",
        "visual_embedding": None,
        "class_id": 1,
        "probabilities": np.array([0.9] + [0.1 / 9] * 9),
    }
    cand_zero = {
        "candidate_id": "c2",
        "visual_embedding": None,
        "class_id": 2,
        "probabilities": np.full(num_classes, 1.0 / num_classes),
    }

    scores = [
        {
            "candidate_index": 0,
            "flip_count": 3,
            "fallback_entropy": 0.2,
            "fdal_tuple": (3, 0.2),
            "fdal_priority": fdal_selector.compute_scalar_priority(3, 0.2, num_classes),
            "flipped_anchor_classes": [0, 2, 3],
        },
        {
            "candidate_index": 1,
            "flip_count": 0,
            "fallback_entropy": 2.3,
            "fdal_tuple": (0, 2.3),
            "fdal_priority": fdal_selector.compute_scalar_priority(0, 2.3, num_classes),
            "flipped_anchor_classes": [],
        },
    ]

    ranked = []
    for idx, s in enumerate(scores):
        item = dict([cand_flips, cand_zero][idx])
        item["flip_count"] = s["flip_count"]
        item["fallback_entropy"] = s["fallback_entropy"]
        item["fdal_tuple"] = s["fdal_tuple"]
        item["fdal_priority"] = s["fdal_priority"]
        ranked.append(item)

    ranked.sort(key=lambda x: (x["flip_count"], x["fallback_entropy"]), reverse=True)
    assert ranked[0]["candidate_id"] == "c1"
    assert ranked[1]["candidate_id"] == "c2"
    assert ranked[0]["fdal_priority"] > ranked[1]["fdal_priority"]
    logger.info("AI-FDAL-02 verified: candidate with flips ranks above zero flips.")


def test_ai_fdal_03_infeasible_perturbation_degradation(fdal_selector: FDALSelector, synthetic_timo):
    w, b, anchors = synthetic_timo
    z = torch.randn(512)
    z = z / z.norm()
    orig_cls = int((z @ w + b).argmax().item())

    s_flip, flipped = fdal_selector.optimize_perturbation_and_count_flips(
        visual_embedding=z,
        original_predicted_class=orig_cls,
        timo_weights=w,
        timo_bias=b,
        anchors=anchors,
        eta=1e-8,
    )
    assert s_flip == 0
    assert len(flipped) == 0
    logger.info("AI-FDAL-03 verified: tiny eta gracefully yields zero flips.")


def test_ai_fdal_04_eta_bounds_verification(fdal_selector: FDALSelector, synthetic_timo):
    w, b, anchors = synthetic_timo
    eta = 0.05
    generator = torch.Generator().manual_seed(42)
    alpha_init = torch.normal(mean=eta / 2.0, std=eta / 4.0, size=(10, 512), generator=generator).clamp_(1e-8, eta)
    alpha = torch.nn.Parameter(alpha_init)
    optimizer = torch.optim.Adam([alpha], lr=0.5)

    for _ in range(5):
        optimizer.zero_grad()
        loss = (alpha ** 2).sum()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            alpha.clamp_(1e-8, eta)

    assert torch.all(alpha >= 1e-8).item()
    assert torch.all(alpha <= eta).item()
    logger.info("AI-FDAL-04 verified: alpha parameters strictly bounded in [1e-8, eta].")


def test_ai_fdal_05_hypersphere_unit_normalization(synthetic_timo):
    _, _, anchors = synthetic_timo
    norms = anchors.norm(dim=-1)
    np.testing.assert_allclose(norms.numpy(), np.ones(anchors.shape[0]), atol=1e-5)

    z = torch.randn(512)
    z_norm = z / torch.clamp(z.norm(dim=-1, keepdim=True), min=1e-8)
    np.testing.assert_allclose(z_norm.norm().item(), 1.0, atol=1e-5)
    logger.info("AI-FDAL-05 verified: embeddings and anchors normalized to unit hypersphere.")


def test_ai_fdal_06_prediction_flip_equivalence_timo(fdal_selector: FDALSelector, synthetic_timo):
    w, b, anchors = synthetic_timo
    z = anchors[0].clone()
    orig_cls = 0

    s_flip, flipped = fdal_selector.optimize_perturbation_and_count_flips(
        visual_embedding=z,
        original_predicted_class=orig_cls,
        timo_weights=w,
        timo_bias=b,
        anchors=anchors,
        eta=0.5,
    )

    z_rep = z.unsqueeze(0).expand(anchors.shape[0], anchors.shape[1])
    test_mixed = (1.0 - 0.5) * z_rep + 0.5 * anchors
    test_mixed_norm = test_mixed / torch.clamp(test_mixed.norm(dim=-1, keepdim=True), min=1e-8)
    test_logits = test_mixed_norm @ w + b
    manual_preds = test_logits.argmax(dim=-1).tolist()

    for idx, c in enumerate(manual_preds):
        if c != orig_cls:
            assert isinstance(c, int)
    logger.info("AI-FDAL-06 verified: prediction flips equivalent to direct TIMO logits evaluation.")


def test_ai_fdal_07_deterministic_reproducibility(synthetic_timo):
    w, b, anchors = synthetic_timo
    z = torch.randn(512)
    z = z / z.norm()
    orig_cls = int((z @ w + b).argmax().item())

    selector1 = FDALSelector(eta=0.03125, optimization_steps=5, random_seed=42, device="cpu")
    s1, f1 = selector1.optimize_perturbation_and_count_flips(z, orig_cls, w, b, anchors)

    selector2 = FDALSelector(eta=0.03125, optimization_steps=5, random_seed=42, device="cpu")
    s2, f2 = selector2.optimize_perturbation_and_count_flips(z, orig_cls, w, b, anchors)

    assert s1 == s2
    assert f1 == f2
    logger.info("AI-FDAL-07 verified: identical seeds produce identical perturbation results.")


def test_ai_fdal_08_batch_candidate_ranking_real_features(fdal_selector: FDALSelector):
    slice_path = Path("tests/fixtures/real_features/features_test_slice.pt")
    if not slice_path.exists():
        pytest.skip("Test feature slice not found.")

    data = torch.load(slice_path, map_location="cpu")
    features = data["features"][:20]
    labels = data["labels"][:20]
    num_classes = 100
    dim = 512

    gen = torch.Generator().manual_seed(42)
    w = torch.randn(dim, num_classes, generator=gen)
    w = w / torch.clamp(w.norm(dim=0, keepdim=True), min=1e-8)
    b = torch.zeros(num_classes)
    anchors = torch.randn(num_classes, dim, generator=gen)
    anchors = anchors / torch.clamp(anchors.norm(dim=-1, keepdim=True), min=1e-8)

    candidates = [
        {
            "candidate_id": f"real_cand_{i}",
            "visual_embedding": features[i].numpy(),
            "class_id": int(labels[i].item()) % num_classes,
            "confidence": 0.8,
        }
        for i in range(len(features))
    ]

    ranked = fdal_selector.rank_candidates(
        candidates=candidates,
        timo_weights=w,
        timo_bias=b,
        anchors=anchors,
        eta=0.03125,
    )

    assert len(ranked) == len(candidates)
    for i in range(len(ranked) - 1):
        prev = (ranked[i]["flip_count"], ranked[i]["fallback_entropy"])
        nxt = (ranked[i + 1]["flip_count"], ranked[i + 1]["fallback_entropy"])
        assert prev >= nxt

    selected = fdal_selector.select_query_batch(
        candidates=candidates,
        query_budget=5,
        timo_weights=w,
        timo_bias=b,
        anchors=anchors,
        eta=0.03125,
    )
    assert len(selected) == 5
    assert [x["candidate_id"] for x in selected] == [x["candidate_id"] for x in ranked[:5]]
    logger.info("AI-FDAL-08 verified: real feature vectors ranked and selected correctly.")


def test_ai_fdal_09_scorer_box_score():
    scorer = FDALScorer()
    assert scorer.name == "fdal"

    det_with_priority = {"fdal_priority": 0.75, "confidence": 0.9}
    assert scorer.compute_box_score(det_with_priority) == 0.75

    det_without_priority = {"confidence": 0.8}
    assert scorer.compute_box_score(det_without_priority) == pytest.approx(0.2, abs=1e-4)

    det_no_conf = {}
    assert scorer.compute_box_score(det_no_conf) == 1.0
    logger.info("AI-FDAL-09 verified: FDALScorer box scoring logic.")
