from typing import Dict, Any, List
import pytest
from loguru import logger
from src.core.active_learning.detection_scorer import DetectionActiveLearningEngine
from src.core.active_learning.strategies.least_confidence import LeastConfidenceScorer


def test_ai_al_01_r3_cec_scorer_registration():
    engine = DetectionActiveLearningEngine()
    effective_strategy, scorer = engine.get_scorer("r3_cec")
    assert effective_strategy == "r3_cec"
    assert scorer is not None
    assert scorer.name == "r3_cec"


def test_ai_al_02_least_confidence_computation():
    scorer = LeastConfidenceScorer()
    sample_det = {"confidence": 0.82}
    score = scorer.compute_box_score(sample_det)
    assert abs(score - 0.18) < 1e-4

    zero_conf_det = {"confidence": 0.0}
    assert abs(scorer.compute_box_score(zero_conf_det) - 1.0) < 1e-4

    max_conf_det = {"confidence": 1.0}
    assert abs(scorer.compute_box_score(max_conf_det) - 0.0) < 1e-4
    logger.info("AI-AL-02 verified: Least confidence mathematically computes 1.0 - conf.")


def test_ai_al_03_aggregation_methods():
    scorer = LeastConfidenceScorer()
    scores = [0.1, 0.4, 0.7]

    assert abs(scorer.aggregate(scores, "avg") - 0.4) < 1e-4
    assert abs(scorer.aggregate(scores, "max") - 0.7) < 1e-4
    assert abs(scorer.aggregate(scores, "min") - 0.1) < 1e-4
    assert abs(scorer.aggregate(scores, "sum") - 1.2) < 1e-4
    logger.info("AI-AL-03 verified: all 4 aggregation methods yield expected outputs.")


def test_ai_al_04_zero_detection_fallback():
    scorer = LeastConfidenceScorer()
    score, detections = scorer.score_detections([], aggregation_method="avg", no_detection_score=0.25)
    assert score == 0.25
    assert detections == []
    logger.info("AI-AL-04 verified: Zero detections return configured no_detection_score.")


def test_ai_al_05_score_detections_batch():
    scorer = LeastConfidenceScorer()
    raw_detections = [
        {"class_id": 1, "confidence": 0.9},
        {"class_id": 2, "confidence": 0.4},
    ]
    img_score, scored = scorer.score_detections(raw_detections, aggregation_method="avg")
    assert len(scored) == 2
    assert abs(scored[0]["score"] - 0.1) < 1e-4
    assert abs(scored[1]["score"] - 0.6) < 1e-4
    assert abs(img_score - 0.35) < 1e-4
    logger.info("AI-AL-05 verified: scored detections list and aggregate score match formula.")

