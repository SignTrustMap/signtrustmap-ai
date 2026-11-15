import uuid
from typing import Dict, Any, List
import numpy as np
import pytest
from loguru import logger
from sqlalchemy import select

from src.infrastructure.db import get_sync_db
from src.infrastructure.models import CandidateEmbedding, ActiveLearningRecord
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


def test_ai_al_03_aggregation_methods():
    scorer = LeastConfidenceScorer()
    scores = [0.1, 0.4, 0.7]

    assert abs(scorer.aggregate(scores, "avg") - 0.4) < 1e-4
    assert abs(scorer.aggregate(scores, "max") - 0.7) < 1e-4
    assert abs(scorer.aggregate(scores, "min") - 0.1) < 1e-4
    assert abs(scorer.aggregate(scores, "sum") - 1.2) < 1e-4


def test_ai_al_04_zero_detection_fallback():
    scorer = LeastConfidenceScorer()
    score, detections = scorer.score_detections([], aggregation_method="avg", no_detection_score=0.25)
    assert score == 0.25
    assert detections == []


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


def test_ai_al_06_fdal_scorer_registration():
    engine = DetectionActiveLearningEngine()
    effective_strategy, scorer = engine.get_scorer("fdal")
    assert effective_strategy == "fdal"
    assert scorer is not None
    assert scorer.name == "fdal"


def test_ai_al_07_persistence_candidate_embedding_and_record():
    engine = DetectionActiveLearningEngine()
    test_candidate_id = uuid.uuid4()
    test_s3_key = f"test_submission/{test_candidate_id}.jpg"
    dummy_embedding = np.zeros(512, dtype=np.float32)
    dummy_embedding[0] = 1.0

    record = {
        "candidate_id": test_candidate_id,
        "s3_key": test_s3_key,
        "model_id": 2,
        "class_id": 1,
        "box_xyxy": [15.0, 25.0, 55.0, 65.0],
        "detect_confidence": 0.92,
        "classify_confidence": 0.85,
        "strategy": "fdal",
        "acquisition_score": 0.73,
        "flip_count": 4,
        "flip_classes": [3, 7, 15, 22],
        "calibrated_entropy": 0.31,
        "embedding": dummy_embedding,
    }

    engine._persist_evaluation_records([record])

    try:
        with get_sync_db() as session:
            ce_stmt = select(CandidateEmbedding).where(CandidateEmbedding.candidate_id == test_candidate_id)
            ce_row = session.execute(ce_stmt).scalar_one_or_none()
            assert ce_row is not None
            assert ce_row.s3_key == test_s3_key
            assert ce_row.model_id == 2
            assert len(ce_row.embedding) == 512
            assert abs(float(ce_row.embedding[0]) - 1.0) < 1e-4

            al_stmt = select(ActiveLearningRecord).where(ActiveLearningRecord.candidate_id == test_candidate_id)
            al_row = session.execute(al_stmt).scalar_one_or_none()
            assert al_row is not None
            assert al_row.strategy == "fdal"
            assert abs(float(al_row.acquisition_score) - 0.73) < 1e-4
            assert al_row.flip_count == 4
            assert al_row.flip_classes == [3, 7, 15, 22]
            assert abs(float(al_row.calibrated_entropy) - 0.31) < 1e-4
            assert al_row.box_xyxy == [15.0, 25.0, 55.0, 65.0]
    finally:
        with get_sync_db() as session:
            ce_stmt = select(CandidateEmbedding).where(CandidateEmbedding.candidate_id == test_candidate_id)
            row = session.execute(ce_stmt).scalar_one_or_none()
            if row:
                session.delete(row)
                session.commit()

        with get_sync_db() as session:
            al_check = session.execute(
                select(ActiveLearningRecord).where(ActiveLearningRecord.candidate_id == test_candidate_id)
            ).scalar_one_or_none()
            assert al_check is None
