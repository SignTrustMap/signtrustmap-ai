from typing import Dict, Any, List
import pytest
from loguru import logger
from src.core.active_learning.feedback_processor import FeedbackProcessor
from src.core.active_learning.store import al_store


@pytest.fixture
def processor():
    return FeedbackProcessor()


def test_ai_fb_01_not_given_evaluation(processor: FeedbackProcessor):
    item = {
        "object_name": "images/test_01.jpg",
        "box_xyxy": [10.0, 10.0, 50.0, 50.0],
        "evaluation": "NOT_GIVEN",
        "pred_confidence": 0.85,
    }
    res = processor.process_feedback_item(item)
    assert res["category"] == "detector"
    assert res["action"] == "added_hard_negative"
    assert res["sample_id"].startswith("det_")

    counts = al_store.get_counts()
    assert counts["detector"]["pending"] >= 1
    logger.info("AI-FB-01 verified: NOT_GIVEN routed to detector hard-negative.")


def test_ai_fb_02_false_classifier_evaluation(processor: FeedbackProcessor):
    item = {
        "object_name": "images/test_02.jpg",
        "box_xyxy": [20.0, 20.0, 80.0, 80.0],
        "evaluation": "FALSE",
        "pred_class_id": 5,
        "pred_class_name": "Speed_limit_50",
        "pred_confidence": 0.90,
        "oracle_class_id": 10,
        "oracle_class_name": "Speed_limit_60",
        "is_clip": False,
    }
    res = processor.process_feedback_item(item)
    assert res["category"] == "classifier"
    assert res["action"] == "added_classifier_correction"
    assert res["sample_id"].startswith("cls_")

    counts = al_store.get_counts()
    assert counts["classifier"]["pending"] >= 1
    logger.info("AI-FB-02 verified: FALSE routed to classifier correction.")


def test_ai_fb_03_false_clip_evaluation(processor: FeedbackProcessor):
    item = {
        "object_name": "images/test_03.jpg",
        "box_xyxy": [30.0, 30.0, 90.0, 90.0],
        "evaluation": "FALSE",
        "oracle_class_name": "New_rare_sign",
        "is_clip": True,
    }
    res = processor.process_feedback_item(item)
    assert res["category"] == "clip"
    assert res["action"] == "added_clip_staging"
    assert res["sample_id"].startswith("clip_")

    counts = al_store.get_counts()
    assert counts["clip"]["pending"] >= 1
    logger.info("AI-FB-03 verified: FALSE with is_clip=True routed to CLIP staging.")


def test_ai_fb_04_true_evaluation(processor: FeedbackProcessor):
    item = {
        "object_name": "images/test_04.jpg",
        "evaluation": "TRUE",
    }
    res = processor.process_feedback_item(item)
    assert res["category"] == "correct"
    assert res["action"] == "verified_correct"
    assert res["sample_id"] is None
    logger.info("AI-FB-04 verified: TRUE evaluation recorded as verified correct without negative store entry.")


def test_ai_fb_05_process_feedback_batch(processor: FeedbackProcessor):
    batch = [
        {"object_name": "img_a.jpg", "evaluation": "NOT_GIVEN"},
        {"object_name": "img_b.jpg", "evaluation": "FALSE", "is_clip": False, "oracle_class_id": 1},
        {"object_name": "img_c.jpg", "evaluation": "TRUE"},
    ]
    summary = processor.process_feedback_batch(batch, auto_trigger=False)
    assert summary["total_processed"] == 3
    assert summary["detector_added"] == 1
    assert summary["classifier_added"] == 1
    assert summary["correct_count"] == 1
    logger.info("AI-FB-05 verified: Feedback batch correctly aggregated and categorized.")

