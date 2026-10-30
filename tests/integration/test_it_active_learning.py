import shutil
from pathlib import Path
from typing import Dict, Any, List
import pytest
from fastapi.testclient import TestClient
from loguru import logger
from main import app
from src.core.active_learning.store import al_store


@pytest.fixture(scope="module")
def api_client():
    with TestClient(app) as client:
        yield client


def test_ai_it_05_r3_cec_strategy_api_fallback(api_client: TestClient, verified_images_dir: Path, mock_minio: Path):
    src_img = verified_images_dir / "11288.jpg"
    dest_img = mock_minio / "stm-sign-crops" / "raw" / "score_test.jpg"
    dest_img.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_img, dest_img)

    payload = {
        "object_name": "raw/score_test.jpg",
        "strategy": "r3_cec",
        "aggregation_method": "avg",
    }
    resp = api_client.post("/api/v1/active-learning/detection/score", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["strategy"] == "r3_cec"


def test_ai_it_06_feedback_persistence(api_client: TestClient):
    payload = {
        "items": [
            {
                "object_name": "feedback_01.jpg",
                "box_xyxy": [10, 10, 50, 50],
                "evaluation": "FALSE",
                "pred_class_id": 1,
                "pred_class_name": "Speed_limit_50",
                "pred_confidence": 0.88,
                "oracle_class_id": 2,
                "oracle_class_name": "Speed_limit_60",
                "is_clip": False,
            }
        ],
        "auto_trigger": False,
    }
    resp = api_client.post("/api/v1/active-learning/feedback", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["classifier_added"] == 1

    counts = al_store.get_counts()
    assert counts["classifier"]["pending"] >= 1
    logger.info("AI-IT-06 verified: Feedback API persisted entry into active store.")


def test_ai_it_07_batch_feedback_auto_retrain_trigger(api_client: TestClient):
    items = [
        {
            "object_name": f"batch_{i}.jpg",
            "box_xyxy": [0, 0, 10, 10],
            "evaluation": "FALSE",
            "pred_class_id": 1,
            "pred_class_name": "Speed_limit_50",
            "pred_confidence": 0.80,
            "oracle_class_id": 2,
            "oracle_class_name": "Speed_limit_60",
            "is_clip": False,
        }
        for i in range(100)
    ]
    payload = {"items": items, "auto_trigger": True}
    resp = api_client.post("/api/v1/active-learning/feedback", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["classifier_added"] == 100
    assert "classifier" in data["triggered_retrain"]
    logger.info("AI-IT-07 verified: 100 pending samples automatically triggered classifier retraining.")

