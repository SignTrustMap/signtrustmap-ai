from typing import Dict, Any, List
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient
from loguru import logger
from main import app
from src.core.config import settings
from src.pipelines.inference_engine import inference_engine


@pytest.fixture(scope="module")
def api_client():
    with TestClient(app) as client:
        yield client


def test_ai_api_01_submission_unauthorized(api_client: TestClient):
    payload = {"media_objects": [{"media_type": "IMAGE", "bucket": "b", "object_key": "k.jpg"}]}
    resp_no_token = api_client.post("/api/v1/inference/submissions/sub_api_01", json=payload)
    assert resp_no_token.status_code in (403, 422)

    resp_bad_token = api_client.post(
        "/api/v1/inference/submissions/sub_api_01",
        json=payload,
        headers={"X-Internal-Token": "wrong_secret_token"},
    )
    assert resp_bad_token.status_code == 403
    logger.info("AI-API-01 verified: Unauthorized requests rejected with 403.")


def test_ai_api_02_early_classification_unauthorized(api_client: TestClient):
    payload = {
        "media_url": "http://example.com/test.jpg",
        "sign_id": "test_sign",
        "task_id": "test_task",
    }
    resp = api_client.post(
        "/api/v1/inference/early-classification",
        json=payload,
        headers={"X-Internal-Token": "invalid_token_xyz"},
    )
    assert resp.status_code == 403
    logger.info("AI-API-02 verified: Early classification rejects invalid internal token with 403.")


def test_ai_api_03_invalid_payload_validation(api_client: TestClient):
    resp = api_client.post(
        "/api/v1/inference/submissions/sub_api_03",
        json={"invalid_key": "unexpected_data"},
        headers={"X-Internal-Token": settings.INTERNAL_API_TOKEN},
    )
    assert resp.status_code == 422
    logger.info("AI-API-03 verified: Invalid payload schema yields 422 Unprocessable Entity.")


def test_ai_api_04_successful_submission_response_schema(api_client: TestClient, monkeypatch: pytest.MonkeyPatch):
    mock_result = {
        "submission_id": "sub_api_04",
        "media_type": "IMAGE",
        "image_width": 1920,
        "image_height": 1080,
        "total_frames": 1,
        "signs": [
            {
                "track_id": None,
                "class_id": 20,
                "class_name": "No_parking",
                "confidence": 0.95,
                "detect_confidence": 0.92,
                "classify_confidence": 0.95,
                "top1": {"class_id": 20, "class_name": "No_parking", "confidence": 0.95},
                "top5": [],
                "box_xyxy": [100.0, 100.0, 200.0, 200.0],
                "best_frame_key": "signtrustmap/candidates/sub_api_04/frame.jpg",
                "sign_crop_key": "signtrustmap/candidates/sub_api_04/crop.jpg",
                "frame_index": 0,
                "timestamp_seconds": 0.0,
                "category": "Regulatory",
                "name_vi": "Cấm đỗ xe",
                "name_en": "No parking",
                "standard_code": "P.131a",
            }
        ],
    }

    monkeypatch.setattr(inference_engine, "process", lambda sid, objs: mock_result)

    payload = {
        "media_objects": [
            {"media_type": "IMAGE", "bucket": "signtrustmap", "object_key": "raw/image.jpg"}
        ]
    }
    resp = api_client.post(
        "/api/v1/inference/submissions/sub_api_04",
        json=payload,
        headers={"X-Internal-Token": settings.INTERNAL_API_TOKEN},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["submission_id"] == "sub_api_04"
    assert data["media_type"] == "IMAGE"
    assert len(data["signs"]) == 1
    assert data["signs"][0]["class_id"] == 20
    assert data["signs"][0]["standard_code"] == "P.131a"
    logger.info("AI-API-04 verified: Submission inference matches schema and returns 200.")

