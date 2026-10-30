from typing import Dict, Any, List
import pytest
from fastapi.testclient import TestClient
from loguru import logger
from main import app
from src.core.config import settings


@pytest.fixture(scope="module")
def api_client():
    with TestClient(app) as client:
        yield client


def test_ai_it_10_end_to_end_internal_security(api_client: TestClient):
    resp_no_token = api_client.post(
        "/api/v1/inference/submissions/sec_test",
        json={"media_objects": [{"media_type": "IMAGE", "bucket": "b", "object_key": "k.jpg"}]},
    )
    assert resp_no_token.status_code in (401, 403, 422)

    resp_bad_token = api_client.post(
        "/api/v1/inference/submissions/sec_test",
        json={"media_objects": [{"media_type": "IMAGE", "bucket": "b", "object_key": "k.jpg"}]},
        headers={"X-Internal-Token": "bad_token_12345"},
    )
    assert resp_bad_token.status_code in (401, 403)

    resp_rebuild_bad = api_client.post(
        "/api/v1/active-learning/classifier/rebuild",
        headers={"X-Internal-Token": "invalid_rebuild_token"},
    )
    assert resp_rebuild_bad.status_code in (401, 403)

    logger.info("AI-IT-10 verified: Internal endpoints enforce authentication security.")

