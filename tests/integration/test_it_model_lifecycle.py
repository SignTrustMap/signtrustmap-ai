from pathlib import Path
from typing import Dict, Any, List
import pytest
from fastapi.testclient import TestClient
from loguru import logger
from main import app
from src.core.config import settings
from src.core.model_manager import model_manager


@pytest.fixture(scope="module")
def api_client():
    with TestClient(app) as client:
        yield client


def test_ai_it_08_hot_reload_timo(api_client: TestClient):
    resp = api_client.post(
        "/api/v1/active-learning/classifier/rebuild",
        headers={"X-Internal-Token": settings.INTERNAL_API_TOKEN},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "SUCCESS"
    assert "weights_shape" in data
    assert "bias_shape" in data
    assert data["weights_shape"][1] == 100
    assert data["bias_shape"][0] == 100
    logger.info("AI-IT-08 verified: TIMO classifier hot-reloaded in memory via API.")


def test_ai_it_09_model_lifecycle_load_unload(api_client: TestClient):
    load_resp = api_client.post(
        "/api/v1/models/load/detector",
        json={"weights_name": "yolo26n_vtsdb_mtsd_dfg.engine"},
    )
    assert load_resp.status_code == 200
    load_data = load_resp.json()
    assert load_data["status"] == "LOADED"
    assert load_data["loaded_detector"] == "yolo26n_vtsdb_mtsd_dfg.engine"

    unload_resp = api_client.post("/api/v1/models/unload/detector")
    assert unload_resp.status_code == 200
    unload_data = unload_resp.json()
    assert unload_data["status"] in ("UNLOADED", "ALREADY_UNLOADED")
    logger.info("AI-IT-09 verified: Detector model loaded and unloaded cleanly.")

