import pytest
from fastapi.testclient import TestClient
from loguru import logger
from main import app
from src.core.config_manager import config_manager
from src.infrastructure.db import sync_engine
from src.infrastructure.models import AiSystemConfig
from sqlalchemy.orm import Session


@pytest.fixture
def client():
    return TestClient(app)


def test_ai_cfg_01_loaded_canonical_keys():
    configs = config_manager.get_all()
    assert "detector" in configs
    assert "conf_thresh" in configs["detector"]
    assert "classifier" in configs
    assert "conf_thresh" in configs["classifier"]
    assert "temperature" in configs["classifier"]
    assert "prompt_template" in configs["classifier"]
    assert "timo" in configs
    assert "alpha" in configs["timo"]
    assert "beta" in configs["timo"]
    assert "few_shot_support" in configs
    assert "shots_per_class" in configs["few_shot_support"]
    assert "num_anchors" in configs["few_shot_support"]
    assert "num_medoids" in configs["few_shot_support"]
    assert "min_crop_size" in configs["few_shot_support"]
    assert "min_sharpness" in configs["few_shot_support"]
    assert "active_learning" in configs
    assert "strategy" in configs["active_learning"]
    assert "query_budget" in configs["active_learning"]
    assert "aggregation" in configs["active_learning"]
    assert "no_detection_score" in configs["active_learning"]
    assert "min_samples_detector" in configs["active_learning"]
    assert "min_samples_classifier" in configs["active_learning"]
    assert "discrepancy_thresh" in configs["active_learning"]
    assert "auto_trigger_retrain" in configs["active_learning"]
    assert "r3_cec_k_neighbors" in configs["active_learning"]
    assert "r3_cec_temperature" in configs["active_learning"]
    assert "system" in configs
    assert "model_idle_timeout" in configs["system"]
    assert "cuda_device" in configs["system"]
    logger.info("AI-CFG-01 verified: all 23 canonical keys present in runtime config.")


def test_ai_cfg_02_no_legacy_keys():
    assert config_manager.get("detector.default_conf") is None
    assert config_manager.get("classifier.default_conf") is None
    assert config_manager.get("active_learning.default_strategy") is None
    assert config_manager.get("active_learning.default_top_k") is None
    assert config_manager.get("system.default_device") is None
    assert config_manager.get("system.model_idle_timeout_seconds") is None
    logger.info("AI-CFG-02 verified: legacy keys are completely deleted and return None.")


def test_ai_cfg_03_update_group_and_persistence():
    orig = config_manager.detector_conf_thresh
    new_val = 0.65
    config_manager.update_group("detector", {"conf_thresh": new_val})

    assert config_manager.detector_conf_thresh == new_val
    with Session(sync_engine) as session:
        row = session.query(AiSystemConfig).filter_by(config_key="detector.conf_thresh").first()
        assert row is not None
        assert row.config_value == new_val

    config_manager.update_group("detector", {"conf_thresh": orig})
    assert config_manager.detector_conf_thresh == orig
    logger.info("AI-CFG-03 verified: update_group persists changes to AI Database and memory.")


def test_ai_cfg_04_api_routes(client: TestClient):
    resp = client.get("/api/v1/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "detector" in data
    assert "classifier" in data
    assert "active_learning" in data

    resp_group = client.get("/api/v1/config/detector")
    assert resp_group.status_code == 200
    assert "conf_thresh" in resp_group.json()

    patch_resp = client.patch("/api/v1/config/detector", json={"conf_thresh": 0.7})
    assert patch_resp.status_code == 200
    assert patch_resp.json()["conf_thresh"] == 0.7

    invalid_patch = client.patch("/api/v1/config/active_learning", json={"strategy": "invalid_strategy"})
    assert invalid_patch.status_code == 400

    invalid_range = client.patch("/api/v1/config/detector", json={"conf_thresh": 1.5})
    assert invalid_range.status_code == 400

    reset_resp = client.post("/api/v1/config/reset")
    assert reset_resp.status_code == 200
    assert reset_resp.json()["status"] == "RESET_SUCCESSFUL"
    assert reset_resp.json()["config"]["detector"]["conf_thresh"] == 0.6
    logger.info("AI-CFG-04 verified: Config API routes and validations perform correctly.")

