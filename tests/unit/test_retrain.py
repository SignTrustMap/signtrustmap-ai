from pathlib import Path
from typing import Dict, Any, List
import pytest
from loguru import logger
from src.core.active_learning.retrain import trigger_retrain
from src.core.active_learning.store import al_store


@pytest.mark.asyncio
async def test_ai_rt_01_invalid_target():
    res = await trigger_retrain(target="invalid_component")
    assert res["status"] == "FAILED"
    assert "error" in res
    logger.info("AI-RT-01 verified: Invalid target rejected with FAILED.")


@pytest.mark.asyncio
async def test_ai_rt_02_zero_pending_skipped():
    res = await trigger_retrain(target="detector")
    assert res["status"] == "SKIPPED"
    assert res["sample_count"] == 0
    logger.info("AI-RT-02 verified: Zero pending samples results in SKIPPED status.")


@pytest.mark.asyncio
async def test_ai_rt_03_dry_run_success():
    al_store.add_classifier_sample("img_dry.jpg", [0, 0, 10, 10], 1, "c1", 0.9, 2, "c2")
    res = await trigger_retrain(target="classifier", dry_run=True)
    assert res["status"] == "SUCCESS"
    assert res["dry_run"] is True
    assert res["sample_count"] >= 1

    counts = al_store.get_counts()
    assert counts["classifier"]["pending"] >= 1
    assert counts["classifier"]["trained"] == 0
    logger.info("AI-RT-03 verified: Dry run does not mutate pending store state.")


@pytest.mark.asyncio
async def test_ai_rt_04_execution_and_history_recording():
    al_store.add_detector_sample("img_train.jpg", [0, 0, 10, 10], 0.7)
    res = await trigger_retrain(target="detector", dry_run=False, mock_duration_seconds=0.01)
    assert res["status"] == "SUCCESS"
    assert res["new_weights"] == "detector_al_mock_latest.pt"

    counts = al_store.get_counts()
    assert counts["detector"]["trained"] >= 1
    assert counts["detector"]["pending"] == 0

    data = al_store.load_data()
    assert len(data["train_history"]) >= 1
    history_entry = data["train_history"][-1]
    assert history_entry["target_model"] == "DETECTOR"
    assert history_entry["status"] == "SUCCESS"
    logger.info("AI-RT-04 verified: Retraining marks samples as trained and records history.")

