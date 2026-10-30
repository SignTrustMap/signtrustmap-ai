from pathlib import Path
from typing import Dict, Any, List
import pytest
from loguru import logger
from src.core.active_learning.store import ActiveLearningStore


def test_ai_st_01_atomic_save_and_load(tmp_path: Path):
    store_file = tmp_path / "custom_store.json"
    store = ActiveLearningStore(custom_path=store_file)

    initial_data = store.load_data()
    assert "detector" in initial_data
    assert "classifier" in initial_data
    assert "clip" in initial_data

    sample = store.add_detector_sample("obj_01.jpg", [0, 0, 10, 10], 0.7)
    assert sample["id"].startswith("det_")

    reloaded = store.load_data()
    assert len(reloaded["detector"]["pending"]) == 1
    assert reloaded["detector"]["pending"][0]["id"] == sample["id"]
    logger.info("AI-ST-01 verified: Atomic store save and load functional.")


def test_ai_st_02_get_counts_accuracy(tmp_path: Path):
    store_file = tmp_path / "counts_store.json"
    store = ActiveLearningStore(custom_path=store_file)

    store.add_detector_sample("det.jpg", [0, 0, 10, 10], 0.8)
    store.add_classifier_sample("cls.jpg", [0, 0, 20, 20], 1, "c1", 0.9, 2, "c2")
    store.add_clip_sample("clip.jpg", "new_sign", [0, 0, 30, 30])

    counts = store.get_counts()
    assert counts["detector"]["pending"] == 1
    assert counts["classifier"]["pending"] == 1
    assert counts["clip"]["pending"] == 1
    logger.info("AI-ST-02 verified: get_counts reflects active entries.")


def test_ai_st_03_mark_as_trained(tmp_path: Path):
    store_file = tmp_path / "trained_store.json"
    store = ActiveLearningStore(custom_path=store_file)

    s1 = store.add_detector_sample("det1.jpg", [0, 0, 10, 10], 0.8)
    s2 = store.add_detector_sample("det2.jpg", [0, 0, 10, 10], 0.85)

    counts_before = store.get_counts()
    assert counts_before["detector"]["pending"] == 2
    assert counts_before["detector"]["trained"] == 0

    store.mark_as_trained("detector", [s1["id"]])

    counts_after = store.get_counts()
    assert counts_after["detector"]["pending"] == 1
    assert counts_after["detector"]["trained"] == 1
    logger.info("AI-ST-03 verified: mark_as_trained moves samples from pending to trained.")


def test_ai_st_04_set_store_path_isolation(tmp_path: Path):
    path_a = tmp_path / "store_a.json"
    path_b = tmp_path / "store_b.json"

    store = ActiveLearningStore()
    store.set_store_path(path_a)
    store.add_detector_sample("a.jpg", [0, 0, 10, 10], 0.5)

    store.set_store_path(path_b)
    store.add_detector_sample("b.jpg", [0, 0, 10, 10], 0.6)

    store.set_store_path(path_a)
    assert len(store.load_data()["detector"]["pending"]) == 1
    assert store.load_data()["detector"]["pending"][0]["object_name"] == "a.jpg"

    store.set_store_path(path_b)
    assert len(store.load_data()["detector"]["pending"]) == 1
    assert store.load_data()["detector"]["pending"][0]["object_name"] == "b.jpg"
    logger.info("AI-ST-04 verified: set_store_path cleanly isolates data stores.")

