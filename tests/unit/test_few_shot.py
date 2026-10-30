from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pytest
from loguru import logger
from src.core.few_shot_manager import FewShotSupportManager, compute_sharpness


def test_ai_fs_01_compute_sharpness():
    assert compute_sharpness(None) == 0.0
    assert compute_sharpness(np.zeros((0, 0, 3), dtype=np.uint8)) == 0.0

    smooth = np.full((50, 50, 3), 128, dtype=np.uint8)
    assert compute_sharpness(smooth) == 0.0

    textured = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    sharpness = compute_sharpness(textured)
    assert sharpness > 0.0
    logger.info(f"AI-FS-01 verified: Laplacian sharpness computed successfully ({sharpness:.2f}).")


def test_ai_fs_02_preserve_pinned_shots():
    manager = FewShotSupportManager()
    candidates = [
        {"id": "c1", "is_pinned": True, "score": 0.1},
        {"id": "c2", "is_pinned": False, "score": 0.9},
        {"id": "c3", "is_pinned": True, "score": 0.2},
    ]
    feats = np.random.randn(3, 8)
    selected = manager.select_support_shots(candidates, visual_features=feats, total_shots=2)
    selected_ids = {item["id"] for item in selected}
    assert "c1" in selected_ids
    assert "c3" in selected_ids
    logger.info("AI-FS-02 verified: Pinned support shots preserved unconditionally.")


def test_ai_fs_03_pool_within_budget():
    manager = FewShotSupportManager()
    candidates = [
        {"id": "s1", "box_xyxy": [0, 0, 10, 10]},
        {"id": "s2", "box_xyxy": [0, 0, 20, 20]},
    ]
    selected = manager.select_support_shots(candidates, total_shots=5, num_anchors=1)
    assert len(selected) == 2
    assert selected[0]["role"] == "ANCHOR"
    assert selected[1]["role"] == "MEDOID"
    logger.info("AI-FS-03 verified: Small pool retains all candidates with appropriate roles.")


def test_ai_fs_04_default_configurations():
    manager = FewShotSupportManager()
    cfg = manager.get_support_config()
    assert cfg["total_shots_per_class"] >= 1
    assert cfg["num_anchors"] >= 1
    assert cfg["preserve_pinned_shots"] is True

    timo_cfg = manager.get_timo_config()
    assert timo_cfg["alpha"] > 0
    assert 0.0 < timo_cfg["beta"] <= 1.0
    logger.info("AI-FS-04 verified: Support manager configurations match expected defaults.")

