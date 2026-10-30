from pathlib import Path
from typing import Dict, Any, List
import time
import pytest
from loguru import logger
from src.core.model_manager import ModelManager


def test_ai_mm_01_lazy_loading_cache():
    manager = ModelManager()
    det1 = manager.get_detector()
    det2 = manager.get_detector()
    assert det1 is det2
    logger.info("AI-MM-01 verified: ModelManager caches loaded detector instance.")


def test_ai_mm_02_touch_updates_timestamp():
    manager = ModelManager()
    before = time.time()
    time.sleep(0.01)
    manager.touch()
    assert manager._last_accessed >= before
    logger.info("AI-MM-02 verified: Touch method updates last_accessed timestamp.")


def test_ai_mm_03_unload_resources():
    manager = ModelManager()
    _ = manager.get_detector()
    assert manager._detector is not None

    unloaded = manager.unload_detector()
    assert unloaded is True
    assert manager._detector is None
    assert manager._loaded_detector_path is None
    logger.info("AI-MM-03 verified: Unload releases detector and frees memory.")

