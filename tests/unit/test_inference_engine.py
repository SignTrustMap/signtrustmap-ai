from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import MagicMock
import pytest
from loguru import logger
from src.infrastructure.minio_client import minio_client
from src.pipelines.inference_engine import SubmissionInferenceEngine


def test_ai_inf_01_primary_media_selection_priority():
    media_list = [
        {"media_type": "GPX", "bucket": "b", "object_key": "track.gpx"},
        {"media_type": "IMAGE", "bucket": "b", "object_key": "frame.jpg"},
        {"media_type": "VIDEO", "bucket": "b", "object_key": "clip.mp4"},
    ]
    primary = SubmissionInferenceEngine._primary_media(media_list)
    assert primary["media_type"] == "VIDEO"
    assert primary["object_key"] == "clip.mp4"
    logger.info("AI-INF-01 verified: VIDEO prioritized over IMAGE and GPX.")


def test_ai_inf_02_gpx_only_error():
    media_list = [
        {"media_type": "GPX", "bucket": "b", "object_key": "track.gpx"}
    ]
    with pytest.raises(ValueError, match="A VIDEO or IMAGE MinIO object is required"):
        SubmissionInferenceEngine._primary_media(media_list)
    logger.info("AI-INF-02 verified: ValueError raised when no visual media present.")


def test_ai_inf_03_minio_download_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(minio_client, "download_file", lambda b, o, l: False)
    engine = SubmissionInferenceEngine()
    media_list = [{"media_type": "IMAGE", "bucket": "b", "object_key": "img.jpg"}]
    with pytest.raises(RuntimeError, match="Unable to download"):
        engine.process("sub_inf_03", media_list)
    logger.info("AI-INF-03 verified: RuntimeError raised on MinIO download failure.")


def test_ai_inf_04_thread_lock_synchronization():
    engine = SubmissionInferenceEngine()
    assert not engine._inference_lock.locked()
    with engine._inference_lock:
        assert engine._inference_lock.locked()
    assert not engine._inference_lock.locked()
    logger.info("AI-INF-04 verified: threading Lock functions correctly.")

