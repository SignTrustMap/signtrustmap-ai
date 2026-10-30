import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import MagicMock
import cv2
import numpy as np
import pytest
from loguru import logger
from src.core.classifier import SignClassifier
from src.core.detector import SignDetector
from src.infrastructure.minio_client import minio_client
from src.pipelines.video_pipeline import _best_candidate, _crop_box, _top_candidates, infer_video


def test_ai_vid_01_track_deduplication(detector: SignDetector, classifier: SignClassifier, real_videos_dir: Path, mock_minio):
    vid_path = str(real_videos_dir / "clip_000505.mp4")
    res = infer_video(
        submission_id="sub_vid_01",
        detector=detector,
        video_path=vid_path,
        classifier=classifier,
        conf=0.5,
    )

    assert "signs" in res
    assert "total_frames" in res
    assert res["total_frames"] == 169
    assert len(res["signs"]) == 3

    track_ids = {s["track_id"] for s in res["signs"]}
    assert len(track_ids) == 3
    for s in res["signs"]:
        assert "candidates/sub_vid_01/" in s["best_frame_key"]
        assert "candidates/sub_vid_01/" in s["sign_crop_key"]
        assert s["standard_code"] is not None
    logger.info("AI-VID-01 verified: exactly 3 track representatives produced.")


def test_ai_vid_02_top_candidates_filtering():
    raw_detections = [
        {"confidence": 0.45, "box_xyxy": [0, 0, 10, 10]},
        {"confidence": 0.70, "box_xyxy": [0, 0, 20, 20]},
        {"confidence": 0.85, "box_xyxy": [0, 0, 30, 30]},
        {"confidence": 0.90, "box_xyxy": [0, 0, 40, 40]},
        {"confidence": 0.95, "box_xyxy": [0, 0, 50, 50]},
    ]
    top = _top_candidates(raw_detections, confidence=0.5)
    assert len(top) == 3
    assert top[0]["confidence"] == 0.95
    assert top[1]["confidence"] == 0.90
    assert top[2]["confidence"] == 0.85
    logger.info("AI-VID-02 verified: top candidates correctly filtered and capped at 3.")


def test_ai_vid_03_best_candidate_selection(real_videos_dir: Path):
    vid_path = str(real_videos_dir / "clip_000505.mp4")
    candidates = [
        {
            "id": 1,
            "class_id": 20,
            "class_name": "No_parking",
            "confidence": 0.80,
            "detect_confidence": 0.80,
            "box_xyxy": [1220.0, 180.0, 1340.0, 290.0],
            "frame_idx": 95,
            "timestamp_seconds": 3.96,
        },
        {
            "id": 1,
            "class_id": 20,
            "class_name": "No_parking",
            "confidence": 0.95,
            "detect_confidence": 0.95,
            "box_xyxy": [1224.42, 180.79, 1340.55, 290.19],
            "frame_idx": 98,
            "timestamp_seconds": 4.08,
        },
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        fake_classifier = MagicMock()
        fake_classifier.crop_box.return_value = np.zeros((50, 50, 3), dtype=np.uint8)
        fake_classifier.classify_crop.return_value = {
            "top1": {"class_id": 20, "class_name": "No_parking", "classify_confidence": 0.99},
            "top5": [],
            "class_id": 20,
            "class_name": "No_parking",
            "classify_confidence": 0.99,
        }

        best = _best_candidate(candidates, vid_path, tmp_dir, track_id=1, classifier=fake_classifier)
        assert best is not None
        assert best["candidate"]["frame_idx"] == 98
        assert best["class_id"] == 20
        logger.info("AI-VID-03 verified: best candidate accurately identified.")


def test_ai_vid_04_invalid_video_path(detector: SignDetector):
    invalid_path = "/nonexistent/path/video.mp4"
    with pytest.raises(ValueError, match="Cannot open video file"):
        infer_video("sub_vid_04", detector, invalid_path)
    logger.info("AI-VID-04 verified: ValueError on invalid video path.")


def test_ai_vid_05_minio_upload_failure(detector: SignDetector, real_videos_dir: Path, monkeypatch: pytest.MonkeyPatch):
    def failing_upload(*args, **kwargs):
        return False, None

    monkeypatch.setattr(minio_client, "upload_file", failing_upload)
    vid_path = str(real_videos_dir / "clip_000505.mp4")
    with pytest.raises(RuntimeError, match="Could not upload artifacts"):
        infer_video("sub_vid_05", detector, vid_path, conf=0.5)
    logger.info("AI-VID-05 verified: RuntimeError raised when upload fails.")


def test_ai_vid_06_video_with_no_signs(real_videos_dir: Path, mock_minio):
    fake_detector = MagicMock()
    fake_detector.track.return_value = iter([(i, []) for i in range(10)])

    vid_path = str(real_videos_dir / "clip_000505.mp4")
    res = infer_video("sub_vid_06", fake_detector, vid_path)
    assert res["signs"] == []
    assert res["total_frames"] == 10
    assert res["video_fps"] > 0
    logger.info("AI-VID-06 verified: video with zero signs returns empty list.")

