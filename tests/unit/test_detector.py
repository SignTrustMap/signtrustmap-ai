from pathlib import Path
from typing import Dict, Any, List
import pytest
from loguru import logger
from src.core.detector import SignDetector


def test_ai_det_01_valid_detection(detector: SignDetector, verified_images_dir: Path):
    img_path = str(verified_images_dir / "11288.jpg")
    detections = detector.detect(img_path, conf=0.4)
    assert isinstance(detections, list)
    assert len(detections) >= 1
    det = detections[0]
    assert "class_id" in det
    assert "class_name" in det
    assert "confidence" in det
    assert det["confidence"] >= 0.4
    assert len(det["box_xyxy"]) == 4
    x1, y1, x2, y2 = det["box_xyxy"]
    assert x2 > x1 and y2 > y1
    logger.info(f"AI-DET-01 verified with {len(detections)} detection(s).")


def test_ai_det_02_multi_box_detection(detector: SignDetector, verified_images_dir: Path):
    img_path = str(verified_images_dir / "10000.jpg")
    detections = detector.detect(img_path, conf=0.3)
    assert isinstance(detections, list)
    assert len(detections) >= 1
    for d in detections:
        assert d["confidence"] >= 0.3
        assert len(d["box_xyxy"]) == 4
    logger.info(f"AI-DET-02 verified with {len(detections)} detection(s).")


def test_ai_det_03_blank_canvas_zero_detections(detector: SignDetector, verified_images_dir: Path):
    blank_path = str(verified_images_dir / "blank_canvas.jpg")
    detections = detector.detect(blank_path, conf=0.5)
    assert detections == []
    logger.info("AI-DET-03 verified: zero detections on blank canvas.")


def test_ai_det_04_nonexistent_image_exception(detector: SignDetector):
    nonexistent = "/path/to/nonexistent_image_123456789.jpg"
    with pytest.raises(Exception):
        detector.detect(nonexistent)
    logger.info("AI-DET-04 verified: exception raised for invalid path.")


def test_ai_trk_01_real_video_tracks(detector: SignDetector, real_videos_dir: Path, verified_tracks_metadata: Dict[str, Any]):
    vid_path = str(real_videos_dir / "clip_000505.mp4")
    generator = detector.track(vid_path, conf=0.5)
    
    discovered_tracks = set()
    frame_count = 0
    for frame_idx, tracked in generator:
        frame_count += 1
        for obj in tracked:
            discovered_tracks.add(obj["id"])
    
    assert frame_count == verified_tracks_metadata["total_frames"]
    assert len(discovered_tracks) >= len(verified_tracks_metadata["tracks"])
    logger.info(f"AI-TRK-01 verified: tracked {len(discovered_tracks)} distinct tracks across {frame_count} frames.")


def test_ai_trk_02_empty_frames_handling(detector: SignDetector, real_videos_dir: Path):
    vid_path = str(real_videos_dir / "clip_000505.mp4")
    generator = detector.track(vid_path, conf=0.5)
    first_frame_idx, first_frame_tracked = next(generator)
    assert isinstance(first_frame_tracked, list)
    assert len(first_frame_tracked) == 0
    for _ in generator:
        pass
    logger.info("AI-TRK-02 verified: early frame yields empty list safely.")


def test_ai_trk_03_multi_object_frame(detector: SignDetector, real_videos_dir: Path):
    vid_path = str(real_videos_dir / "clip_000505.mp4")
    generator = detector.track(vid_path, conf=0.5)
    
    multi_obj_found = False
    for frame_idx, tracked in generator:
        if len(tracked) >= 2:
            multi_obj_found = True
            ids = [obj["id"] for obj in tracked]
            assert len(ids) == len(set(ids))
    assert multi_obj_found
    logger.info("AI-TRK-03 verified: multi-object frame tracked with unique IDs.")
