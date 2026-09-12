import os
import shutil
import tempfile

import cv2

from src.core.class_manager import class_manager
from src.infrastructure.minio_client import minio_client
from src.pipelines.frame_extractor import extract_video_frame


def infer_video(
    submission_id: str,
    detector,
    video_path: str,
    classifier=None,
    conf: float = 0.6,
    device: int = 0,
) -> dict:
    """Track signs in a video and return one candidate per track."""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    groups = {}
    frames_processed = 0
    for frame_idx, tracked_objects in detector.track(video_path, conf=conf, device=device):
        frames_processed = frame_idx + 1
        for tracked_object in tracked_objects:
            groups.setdefault(tracked_object["id"], []).append({
                **tracked_object,
                "frame_idx": frame_idx,
                "timestamp_seconds": round(frame_idx / fps, 3),
            })

    artifact_dir = tempfile.mkdtemp(prefix=f"stm-{submission_id}-")
    signs = []
    try:
        for track_id, detections in groups.items():
            candidates = _top_candidates(detections, conf)
            best = _best_candidate(
                candidates,
                video_path,
                artifact_dir,
                track_id,
                classifier,
            )
            if best is None:
                continue

            frame_key = f"candidates/{submission_id}/track-{track_id}-frame.jpg"
            frame_uploaded, frame_storage_key = minio_client.upload_file(
                best["frame_path"], frame_key, content_type="image/jpeg"
            )
            crop = _crop_box(best["frame"], best["candidate"]["box_xyxy"])
            crop_path = os.path.join(artifact_dir, f"track-{track_id}-crop.jpg")
            if crop.size == 0 or not cv2.imwrite(crop_path, crop):
                raise RuntimeError(f"Could not create crop for track {track_id}")
            crop_key = f"candidates/{submission_id}/track-{track_id}-crop.jpg"
            crop_uploaded, crop_storage_key = minio_client.upload_file(
                crop_path, crop_key, content_type="image/jpeg"
            )
            if not frame_uploaded or not crop_uploaded or not frame_storage_key or not crop_storage_key:
                raise RuntimeError(f"Could not upload artifacts for track {track_id}")

            candidate = best["candidate"]
            classify_confidence = best["classify_confidence"]
            sign_meta = class_manager.get_sign_metadata(best["class_id"])
            signs.append({
                "track_id": track_id,
                "class_id": best["class_id"],
                "class_name": best["class_name"],
                "confidence": classify_confidence if classify_confidence is not None else candidate["confidence"],
                "detect_confidence": candidate.get("detect_confidence", candidate["confidence"]),
                "classify_confidence": classify_confidence,
                "box_xyxy": candidate["box_xyxy"],
                "best_frame_key": frame_storage_key,
                "sign_crop_key": crop_storage_key,
                "frame_index": candidate["frame_idx"],
                "timestamp_seconds": candidate["timestamp_seconds"],
                "category": sign_meta.get("category"),
                "name_vi": sign_meta.get("name_vi"),
                "name_en": sign_meta.get("name_en"),
                "standard_code": sign_meta.get("standard_code"),
            })
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)

    return {
        "video_fps": round(fps, 2),
        "total_frames": frames_processed or total_video_frames,
        "signs": signs,
    }


def _top_candidates(detections: list[dict], confidence: float) -> list[dict]:
    return sorted(
        (item for item in detections if item.get("confidence", 0.0) >= confidence),
        key=lambda item: (
            item.get("confidence", 0.0),
            (item["box_xyxy"][2] - item["box_xyxy"][0])
            * (item["box_xyxy"][3] - item["box_xyxy"][1]),
        ),
        reverse=True,
    )[:3]


def _best_candidate(candidates, video_path, artifact_dir, track_id, classifier):
    best = None
    for candidate in candidates:
        frame_path = os.path.join(
            artifact_dir,
            f"track-{track_id}-frame-{candidate['frame_idx']}.jpg",
        )
        if not extract_video_frame(video_path, candidate["frame_idx"], frame_path):
            continue
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        class_id = candidate["class_id"]
        class_name = candidate["class_name"]
        classify_confidence = None
        if classifier is not None:
            result = classifier.classify_crop(
                classifier.crop_box(frame, candidate["box_xyxy"]),
            )
            if result.get("class_id") is not None:
                class_id = result["class_id"]
                class_name = result["class_name"]
                classify_confidence = result.get("classify_confidence")
        score = candidate.get("detect_confidence", candidate["confidence"])
        score *= classify_confidence if classify_confidence is not None else 1.0
        if best is None or score > best["score"]:
            best = {
                "candidate": candidate,
                "class_id": class_id,
                "class_name": class_name,
                "classify_confidence": classify_confidence,
                "frame": frame,
                "frame_path": frame_path,
                "score": score,
            }
    return best


def _crop_box(image, box_xyxy):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in box_xyxy]
    return image[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
