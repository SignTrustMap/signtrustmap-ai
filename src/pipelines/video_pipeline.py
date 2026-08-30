import os
import shutil
import cv2
from loguru import logger
from src.core.frame_extractor import extract_video_frame
from src.infrastructure.minio_client import upload_frame_to_minio
from src.pipelines.utils import send_callback


def run_video_pipeline(
    task_id: str,
    callback_url: str,
    detector,
    video_path: str,
    static_dir: str,
    conf: float = 0.6,
    device: int = 0,
) -> None:
    logger.info(f"[{task_id}] Starting pure YOLO video tracking pipeline for {video_path} (conf={conf}, device={device})")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        msg = f"Cannot open video file: {video_path}"
        logger.error(f"[{task_id}] {msg}")
        payload = {"task_id": task_id, "status": "FAILED", "video_fps": None, "total_frames": 0, "total_signs": 0, "result": None, "error": msg}
        send_callback(callback_url, payload)
        clean_temp_file(video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    logger.debug(f"[{task_id}] Video metadata: FPS={fps:.2f}, total_frames={total_video_frames}")
    groups = {}
    frames_processed = 0

    try:
        for frame_idx, tracked_objects in detector.track(video_path, conf=conf, device=device):
            frames_processed = frame_idx + 1
            frame_seconds = round(frame_idx / fps, 3)

            for obj in tracked_objects:
                obj_id = obj["id"]
                logger.trace(f"[{task_id}] Frame {frame_idx} ({frame_seconds}s): Track ID {obj_id} ({obj['class_name']}) conf={obj['confidence']:.4f}")

                if obj_id not in groups:
                    groups[obj_id] = []

                groups[obj_id].append({
                    "frame_idx": frame_idx,
                    "timestamp_seconds": frame_seconds,
                    "class_id": obj["class_id"],
                    "class_name": obj["class_name"],
                    "confidence": obj["confidence"],
                    "box_xyxy": obj["box_xyxy"],
                })

    except Exception as e:
        msg = f"Exception occurred during YOLO video tracking: {str(e)}"
        logger.exception(f"[{task_id}] Video tracking failed: {msg}")
        payload = {"task_id": task_id, "status": "FAILED", "video_fps": fps, "total_frames": frames_processed, "total_signs": 0, "result": None, "error": msg}
        send_callback(callback_url, payload)
        clean_temp_file(video_path)
        return

    logger.debug(f"[{task_id}] Tracking complete. Detected {len(groups)} unique tracking sign group(s) across {frames_processed} frame(s).")

    final_signs = []
    task_frames_dir = os.path.join(static_dir, task_id)
    os.makedirs(task_frames_dir, exist_ok=True)

    for obj_id, detections in groups.items():
        valid_detections = [d for d in detections if d.get("confidence", 0.0) >= conf]
        if not valid_detections:
            continue

        best_det = max(
            valid_detections,
            key=lambda x: (
                x.get("confidence", 0.0),
                (x["box_xyxy"][2] - x["box_xyxy"][0]) * (x["box_xyxy"][3] - x["box_xyxy"][1]) if "box_xyxy" in x else 0.0
            )
        )

        frame_filename = f"sign_{obj_id}_conf_{best_det['confidence']:.4f}.jpg"
        output_frame_path = os.path.join(task_frames_dir, frame_filename)

        frame_url = None
        if extract_video_frame(video_path, best_det["frame_idx"], output_frame_path):
            upload_success, upload_url = upload_frame_to_minio(output_frame_path, f"{task_id}/sign_{obj_id}.jpg")
            if upload_success:
                frame_url = upload_url

        final_signs.append({
            "id": obj_id,
            "class_id": best_det["class_id"],
            "class_name": best_det["class_name"],
            "confidence": best_det["confidence"],
            "best_frame_idx": best_det["frame_idx"],
            "best_timestamp_seconds": best_det["timestamp_seconds"],
            "box_xyxy": best_det["box_xyxy"],
            "occurrences": len(valid_detections),
            "frame_image_url": frame_url,
        })

    if os.path.exists(task_frames_dir):
        shutil.rmtree(task_frames_dir, ignore_errors=True)

    logger.info(f"[{task_id}] Video pipeline completed. Tracked {len(final_signs)} unique traffic sign(s).")

    payload = {
        "task_id": task_id,
        "status": "COMPLETED",
        "video_fps": round(fps, 2),
        "total_frames": frames_processed if frames_processed > 0 else total_video_frames,
        "total_signs": len(final_signs),
        "result": final_signs,
        "error": None,
    }
    send_callback(callback_url, payload)
    clean_temp_file(video_path)


def clean_temp_file(file_path: str) -> None:
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Removed temporary file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up temp file {file_path}: {e}")
