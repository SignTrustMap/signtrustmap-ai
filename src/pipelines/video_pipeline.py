import os
import json
import shutil
import cv2
import subprocess
from datetime import datetime, timedelta
from loguru import logger
from src.core.gpx_parser import (
    parse_gpx_points,
    extract_gps_from_video,
    get_interpolated_gps_track,
    interpolate_position,
    parse_gpx_time
)
from src.core.geolocator import geolocate_sign_group
from src.core.frame_extractor import extract_video_frame
from src.infrastructure.minio_client import upload_frame_to_minio
from src.pipelines.utils import send_callback

def get_video_creation_time_ict(file_path):
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_entries", "format_tags=creation_time",
        file_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        data = json.loads(result.stdout)
        creation_time = data.get("format", {}).get("tags", {}).get("creation_time")
        if creation_time:
            parsed = parse_gpx_time(creation_time)
            if parsed:
                return parsed
    except Exception:
        pass
    try:
        timestamp = os.path.getmtime(file_path)
        return datetime.fromtimestamp(timestamp)
    except Exception:
        return datetime.now()

def run_video_pipeline(task_id, callback_url, detector, video_path, gpx_path, static_dir, fov=90.0, conf=0.6, device=0):
    logger.info(f"[{task_id}] Starting video pipeline processing.")
    
    points = []
    if gpx_path and os.path.exists(gpx_path):
        logger.info(f"[{task_id}] Parsing provided GPX file: {gpx_path}")
        points = parse_gpx_points(gpx_path)
    else:
        logger.info(f"[{task_id}] Extracting telemetry from video file GPMF...")
        points = extract_gps_from_video(video_path)
        
    if not points:
        msg = "No GPS points found. Telemetry extraction failed and no valid GPX file was provided."
        logger.error(f"[{task_id}] {msg}")
        payload = {"task_id": task_id, "status": "FAILED", "result": None, "error": msg}
        send_callback(callback_url, payload)
        clean_temp_files(video_path, gpx_path)
        return
        
    track_data = get_interpolated_gps_track(points)
    if not track_data:
        msg = "Failed to interpolate GPS track. Ensure the GPX has at least 2 unique timestamp points."
        logger.error(f"[{task_id}] {msg}")
        payload = {"task_id": task_id, "status": "FAILED", "result": None, "error": msg}
        send_callback(callback_url, payload)
        clean_temp_files(video_path, gpx_path)
        return
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        msg = f"Cannot open video file: {video_path}"
        logger.error(f"[{task_id}] {msg}")
        payload = {"task_id": task_id, "status": "FAILED", "result": None, "error": msg}
        send_callback(callback_url, payload)
        clean_temp_files(video_path, gpx_path)
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    cap.release()
    
    base_datetime = get_video_creation_time_ict(video_path)
    logger.debug(f"[{task_id}] Video profile - FPS: {fps}, start time: {base_datetime}")
    groups = {}
    
    logger.info(f"[{task_id}] Running YOLO tracking...")
    try:
        for frame_idx, tracked_objects in detector.track(video_path, conf=conf, device=device):
            frame_offset_seconds = frame_idx / fps
            frame_time = base_datetime + timedelta(seconds=frame_offset_seconds)
            query_ts = frame_time.timestamp()
            
            cam_pos = interpolate_position(track_data, query_ts)
            if cam_pos is None:
                logger.trace(f"[{task_id}] Frame {frame_idx}: No interpolated position for timestamp {query_ts}")
                continue
                
            for obj in tracked_objects:
                obj_id = obj["id"]
                logger.trace(f"[{task_id}] Frame {frame_idx}: Tracked ID {obj_id} ({obj['class_name']}) conf={obj['confidence']:.4f}")
                if obj_id not in groups:
                    groups[obj_id] = []
                    
                groups[obj_id].append({
                    "frame": frame_idx,
                    "video_seconds": round(frame_offset_seconds, 3),
                    "timestamp": frame_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "camera_lat": cam_pos["lat"],
                    "camera_lon": cam_pos["lon"],
                    "camera_bearing": cam_pos["bearing"],
                    "class_name": obj["class_name"],
                    "confidence": obj["confidence"],
                    "box_xyxy": obj["box_xyxy"]
                })
    except Exception as e:
        msg = f"Exception occurred during YOLO tracking: {str(e)}"
        logger.exception(f"[{task_id}] Video pipeline tracking failed:")
        payload = {"task_id": task_id, "status": "FAILED", "result": None, "error": msg}
        send_callback(callback_url, payload)
        clean_temp_files(video_path, gpx_path)
        return
        
    logger.debug(f"[{task_id}] Tracking complete. Detected {len(groups)} unique tracking group(s).")
    
    final_signs = []
    task_frames_dir = os.path.join(static_dir, task_id)
    os.makedirs(task_frames_dir, exist_ok=True)
    
    logger.info(f"[{task_id}] Geolocating and extracting best frames for each sign.")
    for obj_id, group in groups.items():
        valid_group = [det for det in group if det.get("confidence", 0.0) >= conf]
        if not valid_group:
            logger.debug(f"[{task_id}] Sign {obj_id} has no frame meeting confidence threshold {conf}.")
            continue
            
        logger.debug(f"[{task_id}] Geolocating sign ID {obj_id} from {len(valid_group)} valid frame(s).")
        sign_lat, sign_lon, avg_dist, avg_height = geolocate_sign_group(valid_group, fov=fov)
        
        best_det = max(
            valid_group,
            key=lambda x: (
                x.get("confidence", 0.0),
                (x["box_xyxy"][3] - x["box_xyxy"][1]) if "box_xyxy" in x else 0.0
            )
        )
        
        frame_filename = f"sign_{obj_id}_conf_{best_det['confidence']:.4f}.jpg"
        output_frame_path = os.path.join(task_frames_dir, frame_filename)
        
        success = extract_video_frame(video_path, best_det["frame"], output_frame_path)
        frame_url = None
        if success:
            upload_success, upload_url = upload_frame_to_minio(output_frame_path, f"{task_id}/sign_{obj_id}.jpg")
            frame_url = upload_url
            
        final_signs.append({
            "id": obj_id,
            "class_name": best_det["class_name"],
            "confidence": best_det["confidence"],
            "timestamp": best_det["timestamp"],
            "box_xyxy": best_det["box_xyxy"],
            "frame_image_url": frame_url,
            "gps": {
                "latitude": sign_lat,
                "longitude": sign_lon,
                "distance_meters": avg_dist,
                "height_meters": avg_height
            }
        })
        
    if os.path.exists(task_frames_dir):
        shutil.rmtree(task_frames_dir, ignore_errors=True)
        
    logger.info(f"[{task_id}] Video pipeline completed. Found {len(final_signs)} signs.")
    
    payload = {
        "task_id": task_id,
        "status": "COMPLETED",
        "result": final_signs,
        "error": None
    }
    send_callback(callback_url, payload)
    
    clean_temp_files(video_path, gpx_path)

def clean_temp_files(video_path, gpx_path):
    try:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
            logger.debug(f"Removed temporary video: {video_path}")
        if gpx_path and os.path.exists(gpx_path):
            os.remove(gpx_path)
            logger.debug(f"Removed temporary GPX: {gpx_path}")
    except Exception as e:
        logger.error(f"Error cleaning up temp files: {e}")
