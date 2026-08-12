import os
from src.core.geolocator import geolocate_sign_group
from loguru import logger
from src.pipelines.utils import send_callback

def run_image_pipeline(task_id, callback_url, detector, image_path, timestamp, latitude, longitude, camera_bearing=0.0, fov=90.0, conf=0.6):
    filename = os.path.basename(image_path)
    logger.info(f"[{task_id}] Executing image pipeline for {filename}...")
    logger.debug(f"[{task_id}] Input params - timestamp: {timestamp}, lat: {latitude}, lon: {longitude}, bearing: {camera_bearing}, fov: {fov}")
    
    try:
        detections = detector.detect(image_path, conf=conf)
        localized_signs = []
        
        for det in detections:
            frame_data = [{
                "camera_lat": latitude,
                "camera_lon": longitude,
                "camera_bearing": camera_bearing,
                "box_xyxy": det["box_xyxy"]
            }]
            
            sign_lat, sign_lon, dist, height = geolocate_sign_group(frame_data, fov=fov)
            
            localized_signs.append({
                "class_name": det["class_name"],
                "confidence": det["confidence"],
                "box_xyxy": det["box_xyxy"],
                "timestamp": timestamp,
                "gps": {
                    "latitude": sign_lat,
                    "longitude": sign_lon,
                    "distance_meters": dist,
                    "height_meters": height
                }
            })
            
        logger.info(f"[{task_id}] Image pipeline complete. Localized {len(localized_signs)} sign(s).")
        
        payload = {
            "task_id": task_id,
            "status": "COMPLETED",
            "result": localized_signs,
            "error": None
        }
        send_callback(callback_url, payload)
        
    except Exception as e:
        msg = f"Exception occurred during image processing: {str(e)}"
        logger.exception(f"[{task_id}] Image pipeline failed:")
        payload = {
            "task_id": task_id,
            "status": "FAILED",
            "result": None,
            "error": msg
        }
        send_callback(callback_url, payload)
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)
            logger.debug(f"[{task_id}] Removed temporary image file: {image_path}")
