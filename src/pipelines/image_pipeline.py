import os
import cv2
from loguru import logger
from src.pipelines.utils import send_callback


def run_image_pipeline(task_id: str, callback_url: str, detector, image_path: str, conf: float = 0.6, device: int = 0):
    filename = os.path.basename(image_path)
    logger.info(f"[{task_id}] Executing pure YOLO image pipeline for {filename} (conf={conf}, device={device})...")

    try:
        img = cv2.imread(image_path)
        img_height, img_width = (None, None)
        if img is not None:
            img_height, img_width = img.shape[:2]

        detections = detector.detect(image_path, conf=conf, device=device)
        sign_results = []

        for det in detections:
            sign_results.append({
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "confidence": det["confidence"],
                "box_xyxy": det["box_xyxy"],
            })

        logger.info(f"[{task_id}] YOLO image detection complete. Found {len(sign_results)} sign(s).")

        payload = {
            "task_id": task_id,
            "status": "COMPLETED",
            "image_width": img_width,
            "image_height": img_height,
            "total_detections": len(sign_results),
            "result": sign_results,
            "error": None,
        }
        send_callback(callback_url, payload)

    except Exception as e:
        msg = f"Exception occurred during image detection: {str(e)}"
        logger.exception(f"[{task_id}] Image pipeline failed: {msg}")
        payload = {
            "task_id": task_id,
            "status": "FAILED",
            "image_width": None,
            "image_height": None,
            "total_detections": 0,
            "result": None,
            "error": msg,
        }
        send_callback(callback_url, payload)

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)
            logger.debug(f"[{task_id}] Removed temporary image file: {image_path}")
