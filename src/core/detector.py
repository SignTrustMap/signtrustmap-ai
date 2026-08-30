import os
from ultralytics import YOLO
from loguru import logger
from src.core.config import settings
from src.infrastructure.minio_client import minio_client


class SignDetector:
    def __init__(self, weights_path=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        if weights_path is None:
            if settings.DETECTOR_S3_KEY:
                model_filename = os.path.basename(settings.DETECTOR_S3_KEY)
                local_dir = os.path.join(base_dir, settings.LOCAL_DETECTORS_DIR)
                weights_path = os.path.join(local_dir, model_filename)
                
                if not os.path.exists(weights_path):
                    logger.info(f"Detector file {model_filename} not found locally at {weights_path}.")
                    success = minio_client.download_file(settings.DETECTOR_S3_KEY, weights_path)
                    if not success:
                        logger.warning(f"Could not download detector {settings.DETECTOR_S3_KEY} from MinIO. Trying fallback detector path.")
                        weights_path = os.path.join(base_dir, settings.FALLBACK_DETECTOR_PATH)
            else:
                weights_path = os.path.join(base_dir, settings.FALLBACK_DETECTOR_PATH)
                
        if not os.path.exists(weights_path):
            logger.error(f"Detector weights file not found at: {weights_path}")
            raise FileNotFoundError(f"Detector weights file not found at: {weights_path}")
            
        logger.info(f"Loading detector weights from: {weights_path}")
        self.model = YOLO(weights_path)

    def detect(self, image_source, conf=0.6, device=0):
        logger.info(f"Running detection on image: {image_source}")
        results = self.model(image_source, conf=conf, device=device, verbose=False)
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            clss = results[0].boxes.cls.int().tolist()
            confs = results[0].boxes.conf.tolist()
            boxes = results[0].boxes.xyxy.tolist()
            
            for class_id, confidence, box in zip(clss, confs, boxes):
                logger.trace(f"Detected box: class={self.model.names[class_id]}, conf={confidence:.4f}, box={box}")
                detections.append({
                    "class_id": class_id,
                    "class_name": self.model.names[class_id],
                    "confidence": round(confidence, 4),
                    "detect_confidence": round(confidence, 4),
                    "classify_confidence": None,
                    "box_xyxy": [round(b, 2) for b in box]
                })
        logger.debug(f"Detected {len(detections)} object(s) in image.")
        return detections

    def track(self, video_source, conf=0.6, device=0):
        logger.info(f"Starting tracking on video: {video_source}")
        results = self.model.track(
            source=video_source,
            stream=True,
            persist=True,
            device=device,
            conf=conf,
            verbose=False
        )
        
        for frame_idx, r in enumerate(results):
            tracked_objects = []
            if r.boxes is not None and r.boxes.id is not None:
                ids = r.boxes.id.int().tolist()
                confs = r.boxes.conf.tolist()
                clss = r.boxes.cls.int().tolist()
                boxes = r.boxes.xyxy.tolist()
                
                for obj_id, confidence, class_id, box in zip(ids, confs, clss, boxes):
                    tracked_objects.append({
                        "id": obj_id,
                        "class_id": class_id,
                        "class_name": self.model.names[class_id],
                        "confidence": round(confidence, 4),
                        "detect_confidence": round(confidence, 4),
                        "classify_confidence": None,
                        "box_xyxy": [round(b, 2) for b in box]
                    })
            logger.trace(f"Frame {frame_idx}: tracked {len(tracked_objects)} object(s).")
            yield frame_idx, tracked_objects
