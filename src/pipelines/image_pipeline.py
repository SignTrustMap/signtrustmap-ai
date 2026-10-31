import os
import shutil
import tempfile

import cv2

from src.core.class_manager import class_manager
from src.infrastructure.minio_client import minio_client


def infer_image(
    submission_id: str,
    detector,
    image_path: str,
    source_object_key: str,
    classifier=None,
    conf: float = 0.6,
    device: int = 0,
) -> dict:
    """Run image inference and return storage-backed candidate data."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot decode image file: {image_path}")

    image_height, image_width = image.shape[:2]
    detections = detector.detect(image_path, conf=conf, device=device)
    classification_results = []
    if classifier is not None and detections:
        crops = [classifier.crop_box(image, item["box_xyxy"]) for item in detections]
        classification_results = classifier.classify_crops_batch(crops)

    signs = []
    artifact_dir = tempfile.mkdtemp(prefix=f"stm-{submission_id}-")
    try:
        for index, detection in enumerate(detections):
            class_result = classification_results[index] if classification_results else {}
            top1 = class_result.get("top1")
            top5 = class_result.get("top5", [])
            all_predictions = class_result.get("all_predictions", [])
            has_classification = (top1 is not None and top1.get("class_id") is not None) or class_result.get("class_id") is not None
            class_id = (top1.get("class_id") if top1 else None) or class_result.get("class_id") or detection["class_id"]
            class_name = (top1.get("class_name") if top1 else None) or class_result.get("class_name") or detection["class_name"]
            classify_confidence = (top1.get("classify_confidence") if top1 else None) or class_result.get("classify_confidence") if has_classification else None
            confidence = classify_confidence if classify_confidence is not None else detection["confidence"]

            x1, y1, x2, y2 = [int(round(value)) for value in detection["box_xyxy"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(image_width, x2)
            y2 = min(image_height, y2)
            crop = image[y1:y2, x1:x2]
            crop_key = f"candidates/{submission_id}/image-{index}.jpg"
            crop_path = os.path.join(artifact_dir, f"image-{index}.jpg")
            if crop.size == 0 or not cv2.imwrite(crop_path, crop):
                raise RuntimeError(f"Could not create crop for detection {index}")
            uploaded, crop_storage_key = minio_client.upload_file(
                crop_path,
                crop_key,
                content_type="image/jpeg",
            )
            if not uploaded or not crop_storage_key:
                raise RuntimeError(f"Could not upload crop for detection {index}")

            sign_meta = class_manager.get_sign_metadata(class_id)
            signs.append({
                "track_id": None,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "detect_confidence": detection.get("detect_confidence", detection["confidence"]),
                "classify_confidence": classify_confidence,
                "top1": top1,
                "top5": top5,
                "all_predictions": all_predictions,
                "box_xyxy": detection["box_xyxy"],
                "best_frame_key": source_object_key,
                "sign_crop_key": crop_storage_key,
                "frame_index": 0,
                "timestamp_seconds": 0.0,
                "category": sign_meta.get("category"),
                "name_vi": sign_meta.get("name_vi"),
                "name_en": sign_meta.get("name_en"),
                "standard_code": sign_meta.get("standard_code"),
            })
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)

    return {
        "image_width": image_width,
        "image_height": image_height,
        "total_frames": 1,
        "signs": signs,
    }
