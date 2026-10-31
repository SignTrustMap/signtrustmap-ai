import hmac
import os
import tempfile
import urllib.parse
from typing import Optional

import cv2
import httpx
from fastapi import APIRouter, Header, HTTPException, status
from loguru import logger
from starlette.concurrency import run_in_threadpool

from src.api.schemas.inference import (
    EarlyClassificationRequest,
    EarlyClassificationResponse,
    SubmissionInferenceRequest,
    SubmissionInferenceResponse,
)
from src.core.class_manager import class_manager
from src.core.config import settings
from src.core.config_manager import config_manager
from src.core.model_manager import model_manager
from src.infrastructure.minio_client import minio_client
from src.pipelines.inference_engine import inference_engine


router = APIRouter(tags=["Inference"])


def _extract_s3_reference(url_or_key: str) -> tuple[str, str]:
    """Extract (bucket, object_key) from an S3 URL or relative key."""
    clean = url_or_key.strip()
    if clean.startswith("http://") or clean.startswith("https://"):
        parsed = urllib.parse.urlparse(clean)
        path = parsed.path.lstrip("/")
        parts = path.split("/", 1)
        if len(parts) == 2:
            bucket, key = parts
            return bucket, key
        return settings.MINIO_OUTPUT_BUCKET, path
    clean = clean.lstrip("/")
    parts = clean.split("/", 1)
    if len(parts) == 2 and parts[0] == settings.MINIO_OUTPUT_BUCKET:
        return parts[0], parts[1]
    return settings.MINIO_OUTPUT_BUCKET, clean


def _download_image_to_local(
    media_url: Optional[str],
    bucket: Optional[str],
    object_key: Optional[str],
    local_path: str,
) -> bool:
    """Download image to local_path from MinIO or direct HTTP."""
    if object_key:
        b = bucket or settings.MINIO_OUTPUT_BUCKET
        if minio_client.download_file(b, object_key, local_path):
            return True

    if media_url:
        b, key = _extract_s3_reference(media_url)
        if minio_client.download_file(b, key, local_path):
            return True

        if media_url.startswith("http://") or media_url.startswith("https://"):
            try:
                with httpx.Client(timeout=15.0) as client:
                    resp = client.get(media_url)
                    if resp.status_code == 200 and resp.content:
                        with open(local_path, "wb") as f:
                            f.write(resp.content)
                        return True
            except Exception as e:
                logger.warning(f"HTTP download failed for {media_url}: {e}")

    return False


def _perform_early_classification(
    request_data: dict,
) -> dict:
    media_url = request_data.get("media_url")
    bucket = request_data.get("bucket")
    object_key = request_data.get("object_key")
    sign_id = request_data.get("sign_id")
    task_id = request_data.get("task_id")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        success = _download_image_to_local(media_url, bucket, object_key, tmp_path)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Could not retrieve image from provided source: media_url={media_url}, key={object_key}",
            )

        image = cv2.imread(tmp_path)
        if image is None or image.size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Downloaded image could not be decoded as a valid image",
            )

        detector = model_manager.get_detector()
        classifier = model_manager.get_classifier()

        detections = []
        try:
            detections = detector.detect(
                tmp_path,
                conf=0.35,
                device=config_manager.default_device,
            )
        except Exception as e:
            logger.warning(f"Detection step skipped or failed: {e}")

        signs = []
        best_class_id = None
        best_class_name = None
        best_confidence = 0.0
        best_detect_conf = None
        best_classify_conf = None
        best_top1 = None
        best_top5 = []
        best_all_predictions = []

        if detections and classifier is not None:
            crops = [classifier.crop_box(image, item["box_xyxy"]) for item in detections]
            class_results = classifier.classify_crops_batch(crops)
            for idx, item in enumerate(detections):
                c_res = class_results[idx] if idx < len(class_results) else {}
                top1 = c_res.get("top1")
                top5 = c_res.get("top5", [])
                all_predictions = c_res.get("all_predictions", [])
                cid = (top1.get("class_id") if top1 else None) or c_res.get("class_id", item.get("class_id"))
                cname = (top1.get("class_name") if top1 else None) or c_res.get("class_name", item.get("class_name"))
                c_conf = (top1.get("classify_confidence") if top1 else None) or c_res.get("classify_confidence", 0.0)
                d_conf = item.get("confidence", 0.0)
                conf = c_conf if c_conf > 0 else d_conf
                meta = class_manager.get_sign_metadata(cid) if cid is not None else {}

                sign_result = {
                    "track_id": None,
                    "class_id": cid,
                    "class_name": cname,
                    "confidence": conf,
                    "detect_confidence": d_conf,
                    "classify_confidence": c_conf,
                    "top1": top1,
                    "top5": top5,
                    "all_predictions": all_predictions,
                    "box_xyxy": tuple(item["box_xyxy"]),
                    "best_frame_key": object_key or media_url or "",
                    "sign_crop_key": object_key or media_url or "",
                    "frame_index": 0,
                    "timestamp_seconds": 0.0,
                    "category": meta.get("category"),
                    "name_vi": meta.get("name_vi"),
                    "name_en": meta.get("name_en"),
                    "standard_code": meta.get("standard_code"),
                }
                signs.append(sign_result)

                if conf > best_confidence:
                    best_confidence = conf
                    best_class_id = cid
                    best_class_name = cname
                    best_detect_conf = d_conf
                    best_classify_conf = c_conf
                    best_top1 = top1
                    best_top5 = top5
                    best_all_predictions = all_predictions

        if not signs and classifier is not None:
            c_res = classifier.classify_crop(image)
            best_top1 = c_res.get("top1")
            best_top5 = c_res.get("top5", [])
            best_all_predictions = c_res.get("all_predictions", [])
            best_class_id = (best_top1.get("class_id") if best_top1 else None) or c_res.get("class_id")
            best_class_name = (best_top1.get("class_name") if best_top1 else None) or c_res.get("class_name")
            best_classify_conf = (best_top1.get("classify_confidence") if best_top1 else None) or c_res.get("classify_confidence", 0.0)
            best_confidence = best_classify_conf
            meta = class_manager.get_sign_metadata(best_class_id) if best_class_id is not None else {}
            h, w = image.shape[:2]

            sign_result = {
                "track_id": None,
                "class_id": best_class_id,
                "class_name": best_class_name,
                "confidence": best_confidence,
                "detect_confidence": 0.0,
                "classify_confidence": best_classify_conf,
                "top1": best_top1,
                "top5": best_top5,
                "all_predictions": best_all_predictions,
                "box_xyxy": (0.0, 0.0, float(w), float(h)),
                "best_frame_key": object_key or media_url or "",
                "sign_crop_key": object_key or media_url or "",
                "frame_index": 0,
                "timestamp_seconds": 0.0,
                "category": meta.get("category"),
                "name_vi": meta.get("name_vi"),
                "name_en": meta.get("name_en"),
                "standard_code": meta.get("standard_code"),
            }
            signs.append(sign_result)

        meta = class_manager.get_sign_metadata(best_class_id) if best_class_id is not None else {}
        return {
            "status": "CLASSIFIED" if best_class_id is not None else "UNKNOWN",
            "class_id": best_class_id,
            "class_name": best_class_name,
            "confidence": round(best_confidence, 4),
            "detect_confidence": round(best_detect_conf, 4) if best_detect_conf is not None else None,
            "classify_confidence": round(best_classify_conf, 4) if best_classify_conf is not None else None,
            "top1": best_top1,
            "top5": best_top5,
            "all_predictions": best_all_predictions,
            "category": meta.get("category"),
            "name_vi": meta.get("name_vi"),
            "name_en": meta.get("name_en"),
            "standard_code": meta.get("standard_code"),
            "sign_id": sign_id,
            "task_id": task_id,
            "media_url": media_url,
            "signs": signs,
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.post(
    "/submissions/{submission_id}",
    response_model=SubmissionInferenceResponse,
    summary="Run inference for submission media stored in MinIO",
)
async def process_submission(
    submission_id: str,
    request: SubmissionInferenceRequest,
    internal_token: str = Header(alias="X-Internal-Token"),
) -> dict:
    """Execute inference synchronously; Celery owns retries and orchestration."""
    if not hmac.compare_digest(internal_token, settings.INTERNAL_API_TOKEN):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid internal token")
    media_objects = [item.model_dump() for item in request.media_objects]
    return await run_in_threadpool(
        inference_engine.process,
        submission_id,
        media_objects,
    )


@router.post(
    "/early-classification",
    response_model=EarlyClassificationResponse,
    summary="Run early classification for revalidated or changed sign images",
)
@router.post(
    "/classify",
    response_model=EarlyClassificationResponse,
    summary="Direct sign classification alias",
)
async def early_classification(
    request: EarlyClassificationRequest,
    internal_token: Optional[str] = Header(None, alias="X-Internal-Token"),
) -> dict:
    """Perform early classification on revalidated/changed traffic sign images."""
    if internal_token and not hmac.compare_digest(internal_token, settings.INTERNAL_API_TOKEN):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid internal token")
    return await run_in_threadpool(
        _perform_early_classification,
        request.model_dump(),
    )

