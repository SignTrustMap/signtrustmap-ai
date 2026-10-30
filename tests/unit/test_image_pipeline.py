from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import MagicMock
import pytest
from loguru import logger
from src.core.classifier import SignClassifier
from src.core.detector import SignDetector
from src.infrastructure.minio_client import minio_client
from src.pipelines.image_pipeline import infer_image


def test_ai_img_01_class_id_zero_integrity(monkeypatch: pytest.MonkeyPatch, verified_images_dir: Path, mock_minio):
    fake_detector = MagicMock()
    fake_detector.detect.return_value = [
        {
            "class_id": 99,
            "class_name": "Generic_sign",
            "confidence": 0.85,
            "detect_confidence": 0.85,
            "box_xyxy": [10.0, 10.0, 50.0, 50.0],
        }
    ]

    fake_classifier = MagicMock()
    fake_classifier.crop_box.return_value = MagicMock()
    fake_classifier.classify_crops_batch.return_value = [
        {
            "top1": {
                "class_id": 0,
                "class_name": "No_entry",
                "confidence": 0.95,
                "classify_confidence": 0.95,
            },
            "top5": [
                {
                    "class_id": 0,
                    "class_name": "No_entry",
                    "confidence": 0.95,
                    "classify_confidence": 0.95,
                }
            ],
            "class_id": 0,
            "class_name": "No_entry",
            "classify_confidence": 0.95,
        }
    ]

    img_path = str(verified_images_dir / "1007.jpg")
    res = infer_image(
        submission_id="sub_test_01",
        detector=fake_detector,
        image_path=img_path,
        source_object_key="raw/1007.jpg",
        classifier=fake_classifier,
    )

    assert len(res["signs"]) == 1
    assert res["signs"][0]["class_id"] == 0


def test_ai_img_02_valid_inference(detector: SignDetector, classifier: SignClassifier, verified_images_dir: Path, mock_minio):
    img_path = str(verified_images_dir / "11288.jpg")
    res = infer_image(
        submission_id="sub_test_02",
        detector=detector,
        image_path=img_path,
        source_object_key="raw/11288.jpg",
        classifier=classifier,
        conf=0.4,
    )

    assert res["image_width"] > 0
    assert res["image_height"] > 0
    assert res["total_frames"] == 1
    assert len(res["signs"]) >= 1

    sign = res["signs"][0]
    assert sign["sign_crop_key"] is not None
    assert "candidates/sub_test_02/" in sign["sign_crop_key"]
    assert sign["name_vi"] is not None
    logger.info(f"AI-IMG-02 verified: detected {len(res['signs'])} sign(s).")


def test_ai_img_03_zero_detection_blank_canvas(detector: SignDetector, classifier: SignClassifier, verified_images_dir: Path, mock_minio):
    blank_path = str(verified_images_dir / "blank_canvas.jpg")
    res = infer_image(
        submission_id="sub_test_03",
        detector=detector,
        image_path=blank_path,
        source_object_key="raw/blank.jpg",
        classifier=classifier,
    )
    assert res["signs"] == []
    logger.info("AI-IMG-03 verified: zero detections on blank canvas.")


def test_ai_img_04_minio_upload_failure(detector: SignDetector, classifier: SignClassifier, verified_images_dir: Path, monkeypatch: pytest.MonkeyPatch):
    def failing_upload(*args, **kwargs):
        return False, None

    monkeypatch.setattr(minio_client, "upload_file", failing_upload)
    img_path = str(verified_images_dir / "11288.jpg")
    with pytest.raises(RuntimeError, match="Could not upload crop"):
        infer_image(
            submission_id="sub_test_04",
            detector=detector,
            image_path=img_path,
            source_object_key="raw/11288.jpg",
            classifier=classifier,
            conf=0.4,
        )
    logger.info("AI-IMG-04 verified: RuntimeError raised on MinIO upload failure.")

