from pathlib import Path
from typing import Dict, Any, List
import cv2
import numpy as np
import pytest
import torch
from loguru import logger
from src.core.classifier import SignClassifier


def test_ai_crp_01_valid_crop(verified_images_dir: Path):
    img = cv2.imread(str(verified_images_dir / "11288.jpg"))
    box = [530.0, 310.0, 570.0, 350.0]
    crop = SignClassifier.crop_box(img, box, padding_ratio=0.1)
    assert crop is not None
    assert isinstance(crop, np.ndarray)
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    logger.info("AI-CRP-01 verified: valid crop extracted.")


def test_ai_crp_02_boundary_crop_clipping(verified_images_dir: Path):
    img = cv2.imread(str(verified_images_dir / "11288.jpg"))
    h, w = img.shape[:2]
    out_of_bounds_box = [-50.0, -20.0, w + 100.0, h + 80.0]
    crop = SignClassifier.crop_box(img, out_of_bounds_box, padding_ratio=0.1)
    assert crop is not None
    assert crop.shape[0] == h
    assert crop.shape[1] == w
    logger.info("AI-CRP-02 verified: out of bounds box clipped to image dimensions.")


def test_ai_crp_03_invalid_box_inverted(verified_images_dir: Path):
    img = cv2.imread(str(verified_images_dir / "11288.jpg"))
    inverted_box = [100.0, 100.0, 50.0, 50.0]
    crop = SignClassifier.crop_box(img, inverted_box)
    assert crop is None
    logger.info("AI-CRP-03 verified: inverted box returns None.")


def test_ai_cls_01_classify_crop_top1_and_top5(classifier: SignClassifier, verified_images_dir: Path):
    img = cv2.imread(str(verified_images_dir / "11288.jpg"))
    h, w = img.shape[:2]
    box = [w * 0.40, h * 0.40, w * 0.45, h * 0.48]
    crop = SignClassifier.crop_box(img, box)
    assert crop is not None

    res = classifier.classify_crop(crop, top_k=5)
    assert "top1" in res
    assert "top5" in res
    assert res["top1"]["class_id"] is not None
    assert res["top1"]["confidence"] > 0.0
    assert len(res["top5"]) == 5

    confs = [item["confidence"] for item in res["top5"]]
    assert confs == sorted(confs, reverse=True)
    logger.info(f"AI-CLS-01 verified: top1 class={res['top1']['class_name']} conf={res['top1']['confidence']}")


def test_ai_cls_02_classify_crops_batch(classifier: SignClassifier, verified_images_dir: Path):
    img = cv2.imread(str(verified_images_dir / "11288.jpg"))
    h, w = img.shape[:2]
    crop1 = SignClassifier.crop_box(img, [w * 0.40, h * 0.40, w * 0.45, h * 0.48])
    crop2 = SignClassifier.crop_box(img, [w * 0.60, h * 0.30, w * 0.68, h * 0.40])
    
    results = classifier.classify_crops_batch([crop1, crop2], batch_size=2, top_k=5)
    assert len(results) == 2
    for r in results:
        assert "top1" in r
        assert "top5" in r
        assert r["top1"]["class_id"] is not None
    logger.info("AI-CLS-02 verified: batch classification matches input length.")


def test_ai_cls_03_classify_none_crop(classifier: SignClassifier):
    res = classifier.classify_crop(None)
    assert res["class_id"] is None
    assert res["classify_confidence"] == 0.0
    assert res["top1"]["class_id"] is None
    assert res["top1"]["confidence"] == 0.0
    assert res["top5"] == []
    logger.info("AI-CLS-03 verified: None crop returns empty prediction dict.")


def test_ai_cls_04_batch_with_none_elements(classifier: SignClassifier, verified_images_dir: Path):
    img = cv2.imread(str(verified_images_dir / "11288.jpg"))
    h, w = img.shape[:2]
    crop = SignClassifier.crop_box(img, [w * 0.40, h * 0.40, w * 0.45, h * 0.48])

    batch = [crop, None, crop]
    results = classifier.classify_crops_batch(batch, batch_size=2)
    assert len(results) == 3
    assert results[0]["top1"]["class_id"] is not None
    assert results[1]["top1"]["class_id"] is None
    assert results[2]["top1"]["class_id"] is not None
    logger.info("AI-CLS-04 verified: mixed batch preserves exact index ordering.")


def test_ai_cls_05_probability_distribution_sum(classifier: SignClassifier, verified_images_dir: Path):
    img = cv2.imread(str(verified_images_dir / "11288.jpg"))
    h, w = img.shape[:2]
    crop = SignClassifier.crop_box(img, [w * 0.40, h * 0.40, w * 0.45, h * 0.48])
    assert crop is not None

    res = classifier.classify_crop(crop, top_k=100)
    total_prob = sum(item["confidence"] for item in res["top5"])
    assert total_prob >= 0.99
    logger.info(f"AI-CLS-05 verified: total top-k probability sum = {total_prob:.4f}")

