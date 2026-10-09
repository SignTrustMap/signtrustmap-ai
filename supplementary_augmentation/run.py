import gc
import glob
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
from loguru import logger
from src.core.classifier import SignClassifier
from supplementary_augmentation.generator import SupplementaryAugmentor


def draw_bounding_boxes(
    image_bgr: np.ndarray,
    labels: List[Dict[str, Any]],
) -> np.ndarray:
    annotated = image_bgr.copy()
    color_map = {
        0: (0, 220, 0),
        1: (255, 180, 0),
        2: (0, 100, 255),
    }

    for item in labels:
        cid = item.get("class_id", 0)
        cname = item.get("class", "sign")
        color = color_map.get(cid, (255, 255, 255))
        x1, y1, x2, y2 = item["bbox"]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        desc = cname
        if cid == 0:
            code = item.get("standard_code") or item.get("class_name")
            desc = f"0: main [{code}]" if code else "0: main_sign"
        elif cid == 1:
            sub = item.get("subclass", "")
            desc = f"1: supp [{sub}]"
        elif cid == 2:
            txt = item.get("text", "").replace("\n", " ")
            desc = f"2: text [{txt[:18]}]"

        text_y = max(16, y1 - 6)
        (tw, th), _ = cv2.getTextSize(desc, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated, (x1, text_y - th - 3), (x1 + tw + 4, text_y + 2), (20, 20, 20), -1)
        cv2.putText(annotated, desc, (x1 + 2, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return annotated


def create_zoom_comparison(
    original_bgr: np.ndarray,
    augmented_bgr: np.ndarray,
    labels: List[Dict[str, Any]],
    margin: int = 30,
    target_height: int = 320,
) -> Optional[np.ndarray]:
    if not labels:
        return None

    img_h, img_w = original_bgr.shape[:2]
    all_x1 = min(item["bbox"][0] for item in labels)
    all_y1 = min(item["bbox"][1] for item in labels)
    all_x2 = max(item["bbox"][2] for item in labels)
    all_y2 = max(item["bbox"][3] for item in labels)

    crop_x1 = max(0, all_x1 - margin)
    crop_y1 = max(0, all_y1 - margin)
    crop_x2 = min(img_w, all_x2 + margin)
    crop_y2 = min(img_h, all_y2 + margin)

    orig_crop = original_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
    aug_crop = augmented_bgr[crop_y1:crop_y2, crop_x1:crop_x2]

    if orig_crop.size == 0 or aug_crop.size == 0:
        return None

    ch, cw = orig_crop.shape[:2]
    scale = target_height / max(1.0, float(ch))
    nw = int(round(cw * scale))
    nh = target_height

    orig_resized = cv2.resize(orig_crop, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    aug_resized = cv2.resize(aug_crop, (nw, nh), interpolation=cv2.INTER_LANCZOS4)

    banner_h = 24
    banner_orig = np.zeros((banner_h, nw, 3), dtype=np.uint8) + 40
    cv2.putText(banner_orig, "ORIGINAL", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    banner_aug = np.zeros((banner_h, nw, 3), dtype=np.uint8) + 40
    cv2.putText(banner_aug, "AUGMENTED", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)

    col_orig = np.vstack([banner_orig, orig_resized])
    col_aug = np.vstack([banner_aug, aug_resized])
    divider_full = np.zeros((nh + banner_h, 4, 3), dtype=np.uint8) + 180

    comparison = np.hstack([col_orig, divider_full, col_aug])
    return comparison


def load_vtsdb_samples(max_samples: int = 120) -> List[Dict[str, Any]]:
    label_files = sorted(glob.glob("data/VTSDB100/labels/train/*.txt")) + sorted(glob.glob("data/VTSDB100/labels/val/*.txt"))
    selected = []
    class_counts: Dict[int, int] = {}
    max_per_class = max(1, math.ceil(max_samples / 60))

    for lp in label_files:
        try:
            with open(lp, "r", encoding="utf-8") as f:
                lines = [line.strip().split() for line in f if line.strip()]
            if len(lines) != 1:
                continue
            cls_id = int(lines[0][0])
            if class_counts.get(cls_id, 0) >= max_per_class:
                continue
            xc, yc, bw, bh = float(lines[0][1]), float(lines[0][2]), float(lines[0][3]), float(lines[0][4])
            if 0.06 <= bw <= 0.25 and 0.09 <= bh <= 0.30 and (yc + bh * 1.0 < 0.85) and (yc - bh / 2 > 0.06):
                split = "train" if "/train/" in lp else "val"
                stem = Path(lp).stem
                img_path = Path(f"data/VTSDB100/images/{split}/{stem}.jpg")
                if not img_path.exists():
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]
                px_w = int(round(bw * w))
                px_h = int(round(bh * h))
                if px_w >= 45 and px_h >= 45:
                    x1 = int(round((xc - bw / 2.0) * w))
                    y1 = int(round((yc - bh / 2.0) * h))
                    x2 = int(round((xc + bw / 2.0) * w))
                    y2 = int(round((yc + bh / 2.0) * h))
                    class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                    selected.append({
                        "sample_id": f"vtsdb_{stem}_cls{cls_id}",
                        "image_path": str(img_path),
                        "pixel_boxes": [[x1, y1, x2, y2]],
                        "ground_truth_class_id": cls_id,
                    })
                    if len(selected) >= max_samples:
                        break
        except Exception:
            continue
    logger.info(f"Loaded {len(selected)} diverse samples from VTSDB100")
    return selected


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "outputs" / "supplementary_augmentation"
    results_dir = output_dir / "results"
    synth_dir = results_dir / "synthetic_images"
    labels_dir = results_dir / "labels"
    ann_dir = results_dir / "annotated"
    zoom_dir = results_dir / "zooms"
    meta_dir = results_dir / "metadata"
    rejected_dir = results_dir / "rejected_real_signs"

    for d in [synth_dir, labels_dir, ann_dir, zoom_dir, meta_dir, rejected_dir]:
        if d.exists():
            for item in d.glob("*"):
                if item.is_file():
                    item.unlink()
        d.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing SignClassifier on CPU to preserve Jetson memory...")
    classifier = None
    try:
        classifier = SignClassifier(device="cpu")
    except Exception as e:
        logger.warning(f"Could not load SignClassifier ({e}), proceeding without CLIP...")

    augmentor = SupplementaryAugmentor(classifier=classifier)

    vtsdb_samples = load_vtsdb_samples(max_samples=155)
    modes_cycle = ["supplementary_1", "supplementary_2", "text_sign"]
    work_items: List[Dict[str, Any]] = []

    for idx, sample in enumerate(vtsdb_samples):
        work_items.append({
            "sample_id": sample["sample_id"],
            "image_path": sample["image_path"],
            "pixel_boxes": sample["pixel_boxes"],
            "ground_truth_class_id": sample.get("ground_truth_class_id"),
            "mode": modes_cycle[idx % len(modes_cycle)],
        })

    generated_summary = []
    rejected_summary = []

    for item in work_items:
        sample_id = item["sample_id"]
        mode = item["mode"]
        img_path = Path(item["image_path"])
        if not img_path.exists():
            continue

        raw_bgr = cv2.imread(str(img_path))
        if raw_bgr is None:
            continue

        base_name = f"{sample_id}_{mode}"
        logger.info(f"Generating sample: {sample_id} | Mode: {mode}")
        known_cids = [item.get("ground_truth_class_id")] if item.get("ground_truth_class_id") is not None else None
        aug_bgr, labels = augmentor.augment_image(
            image_bgr=raw_bgr,
            main_boxes=item["pixel_boxes"],
            forced_mode=mode,
            known_class_ids=known_cids,
        )

        if augmentor.last_rejected:
            for rej in augmentor.last_rejected:
                rx1, ry1, rx2, ry2 = rej["roi_box"]
                roi_crop = raw_bgr[ry1:ry2, rx1:rx2]

                rej_vis = raw_bgr.copy()
                mx1, my1, mx2, my2 = rej["main_box"]
                cv2.rectangle(rej_vis, (mx1, my1), (mx2, my2), (0, 220, 0), 2)
                desc = f"0: main [{rej.get('standard_code') or ''}]"
                cv2.putText(rej_vis, desc, (mx1, max(16, my1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1, cv2.LINE_AA)

                cv2.rectangle(rej_vis, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
                reason_desc = f"REJECTED: REAL SIGN ({rej['reason']})"
                cv2.putText(rej_vis, reason_desc, (rx1, max(16, ry1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

                crop_path = rejected_dir / f"{base_name}_real_sign_crop.jpg"
                vis_path = rejected_dir / f"{base_name}_rejected_vis.jpg"
                meta_path = rejected_dir / f"{base_name}_rejected.json"

                if roi_crop.size > 0:
                    cv2.imwrite(str(crop_path), roi_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(vis_path), rej_vis, [cv2.IMWRITE_JPEG_QUALITY, 95])

                rej_record = {
                    "sample_id": sample_id,
                    "mode": mode,
                    "main_box": rej["main_box"],
                    "standard_code": rej.get("standard_code"),
                    "class_name": rej.get("class_name"),
                    "roi_box": rej["roi_box"],
                    "reason": rej["reason"],
                    "metrics": rej["metrics"],
                    "vis_file": str(vis_path),
                    "crop_file": str(crop_path) if roi_crop.size > 0 else None,
                }
                with open(meta_path, "w", encoding="utf-8") as rf:
                    json.dump(rej_record, rf, indent=2, ensure_ascii=False)

                rejected_summary.append(rej_record)
                logger.warning(f"Rejected anchor for {sample_id} ({rej['reason']}). Saved to {rejected_dir}")

        if len(labels) <= len(item["pixel_boxes"]):
            del raw_bgr
            del aug_bgr
            gc.collect()
            continue

        annotated_bgr = draw_bounding_boxes(aug_bgr, labels)
        zoom_comparison = create_zoom_comparison(raw_bgr, aug_bgr, labels)

        aug_file = synth_dir / f"{base_name}_synth.jpg"
        ann_file = ann_dir / f"{base_name}_annotated.jpg"
        zoom_file = zoom_dir / f"{base_name}_zoom.jpg"
        txt_file = labels_dir / f"{base_name}.txt"
        meta_file = meta_dir / f"{base_name}_meta.json"

        cv2.imwrite(str(aug_file), aug_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(ann_file), annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if zoom_comparison is not None:
            cv2.imwrite(str(zoom_file), zoom_comparison, [cv2.IMWRITE_JPEG_QUALITY, 95])

        yolo_entries = augmentor.to_yolo_annotations(labels, aug_bgr.shape[:2])
        with open(txt_file, "w", encoding="utf-8") as tf:
            for entry in yolo_entries:
                tf.write(f"{entry[0]} {entry[1]} {entry[2]} {entry[3]} {entry[4]}\n")

        meta_record = {
            "sample_id": sample_id,
            "mode": mode,
            "image_size": list(aug_bgr.shape[:2]),
            "labels": labels,
            "yolo_entries": yolo_entries,
        }
        with open(meta_file, "w", encoding="utf-8") as mf:
            json.dump(meta_record, mf, indent=2, ensure_ascii=False)

        generated_summary.append({
            "sample_id": sample_id,
            "mode": mode,
            "num_labels": len(labels),
            "aug_file": str(aug_file),
            "ann_file": str(ann_file),
            "zoom_file": str(zoom_file) if zoom_comparison is not None else None,
        })

        del raw_bgr
        del aug_bgr
        del annotated_bgr
        if zoom_comparison is not None:
            del zoom_comparison
        gc.collect()

    summary_file = meta_dir / "generation_manifest.json"
    with open(summary_file, "w", encoding="utf-8") as sf:
        json.dump(generated_summary, sf, indent=2, ensure_ascii=False)

    rejected_manifest_file = rejected_dir / "rejected_manifest.json"
    with open(rejected_manifest_file, "w", encoding="utf-8") as rmf:
        json.dump(rejected_summary, rmf, indent=2, ensure_ascii=False)

    logger.info(f"Augmentation completed: generated {len(generated_summary)} scenarios in {results_dir}")
    logger.info(f"Rejected {len(rejected_summary)} anchors with existing real signs in {rejected_dir}")

    del classifier
    del augmentor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
