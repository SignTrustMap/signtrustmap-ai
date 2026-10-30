import argparse
import concurrent.futures
import gc
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import yaml
from loguru import logger
from src.data.augmentation.generator_v2 import SupplementaryAugmentorV2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", "--dataset-dir", dest="dataset_dir", type=str, default="data/VTSDB100")
    parser.add_argument("--output-dir", type=str, default="data/SynSTMv2")
    parser.add_argument("--config-yaml", type=str, default="configs/training/datasets/SynSTMv2.yaml")
    parser.add_argument("--review-dir", type=str, default="outputs/syn_stm_v2_review")
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--prob-augment", type=float, default=0.70)
    parser.add_argument("--supp-class-id", type=int, default=100)
    parser.add_argument("--text-class-id", type=int, default=101)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def detect_dataset_splits(dataset_dir: Path) -> List[str]:
    images_dir = dataset_dir / "images"
    if images_dir.exists():
        subdirs = [p.name for p in sorted(images_dir.iterdir()) if p.is_dir()]
        if subdirs:
            return subdirs

    subdirs = [p.name for p in sorted(dataset_dir.iterdir()) if p.is_dir() and (p / "images").exists()]
    if subdirs:
        return subdirs

    return ["train", "val", "test"]


def resolve_split_dirs(dataset_dir: Path, split: str) -> Tuple[Path, Path]:
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split
    if images_dir.exists() and labels_dir.exists():
        return images_dir, labels_dir

    split_img = dataset_dir / split / "images"
    split_lbl = dataset_dir / split / "labels"
    if split_img.exists() and split_lbl.exists():
        return split_img, split_lbl

    return images_dir, labels_dir


def load_yolo_split_files(
    dataset_dir: Path,
    split: str,
    max_count: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> List[Tuple[Path, Path]]:
    images_dir, labels_dir = resolve_split_dirs(dataset_dir, split)
    if not labels_dir.exists():
        return []

    label_files = sorted(list(labels_dir.glob("*.txt")))
    pairs: List[Tuple[Path, Path]] = []
    valid_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    for lp in label_files:
        stem = lp.stem
        found_img: Optional[Path] = None
        for ext in valid_exts:
            ip = images_dir / f"{stem}{ext}"
            if ip.exists():
                found_img = ip
                break
        if found_img is not None:
            pairs.append((found_img, lp))

    if rng and pairs:
        rng.shuffle(pairs)

    if max_count is not None and max_count > 0:
        pairs = pairs[:max_count]

    return pairs


def parse_yolo_label_file(
    label_path: Path,
    img_w: int,
    img_h: int,
) -> Tuple[List[int], List[List[float]]]:
    class_ids: List[int] = []
    boxes: List[List[float]] = []

    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip().split() for line in f if line.strip()]

        for tokens in lines:
            if len(tokens) >= 5:
                cid = int(tokens[0])
                xc, yc = float(tokens[1]) * img_w, float(tokens[2]) * img_h
                bw, bh = float(tokens[3]) * img_w, float(tokens[4]) * img_h
                x1 = max(0.0, xc - bw / 2.0)
                y1 = max(0.0, yc - bh / 2.0)
                x2 = min(float(img_w), xc + bw / 2.0)
                y2 = min(float(img_h), yc + bh / 2.0)
                class_ids.append(cid)
                boxes.append([x1, y1, x2, y2])
    except Exception as e:
        logger.warning(f"Failed parsing label file {label_path}: {e}")

    return class_ids, boxes


def save_yolo_labels(
    out_label_path: Path,
    labels: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
) -> None:
    lines: List[str] = []
    for item in labels:
        cid = int(item["class_id"])
        x1, y1, x2, y2 = item["bbox"]
        xc = ((x1 + x2) / 2.0) / float(img_w)
        yc = ((y1 + y2) / 2.0) / float(img_h)
        bw = (x2 - x1) / float(img_w)
        bh = (y2 - y1) / float(img_h)
        xc = max(0.0, min(1.0, xc))
        yc = max(0.0, min(1.0, yc))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))
        lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    out_label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_label_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def draw_debug_visualization(
    image_bgr: np.ndarray,
    labels: List[Dict[str, Any]],
    supp_class_id: int,
    text_class_id: int,
) -> np.ndarray:
    vis = image_bgr.copy()
    for item in labels:
        cid = int(item["class_id"])
        x1, y1, x2, y2 = [int(round(c)) for c in item["bbox"]]
        if cid == supp_class_id:
            color = (255, 180, 0)
            tag = "supp"
        elif cid == text_class_id:
            color = (0, 100, 255)
            orient = item.get("orientation", "")
            tag = f"text_{orient}" if orient else "text"
        else:
            color = (0, 220, 0)
            tag = "main"

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, tag, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return vis


def process_chunk_worker(
    pairs_chunk: List[Tuple[Path, Path]],
    split: str,
    output_dir_str: str,
    review_dir_str: Optional[str],
    supp_class_id: int,
    text_class_id: int,
    prob_augment: float,
    seed: int,
    worker_id: int,
) -> Dict[str, int]:
    output_dir = Path(output_dir_str)
    review_dir = Path(review_dir_str) if review_dir_str else None
    rng = random.Random(seed + worker_id * 1000)

    augmentor = SupplementaryAugmentorV2(
        classifier=None,
        supp_class_id=supp_class_id,
        text_class_id=text_class_id,
        preserve_main_class_id=True,
        seed=seed + worker_id * 1000,
    )

    stats = {
        "total_processed": 0,
        "total_augmented": 0,
        "total_retained_original": 0,
        "supplementary_count": 0,
        "text_sign_count": 0,
        "vertical_text_count": 0,
        "square_text_count": 0,
        "horizontal_text_count": 0,
        "rejected_due_to_existing_sign": 0,
    }

    for idx, (img_path, lbl_path) in enumerate(pairs_chunk):
        raw_bgr = cv2.imread(str(img_path))
        if raw_bgr is None:
            continue

        h, w = raw_bgr.shape[:2]
        cids, boxes = parse_yolo_label_file(lbl_path, img_w=w, img_h=h)

        should_augment = False
        if boxes and rng.random() < prob_augment:
            should_augment = True

        out_img = raw_bgr
        out_labels: List[Dict[str, Any]] = []
        is_augmented = False

        if should_augment:
            forced_mode = rng.choices(
                ["supplementary_1", "supplementary_2", "text_sign"],
                weights=[0.45, 0.10, 0.45],
                k=1,
            )[0]

            forced_orientation = rng.choices(
                ["vertical", "square", "horizontal"],
                weights=[0.40, 0.25, 0.35],
                k=1,
            )[0]

            aug_bgr, gen_labels = augmentor.augment_image(
                image_bgr=raw_bgr,
                main_boxes=boxes,
                forced_mode=forced_mode,
                forced_orientation=forced_orientation,
                known_class_ids=cids,
            )

            if augmentor.last_rejected:
                stats["rejected_due_to_existing_sign"] += len(augmentor.last_rejected)

            num_added = len(gen_labels) - len(boxes)
            if num_added > 0:
                out_img = aug_bgr
                out_labels = gen_labels
                is_augmented = True
                stats["total_augmented"] += 1
                for lbl in gen_labels:
                    if lbl.get("class_id") == supp_class_id:
                        stats["supplementary_count"] += 1
                    elif lbl.get("class_id") == text_class_id:
                        stats["text_sign_count"] += 1
                        orient = lbl.get("orientation")
                        if orient == "vertical":
                            stats["vertical_text_count"] += 1
                        elif orient == "square":
                            stats["square_text_count"] += 1
                        else:
                            stats["horizontal_text_count"] += 1
            else:
                is_augmented = False

        if not is_augmented:
            stats["total_retained_original"] += 1
            for cid, bx in zip(cids, boxes):
                out_labels.append({
                    "class": "main_sign",
                    "class_id": cid,
                    "bbox": bx,
                })

        dest_img_path = output_dir / "images" / split / img_path.name
        dest_lbl_path = output_dir / "labels" / split / f"{img_path.stem}.txt"
        dest_meta_path = output_dir / "metadata" / split / f"{img_path.stem}.json"

        cv2.imwrite(str(dest_img_path), out_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        save_yolo_labels(dest_lbl_path, out_labels, img_w=w, img_h=h)

        sample_meta = {
            "sample_id": img_path.stem,
            "split": split,
            "image_file": img_path.name,
            "is_augmented": is_augmented,
            "num_signs": len(out_labels),
            "labels": out_labels,
        }
        with open(dest_meta_path, "w", encoding="utf-8") as f:
            json.dump(sample_meta, f, indent=2, ensure_ascii=False)

        stats["total_processed"] += 1

        if review_dir and is_augmented and stats["total_augmented"] <= 40:
            review_vis = draw_debug_visualization(
                out_img,
                out_labels,
                supp_class_id=supp_class_id,
                text_class_id=text_class_id,
            )
            rev_path = review_dir / f"{split}_{img_path.stem}_augmented.jpg"
            cv2.imwrite(str(rev_path), review_vis, [cv2.IMWRITE_JPEG_QUALITY, 92])

        if idx > 0 and idx % 50 == 0:
            gc.collect()

    return stats


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    config_yaml_path = Path(args.config_yaml).resolve()

    if not dataset_dir.exists():
        logger.error(f"Input dataset directory does not exist: {dataset_dir}")
        sys.exit(1)

    splits = args.splits if args.splits else detect_dataset_splits(dataset_dir)
    logger.info(f"Target dataset splits: {splits}")

    for split in splits:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "metadata" / split).mkdir(parents=True, exist_ok=True)

    review_dir: Optional[Path] = None
    if args.review_dir:
        review_dir = Path(args.review_dir).resolve()
        review_dir.mkdir(parents=True, exist_ok=True)

    if config_yaml_path.exists():
        shutil.copy(config_yaml_path, output_dir / "data.yaml")

    all_split_pairs: Dict[str, List[Tuple[Path, Path]]] = {}
    for split in splits:
        pairs = load_yolo_split_files(
            dataset_dir=dataset_dir,
            split=split,
            rng=rng,
        )
        all_split_pairs[split] = pairs

    total_available = sum(len(p) for p in all_split_pairs.values())
    if total_available == 0:
        logger.error(f"No valid image-label pairs found in {dataset_dir}")
        sys.exit(1)

    split_counts: Dict[str, int] = {}
    if args.max_samples > 0 and args.max_samples < total_available:
        allocated = 0
        for i, split in enumerate(splits):
            if i == len(splits) - 1:
                split_counts[split] = max(1, args.max_samples - allocated)
            else:
                cnt = int(round(args.max_samples * len(all_split_pairs[split]) / total_available))
                cnt = max(1, min(cnt, len(all_split_pairs[split])))
                split_counts[split] = cnt
                allocated += cnt
    else:
        for split in splits:
            split_counts[split] = len(all_split_pairs[split])

    logger.info(f"Sample allocation across splits: {split_counts}")

    stats = {
        "total_processed": 0,
        "total_augmented": 0,
        "total_retained_original": 0,
        "supplementary_count": 0,
        "text_sign_count": 0,
        "vertical_text_count": 0,
        "square_text_count": 0,
        "horizontal_text_count": 0,
        "rejected_due_to_existing_sign": 0,
    }

    num_workers = max(1, args.num_workers)

    for split in splits:
        target_cnt = split_counts[split]
        pairs = all_split_pairs[split][:target_cnt]
        logger.info(f"Processing split '{split}': {len(pairs)} images with {num_workers} workers")

        if num_workers == 1 or len(pairs) <= 1:
            w_stats = process_chunk_worker(
                pairs_chunk=pairs,
                split=split,
                output_dir_str=str(output_dir),
                review_dir_str=str(review_dir) if review_dir else None,
                supp_class_id=args.supp_class_id,
                text_class_id=args.text_class_id,
                prob_augment=args.prob_augment,
                seed=args.seed,
                worker_id=0,
            )
            for k in stats:
                stats[k] += w_stats.get(k, 0)
        else:
            k = min(num_workers, len(pairs))
            chunk_size = (len(pairs) + k - 1) // k
            chunks = [pairs[i * chunk_size : (i + 1) * chunk_size] for i in range(k)]

            with concurrent.futures.ProcessPoolExecutor(max_workers=k) as executor:
                futures = [
                    executor.submit(
                        process_chunk_worker,
                        chunk,
                        split,
                        str(output_dir),
                        str(review_dir) if review_dir else None,
                        args.supp_class_id,
                        args.text_class_id,
                        args.prob_augment,
                        args.seed,
                        w_idx,
                    )
                    for w_idx, chunk in enumerate(chunks) if chunk
                ]
                for f in concurrent.futures.as_completed(futures):
                    w_stats = f.result()
                    for key in stats:
                        stats[key] += w_stats.get(key, 0)

    summary_stats = {
        "dataset_name": output_dir.name,
        "classes_count": 102,
        "statistics": stats,
    }

    yaml_stats_path = output_dir / "metadata.yaml"
    with open(yaml_stats_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary_stats, f, sort_keys=False, allow_unicode=True)

    logger.info(f"Dataset generation completed: {stats}")
    logger.info(f"Dataset location: {output_dir}")
    logger.info(f"Dataset config: {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
