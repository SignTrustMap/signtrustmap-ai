import argparse
import os
import re
from PIL import Image
from loguru import logger


def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>| ]', "_", name)


def parse_dataset_yaml(yaml_path: str) -> tuple[str, list[str], dict[int, str]]:
    data_root = "data"
    sub_paths = []
    names = {}

    with open(yaml_path, "r", encoding="utf-8") as f:
        in_names = False
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            if stripped.startswith("path:"):
                data_root = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            elif stripped.startswith("names:"):
                in_names = True
                continue
            elif in_names:
                if ":" in stripped:
                    parts = stripped.split(":", 1)
                    cls_key = parts[0].strip()
                    cls_val = parts[1].split("#")[0].strip()
                    try:
                        cls_id = int(cls_key)
                        names[cls_id] = cls_val
                    except ValueError:
                        pass
            else:
                if stripped.startswith("- "):
                    p = stripped[2:].strip().strip('"').strip("'")
                    sub_paths.append(p)

    return data_root, sub_paths, names


def get_dataset_info(names: dict[int, str]) -> list[tuple[str, int, int]]:
    datasets = [
        ("DFG", 0, 200),
        ("SynSTM", 200, 102),
        ("MTSD", 302, 401),
    ]
    return datasets


def extract_crops(
    yaml_path: str = "configs/training/all.yaml",
    output_base_dir: str = "runs/sign_crops",
    samples_per_class: int = 3,
    padding_ratio: float = 0.10,
) -> None:
    data_root, _, names = parse_dataset_yaml(yaml_path)
    dataset_configs = get_dataset_info(names)

    total_classes = len(names)
    saved_counts: dict[int, int] = {cid: 0 for cid in names}
    total_saved = 0

    splits = ["train", "val", "test"]

    for dataset_name, offset, count in dataset_configs:
        dataset_dir = os.path.join(data_root, dataset_name)
        if not os.path.exists(dataset_dir):
            logger.warning(f"Dataset directory not found: {dataset_dir}")
            continue

        dataset_out_dir = os.path.join(output_base_dir, dataset_name)
        os.makedirs(dataset_out_dir, exist_ok=True)

        target_class_ids = set(range(offset, offset + count)) & set(names.keys())

        for split in splits:
            labels_dir = os.path.join(dataset_dir, "labels", split)
            images_dir = os.path.join(dataset_dir, "images", split)

            if not os.path.exists(labels_dir) or not os.path.exists(images_dir):
                continue

            label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]

            for lf in label_files:
                all_targets_met = all(
                    saved_counts[cid] >= samples_per_class for cid in target_class_ids
                )
                if all_targets_met:
                    break

                label_path = os.path.join(labels_dir, lf)
                base_stem = os.path.splitext(lf)[0]

                img_path = None
                for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
                    candidate = os.path.join(images_dir, base_stem + ext)
                    if os.path.exists(candidate):
                        img_path = candidate
                        break

                if img_path is None:
                    continue

                with open(label_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                img_opened = None
                img_w = 0
                img_h = 0

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    try:
                        local_cid = int(parts[0])
                        xc = float(parts[1])
                        yc = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                    except ValueError:
                        continue

                    global_cid = local_cid + offset
                    if global_cid not in names or saved_counts[global_cid] >= samples_per_class:
                        continue

                    if img_opened is None:
                        try:
                            img_opened = Image.open(img_path)
                            img_w, img_h = img_opened.size
                        except Exception as e:
                            logger.error(f"Error opening image {img_path}: {e}")
                            break

                    pad_w = w * padding_ratio
                    pad_h = h * padding_ratio

                    x1 = max(0, int((xc - w / 2.0 - pad_w) * img_w))
                    y1 = max(0, int((yc - h / 2.0 - pad_h) * img_h))
                    x2 = min(img_w, int((xc + w / 2.0 + pad_w) * img_w))
                    y2 = min(img_h, int((yc + h / 2.0 + pad_h) * img_h))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = img_opened.crop((x1, y1, x2, y2))
                    saved_counts[global_cid] += 1
                    sample_idx = saved_counts[global_cid]

                    clean_name = sanitize_filename(names[global_cid])
                    out_filename = f"{global_cid}_{clean_name}_sample_{sample_idx}.jpg"
                    out_path = os.path.join(dataset_out_dir, out_filename)

                    try:
                        if crop.mode != "RGB":
                            crop = crop.convert("RGB")
                        crop.save(out_path, "JPEG", quality=95)
                        total_saved += 1
                    except Exception as e:
                        logger.error(f"Failed to save crop {out_path}: {e}")
                    finally:
                        crop.close()

                if img_opened is not None:
                    img_opened.close()

    full_count = sum(1 for cid in names if saved_counts[cid] >= samples_per_class)
    partial_count = sum(1 for cid in names if 0 < saved_counts[cid] < samples_per_class)
    empty_count = sum(1 for cid in names if saved_counts[cid] == 0)

    logger.info(
        f"Completed extraction: saved {total_saved} crops across {total_classes} classes to '{output_base_dir}'."
    )
    logger.info(
        f"Classes with {samples_per_class}/{samples_per_class} samples: {full_count}, partial: {partial_count}, missing: {empty_count}."
    )

    if empty_count > 0:
        missing_ids = [cid for cid in names if saved_counts[cid] == 0]
        logger.debug(f"Missing class IDs ({empty_count}): {missing_ids[:20]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-yaml", type=str, default="configs/training/all.yaml")
    parser.add_argument("--output-dir", type=str, default="runs/sign_crops")
    parser.add_argument("--samples-per-class", type=int, default=3)
    parser.add_argument("--padding", type=float, default=0.10)
    args = parser.parse_args()

    extract_crops(
        yaml_path=args.data_yaml,
        output_base_dir=args.output_dir,
        samples_per_class=args.samples_per_class,
        padding_ratio=args.padding,
    )
