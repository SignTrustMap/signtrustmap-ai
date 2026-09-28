from PIL import Image
import os
import subprocess
import zipfile
import tarfile
import json
import shutil
import time
import glob
from multiprocessing import Pool, cpu_count

def download_if_needed(target_dir):
    urls = [
        "https://go.vicos.si/dfgaugannot",
        "https://go.vicos.si/dfgaugannotdiff",
        "https://go.vicos.si/dfgimages",
        "https://go.vicos.si/dfgcat"
    ]
    
    expected_files = [
        "DFG-tsd-aug-annot-json.zip",
        "DFG-tsd-aug-annot-json-with-difficult-samples.zip",
        "JPEGImages.tar.bz2",
        "DFG-tsd-category-info.tar.bz2"
    ]

    os.makedirs(target_dir, exist_ok=True)
    
    all_exist = True
    for file in expected_files:
        file_path = os.path.join(target_dir, file)
        if not os.path.exists(file_path):
            all_exist = False
            break

    if all_exist:
        print(f"DFG dataset files already exist in {target_dir}. Skipping download.")
        return True

    print(f"Downloading DFG dataset to {target_dir}...")
    for url in urls:
        cmd = ["wget", "--trust-server-names", url]
        print(f"Running command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, cwd=target_dir, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to execute command: {' '.join(cmd)}. Error: {e}")
            return False
    return True

def extract_if_needed(target_dir):
    zip_path = os.path.join(target_dir, "DFG-tsd-aug-annot-json.zip")
    train_json_path = os.path.join(target_dir, "train.json")
    test_json_path = os.path.join(target_dir, "test.json")
    
    if not (os.path.exists(train_json_path) and os.path.exists(test_json_path)):
        if os.path.exists(zip_path):
            print(f"Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            print("Extraction of annotations complete.")
        else:
            print(f"Error: {zip_path} not found.")
            return False
    else:
        print("Annotations already extracted.")

    tar_path = os.path.join(target_dir, "JPEGImages.tar.bz2")
    images_dir = os.path.join(target_dir, "JPEGImages")
    expected_image_count = 15732
    
    if not os.path.exists(images_dir):
        train_img_dir = os.path.join(target_dir, "images", "train")
        val_img_dir = os.path.join(target_dir, "images", "val")
        train_count = len([f for f in os.listdir(train_img_dir) if f.lower().endswith(".jpg")]) if os.path.exists(train_img_dir) else 0
        val_count = len([f for f in os.listdir(val_img_dir) if f.lower().endswith(".jpg")]) if os.path.exists(val_img_dir) else 0
        
        if train_count + val_count >= expected_image_count:
            print(f"Images are already moved to train/val directories ({train_count + val_count} files).")
            return True
            
        if os.path.exists(tar_path):
            print(f"Extracting {tar_path} (this may take a few minutes)...")
            with tarfile.open(tar_path, "r:bz2") as tar_ref:
                tar_ref.extractall(target_dir)
            print("Extraction of images complete.")
        else:
            print(f"Error: {tar_path} not found.")
            return False
    else:
        current_count = len([f for f in os.listdir(images_dir) if f.lower().endswith(".jpg")])
        if current_count < expected_image_count:
            print(f"JPEGImages directory found but incomplete ({current_count}/{expected_image_count} files).")
            print("Waiting for background extraction to complete...")
            while current_count < expected_image_count:
                time.sleep(10)
                current_count = len([f for f in os.listdir(images_dir) if f.lower().endswith(".jpg")])
                print(f"Progress: {current_count}/{expected_image_count} files extracted.")
            print("Background extraction complete.")
        else:
            print(f"Images already fully extracted ({current_count} files).")
            
    return True

def prepare_yolo_dataset(target_dir):
    os.makedirs(os.path.join(target_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "labels", "val"), exist_ok=True)

    train_json = os.path.join(target_dir, "train.json")
    test_json = os.path.join(target_dir, "test.json")
    images_src_dir = os.path.join(target_dir, "JPEGImages")

    print("Loading train.json to extract categories...")
    with open(train_json, "r") as f:
        train_data = json.load(f)

    categories = train_data["categories"]
    category_names = {cat["id"]: cat["name"] for cat in categories}
    print(f"Extracted {len(categories)} categories.")

    blacklist_names = {
        "I-38", "I-39-1", "I-39-2", "I-39-3", "II-39", "III-120", "III-120.1",
        "III-123", "III-124", "III-14", "III-14.1", "III-15", "III-202-5",
        "III-203-2", "III-206-1", "III-27", "III-30-30", "III-64", "III-68",
        "III-84", "III-84-1", "III-85.1", "III-86-1", "III-86-2", "III-87",
        "III-90.1", "III-90.2", "III-91", "IV-1", "IV-1.1", "IV-11", "IV-12",
        "IV-12.1", "IV-16", "IV-20-1", "IV-3-4", "IV-3-5", "X-1.1", "X-1.2",
        "X-6-3"
    }

    def convert_split(json_file_path, split_name):
        print(f"Converting split: {split_name}...")
        with open(json_file_path, "r") as f:
            data = json.load(f)

        annotations_by_image = {}
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)

        count = 0
        for img in data["images"]:
            img_id = img["id"]
            file_name = img["file_name"]
            img_width = img["width"]
            img_height = img["height"]

            src_img_path = os.path.join(images_src_dir, file_name)
            dst_img_path = os.path.join(target_dir, "images", split_name, file_name)

            if os.path.exists(src_img_path):
                shutil.move(src_img_path, dst_img_path)
            elif not os.path.exists(dst_img_path):
                print(f"Warning: Image {file_name} not found in source or destination.")
                continue

            label_file_name = os.path.splitext(file_name)[0] + ".txt"
            label_file_path = os.path.join(target_dir, "labels", split_name, label_file_name)

            anns = annotations_by_image.get(img_id, [])
            with open(label_file_path, "w") as lf:
                for ann in anns:
                    cat_id = ann["category_id"]
                    bbox = ann["bbox"]
                    x_min, y_min, w, h = bbox

                    x_center = x_min + w / 2.0
                    y_center = y_min + h / 2.0
                    
                    x_center_norm = x_center / img_width
                    y_center_norm = y_center / img_height
                    w_norm = w / img_width
                    h_norm = h / img_height

                    lf.write(f"{cat_id} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")
            count += 1

        print(f"Converted {count} images and labels for split: {split_name}.")

    convert_split(train_json, "train")
    convert_split(test_json, "val")

    yaml_path = os.path.join(target_dir, "dataset.yaml")
    print(f"Generating YOLO dataset configuration file: {yaml_path}")

    yaml_content = f"""path: {os.path.abspath(target_dir)}
train: images/train
val: images/val

names:
"""
    for idx in sorted(category_names.keys()):
        name = category_names[idx]
        if name in blacklist_names:
            yaml_content += f"  {idx}: {name}  # blacklist\n"
        else:
            yaml_content += f"  {idx}: {name}\n"

    with open(yaml_path, "w") as yf:
        yf.write(yaml_content)

    try:
        if os.path.exists(images_src_dir) and not os.listdir(images_src_dir):
            os.rmdir(images_src_dir)
            print(f"Removed empty source directory {images_src_dir}.")
    except Exception as e:
        print(f"Could not remove {images_src_dir}: {e}")

    print("YOLO dataset preparation successfully complete.")

_g_mtsd_class_to_id = {}
_g_mtsd_target_dir = ""

def _process_mtsd_item(item):
    key, split, is_part = item
    target_dir = _g_mtsd_target_dir
    class_to_id = _g_mtsd_class_to_id

    filename = f"{key}.jpg"
    src_img = os.path.join(target_dir, "images", filename)
    dst_img = os.path.join(target_dir, "images", split, filename)

    if os.path.exists(src_img):
        try:
            os.rename(src_img, dst_img)
        except OSError:
            shutil.move(src_img, dst_img)
    elif not os.path.exists(dst_img):
        return 0

    ann_sub = "mtsd_v2_partially_annotated" if is_part else "mtsd_v2_fully_annotated"
    ann_path = os.path.join(target_dir, ann_sub, "annotations", f"{key}.json")
    label_path = os.path.join(target_dir, "labels", split, f"{key}.txt")

    if not os.path.exists(ann_path):
        with open(label_path, "w") as f:
            pass
        return 1

    with open(ann_path, "r") as f:
        data = json.load(f)

    w_img = float(data["width"])
    h_img = float(data["height"])

    lines = []
    for obj in data.get("objects", []):
        lbl = obj.get("label")
        if lbl not in class_to_id:
            continue
        cid = class_to_id[lbl]
        bbox = obj.get("bbox", {})
        xmin = max(0.0, float(bbox.get("xmin", 0)))
        ymin = max(0.0, float(bbox.get("ymin", 0)))
        xmax = min(w_img, float(bbox.get("xmax", 0)))
        ymax = min(h_img, float(bbox.get("ymax", 0)))
        
        bw = xmax - xmin
        bh = ymax - ymin
        if bw <= 0 or bh <= 0:
            continue

        xc = xmin + bw / 2.0
        yc = ymin + bh / 2.0

        x_norm = max(0.0, min(1.0, xc / w_img))
        y_norm = max(0.0, min(1.0, yc / h_img))
        w_norm = max(0.0, min(1.0, bw / w_img))
        h_norm = max(0.0, min(1.0, bh / h_img))

        lines.append(f"{cid} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    with open(label_path, "w") as f:
        f.writelines(lines)

    return 1

def _init_mtsd_worker(target_dir, class_to_id):
    global _g_mtsd_target_dir, _g_mtsd_class_to_id
    _g_mtsd_target_dir = target_dir
    _g_mtsd_class_to_id = class_to_id

def prepare_mtsd_yolo_dataset(target_dir):
    os.makedirs(os.path.join(target_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "images", "test"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "labels", "val"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "labels", "test"), exist_ok=True)

    mtsd_yaml_path = os.path.join(os.path.dirname(__file__), "MTSD.yaml")
    data_yaml_path = os.path.join(target_dir, "data.yaml")

    if os.path.exists(mtsd_yaml_path):
        with open(mtsd_yaml_path, "r") as f:
            lines = f.readlines()
        sorted_classes = []
        in_names = False
        for line in lines:
            if line.strip().startswith("names:"):
                in_names = True
                continue
            if in_names and ":" in line:
                sorted_classes.append(line.split(":", 1)[1].strip())
    else:
        print("Collecting unique MTSD class labels...")
        classes = set()
        for d in [os.path.join(target_dir, "mtsd_v2_fully_annotated", "annotations", "*.json"),
                  os.path.join(target_dir, "mtsd_v2_partially_annotated", "annotations", "*.json")]:
            for f in glob.iglob(d):
                with open(f, "r") as fp:
                    data = json.load(fp)
                    for obj in data.get("objects", []):
                        classes.add(obj["label"])
        sorted_classes = sorted(list(classes))

    class_to_id = {cls_name: i for i, cls_name in enumerate(sorted_classes)}
    print(f"Loaded {len(sorted_classes)} unique MTSD classes (indexed 0 to {len(sorted_classes)-1}).")

    if not os.path.exists(mtsd_yaml_path) or not os.path.exists(data_yaml_path):
        yaml_content = f"""path: {os.path.abspath(target_dir)}
train: images/train
val: images/val
test: images/test

names:
"""
        for idx, name in enumerate(sorted_classes):
            yaml_content += f"  {idx}: {name}\n"

        with open(mtsd_yaml_path, "w") as yf:
            yf.write(yaml_content)
        with open(data_yaml_path, "w") as yf:
            yf.write(yaml_content)
        print(f"Generated {mtsd_yaml_path} and {data_yaml_path}")

    items = []
    
    fully_train_split = os.path.join(target_dir, "mtsd_v2_fully_annotated", "splits", "train.txt")
    if os.path.exists(fully_train_split):
        with open(fully_train_split, "r") as f:
            for line in f:
                k = line.strip()
                if k:
                    items.append((k, "train", False))

    part_train_split = os.path.join(target_dir, "mtsd_v2_partially_annotated", "splits", "train.txt")
    if os.path.exists(part_train_split):
        with open(part_train_split, "r") as f:
            for line in f:
                k = line.strip()
                if k:
                    items.append((k, "train", True))

    val_split = os.path.join(target_dir, "mtsd_v2_fully_annotated", "splits", "val.txt")
    if os.path.exists(val_split):
        with open(val_split, "r") as f:
            for line in f:
                k = line.strip()
                if k:
                    items.append((k, "val", False))

    test_split = os.path.join(target_dir, "mtsd_v2_fully_annotated", "splits", "test.txt")
    if os.path.exists(test_split):
        with open(test_split, "r") as f:
            for line in f:
                k = line.strip()
                if k:
                    items.append((k, "test", False))

    total_items = len(items)
    print(f"Processing {total_items} items across train/val/test splits using multi-core parallel processing...")

    num_workers = min(cpu_count(), 6)
    t0 = time.time()
    
    with Pool(num_workers, initializer=_init_mtsd_worker, initargs=(target_dir, class_to_id)) as pool:
        processed_counts = pool.map(_process_mtsd_item, items, chunksize=500)
    
    t1 = time.time()
    total_processed = sum(processed_counts)
    print(f"MTSD YOLO conversion complete: {total_processed}/{total_items} items processed in {t1 - t0:.2f}s.")

def _optimize_mtsd_single_item(item):
    key, split, is_part, max_long_side, quality = item
    target_dir = _g_mtsd_target_dir
    class_to_id = _g_mtsd_class_to_id

    filename = f"{key}.jpg"
    img_path = os.path.join(target_dir, "images", split, filename)
    lbl_path = os.path.join(target_dir, "labels", split, f"{key}.txt")

    if not os.path.exists(img_path):
        return 0

    ann_sub = "mtsd_v2_partially_annotated" if is_part else "mtsd_v2_fully_annotated"
    ann_path = os.path.join(target_dir, ann_sub, "annotations", f"{key}.json")

    is_pano = False
    data = None
    if os.path.exists(ann_path):
        try:
            with open(ann_path, "r") as f:
                data = json.load(f)
            is_pano = bool(data.get("ispano", False))
        except Exception:
            pass

    if is_pano and data is not None:
        try:
            with Image.open(img_path) as orig_img:
                orig_img = orig_img.convert("RGB")
                W, H = orig_img.size
                tile_w = int(0.3 * W)
                step = int(0.25 * W)

                for i in range(4):
                    x1 = i * step
                    x2 = min(W, x1 + tile_w)
                    tw = x2 - x1
                    th = H
                    
                    tile_img = orig_img.crop((x1, 0, x2, H))
                    tile_max = max(tile_img.size)
                    if tile_max > max_long_side:
                        scale = max_long_side / float(tile_max)
                        new_tw = int(round(tile_img.size[0] * scale))
                        new_th = int(round(tile_img.size[1] * scale))
                        tile_img = tile_img.resize((new_tw, new_th), Image.BILINEAR)

                    tile_img_path = os.path.join(target_dir, "images", split, f"{key}_p{i}.jpg")
                    tile_lbl_path = os.path.join(target_dir, "labels", split, f"{key}_p{i}.txt")

                    tile_img.save(tile_img_path, "JPEG", quality=quality, optimize=True)
                    tile_img.close()

                    tile_lines = []
                    for obj in data.get("objects", []):
                        lbl = obj.get("label")
                        if lbl not in class_to_id:
                            continue
                        cid = class_to_id[lbl]
                        bbox = obj.get("bbox", {})
                        xmin = max(0.0, float(bbox.get("xmin", 0)))
                        ymin = max(0.0, float(bbox.get("ymin", 0)))
                        xmax = min(float(W), float(bbox.get("xmax", 0)))
                        ymax = min(float(H), float(bbox.get("ymax", 0)))
                        
                        box_w = xmax - xmin
                        if box_w <= 0:
                            continue

                        ix_min = max(xmin, float(x1))
                        ix_max = min(xmax, float(x2))
                        if ix_max > ix_min and (ix_max - ix_min) / box_w >= 0.4:
                            lx_min = ix_min - x1
                            lx_max = ix_max - x1
                            bw = lx_max - lx_min
                            bh = ymax - ymin
                            if bw <= 0 or bh <= 0:
                                continue
                            xc = lx_min + bw / 2.0
                            yc = ymin + bh / 2.0
                            
                            x_norm = max(0.0, min(1.0, xc / float(tw)))
                            y_norm = max(0.0, min(1.0, yc / float(th)))
                            w_norm = max(0.0, min(1.0, bw / float(tw)))
                            h_norm = max(0.0, min(1.0, bh / float(th)))

                            tile_lines.append(f"{cid} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

                    with open(tile_lbl_path, "w") as lf:
                        lf.writelines(tile_lines)

            if os.path.exists(img_path):
                os.remove(img_path)
            if os.path.exists(lbl_path):
                os.remove(lbl_path)
            return 1
        except Exception:
            return 0
    else:
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                W, H = img.size
                max_dim = max(W, H)
                if max_dim > max_long_side:
                    scale = max_long_side / float(max_dim)
                    new_w = int(round(W * scale))
                    new_h = int(round(H * scale))
                    img = img.resize((new_w, new_h), Image.BILINEAR)
                tmp_path = img_path + ".tmp.jpg"
                img.save(tmp_path, "JPEG", quality=quality, optimize=True)
            os.replace(tmp_path, img_path)
            return 1
        except Exception:
            return 0

def optimize_mtsd_dataset(target_dir, max_long_side=1280, quality=88, max_workers=None):
    if max_workers is None:
        max_workers = min(cpu_count(), 6)
    mtsd_yaml_path = os.path.join(os.path.dirname(__file__), "MTSD.yaml")
    with open(mtsd_yaml_path, "r") as f:
        lines = f.readlines()
    sorted_classes = []
    in_names = False
    for line in lines:
        if line.strip().startswith("names:"):
            in_names = True
            continue
        if in_names and ":" in line:
            sorted_classes.append(line.split(":", 1)[1].strip())
    class_to_id = {cls_name: i for i, cls_name in enumerate(sorted_classes)}

    items = []
    for split, is_part, txt_file in [
        ("train", False, os.path.join(target_dir, "mtsd_v2_fully_annotated", "splits", "train.txt")),
        ("train", True, os.path.join(target_dir, "mtsd_v2_partially_annotated", "splits", "train.txt")),
        ("val", False, os.path.join(target_dir, "mtsd_v2_fully_annotated", "splits", "val.txt")),
        ("test", False, os.path.join(target_dir, "mtsd_v2_fully_annotated", "splits", "test.txt"))
    ]:
        if os.path.exists(txt_file):
            with open(txt_file, "r") as f:
                for line in f:
                    k = line.strip()
                    if k:
                        items.append((k, split, is_part, max_long_side, quality))

    total_items = len(items)
    print(f"Optimizing {total_items} MTSD images (max_long_side={max_long_side}, quality={quality}, workers={max_workers})...")
    
    t0 = time.time()
    with Pool(max_workers, initializer=_init_mtsd_worker, initargs=(target_dir, class_to_id)) as pool:
        processed_counts = pool.map(_optimize_mtsd_single_item, items, chunksize=100)
    t1 = time.time()

    total_processed = sum(processed_counts)
    print(f"MTSD image optimization complete: {total_processed}/{total_items} processed in {t1 - t0:.2f}s.")

def main():
    import sys
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    if dataset_name in ("all", "dfg", "DFG"):
        dfg_dir = "/media/taiduc_orico_ssd/DucPT/signtrustmap/signtrustmap-ai/data/DFG"
        if download_if_needed(dfg_dir):
            if extract_if_needed(dfg_dir):
                prepare_yolo_dataset(dfg_dir)

    if dataset_name in ("all", "mtsd", "MTSD"):
        mtsd_dir = "/media/taiduc_orico_ssd/DucPT/signtrustmap/signtrustmap-ai/data/MTSD"
        if os.path.exists(mtsd_dir):
            prepare_mtsd_yolo_dataset(mtsd_dir)

    if dataset_name in ("mtsd_optimize", "optimize_mtsd"):
        mtsd_dir = "/media/taiduc_orico_ssd/DucPT/signtrustmap/signtrustmap-ai/data/MTSD"
        if os.path.exists(mtsd_dir):
            optimize_mtsd_dataset(mtsd_dir, max_long_side=1280, quality=88, max_workers=6)

if __name__ == "__main__":
    main()
