# ==============================================================================
# SignTrustMap - Few-Shot Visual Support Cache Builder & Diversity Clusterer
#
# Datasets Credited:
#   1. Kaggle 39-crop Vietnamese Traffic Signs:
#      https://www.kaggle.com/datasets/nguyenquyhcmus/vietnamese-traffic-signs
#   2. ALTSI 56 Detection Dataset (Roboflow v3 / Kaggle):
#      https://www.kaggle.com/datasets/hongluvit/vietnam-traffic-sign
#      https://universe.roboflow.com/nam-b0k5i/traffic-sign-dataset-3q95f
#   3. Hugging Face Vietnamese Traffic Signs (Metadata reference):
#      https://huggingface.co/datasets/star092304/Traffic-sign-detection-VietNam
#   4. VTSDB100 Local Benchmark Dataset (data/VTSDB100)
#   5. Wikimedia Commons QCVN 41:2019 Vector Diagrams:
#      https://commons.wikimedia.org/wiki/Category:Diagrams_of_road_signs_of_Vietnam
#
# Selection Algorithm:
#   1. Candidate Pooling & Pre-filtering:
#      - Extracts crops from indexed datasets with resolution >= 16-20 px.
#      - Computes Laplacian variance sharpness & hashes visual thumbnails for deduplication.
#   2. Top-4 Representative Anchors:
#      - Sorts candidates by area and sharpness to select 4 high-clarity frontal views.
#   3. Top-12 Diverse Medoids (K-Means++ on MobileCLIP-S2):
#      - Extracts 512-dim visual embeddings via MobileCLIP-S2 TensorRT Engine on GPU.
#      - Clusters candidate embeddings using K-Means++ (k=12) across different viewing angles,
#        lighting, distances, and backgrounds; selects medoids closest to cluster centroids.
# ==============================================================================

import json
import os
import random
from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
from PIL import Image
from loguru import logger
from scipy.cluster.vq import kmeans2
from src.core.classifier import TensorRTVisualRunner

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def normalize_code(c: str) -> str:
    return c.replace(".", "").replace("-", "").replace("*", "").replace(" ", "").upper()

def compute_sharpness(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def build_mapping(meta_100: dict, classes_100: dict, k39_base: Path, altsi_names: list) -> dict:
    k39_folders = [d.name for d in k39_base.iterdir() if d.is_dir()] if k39_base.exists() else []
    mapping = {}
    for str_cid, cname in classes_100.items():
        cid = int(str_cid)
        meta = meta_100.get(str_cid, {})
        std_code = meta.get("standard_code", "")
        norm_std = normalize_code(std_code)

        k39_match = None
        for kf in k39_folders:
            norm_kf = normalize_code(kf)
            if norm_kf == norm_std:
                k39_match = kf
                break
            elif norm_std and (norm_kf.startswith(norm_std) or norm_std.startswith(norm_kf)):
                if "SPEED_LIMIT" in meta.get("category", ""):
                    speed_val = cname.replace("Speed_limit_", "").replace("Min_speed_limit_", "")
                    if speed_val in kf:
                        k39_match = kf
                        break
                else:
                    k39_match = kf
                    break

        altsi_match = None
        for an in altsi_names:
            norm_an = normalize_code(an)
            if norm_an == norm_std:
                altsi_match = an
                break
            elif norm_std and (norm_an.startswith(norm_std) or norm_std.startswith(norm_an)):
                altsi_match = an
                break

        mapping[str_cid] = {
            "class_id": cid,
            "class_name": cname,
            "standard_code": std_code,
            "name_vi": meta.get("name_vi", ""),
            "name_en": meta.get("name_en", ""),
            "category": meta.get("category", ""),
            "k39_folder": k39_match,
            "altsi_name": altsi_match,
            "vtsdb100_id": cid,
        }
    return mapping

def index_altsi_labels(altsi_root: Path):
    candidates = {}
    label_dirs = [altsi_root / "train" / "labels", altsi_root / "test" / "labels"]
    total_files = 0
    for ldir in label_dirs:
        if not ldir.exists():
            continue
        for lf in ldir.glob("*.txt"):
            total_files += 1
            with open(lf, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_idx = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:5])
                        min_px = min(bw * 640, bh * 640)
                        if min_px >= 20:
                            if cls_idx not in candidates:
                                candidates[cls_idx] = []
                            candidates[cls_idx].append((min_px, lf, (xc, yc, bw, bh)))
    for c in candidates:
        candidates[c].sort(key=lambda x: x[0], reverse=True)
        candidates[c] = candidates[c][:35]
    logger.info(f"Indexed {total_files} ALTSI label files.")
    return candidates

def index_vtsdb_labels(vtsdb_root: Path):
    candidates = {}
    label_dirs = [
        vtsdb_root / "labels" / "train",
        vtsdb_root / "labels" / "val",
        vtsdb_root / "labels" / "test"
    ]
    total_files = 0
    for ldir in label_dirs:
        if not ldir.exists():
            continue
        for lf in ldir.glob("*.txt"):
            total_files += 1
            with open(lf, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_idx = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:5])
                        min_px = min(bw * 640, bh * 360)
                        if min_px >= 18:
                            if cls_idx not in candidates:
                                candidates[cls_idx] = []
                            candidates[cls_idx].append((min_px, lf, (xc, yc, bw, bh)))
    for c in candidates:
        candidates[c].sort(key=lambda x: x[0], reverse=True)
        candidates[c] = candidates[c][:35]
    logger.info(f"Indexed {total_files} VTSDB100 label files.")
    return candidates

def crop_yolo_box(img_path: Path, coords):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    ih, iw = img.shape[:2]
    xc, yc, bw, bh = coords
    xmin = max(0, int((xc - bw / 2.0) * iw))
    ymin = max(0, int((yc - bh / 2.0) * ih))
    xmax = min(iw, int((xc + bw / 2.0) * iw))
    ymax = min(ih, int((yc + bh / 2.0) * ih))
    if xmax <= xmin or ymax <= ymin:
        return None
    return img[ymin:ymax, xmin:xmax]

def extract_embeddings_trt(runner, preprocess, bgr_images, device: str = "cuda:0"):
    feats = []
    with torch.no_grad():
        for bgr in bgr_images:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(rgb)
            tensor_chw = preprocess(pil_im).to(device)
            f = runner.encode_image(tensor_chw.unsqueeze(0))
            feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)

def select_diverse_medoids(feats: np.ndarray, k: int = 12):
    n_samples = feats.shape[0]
    if n_samples <= k:
        return list(range(n_samples))
    try:
        centroids, labels = kmeans2(feats, k, minit="++", iter=15)
    except Exception:
        centroids, labels = kmeans2(feats, k, minit="points", iter=15)

    centroids = centroids / np.linalg.norm(centroids, axis=-1, keepdims=True)
    chosen_indices = []
    used_set = set()

    for cluster_id in range(k):
        members = np.where(labels == cluster_id)[0]
        if len(members) == 0:
            continue
        c = centroids[cluster_id]
        sims = np.dot(feats[members], c)
        best_member = members[np.argmax(sims)]
        if best_member not in used_set:
            chosen_indices.append(int(best_member))
            used_set.add(best_member)

    if len(chosen_indices) < k:
        remaining = [idx for idx in range(n_samples) if idx not in used_set]
        if remaining and chosen_indices:
            chosen_feats = feats[chosen_indices]
            while len(chosen_indices) < k and remaining:
                sims_to_chosen = np.max(np.dot(feats[remaining], chosen_feats.T), axis=1)
                farthest_idx = remaining[np.argmin(sims_to_chosen)]
                chosen_indices.append(int(farthest_idx))
                used_set.add(farthest_idx)
                remaining.remove(farthest_idx)
                chosen_feats = feats[chosen_indices]

    return chosen_indices

def generate_audit_report(manifest: dict, audit_md_path: Path):
    full_16 = sum(1 for v in manifest.values() if v["num_real_samples"] >= 16)
    full_8 = sum(1 for v in manifest.values() if 8 <= v["num_real_samples"] < 16)
    low_1 = sum(1 for v in manifest.values() if 1 <= v["num_real_samples"] < 8)
    zero_s = sum(1 for v in manifest.values() if v["num_real_samples"] == 0)
    total_crops = sum(v["num_real_samples"] for v in manifest.values())

    lines = []
    lines.append("# Báo Cáo Thống Kê & Bộ Sưu Tập Toàn Bộ Ảnh Mẫu Few-Shot (100 Lớp Biển Báo)\n")
    lines.append(f"- **Tổng số lớp**: {len(manifest)}")
    lines.append(f"- **Số lớp đạt 16 ảnh mẫu thực tế**: {full_16} lớp")
    lines.append(f"- **Số lớp đạt 8–15 ảnh mẫu thực tế**: {full_8} lớp")
    lines.append(f"- **Số lớp có 1–7 ảnh mẫu thực tế**: {low_1} lớp")
    lines.append(f"- **Số lớp hiếm (0 ảnh thực tế)**: {zero_s} lớp")
    lines.append(f"- **Tổng số ảnh crop thực tế**: {total_crops} ảnh\n")

    lines.append("## 1. Bảng Tổng Quan Nhanh 100 Lớp Biển Báo\n")
    lines.append("| ID | Tên Lớp | Mã QCVN | Bản Vẽ Chuẩn | Số Mẫu Thực Tế | Kích Thước TB | Chi Tiết Gallery |")
    lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: |")

    for cid in sorted([int(k) for k in manifest.keys()]):
        v = manifest[str(cid)]
        cname = v["class_name"]
        std_code = v.get("standard_code", "")
        name_vi = v.get("name_vi", "")
        samples = v.get("samples", [])
        cls_folder = f"{cid:03d}_{cname}"

        canonical_tag = f'<img src="support_cache/{cls_folder}/00_canonical_drawing.png" width="40" alt="QCVN">' if v.get("has_canonical") else "N/A"

        avg_sz = "N/A"
        if samples:
            avg_w = int(np.mean([s["width"] for s in samples]))
            avg_h = int(np.mean([s["height"] for s in samples]))
            avg_sz = f"{avg_w}×{avg_h} px"

        anchor = f"class-{cid:03d}"
        lines.append(f"| {cid:03d} | **{cname}**<br>_{name_vi}_ | `{std_code}` | {canonical_tag} | **{v['num_real_samples']}/16** | {avg_sz} | [Xem 16 ảnh](#{anchor}) |")

    lines.append("\n---\n")
    lines.append("## 2. Chi Tiết Toàn Bộ Ảnh Mẫu Cho Từng Lớp (Bản Vẽ Chuẩn + Đầy Đủ Ảnh Thực Tế)\n")

    for cid in sorted([int(k) for k in manifest.keys()]):
        v = manifest[str(cid)]
        cname = v["class_name"]
        std_code = v.get("standard_code", "")
        name_vi = v.get("name_vi", "")
        category = v.get("category", "")
        samples = v.get("samples", [])
        cls_folder = f"{cid:03d}_{cname}"

        lines.append(f'<a id="class-{cid:03d}"></a>')
        lines.append(f"### Lớp {cid:03d}: {cname} (`{std_code}`) — _{name_vi}_\n")
        lines.append(f"- **Mã chuẩn QCVN**: `{std_code}`")
        lines.append(f"- **Nhóm biển**: `{category}`")
        lines.append(f"- **Số lượng mẫu crop thực tế**: **{v['num_real_samples']}/16**")
        lines.append(f"- **Thư mục cache**: `support_cache/{cls_folder}/`\n")

        table_rows = []
        headers = ["Bản Vẽ Chuẩn (QCVN)"] + [f"Mẫu #{i+1:02d}" for i in range(len(samples))]
        table_rows.append("| " + " | ".join(headers) + " |")
        table_rows.append("| " + " | ".join([":---:"] * len(headers)) + " |")

        imgs = []
        if v.get("has_canonical"):
            imgs.append(f'<img src="support_cache/{cls_folder}/00_canonical_drawing.png" width="80" alt="Canonical">')
        else:
            imgs.append("*(Không có)*")

        for s in samples:
            imgs.append(f'<img src="support_cache/{cls_folder}/{s["filename"]}" width="80" alt="{s["filename"]}">')

        table_rows.append("| " + " | ".join(imgs) + " |")

        meta_row = ["**QCVN**"]
        for s in samples:
            meta_row.append(f'{s["width"]}×{s["height"]}<br>_{s["source"]}_')
        table_rows.append("| " + " | ".join(meta_row) + " |")

        lines.extend(table_rows)
        lines.append("\n")

    with open(audit_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info(f"Visual audit report written -> {audit_md_path}")

def main():
    set_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    few_shot_root = Path("outputs/few_shot")
    target_cache_dir = few_shot_root / "support_cache"
    target_cache_dir.mkdir(parents=True, exist_ok=True)

    with open("configs/sign_metadata.json", "r", encoding="utf-8") as f:
        meta_100 = json.load(f)

    with open("configs/classes.json", "r", encoding="utf-8") as f:
        classes_100 = json.load(f)

    k39_base = few_shot_root / "raw_datasets/vietnamese_traffic_signs_39_crop/dataset_cropped_for_lenet/dataset_cropped_for_lenet"
    altsi_base = few_shot_root / "raw_datasets/vietnam_traffic_sign_altsi_56"
    vtsdb_base = Path("data/VTSDB100")
    rep_signs_dir = Path("outputs/representative_signs")

    altsi_names = []
    altsi_yaml_path = altsi_base / "data.yaml"
    if altsi_yaml_path.exists():
        with open(altsi_yaml_path, "r", encoding="utf-8") as yf:
            altsi_names = yaml.safe_load(yf).get("names", [])

    logger.info("Building class mapping...")
    mapping = build_mapping(meta_100, classes_100, k39_base, altsi_names)
    with open(few_shot_root / "class_sources_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    logger.info("Indexing labels...")
    altsi_candidates = index_altsi_labels(altsi_base)
    vtsdb_candidates = index_vtsdb_labels(vtsdb_base)

    engine_path = "weights/classifiers/mobileclip_s2.engine"
    logger.info(f"Loading MobileCLIP-S2 TensorRT Engine from {engine_path} on {device}...")
    runner = TensorRTVisualRunner(engine_path, device=device)
    preprocess = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
    ])
    logger.info("TensorRT Engine ready! Starting fast stratified sample extraction...")

    manifest = {}

    for str_cid, cname in classes_100.items():
        cid = int(str_cid)
        cls_info = mapping.get(str_cid, {})
        cls_dir = target_cache_dir / f"{cid:03d}_{cname}"
        cls_dir.mkdir(parents=True, exist_ok=True)

        candidate_pool = []
        seen_hashes = set()

        k39_folder = cls_info.get("k39_folder")
        if k39_folder and (k39_base / k39_folder).exists():
            k_imgs = sorted(list((k39_base / k39_folder).glob("*.png")) + list((k39_base / k39_folder).glob("*.jpg")))
            for p in k_imgs[:40]:
                try:
                    bgr = cv2.imread(str(p))
                    if bgr is None:
                        continue
                    h, w = bgr.shape[:2]
                    if min(w, h) < 20:
                        continue
                    sh = compute_sharpness(bgr)
                    ch = (h, w, int(np.mean(bgr)))
                    if ch in seen_hashes:
                        continue
                    seen_hashes.add(ch)
                    candidate_pool.append({
                        "bgr": bgr,
                        "source": "Kaggle_39_crop",
                        "original_path": str(p),
                        "width": w,
                        "height": h,
                        "sharpness": round(sh, 2),
                    })
                except Exception:
                    continue

        altsi_name = cls_info.get("altsi_name")
        if altsi_name and altsi_name in altsi_names:
            altsi_idx = altsi_names.index(altsi_name)
            for min_px, lf, coords in altsi_candidates.get(altsi_idx, []):
                split_dir = lf.parent.parent
                img_name = lf.stem + ".jpg"
                img_path = split_dir / "images" / img_name
                if not img_path.exists():
                    continue
                crop = crop_yolo_box(img_path, coords)
                if crop is None or crop.shape[0] < 20 or crop.shape[1] < 20:
                    continue
                sh = compute_sharpness(crop)
                ch = (crop.shape[0], crop.shape[1], int(np.mean(crop)))
                if ch in seen_hashes:
                    continue
                seen_hashes.add(ch)
                candidate_pool.append({
                    "bgr": crop,
                    "source": "ALTSI_56_det",
                    "original_path": str(img_path),
                    "width": crop.shape[1],
                    "height": crop.shape[0],
                    "sharpness": round(sh, 2),
                })

        for min_px, lf, coords in vtsdb_candidates.get(cid, []):
            split_name = lf.parent.name
            img_name = lf.stem + ".jpg"
            img_path = vtsdb_base / "images" / split_name / img_name
            if not img_path.exists():
                continue
            crop = crop_yolo_box(img_path, coords)
            if crop is None or crop.shape[0] < 16 or crop.shape[1] < 16:
                continue
            sh = compute_sharpness(crop)
            ch = (crop.shape[0], crop.shape[1], int(np.mean(crop)))
            if ch in seen_hashes:
                continue
            seen_hashes.add(ch)
            candidate_pool.append({
                "bgr": crop,
                "source": "VTSDB100_det",
                "original_path": str(img_path),
                "width": crop.shape[1],
                "height": crop.shape[0],
                "sharpness": round(sh, 2),
            })

        final_selected = []
        if len(candidate_pool) <= 16:
            for item in candidate_pool:
                final_selected.append((item, "all_available_candidate"))
        else:
            candidate_pool.sort(key=lambda x: (min(x["width"], x["height"]) >= 32, x["width"] * x["height"], x["sharpness"]), reverse=True)
            rep_4 = candidate_pool[:4]
            for item in rep_4:
                final_selected.append((item, "representative_anchor"))

            remaining = candidate_pool[4:35]
            rem_bgrs = [x["bgr"] for x in remaining]
            rem_feats = extract_embeddings_trt(runner, preprocess, rem_bgrs, device=device)
            chosen_rel_indices = select_diverse_medoids(rem_feats, k=12)
            for r_idx in chosen_rel_indices:
                final_selected.append((remaining[r_idx], "kmeans_cluster_medoid"))

        manifest_samples = []
        for idx, (item, role) in enumerate(final_selected):
            fname = f"sample_{idx+1:02d}.jpg"
            out_p = cls_dir / fname
            cv2.imwrite(str(out_p), item["bgr"])
            manifest_samples.append({
                "filename": fname,
                "selection_role": role,
                "source": item["source"],
                "original_path": item["original_path"],
                "width": item["width"],
                "height": item["height"],
                "sharpness": item["sharpness"],
                "is_canonical": False
            })

        rep_candidates = [
            rep_signs_dir / f"{cname.upper()}.png",
            rep_signs_dir / f"{cname}.png"
        ]
        has_canonical = False
        for rep_p in rep_candidates:
            if rep_p.exists():
                rep_img = cv2.imread(str(rep_p))
                if rep_img is not None:
                    cv2.imwrite(str(cls_dir / "00_canonical_drawing.png"), rep_img)
                    has_canonical = True
                    break

        manifest[str_cid] = {
            "class_id": cid,
            "class_name": cname,
            "standard_code": cls_info.get("standard_code", ""),
            "name_vi": cls_info.get("name_vi", ""),
            "name_en": cls_info.get("name_en", ""),
            "category": cls_info.get("category", ""),
            "num_real_samples": len(final_selected),
            "has_canonical": has_canonical,
            "samples": manifest_samples
        }
        logger.info(f"Class {cid:03d} ({cname}): selected {len(final_selected)} samples (pool={len(candidate_pool)})")

    manifest_path = target_cache_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info(f"Manifest written -> {manifest_path}")

    audit_md_path = few_shot_root / "support_cache_audit.md"
    generate_audit_report(manifest, audit_md_path)
    logger.info("All-in-one TensorRT few-shot pipeline finished successfully!")

if __name__ == "__main__":
    main()
