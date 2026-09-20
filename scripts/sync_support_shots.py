import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import boto3
from botocore.client import Config
import cv2
import numpy as np
from PIL import Image
import torch
from loguru import logger
from torchvision import transforms

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR.parent / "signtrustmap-backend"
CONFIGS_DIR = ROOT_DIR / "configs"
SUPPORT_CACHE_DIR = ROOT_DIR / "outputs" / "few_shot" / "support_cache"
DEFAULT_SQL_PATH = BACKEND_DIR / "infra" / "postgres" / "migrations" / "013_seed_few_shot_support_shots.sql"
PROTOTYPES_OUTPUT_PATH = ROOT_DIR / "weights" / "classifiers" / "protofuse_prototypes_100.pt"

sys.path.insert(0, str(ROOT_DIR))
from scripts.bootstrap.generate_sql import DB_SIGN_CODE_TO_ID
from src.core.classifier import TensorRTVisualRunner, SignClassifier
from src.core.config import settings


def parse_args():
    parser = argparse.ArgumentParser(description="Sync support shots to MinIO, compute ProtoFuse prototypes, and generate migration SQL.")
    parser.add_argument("--dry-run", action="store_true", help="Do not upload to MinIO, only generate prototypes and SQL.")
    parser.add_argument("--skip-upload", action="store_true", help="Skip MinIO S3 upload step.")
    parser.add_argument("--bucket", type=str, default="stm-catalog-images", help="Target MinIO bucket for catalog images.")
    parser.add_argument("--endpoint", type=str, default="https://s3.signmap.site", help="MinIO S3 endpoint URL.")
    parser.add_argument("--target-sql", type=str, default=str(DEFAULT_SQL_PATH), help="Target path for SQL migration file.")
    return parser.parse_args()


def get_s3_client(endpoint_url: str):
    access_key = settings.MINIO_ACCESS_KEY or "npkhang2005"
    secret_key = settings.MINIO_SECRET_KEY or "1C#ECspo6sYH*Q"
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def load_metadata_mappings() -> Tuple[Dict[str, Dict], Dict[int, str], Dict[str, int]]:
    with open(CONFIGS_DIR / "sign_metadata.json", "r", encoding="utf-8") as f:
        meta_dict = json.load(f)
    with open(CONFIGS_DIR / "classes.json", "r", encoding="utf-8") as f:
        classes_dict = {int(k): v for k, v in json.load(f).items()}

    sign_name_to_id = {}
    for code_key, db_id in DB_SIGN_CODE_TO_ID.items():
        clean_code = code_key.strip().upper()
        sign_name_to_id[clean_code] = db_id

    return meta_dict, classes_dict, sign_name_to_id


def resolve_sign_type_id(class_idx: int, class_name: str, meta: Dict, sign_name_to_id: Dict[str, int]) -> Optional[int]:
    meta_entry = meta.get(str(class_idx), {})
    std_code = meta_entry.get("standard_code", "")
    db_code = meta_entry.get("class_name", class_name).strip().upper()

    if db_code in sign_name_to_id:
        return sign_name_to_id[db_code]

    alt_key = class_name.strip().upper().replace(" ", "_").replace("-", "_")
    if alt_key in sign_name_to_id:
        return sign_name_to_id[alt_key]

    for k, v in sign_name_to_id.items():
        if k == std_code.replace(".", ""):
            return v

    return None


def main():
    args = parse_args()
    logger.info(f"Starting Support Shots sync and ProtoFuse builder (dry_run={args.dry_run}, skip_upload={args.skip_upload})")

    meta_dict, classes_dict, sign_name_to_id = load_metadata_mappings()
    s3 = None if (args.dry_run or args.skip_upload) else get_s3_client(args.endpoint)

    engine_path = ROOT_DIR / "weights" / "classifiers" / "mobileclip_s2.engine"
    logger.info(f"Loading MobileCLIP TensorRT engine: {engine_path}")
    visual_runner = TensorRTVisualRunner(str(engine_path), device="cuda:0")

    target_preprocess = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
    ])

    cached_text_path = ROOT_DIR / "weights" / "embeddings" / "text_embeddings_mobileclip_s2.pt"
    if not cached_text_path.exists():
        raise FileNotFoundError(f"Required text embeddings not found at {cached_text_path}")
    text_features = torch.load(cached_text_path, map_location="cpu")

    sql_entries: List[str] = []
    prototypes_list: List[torch.Tensor] = []

    total_images_processed = 0
    total_images_uploaded = 0

    class_dirs = sorted([d for d in SUPPORT_CACHE_DIR.iterdir() if d.is_dir()], key=lambda p: int(p.name.split("_")[0]))

    for class_dir in class_dirs:
        parts = class_dir.name.split("_", 1)
        class_idx = int(parts[0])
        class_name = parts[1] if len(parts) > 1 else classes_dict.get(class_idx, f"class_{class_idx}")

        meta_entry = meta_dict.get(str(class_idx), {})
        std_code = meta_entry.get("standard_code", class_name).replace(" ", "")
        sign_type_id = resolve_sign_type_id(class_idx, class_name, meta_dict, sign_name_to_id)

        sample_files = sorted([f for f in class_dir.iterdir() if f.is_file() and f.name.startswith("sample_")])
        if not sample_files:
            logger.warning(f"No samples found in {class_dir.name}")
            t_feat = text_features[class_idx]
            prototypes_list.append(t_feat / t_feat.norm(dim=-1, keepdim=True))
            continue

        sample_embeds = []
        for sample_file in sample_files:
            img_bgr = cv2.imread(str(sample_file))
            if img_bgr is None or img_bgr.size == 0:
                continue
            pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            tensor = target_preprocess(pil_img)

            with torch.inference_mode():
                feat = visual_runner.encode_image(tensor.unsqueeze(0).to("cuda:0")).cpu()
                feat = feat / feat.norm(dim=-1, keepdim=True)
                sample_embeds.append(feat)

            s3_key = f"support_shots/{std_code}/{sample_file.name}"
            s3_url = f"{args.endpoint}/{args.bucket}/{s3_key}"

            if s3 is not None:
                content_type = "image/jpeg" if sample_file.suffix.lower() in (".jpg", ".jpeg") else "image/png"
                with open(sample_file, "rb") as f_in:
                    s3.put_object(
                        Bucket=args.bucket,
                        Key=s3_key,
                        Body=f_in,
                        ContentType=content_type,
                    )
                total_images_uploaded += 1

            if sign_type_id is not None:
                sql_entries.append(
                    f"  ({sign_type_id}, '{s3_url}', 'REAL_PHOTO', false, now())"
                )
            total_images_processed += 1

        if sample_embeds:
            class_embeds = torch.cat(sample_embeds, dim=0)
            v_c = torch.mean(class_embeds, dim=0)
            v_c = v_c / v_c.norm(dim=-1, keepdim=True)

            t_c = text_features[class_idx].cpu()
            t_c = t_c / t_c.norm(dim=-1, keepdim=True)

            p_c = 0.5 * v_c + 0.5 * t_c
            p_c = p_c / p_c.norm(dim=-1, keepdim=True)
            prototypes_list.append(p_c)
        else:
            t_feat = text_features[class_idx].cpu()
            prototypes_list.append(t_feat / t_feat.norm(dim=-1, keepdim=True))

        logger.debug(f"Class [{class_idx:03d}] {std_code}: processed {len(sample_embeds)} shots, fused prototype.")

    proto_matrix = torch.stack(prototypes_list, dim=0).float()
    PROTOTYPES_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(proto_matrix, PROTOTYPES_OUTPUT_PATH)
    logger.info(f"Saved ProtoFuse prototype matrix ({proto_matrix.shape}) to {PROTOTYPES_OUTPUT_PATH}")

    target_sql_path = Path(args.target_sql)
    target_sql_path.parent.mkdir(parents=True, exist_ok=True)

    sql_content = [
        "-- =============================================================================",
        "-- 013_seed_few_shot_support_shots.sql",
        f"-- Seeds {len(sql_entries)} real-world support crops into sign_type_images for ProtoFuse & few-shot classification",
        f"-- MinIO Bucket: {args.bucket} (https://s3.signmap.site/{args.bucket}/support_shots/)",
        "-- =============================================================================",
        "",
        "BEGIN;",
        "",
        'INSERT INTO "sign_type_images" ("sign_type_id", "image_url", "image_type", "is_primary", "uploaded_at")',
        "VALUES",
        ",\n".join(sql_entries) + ";",
        "",
        "COMMIT;",
        "",
        "-- Verification audit query",
        'SELECT tst.sign_code, tst.name_vi, COUNT(sti.id) AS real_shot_count',
        'FROM "traffic_sign_types" tst',
        'LEFT JOIN "sign_type_images" sti ON tst.id = sti.sign_type_id AND sti.image_type = \'REAL_PHOTO\'',
        'GROUP BY tst.id, tst.sign_code, tst.name_vi',
        'ORDER BY real_shot_count ASC, tst.sign_code ASC;',
    ]

    with open(target_sql_path, "w", encoding="utf-8") as f_sql:
        f_sql.write("\n".join(sql_content) + "\n")
    logger.info(f"Generated SQL migration with {len(sql_entries)} entries at {target_sql_path}")

    del visual_runner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(f"Sync & ProtoFuse build complete: {total_images_processed} processed, {total_images_uploaded} uploaded.")


if __name__ == "__main__":
    main()
