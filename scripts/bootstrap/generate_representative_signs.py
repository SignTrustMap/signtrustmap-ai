"""
Generates and uploads representative 128x128 PNG icons for all 102 QCVN 41:2019 traffic sign types.
Uploads them to MinIO S3 (stm-sign-crops/representative/{sign_code}.png) and outputs 011_seed_sign_type_images.sql.
"""

import json
import math
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, Any, Optional

import boto3
from botocore.client import Config
from PIL import Image, ImageDraw, ImageFont

# Project paths
ROOT_DIR = Path(__file__).resolve().parents[2]
AI_DIR = ROOT_DIR
BACKEND_DIR = ROOT_DIR.parent / "signtrustmap-backend"
CONFIGS_DIR = AI_DIR / "configs"
OUTPUT_DIR = AI_DIR / "outputs" / "representative_signs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIGRATION_011_PATH = BACKEND_DIR / "infra" / "postgres" / "migrations" / "011_seed_sign_type_images.sql"

# Import database mapping
sys.path.insert(0, str(AI_DIR))
from scripts.bootstrap.generate_sql import DB_SIGN_CODE_TO_ID

FONT_PATH = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
if not os.path.exists(FONT_PATH):
    FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url="https://s3.signmap.site",
        aws_access_key_id="npkhang2005",
        aws_secret_access_key="1C#ECspo6sYH*Q",
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def try_download_wikimedia(std_code: str, dest_path: Path) -> bool:
    """Attempt to download official PNG from Wikimedia Commons thumbnail service."""
    clean_code = std_code.replace(".", "").replace(" ", "").strip()
    candidates = [
        f"Vietnam_road_sign_{clean_code}.svg",
        f"Vietnam_road_sign_{clean_code.upper()}.svg",
        f"Vietnam_road_sign_{clean_code.lower()}.svg",
    ]
    if clean_code == "P124a":
        candidates.insert(0, "Vietnam_road_sign_P124a1.svg")
    if "-" in clean_code:
        # e.g. P127-40 -> Vietnam_road_sign_P127-40.svg or Vietnam_road_sign_P127.svg
        candidates.append(f"Vietnam_road_sign_{clean_code}.svg")

    for title in candidates:
        url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{title}?width=128"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "SignTrustMap/1.0 (academic; contact@signmap.site)"},
        )
        try:
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    data = resp.read()
                    if len(data) > 500:
                        with open(dest_path, "wb") as f:
                            f.write(data)
                        return True
        except Exception:
            pass
        time.sleep(0.05)
    return False


def render_vector_badge(
    sign_code: str,
    std_code: str,
    category: str,
    name_vi: str,
    dest_path: Path,
) -> None:
    """Dynamically render a clean, high-resolution 128x128 badge with transparent background."""
    img = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 1. Speed Limit Signs: Red circular border, white inside, bold speed number
    speed_match = re.search(r"(\d{2,3})", sign_code)
    if ("SPEED_LIMIT" in sign_code or category == "SPEED_LIMIT") and speed_match:
        val = speed_match.group(1)
        draw.ellipse([6, 6, 122, 122], fill="#FFFFFF", outline="#DC2626", width=12)
        font = ImageFont.truetype(FONT_PATH, 44 if len(val) == 2 else 36)
        bbox = font.getbbox(val)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((128 - w) / 2 - bbox[0], (128 - h) / 2 - bbox[1] - 2), val, fill="#0F172A", font=font)
        img.save(dest_path, "PNG")
        return

    # 2. End of Speed Limit / End of All Prohibitions
    if "END_ALL_PROHIBITIONS" in sign_code or "END_MIN_SPEED" in sign_code or "END_NO_OVERTAKING" in sign_code:
        draw.ellipse([6, 6, 122, 122], fill="#FFFFFF", outline="#94A3B8", width=8)
        draw.line([30, 98, 98, 30], fill="#64748B", width=8)
        draw.line([38, 106, 106, 38], fill="#64748B", width=6)
        img.save(dest_path, "PNG")
        return

    # 3. Prohibitory Signs: Red ring, white fill, icon / text
    if category == "PROHIBITORY" or "NO_" in sign_code or sign_code.startswith("P."):
        draw.ellipse([6, 6, 122, 122], fill="#FFFFFF", outline="#DC2626", width=12)

        # Draw red diagonal slash for 'NO_' signs
        has_slash = any(
            k in sign_code
            for k in ["NO_PARKING", "NO_STOPPING", "NO_TURN", "NO_U_TURN", "NO_HONKING", "NO_OVERTAKING"]
        )

        # Specific glyphs / text inside
        if "NO_ENTRY" in sign_code or std_code == "P.102":
            # Red circle with white horizontal bar
            draw.ellipse([6, 6, 122, 122], fill="#DC2626")
            draw.rounded_rectangle([24, 52, 104, 76], radius=4, fill="#FFFFFF")
            img.save(dest_path, "PNG")
            return

        if "NO_PARKING" in sign_code or "NO_STOPPING" in sign_code:
            font = ImageFont.truetype(FONT_PATH, 54)
            bbox = font.getbbox("P")
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(((128 - w) / 2 - bbox[0], (128 - h) / 2 - bbox[1] - 2), "P", fill="#1E293B", font=font)
            # Red diagonal slash
            draw.line([26, 26, 102, 102], fill="#DC2626", width=10)
            if "NO_STOPPING" in sign_code:
                # X-cross
                draw.line([102, 26, 26, 102], fill="#DC2626", width=10)
            img.save(dest_path, "PNG")
            return

        # Generic code text inside white circle
        display_code = std_code if std_code else sign_code.replace("NO_", "NO ")
        if len(display_code) > 6:
            parts = display_code.split("-")
            display_code = parts[0]
        font_size = 28 if len(display_code) <= 5 else 22
        font = ImageFont.truetype(FONT_PATH, font_size)
        bbox = font.getbbox(display_code)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((128 - w) / 2 - bbox[0], (128 - h) / 2 - bbox[1] - 2), display_code, fill="#1E293B", font=font)
        if has_slash:
            draw.line([24, 24, 104, 104], fill="#DC2626", width=10)
        img.save(dest_path, "PNG")
        return

    # 4. Warning Signs: Yellow equilateral triangle with rounded black border
    if category == "WARNING" or "CURVE" in sign_code or "DANGEROUS" in sign_code or "ROAD_NARROWS" in sign_code or "T_JUNCTION" in sign_code or "CROSSROAD" in sign_code or "INTERSECTION" in sign_code:
        pts = [(64, 12), (118, 114), (10, 114)]
        draw.polygon(pts, fill="#FBBF24", outline="#0F172A")
        # Inner triangle border line
        draw.polygon([(64, 24), (110, 108), (18, 108)], fill="#FBBF24", outline="#DC2626", width=5)

        display_text = std_code if std_code else "!"
        if len(display_text) > 6:
            display_text = display_text[:6]
        font_size = 22 if len(display_text) <= 5 else 18
        font = ImageFont.truetype(FONT_PATH, font_size)
        bbox = font.getbbox(display_text)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((128 - w) / 2 - bbox[0], 70 - h / 2), display_text, fill="#0F172A", font=font)
        img.save(dest_path, "PNG")
        return

    # 5. Mandatory Signs: Blue circle with white symbol/arrow
    if category == "MANDATORY" or "ROUNDABOUT" in sign_code or "KEEP_RIGHT" in sign_code or "GO_STRAIGHT" in sign_code or "TURN_" in sign_code:
        draw.ellipse([6, 6, 122, 122], fill="#1D4ED8", outline="#FFFFFF", width=3)
        display_text = std_code if std_code else sign_code[:4]
        font = ImageFont.truetype(FONT_PATH, 24)
        bbox = font.getbbox(display_text)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((128 - w) / 2 - bbox[0], (128 - h) / 2 - bbox[1] - 2), display_text, fill="#FFFFFF", font=font)
        img.save(dest_path, "PNG")
        return

    # 6. Information Signs: Blue rounded rectangle with white text
    draw.rounded_rectangle([10, 14, 118, 114], radius=12, fill="#1D4ED8", outline="#FFFFFF", width=3)
    display_text = "P" if "PARKING" in sign_code else (std_code if std_code else sign_code[:5])
    font_size = 54 if display_text == "P" else (24 if len(display_text) <= 5 else 18)
    font = ImageFont.truetype(FONT_PATH, font_size)
    bbox = font.getbbox(display_text)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((128 - w) / 2 - bbox[0], (128 - h) / 2 - bbox[1] - 2), display_text, fill="#FFFFFF", font=font)
    img.save(dest_path, "PNG")


def main():
    print("=== Generating Representative PNG Icons for all 102 Traffic Sign Types ===")
    with open(CONFIGS_DIR / "sign_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    meta_by_name = {v["class_name"].strip().upper(): v for v in meta.values()}

    s3 = get_s3_client()
    bucket = "stm-sign-crops"

    sql_lines = [
        "-- =============================================================================",
        "-- 011_seed_sign_type_images.sql",
        "-- Seeds representative official graphic sign icons for all 102 traffic sign types.",
        "-- Stored on MinIO S3: https://s3.signmap.site/stm-sign-crops/representative/",
        "-- =============================================================================",
        "",
        "INSERT INTO \"sign_type_images\" (\"sign_type_id\", \"image_url\", \"image_type\", \"is_primary\", \"uploaded_at\")",
        "VALUES",
    ]

    values_list = []
    total_signs = len(DB_SIGN_CODE_TO_ID)
    idx = 0

    for sign_code, sign_type_id in sorted(DB_SIGN_CODE_TO_ID.items(), key=lambda x: x[1]):
        idx += 1
        meta_info = meta_by_name.get(sign_code, {})
        std_code = meta_info.get("standard_code", "")
        category = meta_info.get("category", "PROHIBITORY")
        name_vi = meta_info.get("name_vi", sign_code)

        local_png = OUTPUT_DIR / f"{sign_code}.png"
        s3_key = f"representative/{sign_code}.png"
        s3_url = f"https://s3.signmap.site/{bucket}/{s3_key}"

        # 1. Download from Wikimedia if standard code exists
        downloaded = False
        if std_code and not local_png.exists():
            downloaded = try_download_wikimedia(std_code, local_png)

        # 2. Render clean vector badge if not downloaded
        if not local_png.exists() or local_png.stat().st_size < 300:
            render_vector_badge(sign_code, std_code, category, name_vi, local_png)

        # 3. Upload to MinIO S3
        s3.upload_file(
            str(local_png),
            bucket,
            s3_key,
            ExtraArgs={"ContentType": "image/png", "CacheControl": "max-age=31536000"},
        )

        values_list.append(
            f"  ({sign_type_id}, '{s3_url}', 'REPRESENTATIVE', true, now())"
        )
        print(f"[{idx:03d}/{total_signs}] {sign_code} (ID: {sign_type_id}) -> {s3_url}")

    sql_lines.append(",\n".join(values_list))
    sql_lines.extend([
        "ON CONFLICT (\"id\") DO NOTHING;",
        "",
        "-- Set primary representative image flag",
        "UPDATE \"sign_type_images\" SET \"is_primary\" = true WHERE \"image_type\" = 'REPRESENTATIVE';",
        "",
    ])

    with open(MIGRATION_011_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(sql_lines))

    print(f"\nGenerated {MIGRATION_011_PATH} successfully with {len(values_list)} sign type images.")
    print("All representative images uploaded to S3 successfully!")


if __name__ == "__main__":
    main()
