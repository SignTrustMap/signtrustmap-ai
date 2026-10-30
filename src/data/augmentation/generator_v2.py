import gc
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from src.core.class_manager import class_manager
from src.core.classifier import SignClassifier

DEFAULT_MAIN_SIGN_SIZE_MM: float = 700.0

MAIN_SIGN_PHYSICAL_SIZES: Dict[str, Tuple[float, float]] = {
    "R.122": (600.0, 600.0),
    "P.102": (700.0, 700.0),
    "P.103a": (700.0, 700.0),
    "P.103b": (700.0, 700.0),
    "P.103c": (700.0, 700.0),
    "P.104": (700.0, 700.0),
    "P.106a": (700.0, 700.0),
    "P.106b": (700.0, 700.0),
    "P.123a": (700.0, 700.0),
    "P.123b": (700.0, 700.0),
    "P.124a": (700.0, 700.0),
    "P.124b": (700.0, 700.0),
    "P.127": (700.0, 700.0),
    "P.130": (700.0, 700.0),
    "P.131a": (700.0, 700.0),
    "P.131b": (700.0, 700.0),
    "P.131c": (700.0, 700.0),
    "W.201a": (700.0, 700.0),
    "W.201b": (700.0, 700.0),
    "W.207a": (700.0, 700.0),
    "W.208": (700.0, 700.0),
    "W.210": (700.0, 700.0),
    "W.224": (700.0, 700.0),
    "W.227": (700.0, 700.0),
    "R.301a": (700.0, 700.0),
    "R.301b": (700.0, 700.0),
    "R.302a": (700.0, 700.0),
    "R.303": (700.0, 700.0),
    "I.401": (700.0, 700.0),
    "I.407a": (600.0, 600.0),
    "I.408": (600.0, 600.0),
    "I.423a": (700.0, 700.0),
}

SUPPLEMENTARY_METADATA: Dict[str, Dict[str, Any]] = {
    "S.501": {"width_mm": 500.0, "height_mm": 300.0, "name": "Phạm vi tác dụng của biển"},
    "S.502": {"width_mm": 500.0, "height_mm": 300.0, "name": "Khoảng cách đến đối tượng báo hiệu"},
    "S.503a": {"width_mm": 500.0, "height_mm": 250.0, "name": "Hướng tác dụng hai bên"},
    "S.503b": {"width_mm": 500.0, "height_mm": 250.0, "name": "Hướng tác dụng sang phải"},
    "S.503c": {"width_mm": 500.0, "height_mm": 250.0, "name": "Hướng tác dụng sang trái"},
    "S.503d": {"width_mm": 300.0, "height_mm": 450.0, "name": "Hướng tác dụng phía trước"},
    "S.503e": {"width_mm": 300.0, "height_mm": 450.0, "name": "Hướng tác dụng hai chiều"},
    "S.503f": {"width_mm": 300.0, "height_mm": 450.0, "name": "Hướng tác dụng kết thúc"},
    "S.504": {"width_mm": 500.0, "height_mm": 300.0, "name": "Làn đường"},
    "S.505a": {"width_mm": 600.0, "height_mm": 300.0, "name": "Loại xe"},
    "S.505b": {"width_mm": 600.0, "height_mm": 300.0, "name": "Loại xe hạn chế qua cầu"},
    "S.505c": {"width_mm": 600.0, "height_mm": 300.0, "name": "Biểu tượng xe kết hợp"},
    "S.506a": {"width_mm": 500.0, "height_mm": 500.0, "name": "Hướng đường ưu tiên"},
    "S.506b": {"width_mm": 500.0, "height_mm": 500.0, "name": "Hướng đường ưu tiên tại ngã tư"},
    "S.508a": {"width_mm": 600.0, "height_mm": 300.0, "name": "Thời gian tác dụng giờ"},
    "S.508b": {"width_mm": 600.0, "height_mm": 300.0, "name": "Thời gian tác dụng ngày/thứ"},
    "S.509a": {"width_mm": 600.0, "height_mm": 300.0, "name": "Thuyết minh biển chính"},
    "S.509b": {"width_mm": 600.0, "height_mm": 300.0, "name": "Thuyết minh cấm rẽ"},
}

COMPATIBILITY_RULES: Dict[str, Dict[str, float]] = {
    "P.130": {"S.503d": 0.20, "S.503e": 0.20, "S.503f": 0.15, "S.508a": 0.25, "S.508b": 0.10, "S.505a": 0.05, "S.501": 0.05},
    "P.131a": {"S.503d": 0.20, "S.503e": 0.20, "S.503f": 0.15, "S.508a": 0.25, "S.508b": 0.10, "S.505a": 0.05, "S.501": 0.05},
    "P.131b": {"S.503d": 0.25, "S.503e": 0.25, "S.503f": 0.20, "S.508a": 0.20, "S.505a": 0.10},
    "P.131c": {"S.503d": 0.25, "S.503e": 0.25, "S.503f": 0.20, "S.508a": 0.20, "S.505a": 0.10},
    "P.103a": {"S.508a": 0.35, "S.508b": 0.25, "S.501": 0.15, "S.502": 0.15, "S.504": 0.10},
    "P.104": {"S.508a": 0.35, "S.508b": 0.25, "S.501": 0.15, "S.504": 0.15, "S.505a": 0.10},
    "P.106a": {"S.508a": 0.35, "S.508b": 0.20, "S.505a": 0.20, "S.501": 0.15, "S.504": 0.10},
    "P.106b": {"S.508a": 0.40, "S.508b": 0.25, "S.505a": 0.15, "S.501": 0.20},
    "P.123a": {"S.508a": 0.35, "S.508b": 0.25, "S.505a": 0.20, "S.509b": 0.20},
    "P.123b": {"S.508a": 0.35, "S.508b": 0.25, "S.505a": 0.20, "S.509b": 0.20},
    "P.124a": {"S.508a": 0.35, "S.508b": 0.25, "S.505a": 0.20, "S.509b": 0.20},
    "P.124b": {"S.508a": 0.35, "S.508b": 0.25, "S.505a": 0.20, "S.509b": 0.20},
    "W.208": {"S.506b": 0.70, "S.502": 0.30},
    "R.122": {"S.506b": 0.60, "S.502": 0.40},
    "I.401": {"S.506a": 0.70, "S.501": 0.15, "S.502": 0.15},
    "I.407a": {"S.504": 0.50, "S.501": 0.30, "S.505a": 0.20},
    "I.408": {"S.503a": 0.25, "S.503b": 0.25, "S.503c": 0.25, "S.505a": 0.25},
    "CATEGORY_PROHIBITORY": {"S.508a": 0.30, "S.508b": 0.20, "S.501": 0.20, "S.505a": 0.15, "S.504": 0.10, "S.502": 0.05},
    "CATEGORY_WARNING": {"S.501": 0.40, "S.502": 0.40, "S.504": 0.10, "S.506b": 0.10},
    "CATEGORY_MANDATORY": {"S.504": 0.40, "S.508a": 0.25, "S.508b": 0.15, "S.505a": 0.20},
    "CATEGORY_INFORMATION": {"S.501": 0.35, "S.502": 0.25, "S.504": 0.20, "S.505a": 0.10, "S.506a": 0.10},
}

TEXT_SIGN_PRESETS_V2: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
    "vertical": {
        "PROHIBITORY": [
            {"bg": "red", "fg": "white", "text": "CẤM\nCÁC LOẠI XE 03 BÁNH\nVÀ XE THÔ SƠ 04 BÁNH\n6:00 - 8:00\n16:00 - 19:00"},
            {"bg": "red", "fg": "white", "text": "CẤM XE TẢI\nTRÊN 2.5 TẤN\n6:00 - 9:00\n16:00 - 20:00"},
            {"bg": "red", "fg": "white", "text": "CẤM Ô TÔ KHÁCH\nVÀ XE TẢI\n6:00 - 22:00"},
            {"bg": "red", "fg": "white", "text": "ĐOẠN ĐƯỜNG\nCẤM DỪNG ĐỖ XE\nCẢ HAI CHIỀU"},
            {"bg": "red", "fg": "white", "text": "CẤM MỌI LOẠI XE\nĐI VÀO KHU VỰC\nTỪ 19:00 ĐẾN 24:00"},
            {"bg": "red", "fg": "white", "text": "CẤM XE CƠ GIỚI\n3 BÁNH VÀ 4 BÁNH\n6:00 - 22:00"},
            {"bg": "red", "fg": "white", "text": "CẤM ĐỖ XE\nCẢ HAI CHIỀU\nĐOẠN ĐƯỜNG 500M"},
            {"bg": "red", "fg": "white", "text": "XE MÔ TÔ\nXE GẮN MÁY\nĐƯỢC PHÉP HOẠT ĐỘNG"},
        ],
        "WARNING": [
            {"bg": "yellow", "fg": "black", "text": "CHÚ Ý\nĐOẠN ĐƯỜNG\nHAY XẢY RA\nTAI NẠN"},
            {"bg": "yellow", "fg": "black", "text": "KHU VỰC\nTRƯỜNG HỌC\nĐI CHẬM"},
            {"bg": "yellow", "fg": "black", "text": "ĐOẠN ĐƯỜNG\nTHƯỜNG XUYÊN\nKIỂM TRA TỐC ĐỘ"},
        ],
        "MANDATORY": [
            {"bg": "blue", "fg": "white", "text": "TUYẾN PHỐ ĐI BỘ\nCẤM PHƯƠNG TIỆN\nTHỨ BẢY - CHỦ NHẬT\n18:00 - 23:00"},
            {"bg": "red", "fg": "white", "text": "CẤM ĐI THẲNG\nXE TẢI ĐI\nĐƯỜNG TRÁNH"},
        ],
        "INFORMATION": [
            {"bg": "blue", "fg": "white", "text": "BÃI ĐỖ XE\nCÓ THU PHÍ\nTHEO QUY ĐỊNH"},
            {"bg": "white", "fg": "black", "text": "KHU VỰC\nGIAO NHẬN HÀNG\nCHO PHÉP DỪNG 15P"},
        ],
    },
    "square": {
        "PROHIBITORY": [
            {"bg": "red", "fg": "white", "text": "CẤM ĐỖ XE\nNGÀY CHẴN\n6:00 - 22:00"},
            {"bg": "red", "fg": "white", "text": "CẤM ĐỖ XE\nNGÀY LẺ\n6:00 - 22:00"},
            {"bg": "red", "fg": "white", "text": "CẤM DỪNG\nVÀ ĐỖ XE"},
            {"bg": "red", "fg": "white", "text": "CẤM XE TẢI\nTRÊN 3.5 TẤN"},
            {"bg": "red", "fg": "white", "text": "CẤM Ô TÔ\nQUAY ĐẦU XE"},
        ],
        "WARNING": [
            {"bg": "yellow", "fg": "black", "text": "CÔNG TRƯỜNG\nĐANG THI CÔNG"},
            {"bg": "yellow", "fg": "black", "text": "CHÚ Ý\nGIẢM TỐC ĐỘ"},
        ],
        "MANDATORY": [
            {"bg": "blue", "fg": "white", "text": "HƯỚNG ĐI\nXE BUÝT"},
        ],
        "INFORMATION": [
            {"bg": "blue", "fg": "white", "text": "BÃI ĐỖ XE\nCÓ THU PHÍ"},
        ],
    },
    "horizontal": {
        "PROHIBITORY": [
            {"bg": "red", "fg": "white", "text": "CẤM XE TẢI 6:00 - 22:00"},
            {"bg": "red", "fg": "white", "text": "CẤM ĐỖ XE NGÀY CHẴN"},
            {"bg": "red", "fg": "white", "text": "CẤM ĐỖ XE NGÀY LẺ"},
            {"bg": "red", "fg": "white", "text": "CẤM RẼ TRÁI 6:00 - 9:00 / 16:30 - 19:30"},
            {"bg": "red", "fg": "white", "text": "CẤM DỪNG ĐỖ XE QUÁ 5 PHÚT"},
            {"bg": "white", "fg": "black", "text": "TỪ 6H00 ĐẾN 22H00"},
            {"bg": "white", "fg": "red", "text": "TRỪ XE ƯU TIÊN"},
        ],
        "WARNING": [
            {"bg": "yellow", "fg": "black", "text": "CHÚ Ý ĐOẠN ĐƯỜNG HAY TAI NẠN"},
            {"bg": "yellow", "fg": "black", "text": "CÔNG TRƯỜNG ĐANG THI CÔNG"},
            {"bg": "yellow", "fg": "black", "text": "ĐOẠN ĐƯỜNG THƯỜNG XUYÊN ĐÔNG XE"},
            {"bg": "yellow", "fg": "black", "text": "CHÚ Ý GIẢM TỐC ĐỘ KHU VỰC TRƯỜNG HỌC"},
        ],
        "MANDATORY": [
            {"bg": "red", "fg": "white", "text": "CẤM ĐI THẲNG XE TẢI ĐI ĐƯỜNG TRÁNH"},
            {"bg": "blue", "fg": "white", "text": "XE MÁY ĐI CHẬM NHƯỜNG ĐƯỜNG NGƯỜI ĐI BỘ"},
        ],
        "INFORMATION": [
            {"bg": "blue", "fg": "white", "text": "BÃI ĐỖ XE CÓ THU PHÍ"},
            {"bg": "blue", "fg": "white", "text": "XE BUÝT ĐƯỢC PHÉP HOẠT ĐỘNG"},
            {"bg": "blue", "fg": "white", "text": "KHU VỰC ĐỖ XE Ô TÔ VÀ XE MÁY"},
            {"bg": "blue", "fg": "white", "text": "LỐI ĐI BỘ SANG ĐƯỜNG"},
            {"bg": "blue", "fg": "white", "text": "HƯỚNG ĐI TP. HỒ CHÍ MINH"},
            {"bg": "blue", "fg": "white", "text": "BÃI ĐỖ XE BUÝT"},
        ],
    },
}

FONTS_DIR = Path(__file__).resolve().parents[3] / "data" / "fonts"
FONT_CANDIDATES = [
    str(FONTS_DIR / "DIN1451-Bahnschrift.ttf"),
    str(FONTS_DIR / "BarlowCondensed-Bold.ttf"),
    str(FONTS_DIR / "Barlow-Bold.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]

REF_SYMBOLS_DIR = Path(__file__).resolve().parents[3] / "outputs" / "symbol" / "ref_symbols"


def _get_font(size: int, condensed: bool = False) -> ImageFont.ImageFont:
    preferred = [
        str(FONTS_DIR / "BarlowCondensed-Bold.ttf") if condensed else str(FONTS_DIR / "DIN1451-Bahnschrift.ttf"),
        str(FONTS_DIR / "DIN1451-Bahnschrift.ttf"),
        str(FONTS_DIR / "Barlow-Bold.ttf"),
        str(FONTS_DIR / "BarlowCondensed-Bold.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for candidate in preferred:
        if os.path.exists(candidate):
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _fit_text_font(
    text: str,
    max_w: float,
    max_h: float,
    initial_font_size: int,
    draw: ImageDraw.ImageDraw,
    min_font_size: int = 6,
    condensed: bool = False,
) -> Tuple[ImageFont.ImageFont, int, float, float]:
    font_size = max(min_font_size, initial_font_size)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        lines = [text]

    while font_size > min_font_size:
        font = _get_font(font_size, condensed=condensed)
        line_w_max = 0.0
        line_h_max = 0.0
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            lw = bbox[2] - bbox[0]
            lh = bbox[3] - bbox[1]
            if lw > line_w_max:
                line_w_max = lw
            if lh > line_h_max:
                line_h_max = lh
        total_th = (len(lines) - 1) * (font_size * 1.25) + line_h_max
        if line_w_max <= max_w and total_th <= max_h:
            break
        font_size -= 1

    font = _get_font(font_size, condensed=condensed)
    line_w_max = 0.0
    line_h_max = 0.0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_w_max = max(line_w_max, bbox[2] - bbox[0])
        line_h_max = max(line_h_max, bbox[3] - bbox[1])
    total_th = (len(lines) - 1) * (font_size * 1.25) + line_h_max
    return font, font_size, line_w_max, total_th


def _draw_rounded_rectangle(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float, float, float],
    radius: float,
    fill: Tuple[int, int, int],
    outline: Optional[Tuple[int, int, int]] = None,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def _draw_horizontal_arrow(
    draw: ImageDraw.ImageDraw,
    y: float,
    x_start: float,
    x_end: float,
    direction: str,
    thickness: int,
    arrow_size: float,
    color: Tuple[int, int, int],
) -> None:
    draw.line([(x_start, y), (x_end, y)], fill=color, width=thickness)
    if "right" in direction:
        tip = (x_end, y)
        left = (x_end - arrow_size, y - arrow_size * 0.6)
        right = (x_end - arrow_size, y + arrow_size * 0.6)
        draw.polygon([tip, left, right], fill=color)
    if "left" in direction:
        tip = (x_start, y)
        left = (x_start + arrow_size, y - arrow_size * 0.6)
        right = (x_start + arrow_size, y + arrow_size * 0.6)
        draw.polygon([tip, left, right], fill=color)


def _draw_vertical_arrow(
    draw: ImageDraw.ImageDraw,
    x: float,
    y_start: float,
    y_end: float,
    direction: str,
    thickness: int,
    arrow_size: float,
    color: Tuple[int, int, int],
) -> None:
    draw.line([(x, y_start), (x, y_end)], fill=color, width=thickness)
    if "up" in direction:
        tip = (x, y_start)
        left = (x - arrow_size * 0.6, y_start + arrow_size)
        right = (x + arrow_size * 0.6, y_start + arrow_size)
        draw.polygon([tip, left, right], fill=color)
    if "down" in direction:
        tip = (x, y_end)
        left = (x - arrow_size * 0.6, y_end - arrow_size)
        right = (x + arrow_size * 0.6, y_end - arrow_size)
        draw.polygon([tip, left, right], fill=color)


def get_compatible_supplementary_types(
    standard_code: Optional[str],
    category: Optional[str],
    count: int = 1,
) -> List[str]:
    weights_dict: Optional[Dict[str, float]] = None
    if standard_code and standard_code in COMPATIBILITY_RULES:
        weights_dict = COMPATIBILITY_RULES[standard_code]
    elif category:
        cat_key = f"CATEGORY_{category.upper()}"
        if cat_key in COMPATIBILITY_RULES:
            weights_dict = COMPATIBILITY_RULES[cat_key]

    if not weights_dict:
        weights_dict = COMPATIBILITY_RULES["CATEGORY_PROHIBITORY"]

    candidates = list(weights_dict.keys())
    weights = [weights_dict[k] for k in candidates]

    chosen: List[str] = []
    pool_cands = list(candidates)
    pool_weights = list(weights)
    for _ in range(count):
        if not pool_cands:
            break
        pick = random.choices(pool_cands, weights=pool_weights, k=1)[0]
        chosen.append(pick)
        idx = pool_cands.index(pick)
        pool_cands.pop(idx)
        pool_weights.pop(idx)
    return chosen


def get_random_text_sign_preset_v2(
    category: Optional[str],
    orientation: str = "vertical",
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    r = rng if rng is not None else random
    orient = orientation if orientation in TEXT_SIGN_PRESETS_V2 else "vertical"
    cat_group = TEXT_SIGN_PRESETS_V2[orient]
    cat_key = (category or "PROHIBITORY").upper()
    presets = cat_group.get(cat_key, cat_group.get("PROHIBITORY", []))
    if not presets:
        presets = TEXT_SIGN_PRESETS_V2["vertical"]["PROHIBITORY"]
    preset = r.choice(presets).copy()
    preset["orientation"] = orient
    return preset


def estimate_pixel_scale(
    main_box: List[float],
    standard_code: Optional[str] = None,
) -> float:
    x1, y1, x2, y2 = main_box[:4]
    wm = max(1.0, x2 - x1)
    hm = max(1.0, y2 - y1)
    min_dim = min(wm, hm)

    phys_size_mm = DEFAULT_MAIN_SIGN_SIZE_MM
    if standard_code and standard_code in MAIN_SIGN_PHYSICAL_SIZES:
        phys_w, phys_h = MAIN_SIGN_PHYSICAL_SIZES[standard_code]
        phys_size_mm = min(phys_w, phys_h)

    px_per_mm = min_dim / max(1.0, phys_size_mm)
    return px_per_mm


def compute_supplementary_placement(
    main_box: List[float],
    supp_meta: Dict[str, Any],
    px_per_mm: float,
    parent_box: Optional[List[float]] = None,
    image_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = main_box[:4]
    wm = x2 - x1
    main_cx = (x1 + x2) / 2.0

    ref_box = parent_box if parent_box is not None else main_box
    ref_y2 = ref_box[3]

    supp_w_mm = supp_meta.get("width_mm", 500.0)
    supp_h_mm = supp_meta.get("height_mm", 300.0)

    supp_w = max(12, int(round(supp_w_mm * px_per_mm)))
    supp_h = max(10, int(round(supp_h_mm * px_per_mm)))

    nominal_gap_px = max(2.0, 50.0 * px_per_mm)
    v_jitter = random.uniform(-0.10, 0.10) * nominal_gap_px
    h_jitter = random.uniform(-0.04, 0.04) * wm

    gap_px = nominal_gap_px + v_jitter
    supp_cx = main_cx + h_jitter
    supp_top = ref_y2 + gap_px
    supp_left = supp_cx - (supp_w / 2.0)

    x1_s = int(round(supp_left))
    y1_s = int(round(supp_top))
    x2_s = x1_s + supp_w
    y2_s = y1_s + supp_h

    if image_shape is not None:
        img_h, img_w = image_shape[:2]
        if x1_s < 0:
            diff = -x1_s
            x1_s = 0
            x2_s += diff
        if x2_s > img_w:
            diff = x2_s - img_w
            x2_s = img_w
            x1_s = max(0, x1_s - diff)

    return (x1_s, y1_s, x2_s, y2_s)


def compute_text_sign_placement_v2(
    main_box: List[float],
    px_per_mm: float,
    orientation: Optional[str] = None,
    rng: Optional[random.Random] = None,
    image_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[Tuple[int, int, int, int], Tuple[float, float], str]:
    r = rng if rng is not None else random
    if orientation is None:
        orientation = r.choices(["vertical", "square", "horizontal"], weights=[0.40, 0.25, 0.35], k=1)[0]

    if orientation == "vertical":
        width_mm = r.uniform(360.0, 450.0)
        height_mm = r.uniform(480.0, 620.0)
    elif orientation == "square":
        dim = r.uniform(450.0, 520.0)
        width_mm = dim
        height_mm = dim * r.uniform(0.95, 1.05)
    else:
        width_mm = r.uniform(550.0, 700.0)
        height_mm = r.uniform(280.0, 360.0)

    supp_meta = {"width_mm": width_mm, "height_mm": height_mm}
    target_box = compute_supplementary_placement(
        main_box=main_box,
        supp_meta=supp_meta,
        px_per_mm=px_per_mm,
        parent_box=None,
        image_shape=image_shape,
    )
    return target_box, (width_mm, height_mm), orientation


def _render_qcvn_supplementary_sign_raw(
    subclass: str,
    target_width: int,
    target_height: int,
    used_symbols: Optional[List[str]] = None,
) -> Image.Image:
    scale = 2
    w = max(40, target_width * scale)
    h = max(24, target_height * scale)

    bg_color = (250, 250, 250)
    border_color = (20, 20, 20)
    fg_color = (20, 20, 20)

    if subclass in ["S.510a", "S.510b"]:
        bg_color = (210, 30, 30)
        border_color = (250, 250, 250)
        fg_color = (250, 250, 250)

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin = max(2, int(h * 0.04))
    border_w = max(2, int(min(w, h) * 0.05))
    corner_r = max(4, int(min(w, h) * 0.08))

    _draw_rounded_rectangle(
        draw,
        (margin, margin, w - margin, h - margin),
        radius=corner_r,
        fill=bg_color,
        outline=border_color,
        width=border_w,
    )

    cx = w / 2.0
    cy = h / 2.0

    if subclass == "S.501":
        distances = ["100 m", "200 m", "300 m", "500 m", "800 m", "1 km", "1.5 km"]
        text = random.choice(distances)
        max_text_w = max(10.0, w * 0.50)
        max_text_h = max(10.0, h * 0.65)
        font, font_size, tw, th = _fit_text_font(text, max_text_w, max_text_h, int(h * 0.45), draw)
        bbox = draw.textbbox((0, 0), text, font=font)
        tx = cx - (bbox[2] - bbox[0]) / 2.0
        ty = cy - (bbox[3] - bbox[1]) / 2.0 - bbox[1]
        draw.text((tx, ty), text, font=font, fill=fg_color)

        arr_len = max(6.0, w * 0.14)
        arr_thick = max(2, int(min(w, h) * 0.06))
        arr_sz = max(4.0, min(w, h) * 0.15)
        _draw_vertical_arrow(draw, margin + border_w + arr_len * 0.5, cy + arr_len * 0.5, cy - arr_len * 0.5, "up", arr_thick, arr_sz, fg_color)
        _draw_vertical_arrow(draw, w - (margin + border_w + arr_len * 0.5), cy + arr_len * 0.5, cy - arr_len * 0.5, "up", arr_thick, arr_sz, fg_color)

    elif subclass == "S.502":
        distances = ["50 m", "100 m", "150 m", "200 m", "300 m", "500 m", "1 km"]
        text = random.choice(distances)
        max_text_w = max(10.0, w * 0.70)
        max_text_h = max(10.0, h * 0.65)
        font, font_size, tw, th = _fit_text_font(text, max_text_w, max_text_h, int(h * 0.50), draw)
        bbox = draw.textbbox((0, 0), text, font=font)
        tx = cx - (bbox[2] - bbox[0]) / 2.0
        ty = cy - (bbox[3] - bbox[1]) / 2.0 - bbox[1]
        draw.text((tx, ty), text, font=font, fill=fg_color)

    elif subclass in ["S.503a", "S.503b", "S.503c"]:
        arr_thick = max(2, int(min(w, h) * 0.08))
        arr_sz = max(4.0, min(w, h) * 0.20)
        arrow_w = w * 0.65
        x_s = cx - arrow_w / 2.0
        x_e = cx + arrow_w / 2.0
        d = "both" if subclass == "S.503a" else ("right" if subclass == "S.503b" else "left")
        _draw_horizontal_arrow(draw, cy, x_s, x_e, d, arr_thick, arr_sz, fg_color)

    elif subclass in ["S.503d", "S.503e", "S.503f"]:
        arr_thick = max(2, int(min(w, h) * 0.08))
        arr_sz = max(4.0, min(w, h) * 0.20)
        arrow_h = h * 0.65
        y_s = cy - arrow_h / 2.0
        y_e = cy + arrow_h / 2.0
        d = "up" if subclass == "S.503d" else ("both" if subclass == "S.503e" else "down")
        _draw_vertical_arrow(draw, cx, y_s, y_e, d, arr_thick, arr_sz, fg_color)

    elif subclass == "S.504":
        arr_thick = max(2, int(min(w, h) * 0.10))
        arr_sz = max(5.0, min(w, h) * 0.25)
        arrow_h = h * 0.55
        _draw_vertical_arrow(draw, cx, cy - arrow_h / 2.0, cy + arrow_h / 2.0, "down", arr_thick, arr_sz, fg_color)

    elif subclass in ["S.505a", "S.505b", "S.505c"]:
        loaded = False
        if REF_SYMBOLS_DIR.exists():
            symbol_candidates = ["symbol_truck.png", "symbol_car.png", "symbol_bus.png", "symbol_motorcycle.png"]
            if used_symbols:
                filtered = [s for s in symbol_candidates if s not in used_symbols]
                if filtered:
                    symbol_candidates = filtered
            sym_file = random.choice(symbol_candidates)
            sym_path = REF_SYMBOLS_DIR / sym_file
            if sym_path.exists():
                try:
                    sym_img = Image.open(sym_path).convert("RGBA")
                    max_sym_w = int(w * 0.70)
                    max_sym_h = int(h * 0.60)
                    sym_img.thumbnail((max_sym_w, max_sym_h), Image.Resampling.LANCZOS)
                    sx = int(cx - sym_img.width / 2.0)
                    sy = int(cy - sym_img.height / 2.0)
                    img.paste(sym_img, (sx, sy), sym_img)
                    if used_symbols is not None:
                        used_symbols.append(sym_file)
                    loaded = True
                except Exception:
                    loaded = False

        if not loaded:
            types = ["XE TẢI", "XE CON", "XE BUÝT", "XE MÁY"]
            text = random.choice(types)
            max_text_w = max(10.0, w * 0.75)
            max_text_h = max(10.0, h * 0.55)
            font, font_size, tw, th = _fit_text_font(text, max_text_w, max_text_h, int(h * 0.40), draw)
            bbox = draw.textbbox((0, 0), text, font=font)
            tx = cx - (bbox[2] - bbox[0]) / 2.0
            ty = cy - (bbox[3] - bbox[1]) / 2.0 - bbox[1]
            draw.text((tx, ty), text, font=font, fill=fg_color)

    elif subclass in ["S.506a", "S.506b"]:
        main_road_thick = max(3, int(min(w, h) * 0.16))
        sub_road_thick = max(2, int(min(w, h) * 0.08))
        draw.line([(cx, h - margin - border_w), (cx, margin + border_w)], fill=fg_color, width=main_road_thick)
        draw.line([(margin + border_w, cy), (w - margin - border_w, cy)], fill=(120, 120, 120), width=sub_road_thick)
        if subclass == "S.506b":
            draw.line([(cx, cy), (w - margin - border_w, margin + border_w)], fill=fg_color, width=main_road_thick)

    elif subclass == "S.508a":
        hours = ["6:00 - 8:30\n16:30 - 18:30", "6:00 - 22:00", "9:00 - 16:00\n20:00 - 6:00", "22:00 - 5:00"]
        text = random.choice(hours)
        max_text_w = max(10.0, w * 0.85)
        max_text_h = max(10.0, h * 0.75)
        font, font_size, tw, th = _fit_text_font(text, max_text_w, max_text_h, int(h * 0.35), draw)
        lines = text.split("\n")
        spacing = font_size * 1.25
        t_total_h = (len(lines) - 1) * spacing + font_size
        s_y = cy - t_total_h / 2.0
        for i, l in enumerate(lines):
            bb = draw.textbbox((0, 0), l, font=font)
            lx = cx - (bb[2] - bb[0]) / 2.0
            ly = s_y + i * spacing - bb[1]
            draw.text((lx, ly), l, font=font, fill=fg_color)

    elif subclass == "S.508b":
        days = ["T2 - T6", "THỨ HAI - THỨ SÁU", "CHỦ NHẬT & NGÀY LỄ", "THỨ BẢY & CHỦ NHẬT"]
        text = random.choice(days)
        max_text_w = max(10.0, w * 0.85)
        max_text_h = max(10.0, h * 0.65)
        font, font_size, tw, th = _fit_text_font(text, max_text_w, max_text_h, int(h * 0.40), draw)
        bbox = draw.textbbox((0, 0), text, font=font)
        tx = cx - (bbox[2] - bbox[0]) / 2.0
        ty = cy - (bbox[3] - bbox[1]) / 2.0 - bbox[1]
        draw.text((tx, ty), text, font=font, fill=fg_color)

    elif subclass in ["S.509a", "S.509b"]:
        texts = ["CẤM RẼ TRÁI", "CẤM XE QUAY ĐẦU", "ĐƯỜNG MỘT CHIỀU"]
        text = random.choice(texts)
        max_text_w = max(10.0, w * 0.85)
        max_text_h = max(10.0, h * 0.60)
        font, font_size, tw, th = _fit_text_font(text, max_text_w, max_text_h, int(h * 0.40), draw)
        bbox = draw.textbbox((0, 0), text, font=font)
        tx = cx - (bbox[2] - bbox[0]) / 2.0
        ty = cy - (bbox[3] - bbox[1]) / 2.0 - bbox[1]
        draw.text((tx, ty), text, font=font, fill=fg_color)

    else:
        text = "BIỂN PHỤ"
        max_text_w = max(10.0, w * 0.80)
        max_text_h = max(10.0, h * 0.60)
        font, font_size, tw, th = _fit_text_font(text, max_text_w, max_text_h, int(h * 0.40), draw)
        bbox = draw.textbbox((0, 0), text, font=font)
        tx = cx - (bbox[2] - bbox[0]) / 2.0
        ty = cy - (bbox[3] - bbox[1]) / 2.0 - bbox[1]
        draw.text((tx, ty), text, font=font, fill=fg_color)

    final_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return final_img


_QCVN_TEMPLATE_CACHE: Dict[str, Image.Image] = {}


def render_qcvn_supplementary_sign(
    subclass: str,
    target_width: int,
    target_height: int,
    used_symbols: Optional[List[str]] = None,
) -> Image.Image:
    template = _QCVN_TEMPLATE_CACHE.get(subclass)
    if template is None:
        template = _render_qcvn_supplementary_sign_raw(
            subclass=subclass,
            target_width=300,
            target_height=180,
            used_symbols=used_symbols,
        )
        _QCVN_TEMPLATE_CACHE[subclass] = template

    tw = max(1, target_width)
    th = max(1, target_height)
    return template.resize((tw, th), Image.Resampling.BILINEAR)


def _render_text_sign_raw_v2(
    preset: Dict[str, Any],
    target_width: int,
    target_height: int,
) -> Tuple[Image.Image, str]:
    scale = 2
    w = max(50, target_width * scale)
    h = max(30, target_height * scale)

    bg_type = preset.get("bg", "red").lower()
    text = preset.get("text", "CẤM XE TẢI\n6:00 - 22:00")

    if bg_type == "red":
        bg_color = (205, 30, 30)
        border_color = (235, 235, 235)
        fg_color = (225, 225, 225)
    elif bg_type == "yellow":
        bg_color = (245, 192, 20)
        border_color = (25, 25, 25)
        fg_color = (20, 20, 20)
    elif bg_type == "blue":
        bg_color = (20, 92, 170)
        border_color = (235, 235, 235)
        fg_color = (225, 225, 225)
    else:
        bg_color = (240, 240, 240)
        border_color = (25, 25, 25)
        fg_color = (25, 25, 25) if preset.get("fg") != "red" else (205, 30, 30)

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin = max(2, int(min(w, h) * 0.04))
    border_w = max(2, int(min(w, h) * 0.045))
    corner_r = max(4, int(min(w, h) * 0.065))

    _draw_rounded_rectangle(
        draw,
        (margin, margin, w - margin, h - margin),
        radius=corner_r,
        fill=bg_color,
        outline=border_color,
        width=border_w,
    )

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    num_lines = max(1, len(lines))

    max_text_w = max(10.0, w - 2 * (margin + border_w + 6))
    max_text_h = max(10.0, h - 2 * (margin + border_w + 6))
    init_fs = max(8, int(h / (num_lines + 0.6)))
    condensed = (num_lines > 2) or any(len(l) > 14 for l in lines)
    font, font_size, _, total_text_h = _fit_text_font(text, max_text_w, max_text_h, init_fs, draw, condensed=condensed)

    line_spacing = font_size * 1.25
    total_text_h = (num_lines - 1) * line_spacing + font_size
    start_y = (h - total_text_h) / 2.0

    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        line_x = (w - tw) / 2.0
        line_y = start_y + (i * line_spacing) - bbox[1]
        draw.text((line_x, line_y), line, font=font, fill=fg_color)

    final_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return final_img, text


_TEXT_SIGN_TEMPLATE_CACHE_V2: Dict[str, Image.Image] = {}


def render_text_sign_v2(
    preset: Dict[str, Any],
    target_width: int,
    target_height: int,
) -> Tuple[Image.Image, str]:
    tw = max(1, target_width)
    th = max(1, target_height)
    bg = preset.get("bg", "red").lower()
    text = preset.get("text", "")
    ratio_bucket = round(float(tw) / float(th), 1)
    cache_key = f"{bg}_{text}_{ratio_bucket}"

    template = _TEXT_SIGN_TEMPLATE_CACHE_V2.get(cache_key)
    if template is None:
        base_h = 300
        base_w = max(50, int(round(base_h * ratio_bucket)))
        template, _ = _render_text_sign_raw_v2(preset, target_width=base_w, target_height=base_h)
        _TEXT_SIGN_TEMPLATE_CACHE_V2[cache_key] = template

    return template.resize((tw, th), Image.Resampling.BILINEAR), text


def apply_perspective_transform(
    patch: Image.Image,
    tilt_ratios: Optional[Tuple[float, float, float, float]] = None,
    max_tilt_ratio: float = 0.04,
) -> Image.Image:
    w, h = patch.size
    if w <= 4 or h <= 4:
        return patch

    if tilt_ratios is not None:
        dx0_r, dy0_r, dx1_r, dy1_r = tilt_ratios
    else:
        dx0_r = random.uniform(-max_tilt_ratio, max_tilt_ratio)
        dy0_r = random.uniform(-max_tilt_ratio, max_tilt_ratio)
        dx1_r = random.uniform(-max_tilt_ratio, max_tilt_ratio)
        dy1_r = random.uniform(-max_tilt_ratio, max_tilt_ratio)

    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_pts = np.float32([
        [0 + dx0_r * w, 0 + dy0_r * h],
        [w - dx1_r * w, 0 + dy0_r * h],
        [w - dx1_r * w, h - dy1_r * h],
        [0 + dx0_r * w, h - dy1_r * h],
    ])

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    rgba = np.array(patch)
    warped = cv2.warpPerspective(
        rgba,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return Image.fromarray(warped)


class PoleEnvironment:
    def __init__(
        self,
        reference_crop: Optional[np.ndarray],
        main_w: int,
        main_h: int,
    ):
        self.target_mean: float = 128.0
        self.contrast_factor: float = 1.0
        self.tints: np.ndarray = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.ksize: int = 3
        self.sigma: float = 0.5
        self.noise_amp: float = 1.8
        self.tilt_ratios: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        self.ref_saturation: float = 120.0

        min_dim = min(main_w, main_h)
        if min_dim < 30:
            self.ksize = 3
            self.sigma = random.uniform(0.7, 1.1)
        elif min_dim < 70:
            self.ksize = 3
            self.sigma = random.uniform(0.4, 0.7)
        else:
            self.ksize = 3
            self.sigma = random.uniform(0.3, 0.5)
        self.noise_amp = random.uniform(1.2, 2.5)

        dx = random.uniform(-0.02, 0.02)
        dy = random.uniform(-0.02, 0.02)
        self.tilt_ratios = (dx, dy, dx, dy)

        if reference_crop is not None and reference_crop.size > 0:
            ref_gray = cv2.cvtColor(reference_crop, cv2.COLOR_BGR2GRAY)
            self.target_mean = float(np.mean(ref_gray))
            ref_std = float(np.std(ref_gray))
            self.contrast_factor = np.clip(ref_std / 45.0, 0.75, 1.25)

            ref_hsv = cv2.cvtColor(reference_crop, cv2.COLOR_BGR2HSV)
            self.ref_saturation = float(np.median(ref_hsv[:, :, 1]))

            b = reference_crop[:, :, 0].astype(np.float32)
            g = reference_crop[:, :, 1].astype(np.float32)
            r = reference_crop[:, :, 2].astype(np.float32)
            max_c = np.maximum(np.maximum(r, g), b)
            min_c = np.minimum(np.minimum(r, g), b)
            chroma = max_c - min_c

            neutral_mask = (chroma < 25.0) & (ref_gray > 40) & (ref_gray < 240)
            if np.count_nonzero(neutral_mask) > 15:
                nb = float(np.mean(b[neutral_mask]))
                ng = float(np.mean(g[neutral_mask]))
                nr = float(np.mean(r[neutral_mask]))
                ntot = max(1.0, nb + ng + nr)
                tb = (nb / ntot) * 3.0
                tg = (ng / ntot) * 3.0
                tr = (nr / ntot) * 3.0
                self.tints = np.clip(np.array([tb, tg, tr], dtype=np.float32), 0.92, 1.08)


def match_ambient_lighting_v2(
    patch_bgr: np.ndarray,
    patch_alpha: np.ndarray,
    pole_env: Optional[PoleEnvironment] = None,
    reference_crop: Optional[np.ndarray] = None,
    rng: Optional[random.Random] = None,
) -> np.ndarray:
    if pole_env is None:
        if reference_crop is None or reference_crop.size == 0:
            return patch_bgr
        pole_env = PoleEnvironment(reference_crop, patch_bgr.shape[1], patch_bgr.shape[0])

    patch_gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    mask = patch_alpha > 50
    if not np.any(mask):
        return patch_bgr

    patch_mean = float(np.mean(patch_gray[mask]))
    gain = float(np.clip(pole_env.target_mean / max(1.0, patch_mean), 0.65, 1.20))

    adjusted = patch_bgr.astype(np.float32) * gain * pole_env.tints
    adjusted = np.clip(adjusted, 0.0, 255.0).astype(np.uint8)
    adjusted = np.clip(128.0 + (adjusted.astype(np.float32) - 128.0) * pole_env.contrast_factor, 0.0, 255.0).astype(np.uint8)

    sat_scale = np.clip(pole_env.ref_saturation / 130.0, 0.22, 1.05)
    r = rng if rng is not None else random
    weathering = r.uniform(0.68, 1.0)
    total_sat_factor = float(sat_scale * weathering)

    adj_hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV).astype(np.float32)
    adj_hsv[:, :, 1] = np.clip(adj_hsv[:, :, 1] * total_sat_factor, 0.0, 255.0)
    return cv2.cvtColor(adj_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_degradations_v2(
    patch_bgr: np.ndarray,
    patch_alpha: np.ndarray,
    box_w: int,
    box_h: int,
    pole_env: Optional[PoleEnvironment] = None,
) -> np.ndarray:
    if pole_env is not None:
        ksize = pole_env.ksize
        sigma = pole_env.sigma
        noise_amp = pole_env.noise_amp
    else:
        min_dim = min(box_w, box_h)
        if min_dim < 30:
            ksize = 3
            sigma = random.uniform(0.7, 1.1)
        elif min_dim < 70:
            ksize = 3
            sigma = random.uniform(0.4, 0.7)
        else:
            ksize = 3
            sigma = random.uniform(0.3, 0.5)
        noise_amp = random.uniform(1.2, 2.5)

    blurred = cv2.GaussianBlur(patch_bgr, (ksize, ksize), sigma)
    noise = np.random.normal(0.0, noise_amp, patch_bgr.shape).astype(np.float32)
    noisy = np.clip(blurred.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
    return noisy


def composite_sign_into_image_v2(
    image_bgr: np.ndarray,
    sign_patch: Image.Image,
    target_box: Tuple[int, int, int, int],
    reference_crop: Optional[np.ndarray] = None,
    pole_env: Optional[PoleEnvironment] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = target_box
    img_h, img_w = image_bgr.shape[:2]

    x1_c = max(0, min(img_w - 1, x1))
    y1_c = max(0, min(img_h - 1, y1))
    x2_c = max(0, min(img_w, x2))
    y2_c = max(0, min(img_h, y2))

    w = x2_c - x1_c
    h = y2_c - y1_c
    if w <= 4 or h <= 4:
        return image_bgr, (x1_c, y1_c, x2_c, y2_c)

    tilt_ratios = pole_env.tilt_ratios if pole_env else None
    warped_patch = apply_perspective_transform(sign_patch, tilt_ratios=tilt_ratios)
    if (warped_patch.size[0] != w) or (warped_patch.size[1] != h):
        warped_patch = warped_patch.resize((w, h), Image.Resampling.LANCZOS)

    rgba_np = np.array(warped_patch)
    patch_rgb = rgba_np[:, :, :3]
    patch_bgr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)
    patch_alpha = rgba_np[:, :, 3].astype(np.float32) / 255.0

    lit_bgr = match_ambient_lighting_v2(patch_bgr, rgba_np[:, :, 3], pole_env=pole_env, reference_crop=reference_crop, rng=rng)
    deg_bgr = apply_degradations_v2(lit_bgr, rgba_np[:, :, 3], w, h, pole_env=pole_env)

    canvas = image_bgr.copy()
    roi = canvas[y1_c:y2_c, x1_c:x2_c]

    alpha_3d = np.repeat(patch_alpha[:, :, np.newaxis], 3, axis=2)
    blended = (deg_bgr.astype(np.float32) * alpha_3d + roi.astype(np.float32) * (1.0 - alpha_3d))
    canvas[y1_c:y2_c, x1_c:x2_c] = np.clip(blended, 0.0, 255.0).astype(np.uint8)

    return canvas, (x1_c, y1_c, x2_c, y2_c)


def check_lower_roi_occupancy(
    image_bgr: np.ndarray,
    main_box: List[float],
) -> Tuple[bool, str, Optional[Tuple[int, int, int, int]], Dict[str, Any]]:
    img_h, img_w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(c)) for c in main_box[:4]]
    main_w = x2 - x1
    main_h = y2 - y1
    main_cx = (x1 + x2) / 2.0

    roi_y1 = max(0, min(img_h - 1, y2))
    roi_y2 = min(img_h, int(round(y2 + main_h * 1.5)))
    roi_x1 = max(0, int(round(x1 - main_w * 0.35)))
    roi_x2 = min(img_w, int(round(x2 + main_w * 0.35)))

    roi_w = roi_x2 - roi_x1
    roi_h = roi_y2 - roi_y1
    roi_box = (roi_x1, roi_y1, roi_x2, roi_y2)

    if roi_w < 10 or roi_h < 10:
        return False, "out_of_bounds", roi_box, {"roi_area": float(roi_w * roi_h)}

    roi = image_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    thresh_variants = []
    _, t_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_variants.extend([t_otsu, cv2.bitwise_not(t_otsu)])

    for b_th in [110, 130, 150]:
        _, t_b = cv2.threshold(gray, b_th, 255, cv2.THRESH_BINARY)
        thresh_variants.append(t_b)

    t_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
    thresh_variants.extend([t_adapt, cv2.bitwise_not(t_adapt)])

    blue_mask = cv2.inRange(hsv, np.array([95, 70, 40]), np.array([135, 255, 255]))
    white_mask = cv2.inRange(hsv, np.array([0, 0, 130]), np.array([180, 45, 255]))
    yellow_mask = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([35, 255, 255]))
    red_mask1 = cv2.inRange(hsv, np.array([0, 80, 70]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([170, 80, 70]), np.array([180, 255, 255]))
    red_mask = red_mask1 | red_mask2
    thresh_variants.extend([blue_mask, white_mask, yellow_mask, red_mask])

    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    min_area_thresh = 0.10 * (main_w * main_h)

    candidates = []

    for thresh in thresh_variants:
        for k in [kernel_rect, kernel_horiz]:
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                c_area = cv2.contourArea(cnt)
                if c_area < min_area_thresh:
                    continue
                bx, by, bw_c, bh_c = cv2.boundingRect(cnt)
                cand_abs_x1 = roi_x1 + bx
                cand_abs_y1 = roi_y1 + by
                cand_abs_x2 = cand_abs_x1 + bw_c
                cand_abs_y2 = cand_abs_y1 + bh_c
                cand_cx = (cand_abs_x1 + cand_abs_x2) / 2.0
                gap = cand_abs_y1 - y2

                valid_gap = -2 <= gap <= 0.30 * main_h
                valid_cx = abs(cand_cx - main_cx) <= 0.25 * main_w
                valid_w = 0.45 * main_w <= bw_c <= 1.40 * main_w
                valid_h = 0.20 * main_h <= bh_c <= 1.20 * main_h

                if not (valid_gap and valid_cx and valid_w and valid_h):
                    continue

                rect_ratio = c_area / float(bw_c * bh_c)
                if rect_ratio < 0.50:
                    continue

                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                num_vertices = len(approx)

                vert_score = 1.0 if num_vertices in [4, 5, 6] else 0.7
                rect_score = min(1.0, rect_ratio / 0.85)
                gap_score = max(0.0, 1.0 - (max(0, gap) / (0.30 * main_h)))
                cx_score = max(0.0, 1.0 - (abs(cand_cx - main_cx) / (0.25 * main_w)))
                geom_score = vert_score * 0.3 + rect_score * 0.3 + gap_score * 0.2 + cx_score * 0.2

                cand_patch = image_bgr[cand_abs_y1:cand_abs_y2, cand_abs_x1:cand_abs_x2]
                if cand_patch.size == 0:
                    continue
                c_hsv = cv2.cvtColor(cand_patch, cv2.COLOR_BGR2HSV)
                p_blue = np.count_nonzero(cv2.inRange(c_hsv, np.array([95, 70, 40]), np.array([135, 255, 255]))) / cand_patch.size
                p_white = np.count_nonzero(cv2.inRange(c_hsv, np.array([0, 0, 120]), np.array([180, 45, 255]))) / cand_patch.size
                p_yellow = np.count_nonzero(cv2.inRange(c_hsv, np.array([15, 80, 80]), np.array([35, 255, 255]))) / cand_patch.size
                max_sign_color = max(p_blue, p_white, p_yellow)
                app_score = min(1.0, max_sign_color / 0.35)

                c_gray = cv2.cvtColor(cand_patch, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(c_gray, 50, 150)
                edge_dens = float(np.count_nonzero(edges)) / float(edges.size)
                if 0.02 <= edge_dens <= 0.35:
                    sym_score = 0.9
                elif edge_dens < 0.02:
                    sym_score = 0.4
                else:
                    sym_score = 0.3

                panel_score = geom_score * 0.60 + app_score * 0.25 + sym_score * 0.15
                candidates.append({
                    "score": panel_score,
                    "cand_box": (cand_abs_x1, cand_abs_y1, cand_abs_x2, cand_abs_y2),
                    "metrics": {
                        "panel_score": round(panel_score, 3),
                        "geom_score": round(geom_score, 3),
                        "app_score": round(app_score, 3),
                        "sym_score": round(sym_score, 3),
                        "rect_ratio": round(rect_ratio, 3),
                        "vertices": num_vertices,
                        "bw": bw_c,
                        "bh": bh_c,
                        "gap": gap,
                    }
                })

    if candidates:
        best = max(candidates, key=lambda c: c["score"])
        if best["score"] >= 0.50:
            return False, "geometry_real_panel", best["cand_box"], best["metrics"]

    return True, "clean", roi_box, {}


def check_patch_suspicion(
    canvas_bgr: np.ndarray,
    target_box: Tuple[int, int, int, int],
    detected_panel_box: Optional[Tuple[int, int, int, int]] = None,
) -> bool:
    tx1, ty1, tx2, ty2 = target_box
    th, tw = ty2 - ty1, tx2 - tx1
    if tw <= 6 or th <= 6:
        return True
    if detected_panel_box:
        px1, py1, px2, py2 = detected_panel_box
        ix1 = max(tx1, px1)
        iy1 = max(ty1, py1)
        ix2 = min(tx2, px2)
        iy2 = min(ty2, py2)
        if ix2 > ix1 and iy2 > iy1:
            return True
    patch = canvas_bgr[ty1:ty2, tx1:tx2]
    if patch.size == 0:
        return True
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    blue_m = cv2.inRange(hsv, np.array([95, 70, 40]), np.array([135, 255, 255]))
    yellow_m = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([35, 255, 255]))
    red_m1 = cv2.inRange(hsv, np.array([0, 80, 70]), np.array([10, 255, 255]))
    red_m2 = cv2.inRange(hsv, np.array([170, 80, 70]), np.array([180, 255, 255]))
    saturated_color_ratio = float(np.count_nonzero(blue_m | yellow_m | red_m1 | red_m2)) / float(tw * th)
    if saturated_color_ratio >= 0.45:
        return True
    return False


def has_other_main_sign_below(
    current_box: List[float],
    all_boxes: List[List[float]],
    current_idx: int,
) -> bool:
    x1, y1, x2, y2 = current_box[:4]
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    for j, o_box in enumerate(all_boxes):
        if j == current_idx:
            continue
        ox1, oy1, ox2, oy2 = o_box[:4]
        ow = max(1.0, float(ox2 - ox1))
        oh = max(1.0, float(oy2 - oy1))
        ocx = (ox1 + ox2) / 2.0
        ocy = (oy1 + oy2) / 2.0

        if not (0.35 <= (w / ow) <= 2.85):
            continue

        if abs(cx - ocx) > 0.70 * max(w, ow):
            continue

        horiz_overlap = min(x2, ox2) - max(x1, ox1)
        if horiz_overlap < -0.20 * min(w, ow):
            continue

        if ocy - cy < 0.25 * min(h, oh):
            continue

        if oy2 <= y2 + 0.15 * min(h, oh):
            continue

        vert_gap = oy1 - y2
        if -0.30 * h <= vert_gap <= 1.50 * max(h, oh):
            return True

    return False


def is_target_colliding_with_main_signs(
    target_box: Tuple[int, int, int, int],
    all_boxes: List[List[float]],
    current_idx: int,
    overlap_thresh: float = 0.45,
) -> bool:
    tx1, ty1, tx2, ty2 = target_box
    t_area = float((tx2 - tx1) * (ty2 - ty1))
    if t_area <= 0:
        return True

    for j, o_box in enumerate(all_boxes):
        if j == current_idx:
            continue
        ox1, oy1, ox2, oy2 = [int(round(c)) for c in o_box[:4]]
        ix1 = max(tx1, ox1)
        iy1 = max(ty1, oy1)
        ix2 = min(tx2, ox2)
        iy2 = min(ty2, oy2)
        if ix2 > ix1 and iy2 > iy1:
            inter_area = float((ix2 - ix1) * (iy2 - iy1))
            if (inter_area / t_area) > overlap_thresh:
                return True

    return False


class SupplementaryAugmentorV2:
    def __init__(
        self,
        classifier: Optional[SignClassifier] = None,
        sampling_probs: Optional[Dict[str, float]] = None,
        supp_class_id: int = 100,
        text_class_id: int = 101,
        preserve_main_class_id: bool = True,
        seed: int = 42,
    ):
        self.classifier = classifier
        self.sampling_probs = sampling_probs or {
            "none": 0.25,
            "supplementary_1": 0.35,
            "supplementary_2": 0.10,
            "text_sign": 0.30,
        }
        self.supp_class_id = supp_class_id
        self.text_class_id = text_class_id
        self.preserve_main_class_id = preserve_main_class_id
        self.rng = random.Random(seed)
        self.last_rejected: List[Dict[str, Any]] = []

    def _sample_mode(self, forced_mode: Optional[str] = None) -> str:
        if forced_mode:
            return forced_mode
        modes = list(self.sampling_probs.keys())
        probs = [self.sampling_probs[m] for m in modes]
        return self.rng.choices(modes, weights=probs, k=1)[0]

    def _get_sign_metadata_for_crop(
        self,
        crop: np.ndarray,
        known_class_id: Optional[int] = None,
    ) -> Tuple[Optional[int], str, Optional[str], Optional[str]]:
        if known_class_id is not None:
            class_id = known_class_id
            meta = class_manager.get_sign_metadata(class_id)
            if meta:
                standard_code = meta.get("standard_code")
                category = meta.get("category", "PROHIBITORY")
                class_name = meta.get("name_vi") or meta.get("standard_code", "main_sign")
                return class_id, class_name, standard_code, category

        if self.classifier is None:
            return None, "main_sign", None, "PROHIBITORY"

        clf_res = self.classifier.classify_crop(crop)
        top1 = clf_res.get("top1") or clf_res
        class_id = top1.get("class_id")
        class_name = top1.get("class_name", "main_sign")

        standard_code = None
        category = "PROHIBITORY"
        if class_id is not None:
            meta = class_manager.get_sign_metadata(class_id)
            if meta:
                standard_code = meta.get("standard_code")
                category = meta.get("category", "PROHIBITORY")

        return class_id, class_name, standard_code, category

    def augment_image(
        self,
        image_bgr: np.ndarray,
        main_boxes: List[List[float]],
        forced_mode: Optional[str] = None,
        forced_orientation: Optional[str] = None,
        known_class_ids: Optional[List[Optional[int]]] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        if image_bgr is None or image_bgr.size == 0 or not main_boxes:
            return image_bgr, []

        self.last_rejected = []
        canvas = image_bgr.copy()
        img_h, img_w = canvas.shape[:2]
        output_labels: List[Dict[str, Any]] = []

        for i, main_box in enumerate(main_boxes):
            logger.trace(f"Processing main box {i+1}/{len(main_boxes)}: {main_box}")
            x1, y1, x2, y2 = [int(round(coord)) for coord in main_box[:4]]
            crop = SignClassifier.crop_box(canvas, [x1, y1, x2, y2], padding_ratio=0.05)

            known_cid = known_class_ids[i] if known_class_ids and i < len(known_class_ids) else None
            class_id, class_name, standard_code, category = self._get_sign_metadata_for_crop(crop, known_class_id=known_cid)

            output_labels.append({
                "class": "main_sign",
                "class_id": class_id if (self.preserve_main_class_id and class_id is not None) else 0,
                "bbox": [x1, y1, x2, y2],
                "standard_code": standard_code,
                "class_name": class_name,
            })

            mode = self._sample_mode(forced_mode=forced_mode)
            if mode == "none":
                continue

            if has_other_main_sign_below([x1, y1, x2, y2], main_boxes, i):
                logger.trace(f"Anchor {[x1, y1, x2, y2]} skipped: has other main sign below on same pole.")
                self.last_rejected.append({
                    "main_box": [x1, y1, x2, y2],
                    "standard_code": standard_code,
                    "class_name": class_name,
                    "reason": "has_main_sign_below",
                })
                continue

            is_clean, reject_reason, roi_box, metrics = check_lower_roi_occupancy(canvas, [x1, y1, x2, y2])
            if not is_clean:
                logger.debug(f"Anchor {[x1, y1, x2, y2]} rejected: {reject_reason} | metrics={metrics}")
                self.last_rejected.append({
                    "main_box": [x1, y1, x2, y2],
                    "standard_code": standard_code,
                    "class_name": class_name,
                    "roi_box": list(roi_box) if roi_box else None,
                    "reason": reject_reason,
                    "metrics": metrics,
                })
                continue

            px_per_mm = estimate_pixel_scale([x1, y1, x2, y2], standard_code=standard_code)
            pole_env = PoleEnvironment(
                reference_crop=crop,
                main_w=x2 - x1,
                main_h=y2 - y1,
            )
            used_symbols: List[str] = []

            if mode in ["supplementary_1", "supplementary_2"]:
                count = 1 if mode == "supplementary_1" else 2
                supp_types = get_compatible_supplementary_types(
                    standard_code=standard_code,
                    category=category,
                    count=count,
                )

                parent_box = [x1, y1, x2, y2]
                for s_idx, supp_type in enumerate(supp_types):
                    supp_meta = SUPPLEMENTARY_METADATA.get(
                        supp_type,
                        {"width_mm": 500.0, "height_mm": 300.0, "name": supp_type},
                    )

                    target_box = compute_supplementary_placement(
                        main_box=[x1, y1, x2, y2],
                        supp_meta=supp_meta,
                        px_per_mm=px_per_mm,
                        parent_box=parent_box,
                        image_shape=(img_h, img_w),
                    )

                    tb_w = target_box[2] - target_box[0]
                    tb_h = target_box[3] - target_box[1]

                    if target_box[3] > img_h + 10 or tb_w <= 4 or tb_h <= 4:
                        break

                    if is_target_colliding_with_main_signs(target_box, main_boxes, i, overlap_thresh=0.45):
                        logger.debug(f"Target box {target_box} collides with another main sign. Skipping paste.")
                        self.last_rejected.append({
                            "main_box": [x1, y1, x2, y2],
                            "standard_code": standard_code,
                            "class_name": class_name,
                            "roi_box": list(target_box),
                            "reason": "collision_with_other_main_sign",
                            "metrics": {},
                        })
                        break

                    if check_patch_suspicion(canvas, target_box):
                        logger.debug(f"Target box {target_box} suspicious. Skipping paste.")
                        self.last_rejected.append({
                            "main_box": [x1, y1, x2, y2],
                            "standard_code": standard_code,
                            "class_name": class_name,
                            "roi_box": list(target_box),
                            "reason": "suspicious_target_region",
                            "metrics": {},
                        })
                        break

                    sign_img = render_qcvn_supplementary_sign(
                        subclass=supp_type,
                        target_width=tb_w,
                        target_height=tb_h,
                        used_symbols=used_symbols,
                    )

                    canvas, final_box = composite_sign_into_image_v2(
                        image_bgr=canvas,
                        sign_patch=sign_img,
                        target_box=target_box,
                        reference_crop=crop,
                        pole_env=pole_env,
                        rng=self.rng,
                    )

                    output_labels.append({
                        "class": "supplementary_sign",
                        "class_id": self.supp_class_id,
                        "bbox": list(final_box),
                        "subclass": supp_type,
                        "parent_box": [x1, y1, x2, y2],
                        "parent_code": standard_code,
                    })

                    parent_box = list(final_box)

            elif mode == "text_sign":
                target_box, (w_mm, h_mm), orientation = compute_text_sign_placement_v2(
                    main_box=[x1, y1, x2, y2],
                    px_per_mm=px_per_mm,
                    orientation=forced_orientation,
                    rng=self.rng,
                    image_shape=(img_h, img_w),
                )

                tb_w = target_box[2] - target_box[0]
                tb_h = target_box[3] - target_box[1]

                if target_box[3] <= img_h + 10 and tb_w > 4 and tb_h > 4:
                    if is_target_colliding_with_main_signs(target_box, main_boxes, i, overlap_thresh=0.45):
                        logger.debug(f"Target box {target_box} collides with another main sign. Skipping paste.")
                        self.last_rejected.append({
                            "main_box": [x1, y1, x2, y2],
                            "standard_code": standard_code,
                            "class_name": class_name,
                            "roi_box": list(target_box),
                            "reason": "collision_with_other_main_sign",
                            "metrics": {},
                        })
                        continue

                    if check_patch_suspicion(canvas, target_box):
                        logger.debug(f"Target box {target_box} suspicious. Skipping paste.")
                        self.last_rejected.append({
                            "main_box": [x1, y1, x2, y2],
                            "standard_code": standard_code,
                            "class_name": class_name,
                            "roi_box": list(target_box),
                            "reason": "suspicious_target_region",
                            "metrics": {},
                        })
                        continue

                    preset = get_random_text_sign_preset_v2(
                        category=category,
                        orientation=orientation,
                        rng=self.rng,
                    )

                    sign_img, text_str = render_text_sign_v2(
                        preset=preset,
                        target_width=tb_w,
                        target_height=tb_h,
                    )

                    canvas, final_box = composite_sign_into_image_v2(
                        image_bgr=canvas,
                        sign_patch=sign_img,
                        target_box=target_box,
                        reference_crop=crop,
                        pole_env=pole_env,
                        rng=self.rng,
                    )

                    output_labels.append({
                        "class": "text_sign",
                        "class_id": self.text_class_id,
                        "bbox": list(final_box),
                        "text": text_str,
                        "bg": preset.get("bg", "red"),
                        "orientation": orientation,
                        "aspect_ratio": round(float(tb_w) / max(1.0, float(tb_h)), 3),
                        "parent_box": [x1, y1, x2, y2],
                        "parent_code": standard_code,
                    })

        logger.debug(f"Augmented image: generated {len(output_labels) - len(main_boxes)} synthetic signs.")
        return canvas, output_labels
