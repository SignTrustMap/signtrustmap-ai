import gc
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from loguru import logger
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

TEXT_SIGN_PRESETS: Dict[str, List[Dict[str, str]]] = {
    "PROHIBITORY": [
        {"bg": "red", "fg": "white", "text": "CẤM XE TẢI\n6:00 - 22:00"},
        {"bg": "red", "fg": "white", "text": "CẤM ĐỖ XE\nNGÀY CHẴN"},
        {"bg": "red", "fg": "white", "text": "CẤM XE 2 VÀ 3 BÁNH\n9:00 - 16:00\n20:00 - 6:00"},
        {"bg": "red", "fg": "white", "text": "CẤM Ô TÔ KHÁCH\nTRÊN 16 CHỖ"},
        {"bg": "red", "fg": "white", "text": "CẤM RẼ TRÁI\n6:00 - 9:00\n16:30 - 19:30"},
        {"bg": "red", "fg": "white", "text": "CẤM DỪNG ĐỖ XE\nQUÁ 5 PHÚT"},
    ],
    "WARNING": [
        {"bg": "yellow", "fg": "black", "text": "CHÚ Ý\nĐOẠN ĐƯỜNG HAY TAI NẠN"},
        {"bg": "yellow", "fg": "black", "text": "CÔNG TRƯỜNG\nĐANG THI CÔNG"},
        {"bg": "yellow", "fg": "black", "text": "ĐOẠN ĐƯỜNG\nTHƯỜNG XUYÊN ĐÔNG XE"},
        {"bg": "yellow", "fg": "black", "text": "CHÚ Ý GIẢM TỐC ĐỘ\nKHU VỰC TRƯỜNG HỌC"},
    ],
    "MANDATORY": [
        {"bg": "red", "fg": "white", "text": "CẤM ĐI THẲNG\nXE TẢI ĐI ĐƯỜNG TRÁNH"},
        {"bg": "blue", "fg": "white", "text": "XE MÁY ĐI CHẬM\nNHƯỜNG ĐƯỜNG NGƯỜI ĐI BỘ"},
    ],
    "INFORMATION": [
        {"bg": "blue", "fg": "white", "text": "KHU VỰC ĐỖ XE\nÔ TÔ VÀ XE MÁY"},
        {"bg": "blue", "fg": "white", "text": "LỐI ĐI BỘ\nSANG ĐƯỜNG"},
        {"bg": "blue", "fg": "white", "text": "HƯỚNG ĐI TP. HỒ CHÍ MINH"},
        {"bg": "blue", "fg": "white", "text": "BÃI ĐỖ XE BUÝT"},
    ],
}

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]

REF_SYMBOLS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "symbol" / "ref_symbols"


def _get_font(size: int) -> ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
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
) -> Tuple[ImageFont.ImageFont, int, float, float]:
    font_size = max(min_font_size, initial_font_size)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        lines = [text]

    while font_size > min_font_size:
        font = _get_font(font_size)
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

    font = _get_font(font_size)
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
    for _ in range(count):
        pick = random.choices(candidates, weights=weights, k=1)[0]
        chosen.append(pick)
    return chosen


def get_random_text_sign_preset(category: Optional[str]) -> Dict[str, str]:
    cat = (category or "PROHIBITORY").upper()
    presets = TEXT_SIGN_PRESETS.get(cat, TEXT_SIGN_PRESETS["PROHIBITORY"])
    return random.choice(presets)


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


def compute_text_sign_placement(
    main_box: List[float],
    px_per_mm: float,
    width_mm: float = 600.0,
    height_mm: float = 350.0,
    image_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int, int, int]:
    meta = {"width_mm": width_mm, "height_mm": height_mm}
    return compute_supplementary_placement(
        main_box=main_box,
        supp_meta=meta,
        px_per_mm=px_per_mm,
        parent_box=None,
        image_shape=image_shape,
    )


def render_qcvn_supplementary_sign(
    subclass: str,
    target_width: int,
    target_height: int,
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
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        draw.text((cx - tw / 2.0, cy - th / 2.0 - bbox[1]), text, font=font, fill=fg_color)

        arrow_h = h * 0.6
        arrow_size = min(w, h) * 0.16
        arrow_thick = max(2, int(border_w * 0.7))
        arrow_y1 = cy - arrow_h / 2.0
        arrow_y2 = cy + arrow_h / 2.0

        _draw_vertical_arrow(draw, w * 0.16, arrow_y1, arrow_y2, "up", arrow_thick, arrow_size, fg_color)
        _draw_vertical_arrow(draw, w * 0.84, arrow_y1, arrow_y2, "up", arrow_thick, arrow_size, fg_color)

    elif subclass == "S.502":
        distances = ["50 m", "100 m", "150 m", "200 m", "300 m", "500 m"]
        text = random.choice(distances)
        max_text_w = max(10.0, w - 2 * (margin + border_w + 6))
        max_text_h = max(10.0, h * 0.70)
        font, font_size, tw, th = _fit_text_font(text, max_text_w, max_text_h, int(h * 0.50), draw)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((cx - tw / 2.0, cy - th / 2.0 - bbox[1]), text, font=font, fill=fg_color)

    elif subclass == "S.503a":
        arr_w = w * 0.75
        arr_thick = max(2, int(border_w * 1.0))
        arr_size = min(w, h) * 0.25
        _draw_horizontal_arrow(draw, cy, cx - arr_w / 2.0, cx + arr_w / 2.0, "left_right", arr_thick, arr_size, fg_color)

    elif subclass == "S.503b":
        arr_w = w * 0.70
        arr_thick = max(2, int(border_w * 1.0))
        arr_size = min(w, h) * 0.25
        _draw_horizontal_arrow(draw, cy, cx - arr_w / 2.0, cx + arr_w / 2.0, "right", arr_thick, arr_size, fg_color)

    elif subclass == "S.503c":
        arr_w = w * 0.70
        arr_thick = max(2, int(border_w * 1.0))
        arr_size = min(w, h) * 0.25
        _draw_horizontal_arrow(draw, cy, cx - arr_w / 2.0, cx + arr_w / 2.0, "left", arr_thick, arr_size, fg_color)

    elif subclass == "S.503d":
        arr_h = h * 0.70
        arr_thick = max(2, int(border_w * 1.0))
        arr_size = min(w, h) * 0.25
        _draw_vertical_arrow(draw, cx, cy - arr_h / 2.0, cy + arr_h / 2.0, "up", arr_thick, arr_size, fg_color)

    elif subclass == "S.503e":
        arr_h = h * 0.70
        arr_thick = max(2, int(border_w * 1.0))
        arr_size = min(w, h) * 0.25
        _draw_vertical_arrow(draw, cx, cy - arr_h / 2.0, cy + arr_h / 2.0, "up_down", arr_thick, arr_size, fg_color)

    elif subclass == "S.503f":
        arr_h = h * 0.70
        arr_thick = max(2, int(border_w * 1.0))
        arr_size = min(w, h) * 0.25
        _draw_vertical_arrow(draw, cx, cy - arr_h / 2.0, cy + arr_h / 2.0, "down", arr_thick, arr_size, fg_color)

    elif subclass == "S.504":
        arr_h = h * 0.65
        arr_thick = max(3, int(border_w * 1.2))
        arr_size = min(w, h) * 0.32
        _draw_vertical_arrow(draw, cx, cy - arr_h / 2.0, cy + arr_h / 2.0, "down", arr_thick, arr_size, fg_color)

    elif subclass in ["S.505a", "S.505b", "S.505c"]:
        symbol_files = list(REF_SYMBOLS_DIR.glob("*.webp")) if REF_SYMBOLS_DIR.exists() else []
        placed_icon = False
        if symbol_files:
            sym_file = random.choice(symbol_files)
            try:
                sym_img = Image.open(sym_file).convert("RGBA")
                sym_w, sym_h = sym_img.size
                avail_w = int(w * 0.75)
                avail_h = int(h * 0.70)
                aspect = sym_w / max(1, sym_h)
                if aspect > (avail_w / avail_h):
                    nw = avail_w
                    nh = int(nw / aspect)
                else:
                    nh = avail_h
                    nw = int(nh * aspect)
                sym_resized = sym_img.resize((nw, nh), Image.Resampling.BILINEAR)
                pos_x = int(cx - nw / 2.0)
                pos_y = int(cy - nh / 2.0)
                img.alpha_composite(sym_resized, (pos_x, pos_y))
                draw = ImageDraw.Draw(img)
                placed_icon = True
            except Exception as e:
                logger.trace(f"Failed loading symbol {sym_file}: {e}")

        if not placed_icon:
            text = "XE TẢI"
            max_text_w = max(10.0, w - 2 * (margin + border_w + 6))
            max_text_h = max(10.0, h * 0.70)
            font, font_size, tw, th = _fit_text_font(text, max_text_w, max_text_h, int(h * 0.45), draw)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((cx - tw / 2.0, cy - th / 2.0 - bbox[1]), text, font=font, fill=fg_color)

    elif subclass in ["S.506a", "S.506b"]:
        main_thick = max(6, int(min(w, h) * 0.18))
        minor_thick = max(2, int(min(w, h) * 0.05))

        draw.line([(cx, margin + border_w), (cx, h - margin - border_w)], fill=fg_color, width=main_thick)
        draw.line([(margin + border_w, cy), (w - margin - border_w, cy)], fill=fg_color, width=minor_thick)

    elif subclass == "S.508a":
        time_choices = [
            "6:00 - 9:00",
            "16:00 - 20:00",
            "6h00 - 18h00",
            "7:00 - 8:30",
            "16:30 - 19:00",
            "22:00 - 5:00",
        ]
        text = random.choice(time_choices)
        max_text_w = max(10.0, w - 2 * (margin + border_w + 6))
        max_text_h = max(10.0, h * 0.70)
        font, font_size, tw, th = _fit_text_font(text, max_text_w, max_text_h, int(h * 0.42), draw)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((cx - tw / 2.0, cy - th / 2.0 - bbox[1]), text, font=font, fill=fg_color)

    elif subclass == "S.508b":
        day_choices = [
            "THỨ 2 - THỨ 6",
            "NGÀY CHẴN",
            "NGÀY LẺ",
            "T7, CHỦ NHẬT",
            "T2 - T7",
        ]
        text = random.choice(day_choices)
        max_text_w = max(10.0, w - 2 * (margin + border_w + 6))
        max_text_h = max(10.0, h * 0.70)
        font, font_size, tw, th = _fit_text_font(text, max_text_w, max_text_h, int(h * 0.40), draw)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((cx - tw / 2.0, cy - th / 2.0 - bbox[1]), text, font=font, fill=fg_color)

    elif subclass in ["S.509a", "S.509b"]:
        text = "CẤM ĐỖ XE\nNGÀY LỄ"
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        max_text_w = max(10.0, w - 2 * (margin + border_w + 6))
        max_text_h = max(10.0, h - 2 * (margin + border_w + 6))
        font, font_size, _, total_th = _fit_text_font(text, max_text_w, max_text_h, int(h * 0.32), draw)
        cur_y = cy - total_th / 2.0
        line_spacing = font_size * 1.25
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            tw = bbox[2] - bbox[0]
            draw.text((cx - tw / 2.0, cur_y - bbox[1]), line, font=font, fill=fg_color)
            cur_y += line_spacing

    else:
        text = subclass
        max_text_w = max(10.0, w - 2 * (margin + border_w + 6))
        max_text_h = max(10.0, h * 0.70)
        font, font_size, tw, th = _fit_text_font(text, max_text_w, max_text_h, int(h * 0.40), draw)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((cx - tw / 2.0, cy - th / 2.0 - bbox[1]), text, font=font, fill=fg_color)

    final_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return final_img


def render_text_sign(
    preset: Dict[str, str],
    target_width: int,
    target_height: int,
) -> Tuple[Image.Image, str]:
    scale = 2
    w = max(50, target_width * scale)
    h = max(30, target_height * scale)

    bg_type = preset.get("bg", "red").lower()
    text = preset.get("text", "CẤM XE TẢI\n6:00 - 22:00")

    if bg_type == "red":
        bg_color = (210, 28, 28)
        border_color = (250, 250, 250)
        fg_color = (255, 255, 255)
    elif bg_type == "yellow":
        bg_color = (248, 196, 20)
        border_color = (20, 20, 20)
        fg_color = (15, 15, 15)
    elif bg_type == "blue":
        bg_color = (16, 92, 172)
        border_color = (250, 250, 250)
        fg_color = (255, 255, 255)
    else:
        bg_color = (248, 248, 248)
        border_color = (20, 20, 20)
        fg_color = (20, 20, 20)

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin = max(2, int(h * 0.04))
    border_w = max(2, int(min(w, h) * 0.045))
    corner_r = max(4, int(min(w, h) * 0.06))

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

    max_text_w = max(10.0, w - 2 * (margin + border_w + 8))
    max_text_h = max(10.0, h - 2 * (margin + border_w + 8))
    init_fs = max(10, int(h / (num_lines + 0.8)))
    font, font_size, _, total_text_h = _fit_text_font(text, max_text_w, max_text_h, init_fs, draw)

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


def apply_perspective_transform(
    patch: Image.Image,
    max_tilt_ratio: float = 0.04,
) -> Image.Image:
    w, h = patch.size
    dx_top = random.uniform(-max_tilt_ratio, max_tilt_ratio) * w
    dx_bot = random.uniform(-max_tilt_ratio, max_tilt_ratio) * w
    dy_left = random.uniform(-max_tilt_ratio, max_tilt_ratio) * h
    dy_right = random.uniform(-max_tilt_ratio, max_tilt_ratio) * h

    src_pts = np.float32([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h],
    ])
    dst_pts = np.float32([
        [max(0.0, dx_top), max(0.0, dy_left)],
        [min(float(w), w - max(0.0, -dx_top)), max(0.0, dy_right)],
        [min(float(w), w - max(0.0, -dx_bot)), min(float(h), h - max(0.0, -dy_right))],
        [max(0.0, dx_bot), min(float(h), h - max(0.0, -dy_left))],
    ])

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    patch_np = np.array(patch)
    warped_np = cv2.warpPerspective(
        patch_np,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return Image.fromarray(warped_np)


def match_ambient_lighting(
    patch_bgr: np.ndarray,
    patch_alpha: np.ndarray,
    reference_crop: Optional[np.ndarray] = None,
) -> np.ndarray:
    if reference_crop is None or reference_crop.size == 0:
        return patch_bgr

    ref_gray = cv2.cvtColor(reference_crop, cv2.COLOR_BGR2GRAY)
    ref_mean = float(np.mean(ref_gray))

    patch_gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    mask = patch_alpha > 50
    if not np.any(mask):
        return patch_bgr

    patch_mean = float(np.mean(patch_gray[mask]))

    target_mean = np.clip(ref_mean * 1.05, 90.0, 240.0)
    gain = np.clip(target_mean / max(1.0, patch_mean), 0.65, 1.15)

    ref_b = float(np.mean(reference_crop[:, :, 0]))
    ref_g = float(np.mean(reference_crop[:, :, 1]))
    ref_r = float(np.mean(reference_crop[:, :, 2]))
    ref_total = max(1.0, ref_b + ref_g + ref_r)
    tint_b = (ref_b / ref_total) * 3.0
    tint_g = (ref_g / ref_total) * 3.0
    tint_r = (ref_r / ref_total) * 3.0

    tints = np.array([tint_b, tint_g, tint_r], dtype=np.float32)
    tints = np.clip(tints, 0.85, 1.15)

    adjusted = patch_bgr.astype(np.float32) * gain * tints
    adjusted = np.clip(adjusted, 0.0, 255.0).astype(np.uint8)

    contrast_factor = random.uniform(0.90, 1.05)
    adjusted = np.clip(128.0 + (adjusted.astype(np.float32) - 128.0) * contrast_factor, 0.0, 255.0).astype(np.uint8)

    return adjusted


def apply_degradations(
    patch_bgr: np.ndarray,
    patch_alpha: np.ndarray,
    box_w: int,
    box_h: int,
) -> np.ndarray:
    min_dim = min(box_w, box_h)
    if min_dim < 30:
        ksize = 3
        sigma = random.uniform(0.8, 1.3)
    elif min_dim < 70:
        ksize = 3
        sigma = random.uniform(0.5, 0.9)
    else:
        ksize = 3
        sigma = random.uniform(0.3, 0.6)

    blurred = cv2.GaussianBlur(patch_bgr, (ksize, ksize), sigma)

    noise_amp = random.uniform(1.5, 4.0)
    noise = np.random.normal(0.0, noise_amp, patch_bgr.shape).astype(np.float32)
    noisy = np.clip(blurred.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)

    return noisy


def composite_sign_into_image(
    image_bgr: np.ndarray,
    sign_patch: Image.Image,
    target_box: Tuple[int, int, int, int],
    reference_crop: Optional[np.ndarray] = None,
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

    warped_patch = apply_perspective_transform(sign_patch, max_tilt_ratio=0.03)
    if (warped_patch.size[0] != w) or (warped_patch.size[1] != h):
        warped_patch = warped_patch.resize((w, h), Image.Resampling.LANCZOS)

    rgba_np = np.array(warped_patch)
    patch_rgb = rgba_np[:, :, :3]
    patch_bgr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)
    patch_alpha = rgba_np[:, :, 3].astype(np.float32) / 255.0

    lit_bgr = match_ambient_lighting(patch_bgr, rgba_np[:, :, 3], reference_crop)
    deg_bgr = apply_degradations(lit_bgr, rgba_np[:, :, 3], w, h)

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
    if saturated_color_ratio >= 0.30:
        return True
    return False


class SupplementaryAugmentor:
    def __init__(
        self,
        classifier: Optional[SignClassifier] = None,
        sampling_probs: Optional[Dict[str, float]] = None,
    ):
        self.classifier = classifier
        self.sampling_probs = sampling_probs or {
            "none": 0.50,
            "supplementary_1": 0.35,
            "supplementary_2": 0.10,
            "text_sign": 0.05,
        }
        self.last_rejected: List[Dict[str, Any]] = []

    def _sample_mode(self, forced_mode: Optional[str] = None) -> str:
        if forced_mode:
            return forced_mode
        modes = list(self.sampling_probs.keys())
        probs = [self.sampling_probs[m] for m in modes]
        return random.choices(modes, weights=probs, k=1)[0]

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
        class_id = clf_res.get("class_id")
        class_name = clf_res.get("class_name", "main_sign")

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
                "class_id": 0,
                "bbox": [x1, y1, x2, y2],
                "standard_code": standard_code,
                "class_name": class_name,
            })

            mode = self._sample_mode(forced_mode=forced_mode)
            if mode == "none":
                continue

            is_clean, reject_reason, roi_box, metrics = check_lower_roi_occupancy(canvas, [x1, y1, x2, y2])
            if not is_clean:
                logger.debug(f"Anchor {[x1, y1, x2, y2]} rejected: {reject_reason} | metrics={metrics}")
                self.last_rejected.append({
                    "main_box": [x1, y1, x2, y2],
                    "standard_code": standard_code,
                    "class_name": class_name,
                    "roi_box": list(roi_box),
                    "reason": reject_reason,
                    "metrics": metrics,
                })
                continue

            px_per_mm = estimate_pixel_scale([x1, y1, x2, y2], standard_code=standard_code)

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
                    )

                    canvas, final_box = composite_sign_into_image(
                        image_bgr=canvas,
                        sign_patch=sign_img,
                        target_box=target_box,
                        reference_crop=crop,
                    )

                    output_labels.append({
                        "class": "supplementary_sign",
                        "class_id": 1,
                        "bbox": list(final_box),
                        "subclass": supp_type,
                        "parent_box": [x1, y1, x2, y2],
                        "parent_code": standard_code,
                    })

                    parent_box = list(final_box)

            elif mode == "text_sign":
                preset = get_random_text_sign_preset(category=category)
                target_box = compute_text_sign_placement(
                    main_box=[x1, y1, x2, y2],
                    px_per_mm=px_per_mm,
                    width_mm=600.0,
                    height_mm=350.0,
                    image_shape=(img_h, img_w),
                )

                tb_w = target_box[2] - target_box[0]
                tb_h = target_box[3] - target_box[1]

                if target_box[3] <= img_h + 10 and tb_w > 4 and tb_h > 4:
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

                    sign_img, text_str = render_text_sign(
                        preset=preset,
                        target_width=tb_w,
                        target_height=tb_h,
                    )

                    canvas, final_box = composite_sign_into_image(
                        image_bgr=canvas,
                        sign_patch=sign_img,
                        target_box=target_box,
                        reference_crop=crop,
                    )

                    output_labels.append({
                        "class": "text_sign",
                        "class_id": 2,
                        "bbox": list(final_box),
                        "text": text_str,
                        "bg": preset.get("bg", "red"),
                        "parent_box": [x1, y1, x2, y2],
                        "parent_code": standard_code,
                    })

        logger.debug(f"Augmented image: generated {len(output_labels) - len(main_boxes)} synthetic signs.")
        return canvas, output_labels

    def to_yolo_annotations(
        self,
        labels: List[Dict[str, Any]],
        image_shape: Tuple[int, int],
    ) -> List[Tuple[int, float, float, float, float]]:
        img_h, img_w = image_shape[:2]
        yolo_entries: List[Tuple[int, float, float, float, float]] = []

        for item in labels:
            cid = item.get("class_id", 0)
            x1, y1, x2, y2 = item["bbox"]
            box_w = max(0.0, float(x2 - x1))
            box_h = max(0.0, float(y2 - y1))
            cx = float(x1) + box_w / 2.0
            cy = float(y1) + box_h / 2.0

            norm_cx = round(cx / img_w, 6)
            norm_cy = round(cy / img_h, 6)
            norm_w = round(box_w / img_w, 6)
            norm_h = round(box_h / img_h, 6)
            yolo_entries.append((cid, norm_cx, norm_cy, norm_w, norm_h))

        return yolo_entries
