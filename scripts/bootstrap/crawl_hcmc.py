#!/usr/bin/env python3
"""
Bootstrap traffic sign crawler for Ho Chi Minh City (excluding Phường Tăng Nhơn Phú) using Mapillary Graph API v4.
Ward boundary for Phường Tăng Nhơn Phú is sourced directly from VNSDI
(https://vnsdi.mae.gov.vn/home - Vietnam National Spatial Data Infrastructure, Layer AD03: Địa phận đơn vị hành chính cấp xã).

====================================================================================================
DETERMINISTIC 2-WAY TRACEBACK UUID SPECIFICATION:
Format: [8 hex]-[4 hex]-[4 hex]-[4 hex]-[12 hex] (Compliant with RFC 4122 standard UUID)
----------------------------------------------------------------------------------------------------
- Segment 1 (8 hex): 'b00757ea' (Identifies bootstrap system data)
- Segment 2 (4 hex): Entity type classification:
    + '54b5' -> survey_submissions
    + 'cadd' -> unverified_sign_candidates
    + '5194' -> verified_traffic_signs
    + '8157' -> verified_sign_history
- Segment 3 (4 hex): '4000' (UUID Version 4 marker)
- Segment 4 (4 hex): High 16 bits of 64-bit Mapillary Feature ID (hex[:4])
- Segment 5 (12 hex): Low 48 bits of 64-bit Mapillary Feature ID (hex[4:])

Together, Segment 4 (4 hex) + Segment 5 (12 hex) represent the full 64-bit integer
(16 hex digits), guaranteeing 100% exact, lossless 2-way traceback without integer overflow.

DETERMINISTIC 2-WAY REVERSIBILITY:
1. Forward Direction (Mapillary Feature ID -> Database UUIDs):
   hex_16 = f"{int(feature_id):016x}"
   seg4 = hex_16[:4]
   seg5 = hex_16[4:]
   candidate_uuid = f"b00757ea-cadd-4000-{seg4}-{seg5}"
   verified_sign_uuid = f"b00757ea-5194-4000-{seg4}-{seg5}"

2. Reverse Direction (Database UUID -> Mapillary Feature ID):
   parts = uuid.split("-")
   combined_hex = parts[3] + parts[4]
   mapillary_feature_id = str(int(combined_hex, 16))
====================================================================================================
"""

import argparse
import datetime
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import requests
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MAPPING_FILE = BASE_DIR / "configs" / "mapillary_to_qcvn41_mapping.json"
SIGN_META_FILE = BASE_DIR / "configs" / "sign_metadata.json"
EXCLUDED_BOUNDARIES_FILE = BASE_DIR / "data" / "boundaries" / "tang_nhon_phu_excluded_ward.geojson"
OUTPUT_DIR = BASE_DIR / "outputs"
DEFAULT_OUTPUT_JSON = OUTPUT_DIR / "bootstrap_hcmc.json"

MAPILLARY_GRAPH_URL = "https://graph.mapillary.com"
DEFAULT_TOKEN = os.environ.get(
    "MAPILLARY_TOKEN",
    "MLY|27867926556148903|0b6e227179c9eefd1968a7d26aa042fe",
)

# Global System Submission UUID for HCMC Bootstrap
SYSTEM_SUBMISSION_UUID = "b00757ea-54b5-4000-0000-000000000001"

# HCMC Bounding Box:
# Covers urban core, western and northern corridors
HCMC_BBOX = {
    "min_lat": 10.720,
    "max_lat": 10.865,
    "min_lon": 106.600,
    "max_lon": 106.840,
}


def point_in_ring(x: float, y: float, ring: List[List[float]]) -> bool:
    """Ray casting algorithm for a single polygon ring."""
    n = len(ring)
    inside = False
    p1x, p1y = ring[0]
    for i in range(n + 1):
        p2x, p2y = ring[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def point_in_multipolygon(x: float, y: float, geom: Dict[str, Any]) -> bool:
    """Check if point (x=lon, y=lat) is inside a GeoJSON Polygon or MultiPolygon."""
    gtype = geom.get("type")
    coords = geom.get("coordinates", [])
    if gtype == "Polygon":
        if point_in_ring(x, y, coords[0]):
            for hole in coords[1:]:
                if point_in_ring(x, y, hole):
                    return False
            return True
    elif gtype == "MultiPolygon":
        for poly in coords:
            if point_in_ring(x, y, poly[0]):
                in_hole = False
                for hole in poly[1:]:
                    if point_in_ring(x, y, hole):
                        in_hole = True
                        break
                if not in_hole:
                    return True
    return False


def load_excluded_geometries() -> List[Dict[str, Any]]:
    """Load boundary polygon for Phường Tăng Nhơn Phú from VNSDI dataset in data/boundaries/."""
    if not EXCLUDED_BOUNDARIES_FILE.exists():
        logger.warning(f"Exclusion boundary file not found at {EXCLUDED_BOUNDARIES_FILE}.")
        return []
    with open(EXCLUDED_BOUNDARIES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [feat["geometry"] for feat in data.get("features", [])]


EXCLUDED_GEOMS = load_excluded_geometries()


def is_in_excluded_area(lat: float, lon: float) -> bool:
    """Check if coordinate falls within Phường Tăng Nhơn Phú boundary."""
    if EXCLUDED_GEOMS:
        for geom in EXCLUDED_GEOMS:
            if point_in_multipolygon(lon, lat, geom):
                return True
        return False
    return False


def generate_traceback_uuids(feature_id: str) -> Tuple[str, str]:
    """
    Generate deterministic Candidate UUID and Verified Sign UUID from Mapillary Feature ID.
    Enables full 64-bit lossless 2-way traceback across Segment 4 and Segment 5.
    """
    try:
        numeric_id = int(feature_id)
    except ValueError:
        numeric_id = abs(hash(feature_id))

    # Mapillary Feature IDs are 64-bit integers.
    # 16 hexadecimal digits = 64 bits.
    # Segment 4 (4 hex chars) stores the high 16 bits.
    # Segment 5 (12 hex chars) stores the low 48 bits.
    hex_16 = f"{numeric_id:016x}"
    seg4 = hex_16[:4]
    seg5 = hex_16[4:]

    candidate_uuid = f"b00757ea-cadd-4000-{seg4}-{seg5}"
    verified_sign_uuid = f"b00757ea-5194-4000-{seg4}-{seg5}"

    return candidate_uuid, verified_sign_uuid


def decode_mapillary_id_from_uuid(uuid_str: str) -> Optional[int]:
    """
    Reverse helper: Decode exact 64-bit Mapillary Feature ID from SignTrustMap UUID.
    Combined hex of Segment 4 (high 16 bits) and Segment 5 (low 48 bits) yields
    the exact Mapillary Feature ID without loss.
    """
    parts = uuid_str.split("-")
    if len(parts) == 5 and parts[0] == "b00757ea":
        combined_hex = parts[3] + parts[4]
        return int(combined_hex, 16)
    return None


def load_mapping() -> Dict[str, Any]:
    if not MAPPING_FILE.exists():
        logger.error(f"Mapping file not found at: {MAPPING_FILE}")
        sys.exit(1)
    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_sign_metadata() -> Dict[str, Any]:
    if not SIGN_META_FILE.exists():
        logger.error(f"Sign metadata file not found at: {SIGN_META_FILE}")
        sys.exit(1)
    with open(SIGN_META_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


SESSION = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=60, pool_maxsize=60, max_retries=3)
SESSION.mount("https://", adapter)
SESSION.mount("http://", adapter)


def request_get(url: str, params: Dict[str, Any], timeout: int = 15, max_retries: int = 3) -> Optional[requests.Response]:
    for attempt in range(1, max_retries + 1):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            elif r.status_code in (500, 429):
                time.sleep(0.3 * attempt)
            else:
                time.sleep(0.1)
        except Exception:
            time.sleep(0.3 * attempt)
    return None


def fetch_tile_map_features(token: str, bbox_str: str) -> List[Dict[str, Any]]:
    url = f"{MAPILLARY_GRAPH_URL}/map_features"
    params = {
        "access_token": token,
        "layers": "traffic_signs",
        "bbox": bbox_str,
        "fields": "id,geometry,object_value",
        "limit": 100,
    }
    r = request_get(url, params)
    if r:
        return r.json().get("data", [])
    return []


def fetch_feature_images(token: str, feature_id: str) -> List[Dict[str, Any]]:
    url = f"{MAPILLARY_GRAPH_URL}/{feature_id}"
    params = {
        "access_token": token,
        "fields": "id,geometry,object_value,images",
    }
    r = request_get(url, params)
    if r:
        return r.json().get("images", {}).get("data", [])
    return []


def fetch_image_metadata(token: str, image_id: str) -> Optional[Dict[str, Any]]:
    url = f"{MAPILLARY_GRAPH_URL}/{image_id}"
    params = {
        "access_token": token,
        "fields": "id,captured_at,compass_angle,geometry,thumb_1024_url,thumb_2048_url,thumb_original_url",
    }
    r = request_get(url, params)
    if r:
        return r.json()
    return None


def process_single_sign(
    fid: str,
    feat: Dict[str, Any],
    token: str,
    mapping: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    obj_val = feat.get("object_value", "")
    geom = feat.get("geometry", {})
    coords = geom.get("coordinates", [])
    if not coords or len(coords) < 2:
        return None
    sign_lon, sign_lat = coords[0], coords[1]

    mapped_info = mapping.get(obj_val)

    images_list = fetch_feature_images(token, fid)
    if not images_list:
        return None

    observations = []
    for img_item in images_list[:2]:
        iid = img_item.get("id")
        meta = fetch_image_metadata(token, iid)
        if not meta or not meta.get("captured_at"):
            continue

        ts = meta.get("captured_at")
        dt_iso = datetime.datetime.fromtimestamp(ts / 1000.0, datetime.timezone.utc).isoformat()
        thumb_url = meta.get("thumb_1024_url") or meta.get("thumb_2048_url") or meta.get("thumb_original_url")
        cam_geom = meta.get("geometry", {}).get("coordinates", [])

        observations.append({
            "image_id": iid,
            "captured_at": dt_iso,
            "timestamp_ms": ts,
            "compass_angle": meta.get("compass_angle"),
            "camera_location": cam_geom,
            "image_url": thumb_url,
        })

    if not observations:
        return None

    observations.sort(key=lambda x: x["timestamp_ms"])
    latest = observations[-1]
    history = []
    for idx, obs in enumerate(observations[:-1], start=1):
        history.append({
            "version_number": idx,
            "changed_at": obs["captured_at"],
            "previous_crop_url": obs["image_url"],
            "previous_location": obs["camera_location"],
            "previous_traffic_direction": obs["compass_angle"],
            "change_reason": "REVALIDATION_UPDATE",
        })

    cand_uuid, sign_uuid = generate_traceback_uuids(fid)

    return {
        "type": "Feature",
        "id": fid,
        "geometry": {
            "type": "Point",
            "coordinates": [sign_lon, sign_lat],
        },
        "properties": {
            "mapillary_feature_id": fid,
            "mapillary_mtsd_value": obj_val,
            "traceback": {
                "submission_id": SYSTEM_SUBMISSION_UUID,
                "candidate_id": cand_uuid,
                "verified_sign_id": sign_uuid,
                "decoded_mapillary_id": decode_mapillary_id_from_uuid(sign_uuid),
            },
            "sign_type_id": mapped_info.get("class_id", 999) if mapped_info else 999,
            "sign_code": mapped_info.get("standard_code", obj_val) if mapped_info else obj_val,
            "standard_code": mapped_info.get("standard_code", "UNKNOWN") if mapped_info else "UNKNOWN",
            "category": mapped_info.get("category", "REGULATORY") if mapped_info else "REGULATORY",
            "name_vi": mapped_info.get("name_vi", obj_val) if mapped_info else obj_val,
            "name_en": mapped_info.get("name_en", obj_val) if mapped_info else obj_val,
            "traffic_direction": latest.get("compass_angle"),
            "sign_crop_url": latest.get("image_url"),
            "status": "ACTIVE",
            "freshness_score": 1.0,
            "last_verified_at": latest["captured_at"],
            "total_observations": len(observations),
            "history": history,
        },
    }


def save_geojson_result(output_path: Path, signs: List[Dict[str, Any]]) -> None:
    result_geojson = {
        "type": "FeatureCollection",
        "metadata": {
            "region": "Ho Chi Minh City (Urban corridors, excluding Phường Tăng Nhơn Phú)",
            "exclusion_source": "https://vnsdi.mae.gov.vn/home (Vietnam National Spatial Data Infrastructure, Layer AD03: Địa phận đơn vị hành chính cấp xã)",
            "system_submission_id": SYSTEM_SUBMISSION_UUID,
            "uuid_rule": "Hex Segment Pattern: b00757ea-[entity]-4000-[mapillary_hex_high4]-[mapillary_hex_low12]",
            "extracted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "total_signs": len(signs),
            "signs_with_history": sum(1 for s in signs if len(s["properties"]["history"]) > 0),
        },
        "features": signs,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_geojson, f, indent=2, ensure_ascii=False)


def crawl_hcmc(
    token: str = DEFAULT_TOKEN,
    min_lat: float = HCMC_BBOX["min_lat"],
    max_lat: float = HCMC_BBOX["max_lat"],
    min_lon: float = HCMC_BBOX["min_lon"],
    max_lon: float = HCMC_BBOX["max_lon"],
    step: float = 0.005,
    max_signs: int = 7000,
    output_path: Path = DEFAULT_OUTPUT_JSON,
) -> List[Dict[str, Any]]:
    mapping = load_mapping()
    logger.info(f"Loaded {len(EXCLUDED_GEOMS)} exclusion boundary polygon (Phường Tăng Nhơn Phú).")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_cache_file = OUTPUT_DIR / ".raw_features_hcmc_cache.json"
    extracted_cache_file = OUTPUT_DIR / ".extracted_signs_cache.json"
    raw_features: Dict[str, Dict[str, Any]] = {}

    if raw_cache_file.exists():
        try:
            with open(raw_cache_file, "r", encoding="utf-8") as f:
                raw_features = json.load(f)
            logger.info(f"Loaded {len(raw_features)} cached sign features from {raw_cache_file}")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            raw_features = {}

    if not raw_features:
        logger.info(f"Starting HCMC crawl: Lat [{min_lat}, {max_lat}], Lon [{min_lon}, {max_lon}] (tile step={step})")
        tile_bboxes = []
        lat_curr = min_lat
        while lat_curr < max_lat:
            lon_curr = min_lon
            while lon_curr < max_lon:
                mid_lat = lat_curr + step / 2.0
                mid_lon = lon_curr + step / 2.0
                if not is_in_excluded_area(mid_lat, mid_lon):
                    bbox_str = f"{lon_curr:.4f},{lat_curr:.4f},{lon_curr + step:.4f},{lat_curr + step:.4f}"
                    tile_bboxes.append(bbox_str)
                lon_curr += step
            lat_curr += step

        logger.info(f"Scanning {len(tile_bboxes)} tiles across HCMC with concurrent workers...")
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_tile_map_features, token, bbox): bbox for bbox in tile_bboxes}
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                feats = future.result()
                for f in feats:
                    fid = str(f.get("id"))
                    val = f.get("object_value", "")
                    geom = f.get("geometry", {})
                    coords = geom.get("coordinates", [])
                    if not coords or len(coords) < 2:
                        continue

                    sign_lon, sign_lat = coords[0], coords[1]
                    if is_in_excluded_area(sign_lat, sign_lon):
                        continue

                    if fid not in raw_features and (
                        val in mapping
                        or "regulatory--" in val
                        or "warning--" in val
                        or "information--" in val
                        or "traffic-sign" in val
                    ):
                        raw_features[fid] = f

                if done_count % 30 == 0 or done_count == len(tile_bboxes):
                    logger.info(f"Tile progress: {done_count}/{len(tile_bboxes)} tiles scanned. Signs found: {len(raw_features)}")

        with open(raw_cache_file, "w", encoding="utf-8") as f:
            json.dump(raw_features, f)
        logger.info(f"Cached {len(raw_features)} raw features to {raw_cache_file}")

    logger.info(f"Total potential traffic sign features found in HCMC (excluding Phường Tăng Nhơn Phú): {len(raw_features)}")

    # Load previously extracted cache if any
    extracted_cache: Dict[str, Dict[str, Any]] = {}
    if extracted_cache_file.exists():
        try:
            with open(extracted_cache_file, "r", encoding="utf-8") as f:
                extracted_cache = json.load(f)
            logger.info(f"Loaded {len(extracted_cache)} previously extracted signs from {extracted_cache_file}")
        except Exception as e:
            logger.warning(f"Failed to read extracted cache: {e}")
            extracted_cache = {}

    # Prioritize mapped signs first
    sorted_fids = sorted(
        raw_features.keys(),
        key=lambda fid: (0 if raw_features[fid].get("object_value") in mapping else 1),
    )

    target_fids = sorted_fids[:max_signs]
    # Identify which target FIDs still need to be extracted
    remaining_fids = [fid for fid in target_fids if fid not in extracted_cache]
    logger.info(f"Target: {len(target_fids)} signs. Already cached: {len(target_fids) - len(remaining_fids)}. To fetch: {len(remaining_fids)}")

    processed_dict: Dict[str, Dict[str, Any]] = {fid: extracted_cache[fid] for fid in target_fids if fid in extracted_cache and extracted_cache[fid] is not None}

    if remaining_fids:
        logger.info(f"Extracting observations for {len(remaining_fids)} signs with 40 concurrent workers...")
        with ThreadPoolExecutor(max_workers=40) as executor:
            futures = {
                executor.submit(process_single_sign, fid, raw_features[fid], token, mapping): fid
                for fid in remaining_fids
            }
            newly_extracted = 0
            for future in as_completed(futures):
                fid = futures[future]
                record = future.result()
                extracted_cache[fid] = record
                if record:
                    processed_dict[fid] = record
                    newly_extracted += 1

                total_done = len(target_fids) - len(remaining_fids) + newly_extracted
                if newly_extracted % 200 == 0 or newly_extracted == len(remaining_fids):
                    logger.info(f"Extraction progress: {total_done}/{len(target_fids)} completed (Valid signs: {len(processed_dict)})...")
                    # Checkpoint to disk
                    try:
                        with open(extracted_cache_file, "w", encoding="utf-8") as f:
                            json.dump(extracted_cache, f)
                        save_geojson_result(output_path, list(processed_dict.values()))
                    except Exception as e:
                        logger.warning(f"Error saving checkpoint: {e}")

        # Final cache save
        with open(extracted_cache_file, "w", encoding="utf-8") as f:
            json.dump(extracted_cache, f)

    processed_signs = [processed_dict[fid] for fid in target_fids if fid in processed_dict]
    return processed_signs


def main():
    parser = argparse.ArgumentParser(description="Bootstrap traffic signs across HCMC (excluding Phường Tăng Nhơn Phú).")
    parser.add_argument("--token", default=DEFAULT_TOKEN, help="Mapillary Graph API v4 access token")
    parser.add_argument("--max-signs", type=int, default=7000, help="Maximum number of signs to extract")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_JSON, help="Output JSON file path")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    signs = crawl_hcmc(token=args.token, max_signs=args.max_signs, output_path=args.output)

    save_geojson_result(args.output, signs)
    logger.success(f"Crawl completed! Saved {len(signs)} signs across HCMC to: {args.output}")


if __name__ == "__main__":
    main()

