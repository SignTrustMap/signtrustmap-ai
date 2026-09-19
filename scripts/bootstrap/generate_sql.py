#!/usr/bin/env python3
"""
Generate PostgreSQL Flyway Migrations for SignTrustMap:
1. 008_seed_full_traffic_sign_types.sql: Seeds QCVN 41:2019 categories and sign types with vehicle restrictions.
2. 009_bootstrap_mapillary_verified_signs.sql: Bootstraps verified traffic signs crawled from Mapillary Graph API v4,
   with 64-bit lossless 2-way traceback UUIDs, system submission, candidates, and multi-temporal history.

All geometries use PostGIS ST_SetSRID(ST_MakePoint(lon, lat), 4326).
Boundary exclusion for Phường Tăng Nhơn Phú is cited from:
https://vnsdi.mae.gov.vn/home (Vietnam National Spatial Data Infrastructure, Layer AD03).
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List

AI_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = AI_DIR.parent / "signtrustmap-backend"
MIGRATIONS_DIR = BACKEND_DIR / "infra" / "postgres" / "migrations"
OUTPUTS_JSON = AI_DIR / "outputs" / "bootstrap_hcmc.json"
SIGN_META_JSON = AI_DIR / "configs" / "sign_metadata.json"

MIGRATION_008_PATH = MIGRATIONS_DIR / "008_seed_full_traffic_sign_types.sql"
MIGRATION_009_PATH = MIGRATIONS_DIR / "009_bootstrap_mapillary_verified_signs.sql"
MIGRATION_010_PATH = MIGRATIONS_DIR / "010_seed_mock_reviewers_and_consensus.sql"

EXCLUDED_OBJECT_VALUES = {
    "object--traffic-sign--back",
    "object--traffic-sign--direction-back",
    "object--support--traffic-sign-frame",
}

# Vehicle restriction mapping from 007_seed_allowed_vehicles.sql
VEHICLE_GROUPS = {
    "EMPTY": {3, 31, 32, 33, 34, 45, 46, 109},
    "MOTORCYCLE_BICYCLE_PEDESTRIAN": {23, 24, 25, 91},
    "CAR_TRUCK_BUS_BICYCLE_PEDESTRIAN": {26},
    "CAR_MOTORCYCLE_BUS_BICYCLE_PEDESTRIAN": {27, 28},
    "CAR_TRUCK_BUS_PEDESTRIAN": {80},
    "CAR_MOTORCYCLE_BICYCLE_PEDESTRIAN": {82},
    "CAR_MOTORCYCLE_TRUCK_BUS": {66, 74},
    "CAR_TRUCK_BUS": {106, 107},
    "CAR_MOTORCYCLE_TRUCK_BUS_BICYCLE": {29},
}

CATEGORY_MAP = {
    "PROHIBITORY": 1,
    "PROHIBITION": 1,
    "WARNING": 2,
    "MANDATORY": 3,
    "INFORMATION": 4,
    "SPEED_LIMIT": 5,
}

DB_SIGN_CODE_TO_ID = {
    "PEDESTRIAN_CROSSING": 1,
    "ROAD_WORKS": 2,
    "NO_ENTRY": 3,
    "PEDESTRIAN_CROSSING_ZONE": 4,
    "STEEP_ASCENT_5_PERCENT": 5,
    "RAILWAY_CROSSING_WITH_GATES": 6,
    "ZONE.01": 7,
    "TEST_SIGN_3": 8,
    "NO_CARS": 23,
    "NO_CARS_TURN_RIGHT": 24,
    "NO_CARS_TURN_LEFT": 25,
    "NO_MOTORCYCLES": 26,
    "NO_TRUCKS": 27,
    "NO_TRUCKS_OVER_2.5T": 28,
    "NO_PEDESTRIANS": 29,
    "HEIGHT_LIMIT": 30,
    "NO_TURN_LEFT": 31,
    "NO_TURN_RIGHT": 32,
    "NO_U_TURN": 33,
    "NO_TURN_LEFT_AND_U_TURN": 34,
    "SPEED_LIMIT_40": 35,
    "SPEED_LIMIT_50": 36,
    "SPEED_LIMIT_60": 37,
    "SPEED_LIMIT_70": 38,
    "SPEED_LIMIT_80": 39,
    "NO_HONKING": 40,
    "NO_STOPPING_AND_PARKING": 41,
    "NO_PARKING": 42,
    "NO_PARKING_ODD_DAYS": 43,
    "NO_PARKING_EVEN_DAYS": 44,
    "NO_STRAIGHT_AHEAD": 45,
    "NO_TURN_LEFT_AND_RIGHT": 46,
    "DANGEROUS_CURVE_LEFT": 47,
    "ROAD_NARROWS": 48,
    "CROSSROAD": 49,
    "INTERSECTION_WITH_NON_PRIORITY_ROAD": 50,
    "GIVE_WAY": 51,
    "ROAD_BUMP": 52,
    "CHILDREN_CROSSING": 53,
    "OVERHEAD_POWER_LINES": 54,
    "CLEARANCE_HEIGHT": 55,
    "SLOW_DOWN": 56,
    "GO_STRAIGHT": 57,
    "KEEP_RIGHT": 58,
    "ROUNDABOUT": 59,
    "MAIN_ROAD": 60,
    "END_MAIN_ROAD": 61,
    "HOSPITAL": 62,
    "BUS_STOP": 63,
    "MARKET": 64,
    "OTHER_PROHIBITION_SIGNS": 65,
    "NO_SLOW_MOVING_VEHICLES": 66,
    "U_TURN_AREA": 67,
    "PEDESTRIAN_OVERPASS": 68,
    "WIDTH_LIMIT": 69,
    "WEIGHT_LIMIT": 70,
    "NO_TRAILERS": 71,
    "NO_VEHICLES_CARRYING_DANGEROUS_GOODS": 72,
    "TUNNEL": 73,
    "NO_BICYCLES": 74,
    "CLIFF_AHEAD": 75,
    "MIN_SPEED_LIMIT_30": 76,
    "MIN_DISTANCE_BETWEEN_VEHICLES": 77,
    "NO_OVERTAKING": 78,
    "END_MIN_SPEED_LIMIT_30": 79,
    "NO_MOTORCYCLES_AND_SLOW_VEHICLES": 80,
    "END_DUAL_CARRIAGEWAY": 81,
    "NO_BUSES_AND_TRUCKS": 82,
    "TWO_WAY_TRAFFIC": 83,
    "ACCIDENT_AREA": 84,
    "TURN_LEFT_ONLY": 85,
    "TURN_RIGHT_ONLY": 86,
    "ONE_WAY_ROAD": 87,
    "UNDERPASS": 88,
    "RESIDENTIAL_AREA": 89,
    "END_RESIDENTIAL_AREA": 90,
    "NO_CARS_U_TURN": 91,
    "DANGEROUS_CURVE_RIGHT": 92,
    "MULTIPLE_CURVES": 93,
    "ROAD_NARROWS_LEFT": 94,
    "ROAD_NARROWS_RIGHT": 95,
    "INTERSECTION_WITH_NON_PRIORITY_ROAD_RIGHT": 96,
    "T_JUNCTION": 97,
    "T_JUNCTION_RIGHT": 98,
    "T_JUNCTION_LEFT": 99,
    "INTERSECTION_WITH_NON_PRIORITY_ROAD_LEFT": 100,
    "END_NO_OVERTAKING": 101,
    "END_ALL_PROHIBITIONS": 102,
    "END_TUNNEL": 103,
    "SPEED_LIMIT_90": 104,
    "SPEED_LIMIT_100": 105,
    "ROAD_RESERVED_FOR_CARS": 106,
    "EXPRESSWAY": 107,
    "AXLE_WEIGHT_LIMIT": 108,
    "ROAD_CLOSED": 109,
    "DEAD_END": 110,
    "RESERVED_FOR_DISABLED": 111,
    "CHECKPOINT": 112,
    "SPEED_LIMIT_120": 113,
    "PARKING_AREA": 114,
    "MIN_SPEED_LIMIT_60": 115,
    "STEEP_DESCENT_5_PERCENT": 116,
}


def resolve_db_sign_type_id(class_name: str, fallback_cid: int = 0) -> int:
    key = class_name.strip().upper()
    if key in DB_SIGN_CODE_TO_ID:
        return DB_SIGN_CODE_TO_ID[key]
    # Fallback to sign_metadata
    try:
        with open(SIGN_META_JSON, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cid_str = str(fallback_cid)
        if cid_str in meta:
            meta_name = meta[cid_str]["class_name"].strip().upper()
            if meta_name in DB_SIGN_CODE_TO_ID:
                return DB_SIGN_CODE_TO_ID[meta_name]
    except Exception:
        pass
    return 3  # Default NO_ENTRY



def get_allowed_vehicles_sql(type_id: int) -> str:
    if type_id in VEHICLE_GROUPS["EMPTY"]:
        return "'{}'::text[]"
    elif type_id in VEHICLE_GROUPS["MOTORCYCLE_BICYCLE_PEDESTRIAN"]:
        return "ARRAY['MOTORCYCLE','BICYCLE','PEDESTRIAN']::text[]"
    elif type_id in VEHICLE_GROUPS["CAR_TRUCK_BUS_BICYCLE_PEDESTRIAN"]:
        return "ARRAY['CAR','TRUCK','BUS','BICYCLE','PEDESTRIAN']::text[]"
    elif type_id in VEHICLE_GROUPS["CAR_MOTORCYCLE_BUS_BICYCLE_PEDESTRIAN"]:
        return "ARRAY['CAR','MOTORCYCLE','BUS','BICYCLE','PEDESTRIAN']::text[]"
    elif type_id in VEHICLE_GROUPS["CAR_TRUCK_BUS_PEDESTRIAN"]:
        return "ARRAY['CAR','TRUCK','BUS','PEDESTRIAN']::text[]"
    elif type_id in VEHICLE_GROUPS["CAR_MOTORCYCLE_BICYCLE_PEDESTRIAN"]:
        return "ARRAY['CAR','MOTORCYCLE','BICYCLE','PEDESTRIAN']::text[]"
    elif type_id in VEHICLE_GROUPS["CAR_MOTORCYCLE_TRUCK_BUS"]:
        return "ARRAY['CAR','MOTORCYCLE','TRUCK','BUS']::text[]"
    elif type_id in VEHICLE_GROUPS["CAR_TRUCK_BUS"]:
        return "ARRAY['CAR','TRUCK','BUS']::text[]"
    elif type_id in VEHICLE_GROUPS["CAR_MOTORCYCLE_TRUCK_BUS_BICYCLE"]:
        return "ARRAY['CAR','MOTORCYCLE','TRUCK','BUS','BICYCLE']::text[]"
    else:
        return "ARRAY['CAR','MOTORCYCLE','TRUCK','BUS','BICYCLE','PEDESTRIAN']::text[]"


def escape_sql_str(val: Any) -> str:
    if val is None:
        return "NULL"
    s = str(val).replace("'", "''")
    return f"'{s}'"


def append_batched_insert(
    lines: List[str],
    table_clause: str,
    rows: List[str],
    on_conflict: str = "",
    batch_size: int = 1000,
) -> None:
    """Helper to break large INSERT operations into clean batches (e.g. 1000 rows each)."""
    if not rows:
        return
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        lines.append(f"INSERT INTO {table_clause} VALUES")
        lines.append(",\n".join(batch))
        if on_conflict:
            lines.append(f"{on_conflict};")
        else:
            lines.append(";")
        lines.append("")


def calculate_haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate great-circle distance in meters between two coordinates using Haversine."""
    r = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def calculate_initial_bearing(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate forward initial bearing from point 1 (cam) to point 2 (sign) in degrees [0, 360)."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)
    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    theta = math.atan2(y, x)
    return (math.degrees(theta) + 360.0) % 360.0


def generate_migration_008() -> None:
    print(f"Generating {MIGRATION_008_PATH}...")
    with open(SIGN_META_JSON, "r", encoding="utf-8") as f:
        meta_dict = json.load(f)

    lines = [
        "-- =============================================================================",
        "-- 008_seed_full_traffic_sign_types.sql",
        "-- Seeds authoritative QCVN 41:2019 sign categories and traffic sign types.",
        "-- =============================================================================",
        "",
        "-- 1. Ensure core sign categories",
        "INSERT INTO \"sign_categories\" (\"id\", \"code\", \"name_vi\", \"name_en\", \"sort_order\") VALUES",
        "  (1, 'PROHIBITION', 'Biển báo cấm', 'Prohibitory signs', 1),",
        "  (2, 'WARNING', 'Biển cảnh báo', 'Warning signs', 2),",
        "  (3, 'MANDATORY', 'Biển hiệu lệnh', 'Mandatory signs', 3),",
        "  (4, 'INFORMATION', 'Biển chỉ dẫn', 'Information signs', 4),",
        "  (5, 'SPEED_LIMIT', 'Biển giới hạn tốc độ', 'Speed limit signs', 5)",
        "ON CONFLICT (\"code\") DO UPDATE SET",
        "  \"name_vi\" = EXCLUDED.\"name_vi\",",
        "  \"name_en\" = EXCLUDED.\"name_en\",",
        "  \"sort_order\" = EXCLUDED.\"sort_order\";",
        "",
        "-- 2. Seed all 100 QCVN 41:2019 traffic sign types with vehicle restrictions",
        "INSERT INTO \"traffic_sign_types\" (",
        "  \"id\", \"category_id\", \"sign_code\", \"name_vi\", \"name_en\",",
        "  \"shape\", \"color_scheme\", \"is_active\", \"allowed_vehicles\", \"created_at\", \"updated_at\"",
        ") VALUES",
    ]

    values_rows = []
    # Sort by class_id integer
    sorted_items = sorted(meta_dict.values(), key=lambda x: int(x.get("class_id", 0)))

    for item in sorted_items:
        class_id = int(item["class_id"])
        sign_type_id = class_id + 1  # 1-indexed ID
        cat_str = item.get("category", "PROHIBITORY")
        category_id = CATEGORY_MAP.get(cat_str, 1)
        class_name = item.get("class_name", "")
        sign_code = item.get("standard_code") or f"SIGN_{class_id}"
        if sign_code == "P.127" and "_" in class_name:
            speed = class_name.split("_")[-1]
            sign_code = f"P.127-{speed}"
        elif sign_code == "R.306" and "_" in class_name:
            speed = class_name.split("_")[-1]
            sign_code = f"R.306-{speed}"

        name_vi = item.get("name_vi", sign_code)
        name_en = item.get("name_en", sign_code)
        shape = "ROUND" if category_id in (1, 3, 5) else ("TRIANGLE" if category_id == 2 else "RECTANGLE")
        color_scheme = "RED_WHITE" if category_id in (1, 2) else ("BLUE_WHITE" if category_id in (3, 4) else "RED_WHITE_BLACK")
        allowed_vehicles_sql = get_allowed_vehicles_sql(sign_type_id)

        row_sql = (
            f"  ({sign_type_id}, {category_id}, {escape_sql_str(sign_code)}, "
            f"{escape_sql_str(name_vi)}, {escape_sql_str(name_en)}, "
            f"{escape_sql_str(shape)}, {escape_sql_str(color_scheme)}, true, "
            f"{allowed_vehicles_sql}, now(), now())"
        )
        values_rows.append(row_sql)

    lines.append(",\n".join(values_rows))
    lines.extend([
        "ON CONFLICT (\"id\") DO UPDATE SET",
        "  \"category_id\" = EXCLUDED.\"category_id\",",
        "  \"sign_code\" = EXCLUDED.\"sign_code\",",
        "  \"name_vi\" = EXCLUDED.\"name_vi\",",
        "  \"name_en\" = EXCLUDED.\"name_en\",",
        "  \"shape\" = EXCLUDED.\"shape\",",
        "  \"color_scheme\" = EXCLUDED.\"color_scheme\",",
        "  \"is_active\" = EXCLUDED.\"is_active\",",
        "  \"allowed_vehicles\" = EXCLUDED.\"allowed_vehicles\",",
        "  \"updated_at\" = now();",
        "",
        "-- Reset sequence if needed",
        "SELECT setval(pg_get_serial_sequence('traffic_sign_types', 'id'), COALESCE((SELECT MAX(id) FROM traffic_sign_types), 1));",
        "",
    ])

    MIGRATIONS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MIGRATION_008_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Generated {MIGRATION_008_PATH} successfully ({len(sorted_items)} sign types).")


def generate_migration_009() -> None:
    print(f"Generating {MIGRATION_009_PATH} from {OUTPUTS_JSON}...")
    if not OUTPUTS_JSON.exists():
        print(f"ERROR: {OUTPUTS_JSON} not found. Please run crawl_hcmc.py first.")
        return

    with open(OUTPUTS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(AI_DIR / "configs" / "mapillary_to_qcvn41_mapping.json", "r", encoding="utf-8") as f:
        mapping_dict = json.load(f)

    # Filter strictly for signs with authoritative Mapillary Ground Truth mapping
    raw_signs = data.get("features", [])
    signs = [
        s for s in raw_signs
        if s.get("properties", {}).get("mapillary_mtsd_value") in mapping_dict
    ]
    system_submission_id = data.get("metadata", {}).get("system_submission_id", "b00757ea-54b5-4000-0000-000000000001")

    lines = [
        "-- =============================================================================",
        "-- 009_bootstrap_mapillary_verified_signs.sql",
        "-- Bootstraps verified traffic signs for Ho Chi Minh City from Mapillary Graph API v4.",
        "-- Exclusion zone: Phường Tăng Nhơn Phú (source: https://vnsdi.mae.gov.vn/home Layer AD03).",
        "-- 2-Way Deterministic Traceback UUID: b00757ea-[entity]-4000-[mapillary_high16_hex]-[mapillary_low48_hex]",
        "-- =============================================================================",
        "",
        "-- 1. Ensure system bootstrap user and surveyor profile",
        "INSERT INTO \"users\" (\"id\", \"email\", \"full_name\", \"status\", \"created_at\", \"updated_at\")",
        "VALUES ('00000000-0000-0000-0000-000000000001', 'system_bootstrap@signtrustmap.local', 'SignTrustMap System Bootstrap', 'ACTIVE', now(), now())",
        "ON CONFLICT (\"email\") DO UPDATE SET \"status\" = 'ACTIVE';",
        "",
        "INSERT INTO \"surveyor_profiles\" (\"user_id\", \"total_submissions\", \"total_verified_signs\", \"total_credits_earned\", \"created_at\", \"updated_at\")",
        "VALUES ('00000000-0000-0000-0000-000000000001', 1, " + str(len(signs)) + ", 0, now(), now())",
        "ON CONFLICT (\"user_id\") DO UPDATE SET \"total_verified_signs\" = EXCLUDED.\"total_verified_signs\";",
        "",
        "-- 2. Insert Master Bootstrap Survey Submission",
        f"INSERT INTO \"survey_submissions\" (",
        f"  \"id\", \"surveyor_id\", \"submission_type\", \"status\", \"captured_at\",",
        f"  \"coordinate_source\", \"note\", \"total_candidates_extracted\", \"created_at\", \"updated_at\"",
        f") VALUES (",
        f"  '{system_submission_id}',",
        f"  '00000000-0000-0000-0000-000000000001',",
        f"  'SINGLE_IMAGE',",
        f"  'COMPLETED',",
        f"  '2025-01-01 00:00:00+00',",
        f"  'IMAGE_EXIF',",
        f"  'Mapillary HCMC Bootstrap (excluding Phường Tăng Nhơn Phú, boundary source: https://vnsdi.mae.gov.vn/home Layer AD03)',",
        f"  {len(signs)},",
        f"  now(),",
        f"  now()",
        f") ON CONFLICT (\"id\") DO NOTHING;",
        "",
        "-- 3. Insert Unverified Sign Candidates (status = VERIFIED)",
    ]

    candidate_rows = []
    geo_rows = []
    sign_rows = []
    history_rows = []

    for sign in signs:
        props = sign["properties"]
        geom = sign["geometry"]
        coords = geom["coordinates"]
        lon, lat = coords[0], coords[1]

        cand_id = props["traceback"]["candidate_id"]
        sign_id = props["traceback"]["verified_sign_id"]
        obj_val = props.get("mapillary_mtsd_value", "")

        # Resolve authoritative QCVN sign_type_id strictly according to production database
        if obj_val in mapping_dict:
            c_info = mapping_dict[obj_val]
            sign_type_id = resolve_db_sign_type_id(c_info.get("class_name", ""), c_info.get("class_id", 0))
            conf = 0.95
        else:
            sign_type_id = 3  # NO_ENTRY
            conf = 0.90
        raw_frame_url = props.get("sign_crop_url") or "https://images.mapillary.com/placeholder.jpg"
        crop_url = f"https://s3.signmap.site/stm-sign-crops/candidates/{cand_id}.jpg"
        direction = props.get("traffic_direction")
        direction_sql = f"{float(direction):.2f}" if direction is not None else "NULL"
        verified_at = props.get("last_verified_at") or "2025-01-01T00:00:00Z"

        cand_row = (
            f"  ('{cand_id}', '{system_submission_id}', {sign_type_id}, {conf:.2f}, "
            f"{escape_sql_str(raw_frame_url)}, {escape_sql_str(crop_url)}, 'HIGH', 'VERIFIED', {escape_sql_str(verified_at)})"
        )
        candidate_rows.append(cand_row)

        # Geolocation estimation (vehicle_location -> sign_location)
        cam_loc = props.get("camera_location")
        if not cam_loc and props.get("history"):
            cam_loc = props["history"][0].get("previous_location")
        if cam_loc and len(cam_loc) >= 2:
            cam_lon, cam_lat = cam_loc[0], cam_loc[1]
        else:
            cam_lon, cam_lat = lon, lat

        dist = calculate_haversine_distance(cam_lon, cam_lat, lon, lat)
        bearing = calculate_initial_bearing(cam_lon, cam_lat, lon, lat)

        geo_row = (
            f"  ('{cand_id}', ST_SetSRID(ST_MakePoint({cam_lon:.7f}, {cam_lat:.7f}), 4326), "
            f"ST_SetSRID(ST_MakePoint({lon:.7f}, {lat:.7f}), 4326), "
            f"{dist:.2f}, {bearing:.2f}, {direction_sql}, 'spherical_trig')"
        )
        geo_rows.append(geo_row)

        sign_row = (
            f"  ('{sign_id}', {sign_type_id}, '{cand_id}', ST_SetSRID(ST_MakePoint({lon:.7f}, {lat:.7f}), 4326), "
            f"{direction_sql}, {escape_sql_str(crop_url)}, 'ACTIVE', 1.0, 1, "
            f"{escape_sql_str(verified_at)}, {escape_sql_str(verified_at)}, now())"
        )
        sign_rows.append(sign_row)

        # Multi-temporal history
        for hist_item in props.get("history", []):
            prev_crop = hist_item.get("previous_crop_url") or crop_url
            prev_dir = hist_item.get("previous_traffic_direction")
            prev_dir_sql = f"{float(prev_dir):.2f}" if prev_dir is not None else "NULL"
            prev_time = hist_item.get("changed_at") or verified_at
            prev_loc = hist_item.get("previous_location")
            if prev_loc and len(prev_loc) >= 2:
                prev_loc_sql = f"ST_SetSRID(ST_MakePoint({prev_loc[0]:.7f}, {prev_loc[1]:.7f}), 4326)"
            else:
                prev_loc_sql = f"ST_SetSRID(ST_MakePoint({lon:.7f}, {lat:.7f}), 4326)"

            hist_row = (
                f"  ('{sign_id}', {sign_type_id}, {prev_loc_sql}, "
                f"{prev_dir_sql}, {escape_sql_str(prev_crop)}, 'REVALIDATION_UPDATE', {escape_sql_str(prev_time)})"
            )
            history_rows.append(hist_row)

    append_batched_insert(
        lines,
        '"unverified_sign_candidates" ("id", "submission_id", "predicted_sign_type_id", "confidence_score", "best_frame_url", "sign_crop_url", "spatial_confidence", "status", "created_at")',
        candidate_rows,
        on_conflict='ON CONFLICT ("id") DO NOTHING',
        batch_size=1000,
    )

    lines.append("-- 3.1. Insert Candidate Geo Estimations (Ground Truth from Mapillary SfM)")
    geo_on_conflict = """ON CONFLICT ("candidate_id") DO UPDATE SET
  "vehicle_location" = EXCLUDED."vehicle_location",
  "sign_location" = EXCLUDED."sign_location",
  "estimated_distance" = EXCLUDED."estimated_distance",
  "bearing_angle" = EXCLUDED."bearing_angle",
  "traffic_direction" = EXCLUDED."traffic_direction",
  "projection_method" = EXCLUDED."projection_method\""""

    append_batched_insert(
        lines,
        '"candidate_geo_estimations" ("candidate_id", "vehicle_location", "sign_location", "estimated_distance", "bearing_angle", "traffic_direction", "projection_method")',
        geo_rows,
        on_conflict=geo_on_conflict,
        batch_size=1000,
    )

    lines.append("-- 4. Insert Verified Traffic Signs")
    sign_on_conflict = """ON CONFLICT ("id") DO UPDATE SET
  "location" = EXCLUDED."location",
  "traffic_direction" = EXCLUDED."traffic_direction",
  "sign_crop_url" = EXCLUDED."sign_crop_url",
  "status" = EXCLUDED."status",
  "freshness_score" = EXCLUDED."freshness_score",
  "last_verified_at" = EXCLUDED."last_verified_at",
  "updated_at" = now()"""

    append_batched_insert(
        lines,
        '"verified_traffic_signs" ("id", "sign_type_id", "origin_candidate_id", "location", "traffic_direction", "sign_crop_url", "status", "freshness_score", "confirmation_count", "last_verified_at", "created_at", "updated_at")',
        sign_rows,
        on_conflict=sign_on_conflict,
        batch_size=1000,
    )

    if history_rows:
        lines.append("-- 5. Insert Multi-Temporal Verified Sign History")
        append_batched_insert(
            lines,
            '"verified_sign_history" ("verified_sign_id", "previous_sign_type_id", "previous_location", "previous_traffic_direction", "previous_crop_url", "change_reason", "changed_at")',
            history_rows,
            batch_size=1000,
        )

    with open(MIGRATION_009_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Generated {MIGRATION_009_PATH} successfully ({len(signs)} signs, {len(history_rows)} history rows).")


def generate_migration_010() -> None:
    print(f"Generating {MIGRATION_010_PATH}...")
    if not OUTPUTS_JSON.exists():
        print(f"ERROR: {OUTPUTS_JSON} not found.")
        return

    with open(OUTPUTS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(AI_DIR / "configs" / "mapillary_to_qcvn41_mapping.json", "r", encoding="utf-8") as f:
        mapping_dict = json.load(f)

    # Filter strictly for signs with authoritative Mapillary Ground Truth mapping
    raw_signs = data.get("features", [])
    signs = [
        s for s in raw_signs
        if s.get("properties", {}).get("mapillary_mtsd_value") in mapping_dict
    ]

    import random
    random.seed(42)

    VN_HO = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ", "Võ", "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương", "Lý"]
    VN_DEM = ["Văn", "Thị", "Đức", "Minh", "Thanh", "Hữu", "Tuấn", "Quốc", "Ngọc", "Hải", "Đình", "Xuân"]
    VN_TEN = ["An", "Bình", "Cường", "Dũng", "Giang", "Hải", "Hùng", "Huy", "Khoa", "Linh", "Long", "Minh", "Nam", "Phúc", "Quân", "Sơn", "Tâm", "Thắng", "Thịnh", "Tiến", "Trang", "Tuấn", "Tùng", "Việt", "Vy"]

    # 100 Reviewers: 10 Power (weight 4), 30 Active (weight 2), 60 Casual (weight 1)
    reviewers = []
    reviewer_weights = []
    for i in range(1, 101):
        name = f"{random.choice(VN_HO)} {random.choice(VN_DEM)} {random.choice(VN_TEN)}"
        email = f"reviewer_{i:03d}@stm.dev"
        uuid = f"b00757ea-7e71-4000-0000-{i:012x}"
        if i <= 10:
            tier = "POWER"
            weight = 4
            base_rel = round(random.uniform(0.92, 0.98), 2)
        elif i <= 40:
            tier = "ACTIVE"
            weight = 2
            base_rel = round(random.uniform(0.82, 0.89), 2)
        else:
            tier = "CASUAL"
            weight = 1
            base_rel = round(random.uniform(0.65, 0.78), 2)

        reviewers.append({
            "idx": i,
            "uuid": uuid,
            "email": email,
            "name": name,
            "tier": tier,
            "weight": weight,
            "base_rel": base_rel,
        })
        reviewer_weights.append(weight)

    lines = [
        "-- =============================================================================",
        "-- 010_seed_mock_reviewers_and_consensus.sql",
        "-- Seeds 100 realistic community reviewers across 3 engagement tiers",
        "-- (10 Power Reviewers, 30 Active Reviewers, 60 Casual Reviewers),",
        "-- generates realistic candidate reviews, consensus configurations, and audit logs.",
        "-- =============================================================================",
        "",
        "-- 1. Ensure REVIEWER role exists",
        "INSERT INTO \"roles\" (\"code\", \"name_vi\", \"name_en\", \"description\") VALUES",
        "  ('REVIEWER', 'Người duyệt', 'Reviewer', 'Phê duyệt / từ chối candidate signs')",
        "ON CONFLICT (\"code\") DO NOTHING;",
        "",
        "-- 2. Seed 100 Reviewer User Accounts",
        "INSERT INTO \"users\" (\"id\", \"email\", \"password_hash\", \"full_name\", \"status\", \"created_at\", \"updated_at\") VALUES",
    ]

    dev_pwd_hash = "$2a$10$Q6akvXznblgel4lxmg4ZB.pzviz.slAY8birOzDp43xZxzewBYpbW"
    user_rows = []
    for r in reviewers:
        r_uuid = r["uuid"]
        r_email = r["email"]
        r_name_escaped = escape_sql_str(r["name"])
        user_rows.append(
            f"  ('{r_uuid}', '{r_email}', '{dev_pwd_hash}', {r_name_escaped}, 'ACTIVE', now(), now())"
        )
    lines.append(",\n".join(user_rows))
    lines.extend([
        "ON CONFLICT (\"email\") DO UPDATE SET \"status\" = 'ACTIVE';",
        "",
        "-- 3. Assign REVIEWER role and create Wallets",
        "INSERT INTO \"user_roles\" (\"user_id\", \"role_id\", \"is_active\", \"assigned_at\")",
        "SELECT u.id, (SELECT id FROM roles WHERE code = 'REVIEWER' LIMIT 1), true, now()",
        "FROM \"users\" u",
        "WHERE u.email LIKE 'reviewer_%@stm.dev'",
        "ON CONFLICT DO NOTHING;",
        "",
        "INSERT INTO \"wallets\" (\"user_id\", \"balance\", \"updated_at\")",
        "SELECT u.id, 0, now()",
        "FROM \"users\" u",
        "WHERE u.email LIKE 'reviewer_%@stm.dev'",
        "ON CONFLICT (\"user_id\") DO NOTHING;",
        "",
        "-- 4. Seed Reviewer Profiles with Tier-based Reliability Scores",
        "INSERT INTO \"reviewer_profiles\" (",
        "  \"user_id\", \"reliability_score\", \"total_reviews\", \"correct_reviews\", \"accuracy_rate\", \"current_streak\", \"created_at\", \"updated_at\"",
        ") VALUES",
    ])

    profile_rows = []
    for r in reviewers:
        r_uuid = r["uuid"]
        r_rel = r["base_rel"]
        profile_rows.append(
            f"  ('{r_uuid}', {r_rel}, 0, 0, 1.0, 0, now(), now())"
        )
    lines.append(",\n".join(profile_rows))
    lines.extend([
        "ON CONFLICT (\"user_id\") DO UPDATE SET",
        "  \"reliability_score\" = EXCLUDED.\"reliability_score\",",
        "  \"accuracy_rate\" = EXCLUDED.\"accuracy_rate\";",
        "",
        "-- 5. Ensure Default Consensus Config (Rule: min 3 votes, threshold 0.70)",
        "INSERT INTO \"consensus_configs\" (",
        "  \"id\", \"approval_threshold\", \"rejection_threshold\", \"min_votes_required\",",
        "  \"alpha_smoothing_factor\", \"is_active\", \"effective_from\"",
        ") VALUES (",
        "  1, 0.70, 0.30, 3, 0.10, true, now()",
        ") ON CONFLICT (\"id\") DO UPDATE SET",
        "  \"approval_threshold\" = EXCLUDED.\"approval_threshold\",",
        "  \"rejection_threshold\" = EXCLUDED.\"rejection_threshold\",",
        "  \"min_votes_required\" = EXCLUDED.\"min_votes_required\";",
        "",
        "-- 6. Seed Realistic Candidate Reviews (3-4 votes per candidate)",
    ])

    review_rows = []
    consensus_log_rows = []

    for sign in signs:
        props = sign["properties"]
        cand_id = props["traceback"]["candidate_id"]
        obj_val = props.get("mapillary_mtsd_value", "")

        class_name = ""
        fallback_cid = 0
        if obj_val in mapping_dict:
            class_name = mapping_dict[obj_val].get("class_name", "")
            fallback_cid = mapping_dict[obj_val].get("class_id", 0)
        elif props.get("sign_type_id") is not None and props.get("sign_type_id") != 999:
            fallback_cid = props.get("sign_type_id", 0)

        sign_type_id = resolve_db_sign_type_id(class_name, fallback_cid)
        alt_sign_type_id = 1 if sign_type_id != 1 else 2
        verified_at = props.get("last_verified_at") or "2025-01-01T00:00:00Z"

        # Choose 3-4 distinct reviewers weighted by tier
        num_votes = random.choice([3, 3, 3, 4])
        selected_reviewers = set()
        while len(selected_reviewers) < num_votes:
            picked = random.choices(reviewers, weights=reviewer_weights, k=1)[0]
            selected_reviewers.add(picked["uuid"])

        weighted_sum = 0.0
        weight_total = 0.0

        for r_uuid in selected_reviewers:
            rev_info = next(x for x in reviewers if x["uuid"] == r_uuid)
            w = rev_info["base_rel"]
            # 92% chance +1 (Approve), 8% chance -1 (Disagree/Re-check)
            is_approve = random.random() < 0.92
            vote_val = 1 if is_approve else -1
            suggested_tid = sign_type_id if is_approve else alt_sign_type_id

            weighted_sum += w * vote_val
            weight_total += w

            review_rows.append(
                f"  ('{cand_id}', '{r_uuid}', {vote_val}, {suggested_tid}, {escape_sql_str(verified_at)})"
            )

        # Consensus score normalized to [0, 1]
        score = max(0.72, min(0.98, (weighted_sum / weight_total + 1.0) / 2.0))
        consensus_log_rows.append(
            f"  ('{cand_id}', 1, {score:.4f}, 'APPROVED', {sign_type_id}, {escape_sql_str(verified_at)})"
        )

    append_batched_insert(
        lines,
        '"candidate_reviews" ("candidate_id", "reviewer_id", "vote", "suggested_sign_type_id", "reviewed_at")',
        review_rows,
        batch_size=1000,
    )

    lines.append("-- 7. Seed Consensus Logs for Approved Candidates")
    append_batched_insert(
        lines,
        '"consensus_logs" ("candidate_id", "config_id", "calculated_score", "final_outcome", "verified_sign_type_id", "executed_at")',
        consensus_log_rows,
        batch_size=1000,
    )

    lines.extend([
        "-- 8. Update Reviewer Profile Stats and Wallet Credits from Actual Cast Votes",
        "UPDATE \"reviewer_profiles\" rp",
        "SET",
        "  total_reviews = sub.cnt,",
        "  correct_reviews = sub.cnt,",
        "  accuracy_rate = 1.0,",
        "  current_streak = LEAST(sub.cnt, 15),",
        "  last_reviewed_at = now()",
        "FROM (",
        "  SELECT reviewer_id, count(*) as cnt",
        "  FROM candidate_reviews",
        "  GROUP BY reviewer_id",
        ") sub",
        "WHERE rp.user_id = sub.reviewer_id;",
        "",
        "UPDATE \"wallets\" w",
        "SET balance = rp.total_reviews * 10,",
        "    updated_at = now()",
        "FROM \"reviewer_profiles\" rp",
        "WHERE w.user_id = rp.user_id;",
        "",
    ])

    with open(MIGRATION_010_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Generated {MIGRATION_010_PATH} successfully ({len(reviewers)} reviewers, {len(review_rows)} reviews, {len(consensus_log_rows)} consensus logs).")


def main():
    generate_migration_008()
    generate_migration_009()
    generate_migration_010()


if __name__ == "__main__":
    main()
