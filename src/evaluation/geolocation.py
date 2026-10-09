import math
import xml.etree.ElementTree as ET
from pathlib import Path
from loguru import logger
import numpy as np


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)
    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2.0) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0) ** 2
    return 2.0 * radius * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def great_circle_destination(lat: float, lon: float, distance_m: float, bearing_deg: float) -> tuple[float, float]:
    radius = 6371000.0
    delta = distance_m / radius
    phi1 = math.radians(lat)
    lambda1 = math.radians(lon)
    theta = math.radians(bearing_deg)

    sin_phi2 = math.sin(phi1) * math.cos(delta) + math.cos(phi1) * math.sin(delta) * math.cos(theta)
    sin_phi2 = max(-1.0, min(1.0, sin_phi2))
    phi2 = math.asin(sin_phi2)

    y = math.sin(theta) * math.sin(delta) * math.cos(phi1)
    x = math.cos(delta) - math.sin(phi1) * math.sin(phi2)
    lambda2 = lambda1 + math.atan2(y, x)

    out_lat = math.degrees(phi2)
    out_lon = (math.degrees(lambda2) + 540.0) % 360.0 - 180.0
    return out_lat, out_lon


def parse_artsv2_xml(xml_path: str) -> dict | None:
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        loc = root.find("Location")
        if loc is None:
            return None
        c_lat = float(loc.find("Latitude").text)
        c_lon = float(loc.find("Longitude").text)
        c_dom_elem = loc.find("Cam_DOM")
        c_dom = float(c_dom_elem.text) if c_dom_elem is not None and c_dom_elem.text else 0.0

        filename = root.find("filename").text if root.find("filename") is not None else Path(xml_path).stem + ".jpg"

        objects = []
        for obj in root.findall("object"):
            s_id_elem = obj.find("id")
            s_name_elem = obj.find("name")
            s_loc = obj.find("location")
            bnd = obj.find("bndbox")
            if s_id_elem is None or s_loc is None or bnd is None:
                continue

            sign_id = s_id_elem.text
            sign_name = s_name_elem.text if s_name_elem is not None else "UNKNOWN"
            gt_lat = float(s_loc.find("latitude").text)
            gt_lon = float(s_loc.find("longitude").text)
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)

            objects.append({
                "sign_id": sign_id,
                "name": sign_name,
                "gt_lat": gt_lat,
                "gt_lon": gt_lon,
                "box_xyxy": [xmin, ymin, xmax, ymax],
            })

        return {
            "filename": filename,
            "c_lat": c_lat,
            "c_lon": c_lon,
            "c_dom": c_dom,
            "objects": objects,
        }
    except Exception as exc:
        logger.error(f"Error parsing XML {xml_path}: {exc}")
        return None


def run_capstone_triangulation(
    observations: list[dict],
    focal_px: float,
    image_width: float = 1920.0,
    enforce_height_filter: bool = False,
) -> tuple[float | None, float | None, float | None, float | None]:
    if len(observations) < 2:
        return None, None, None, None

    ref_lat = observations[0]["c_lat"]
    ref_lon = observations[0]["c_lon"]
    deg_to_meter_lon = 111320.0 * math.cos(math.radians(ref_lat))
    deg_to_meter_lat = 111320.0

    as_list, bs_list, cs_list = [], [], []
    for obs in observations:
        box = obs["box_xyxy"]
        x_center = (box[0] + box[2]) / 2.0
        bearing_offset_deg = math.degrees(math.atan((x_center - image_width / 2.0) / focal_px))
        sign_bearing = (obs["heading"] + bearing_offset_deg) % 360.0

        cx = (obs["c_lon"] - ref_lon) * deg_to_meter_lon
        cy = (obs["c_lat"] - ref_lat) * deg_to_meter_lat

        u = math.sin(math.radians(sign_bearing))
        v = math.cos(math.radians(sign_bearing))

        as_list.append(v)
        bs_list.append(u)
        cs_list.append(v * cx - u * cy)

    m_mat = np.array([[a, -b] for a, b in zip(as_list, bs_list)])
    c_vec = np.array(cs_list)

    try:
        res, _, _, _ = np.linalg.lstsq(m_mat, c_vec, rcond=None)
        sx, sy = float(res[0]), float(res[1])

        temp_heights = []
        temp_dists = []
        for obs in observations:
            cx = (obs["c_lon"] - ref_lon) * deg_to_meter_lon
            cy = (obs["c_lat"] - ref_lat) * deg_to_meter_lat
            dist = math.sqrt((sx - cx) ** 2 + (sy - cy) ** 2)
            h_pixel = max(1.0, obs["box_xyxy"][3] - obs["box_xyxy"][1])
            h_physical = (dist * h_pixel) / focal_px
            temp_heights.append(h_physical)
            temp_dists.append(dist)

        avg_height = float(np.mean(temp_heights))
        avg_dist = float(np.mean(temp_dists))

        if enforce_height_filter and not (0.3 <= avg_height <= 1.2):
            return None, None, None, None

        est_lat = ref_lat + (sy / deg_to_meter_lat)
        est_lon = ref_lon + (sx / deg_to_meter_lon)
        return est_lat, est_lon, avg_dist, avg_height
    except Exception:
        return None, None, None, None


def run_capstone_fallback(
    obs: dict,
    focal_px: float,
    image_width: float = 1920.0,
    assumed_height_m: float = 0.6,
) -> tuple[float, float, float]:
    ref_lat = obs["c_lat"]
    ref_lon = obs["c_lon"]
    deg_to_meter_lon = 111320.0 * math.cos(math.radians(ref_lat))
    deg_to_meter_lat = 111320.0

    box = obs["box_xyxy"]
    x_center = (box[0] + box[2]) / 2.0
    bearing_offset_deg = math.degrees(math.atan((x_center - image_width / 2.0) / focal_px))
    sign_bearing = (obs["heading"] + bearing_offset_deg) % 360.0

    h_pixel = max(1.0, box[3] - box[1])
    dist = (focal_px * assumed_height_m) / h_pixel

    sx_fallback = dist * math.sin(math.radians(sign_bearing))
    sy_fallback = dist * math.cos(math.radians(sign_bearing))

    est_lat = ref_lat + (sy_fallback / deg_to_meter_lat)
    est_lon = ref_lon + (sx_fallback / deg_to_meter_lon)
    return est_lat, est_lon, dist


def run_final_optimized_pipeline(
    observations: list[dict],
    focal_px: float,
    catalog_height_m: float,
    image_width: float = 1920.0,
) -> tuple[float, float]:
    obs_sorted = sorted(observations, key=lambda item: item["box_xyxy"][3] - item["box_xyxy"][1], reverse=True)
    best = obs_sorted[0]
    lat, lon, _ = run_capstone_fallback(best, focal_px, image_width, assumed_height_m=catalog_height_m)
    return lat, lon


def assign_adaptive_gps_headings(obs_list: list[dict], min_baseline_m: float) -> list[dict] | None:
    n = len(obs_list)
    if n < 2:
        return None

    headings = [None] * n
    for i in range(n):
        pi = (obs_list[i]["c_lat"], obs_list[i]["c_lon"])
        for j in range(i + 1, n):
            pj = (obs_list[j]["c_lat"], obs_list[j]["c_lon"])
            if haversine(pi[0], pi[1], pj[0], pj[1]) >= min_baseline_m:
                headings[i] = calculate_bearing(pi[0], pi[1], pj[0], pj[1])
                break

    for i in range(n - 1, -1, -1):
        if headings[i] is None:
            pi = (obs_list[i]["c_lat"], obs_list[i]["c_lon"])
            for j in range(i - 1, -1, -1):
                pj = (obs_list[j]["c_lat"], obs_list[j]["c_lon"])
                if haversine(pj[0], pj[1], pi[0], pi[1]) >= min_baseline_m:
                    headings[i] = calculate_bearing(pj[0], pj[1], pi[0], pi[1])
                    break

    for i in range(n):
        if headings[i] is None:
            for j in range(n):
                if headings[j] is not None:
                    headings[i] = headings[j]
                    break

    if any(h is None for h in headings):
        return None

    result = []
    for i in range(n):
        result.append({**obs_list[i], "heading": headings[i]})
    return result


def calculate_metrics(errors: list[float]) -> dict:
    if not errors:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0, "le_2m": 0.0, "le_3m": 0.0, "le_5m": 0.0}
    arr = np.array(errors)
    return {
        "count": len(errors),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "le_2m": float(np.mean(arr <= 2.0) * 100.0),
        "le_3m": float(np.mean(arr <= 3.0) * 100.0),
        "le_5m": float(np.mean(arr <= 5.0) * 100.0),
    }
