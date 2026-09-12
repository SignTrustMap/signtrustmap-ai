import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger


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


def main():
    parser = argparse.ArgumentParser(description="Optimized Benchmark for Capstone Sign GPS Geolocation on ARTSv2")
    parser.add_argument("--annotations-dir", type=str, default="/media/taiduc_orico_ssd/DucPT/signtrustmap/signtrustmap-ai/data/ARTSv2/Annotations")
    parser.add_argument("--catalog-path", type=str, default="/media/taiduc_orico_ssd/DucPT/signtrustmap/signtrustmap-ai/configs/sign_height_catalog.json")
    parser.add_argument("--output-dir", type=str, default="/media/taiduc_orico_ssd/DucPT/signtrustmap/signtrustmap-ai/runs/evaluation")
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--fov", type=float, default=95.8)
    parser.add_argument("--baseline", type=float, default=10.0)
    parser.add_argument("--min-box-height", type=float, default=45.0, help="Minimum observed bounding box height (px) to confirm confident localization")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_xml_files = sorted(os.listdir(args.annotations_dir))
    total_available = len(all_xml_files)
    logger.info(f"Discovered {total_available} XML annotation files in {args.annotations_dir}")

    catalog = {}
    if os.path.exists(args.catalog_path):
        with open(args.catalog_path) as f_cat:
            catalog = json.load(f_cat)
        logger.info(f"Loaded physical height catalog with {len(catalog)} entries from {args.catalog_path}")
    else:
        logger.warning(f"Catalog file not found at {args.catalog_path}, falling back to default 0.78m")
    default_sign_height = catalog.get("__default__", 0.78)

    num_to_take = int(round(total_available * args.sample_ratio)) if args.max_files is None else args.max_files
    num_to_take = max(1, min(total_available, num_to_take))
    eval_files = all_xml_files[:num_to_take]
    logger.info(f"Evaluating {len(eval_files)} files ({len(eval_files)/total_available*100:.1f}% sample of dataset)")

    image_width = 1920.0
    focal_px = (image_width / 2.0) / math.tan(math.radians(args.fov / 2.0))
    logger.debug(f"Pinhole parameters: Image=1920x1080, HFOV={args.fov} deg, Computed Focal Length={focal_px:.2f} px")

    sign_groups = defaultdict(list)
    for idx, fname in enumerate(eval_files):
        xml_path = os.path.join(args.annotations_dir, fname)
        frame_data = parse_artsv2_xml(xml_path)
        if frame_data is None:
            continue
        for obj in frame_data["objects"]:
            sign_groups[obj["sign_id"]].append({
                "frame_idx": idx,
                "filename": frame_data["filename"],
                "name": obj["name"],
                "c_lat": frame_data["c_lat"],
                "c_lon": frame_data["c_lon"],
                "c_dom": frame_data["c_dom"],
                "box_xyxy": obj["box_xyxy"],
                "gt_lat": obj["gt_lat"],
                "gt_lon": obj["gt_lon"],
            })

    logger.info(f"Parsed {len(eval_files)} frames. Found {len(sign_groups)} unique physical signs.")

    triang_gt_errors = []
    opt_gt_errors = []
    opt_filtered_gt_errors = []
    fallback_gt_errors = []

    triang_gps_errors = []
    opt_gps_errors = []
    opt_filtered_gps_errors = []
    fallback_gps_errors = []

    bearing_diffs = []
    total_multi_obs = 0
    total_gps_baseline = 0
    total_confident_signs = 0

    for sign_id, obs_list in sign_groups.items():
        if len(obs_list) < 2:
            continue
        total_multi_obs += 1
        gt_lat = obs_list[0]["gt_lat"]
        gt_lon = obs_list[0]["gt_lon"]
        sign_name = obs_list[0]["name"]
        cat_h = catalog.get(sign_name, default_sign_height)

        max_observed_box_h = max(d["box_xyxy"][3] - d["box_xyxy"][1] for d in obs_list)
        is_confident = (max_observed_box_h >= args.min_box_height)
        if is_confident:
            total_confident_signs += 1

        # 1. Mode Ground Truth Heading (Cam_DOM)
        obs_gt_heading = [{**item, "heading": item["c_dom"]} for item in obs_list]

        lat_t, lon_t, _, _ = run_capstone_triangulation(obs_gt_heading, focal_px, image_width, enforce_height_filter=False)
        if lat_t is not None:
            triang_gt_errors.append(haversine(lat_t, lon_t, gt_lat, gt_lon))

        closest_obs_gt = obs_gt_heading[-1]
        lat_fb, lon_fb, _ = run_capstone_fallback(closest_obs_gt, focal_px, image_width, assumed_height_m=0.6)
        fallback_gt_errors.append(haversine(lat_fb, lon_fb, gt_lat, gt_lon))

        lat_opt_gt, lon_opt_gt = run_final_optimized_pipeline(obs_gt_heading, focal_px, cat_h, image_width)
        err_opt_gt = haversine(lat_opt_gt, lon_opt_gt, gt_lat, gt_lon)
        opt_gt_errors.append(err_opt_gt)
        if is_confident:
            opt_filtered_gt_errors.append(err_opt_gt)

        # 2. Mode Derived GPS Bearing (Adaptive Baseline within sign observations)
        obs_gps_heading = assign_adaptive_gps_headings(obs_list, args.baseline)
        if obs_gps_heading is not None:
            total_gps_baseline += 1
            for item in obs_gps_heading:
                diff = abs((item["heading"] - item["c_dom"] + 180.0) % 360.0 - 180.0)
                bearing_diffs.append(diff)

            lat_t_gps, lon_t_gps, _, _ = run_capstone_triangulation(obs_gps_heading, focal_px, image_width, enforce_height_filter=False)
            if lat_t_gps is not None:
                triang_gps_errors.append(haversine(lat_t_gps, lon_t_gps, gt_lat, gt_lon))

            closest_obs_gps = obs_gps_heading[-1]
            lat_fb_gps, lon_fb_gps, _ = run_capstone_fallback(closest_obs_gps, focal_px, image_width, assumed_height_m=0.6)
            fallback_gps_errors.append(haversine(lat_fb_gps, lon_fb_gps, gt_lat, gt_lon))

            lat_opt_gps, lon_opt_gps = run_final_optimized_pipeline(obs_gps_heading, focal_px, cat_h, image_width)
            err_opt_gps = haversine(lat_opt_gps, lon_opt_gps, gt_lat, gt_lon)
            opt_gps_errors.append(err_opt_gps)
            if is_confident:
                opt_filtered_gps_errors.append(err_opt_gps)

    res_triang_gt = calculate_metrics(triang_gt_errors)
    res_fb_gt = calculate_metrics(fallback_gt_errors)
    res_opt_gt = calculate_metrics(opt_gt_errors)
    res_opt_filt_gt = calculate_metrics(opt_filtered_gt_errors)

    res_triang_gps = calculate_metrics(triang_gps_errors)
    res_fb_gps = calculate_metrics(fallback_gps_errors)
    res_opt_gps = calculate_metrics(opt_gps_errors)
    res_opt_filt_gps = calculate_metrics(opt_filtered_gps_errors)

    res_bearing = calculate_metrics(bearing_diffs)
    coverage_multi = (total_multi_obs / len(sign_groups) * 100.0) if sign_groups else 0.0
    coverage_gps = (total_gps_baseline / total_multi_obs * 100.0) if total_multi_obs else 0.0
    coverage_conf = (total_confident_signs / total_multi_obs * 100.0) if total_multi_obs else 0.0

    logger.info("==========================================================================================")
    logger.info(f"         FINAL OPTIMIZED BENCHMARK REPORT ({len(eval_files)} frames, {len(sign_groups)} signs)         ")
    logger.info("==========================================================================================")
    logger.info(f"Total Unique Signs: {len(sign_groups)}")
    logger.info(f"Signs with >= 2 observations: {total_multi_obs} ({coverage_multi:.1f}% coverage)")
    logger.info(f"Signs with sufficient GPS baseline (>={args.baseline}m): {total_gps_baseline} ({coverage_gps:.1f}% of multi-obs)")
    logger.info(f"Signs confident for final map (max box >= {args.min_box_height}px): {total_confident_signs} ({coverage_conf:.1f}% of multi-obs)")
    logger.info(f"Bearing Error (|GPS_bearing - Cam_DOM|): Median={res_bearing['median']:.2f} deg, Mean={res_bearing['mean']:.2f} deg, P90={res_bearing['p90']:.2f} deg")
    logger.info("------------------------------------------------------------------------------------------")
    logger.info("[MODE 1: Ground Truth Heading (Cam_DOM)]")
    logger.info(f"  1. Capstone Gốc (Pure Triangulation)       : Median = {res_triang_gt['median']:.2f} m | Mean = {res_triang_gt['mean']:.2f} m | P90 = {res_triang_gt['p90']:.2f} m")
    logger.info(f"  2. Capstone Gốc (Fallback Fixed 0.6m)      : Median = {res_fb_gt['median']:.2f} m | Mean = {res_fb_gt['mean']:.2f} m | P90 = {res_fb_gt['p90']:.2f} m")
    logger.info(f"  3. OPTIMIZED ALL (Catalog + Hybrid)        : Median = {res_opt_gt['median']:.2f} m | Mean = {res_opt_gt['mean']:.2f} m | P90 = {res_opt_gt['p90']:.2f} m (<=2m: {res_opt_gt['le_2m']:.1f}%)")
    logger.info(f"  4. OPTIMIZED TARGET (>= {args.min_box_height}px Confident)   : Median = {res_opt_filt_gt['median']:.2f} m | Mean = {res_opt_filt_gt['mean']:.2f} m | P90 = {res_opt_filt_gt['p90']:.2f} m (<=2m: {res_opt_filt_gt['le_2m']:.1f}%)")
    logger.info("------------------------------------------------------------------------------------------")
    logger.info(f"[MODE 2: Derived GPS Bearing (Adaptive Baseline >= {args.baseline}m)]")
    logger.info(f"  1. Capstone Gốc (Pure Triangulation)       : Median = {res_triang_gps['median']:.2f} m | Mean = {res_triang_gps['mean']:.2f} m | P90 = {res_triang_gps['p90']:.2f} m")
    logger.info(f"  2. Capstone Gốc (Fallback Fixed 0.6m)      : Median = {res_fb_gps['median']:.2f} m | Mean = {res_fb_gps['mean']:.2f} m | P90 = {res_fb_gps['p90']:.2f} m")
    logger.info(f"  3. OPTIMIZED ALL (Catalog + Hybrid)        : Median = {res_opt_gps['median']:.2f} m | Mean = {res_opt_gps['mean']:.2f} m | P90 = {res_opt_gps['p90']:.2f} m (<=2m: {res_opt_gps['le_2m']:.1f}%)")
    logger.info(f"  4. OPTIMIZED TARGET (>= {args.min_box_height}px Confident)   : Median = {res_opt_filt_gps['median']:.2f} m | Mean = {res_opt_filt_gps['mean']:.2f} m | P90 = {res_opt_filt_gps['p90']:.2f} m (<=2m: {res_opt_filt_gps['le_2m']:.1f}%)")
    logger.info("==========================================================================================")

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_data = {
        "timestamp": timestamp_str,
        "eval_frames_count": len(eval_files),
        "total_unique_signs": len(sign_groups),
        "multi_obs_signs": total_multi_obs,
        "gps_baseline_signs": total_gps_baseline,
        "confident_signs": total_confident_signs,
        "parameters": {
            "sample_ratio": args.sample_ratio,
            "fov_deg": args.fov,
            "focal_px": focal_px,
            "min_baseline_m": args.baseline,
            "min_box_height_px": args.min_box_height,
            "catalog_path": args.catalog_path,
        },
        "bearing_metrics": res_bearing,
        "derived_gps_heading_mode": {
            "capstone_triangulation_pure": res_triang_gps,
            "capstone_fallback_fixed_0_6m": res_fb_gps,
            "optimized_all": res_opt_gps,
            "optimized_confident_target": res_opt_filt_gps,
        },
    }

    json_report_path = os.path.join(args.output_dir, f"artsv2_optimized_{timestamp_str}.json")
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    logger.info(f"Saved JSON report to {json_report_path}")

    md_report_path = os.path.join(args.output_dir, f"artsv2_optimized_{timestamp_str}.md")
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write(f"# Final Optimized Benchmark Report ({timestamp_str})\n\n")
        f.write(f"- **Evaluated Frames**: {len(eval_files)} ({args.sample_ratio*100:.1f}% sample)\n")
        f.write(f"- **Total Unique Signs**: {len(sign_groups)}\n")
        f.write(f"- **Signs with >= 2 observations**: {total_multi_obs} ({coverage_multi:.1f}%)\n")
        f.write(f"- **Confident Signs for Map (>= {args.min_box_height}px)**: {total_confident_signs} ({coverage_conf:.1f}%)\n")
        f.write(f"- **Bearing Error (GPS vs Cam_DOM)**: Median = {res_bearing['median']:.2f}°, Mean = {res_bearing['mean']:.2f}°, P90 = {res_bearing['p90']:.2f}°\n\n")
        f.write(f"## Comparison on Derived GPS Bearing (Adaptive Baseline >= {args.baseline}m)\n\n")
        f.write("| Algorithm / Configuration | Median Error (m) | Mean Error (m) | P90 Error (m) | <=2m Rate (%) | Evaluated Signs |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **FINAL OPTIMIZED TARGET (Confident >= {args.min_box_height}px)** | **{res_opt_filt_gps['median']:.2f}** | **{res_opt_filt_gps['mean']:.2f}** | **{res_opt_filt_gps['p90']:.2f}** | **{res_opt_filt_gps['le_2m']:.1f}%** | {res_opt_filt_gps['count']} |\n")
        f.write(f"| OPTIMIZED (All Signs, including far ones) | {res_opt_gps['median']:.2f} | {res_opt_gps['mean']:.2f} | {res_opt_gps['p90']:.2f} | {res_opt_gps['le_2m']:.1f}% | {res_opt_gps['count']} |\n")
        f.write(f"| Capstone Fallback (Fixed H=0.6m) | {res_fb_gps['median']:.2f} | {res_fb_gps['mean']:.2f} | {res_fb_gps['p90']:.2f} | {res_fb_gps['le_2m']:.1f}% | {res_fb_gps['count']} |\n")
        f.write(f"| Capstone Gốc (Pure Triangulation) | {res_triang_gps['median']:.2f} | {res_triang_gps['mean']:.2f} | {res_triang_gps['p90']:.2f} | {res_triang_gps['le_2m']:.1f}% | {res_triang_gps['count']} |\n")
    logger.info(f"Saved Markdown report to {md_report_path}")


if __name__ == "__main__":
    main()
