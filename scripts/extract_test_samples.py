import argparse
import bisect
from datetime import datetime, timedelta, timezone
import gc
import json
import math
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional
import xml.etree.ElementTree as ET

import cv2
from mapillary_tools import telemetry
from mapillary_tools.camm import camm_builder, camm_parser
from mapillary_tools.mp4 import simple_mp4_builder
import torch
from loguru import logger
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract video frames and clips containing multiple large, clear traffic signs for testing."
    )
    parser.add_argument(
        "--video",
        type=str,
        default="data/GX011183.MP4",
        help="Path to input video file",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/detectors/yolo26n_vtsdb_mtsd_dfg.engine"
        if os.path.exists("weights/detectors/yolo26n_vtsdb_mtsd_dfg.engine")
        else "weights/detectors/yolo26n_vtsdb_mtsd_dfg.pt",
        help="Path to YOLO detector weights",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/test_samples",
        help="Directory to save extracted samples and metadata",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of best samples to extract",
    )
    parser.add_argument(
        "--min-signs",
        type=int,
        default=2,
        help="Minimum number of signs in frame to be considered priority",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.45,
        help="Minimum confidence threshold for detected signs",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=800.0,
        help="Minimum bounding box area in pixels",
    )
    parser.add_argument(
        "--min-interval",
        type=int,
        default=60,
        help="Minimum frame interval between selected samples to ensure diversity",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=5,
        help="Frame evaluation stride",
    )
    parser.add_argument(
        "--max-scan-frames",
        type=int,
        default=0,
        help="Maximum number of frames to scan (0 for entire video)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device for inference (e.g. '0' or 'cpu')",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default=None,
        help="Output resolution scaling (e.g. '1920:1080', '1920:-2', or 'native')",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        default=True,
        help="Save extracted frame images",
    )
    parser.add_argument(
        "--save-annotated",
        action="store_true",
        default=True,
        help="Save visualized frames with bounding boxes drawn",
    )
    parser.add_argument(
        "--save-clips",
        action="store_true",
        default=True,
        help="Save video clips around sign events",
    )
    parser.add_argument(
        "--clip-duration",
        type=float,
        default=7.0,
        help="Duration in seconds of each extracted video clip",
    )
    parser.add_argument(
        "--clip-lead-ratio",
        type=float,
        default=0.6,
        help="Proportion of clip duration placed before the event timestamp",
    )
    parser.add_argument(
        "--from-metadata",
        type=str,
        default=None,
        help="Path to existing metadata JSON to extract clips/frames without re-scanning",
    )
    parser.add_argument(
        "--gpx",
        type=str,
        default=None,
        help="Path to matching GPX file (auto-detected if None)",
    )
    parser.add_argument(
        "--hfov",
        type=float,
        default=95.8,
        help="Horizontal FOV in degrees for focal length calculation",
    )
    parser.add_argument(
        "--embed-metadata",
        action="store_true",
        default=True,
        help="Embed EXIF GPS/Intrinsics to images and CAMM to video clips",
    )
    return parser.parse_args()


def calculate_frame_score(detections: List[Dict[str, Any]]) -> float:
    base_score = sum(d["confidence"] * math.sqrt(d["box_area"]) for d in detections)
    sign_count_factor = 1.0 + 0.5 * (len(detections) - 1)
    return round(base_score * sign_count_factor, 2)


def draw_detections(
    image: Any,
    detections: List[Dict[str, Any]],
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> Any:
    annotated = image.copy()
    font_scale = max(0.6, 0.6 * scale_x)
    thickness = max(1, int(round(1.5 * scale_x)))
    box_thickness = max(2, int(round(2 * scale_x)))

    for det in detections:
        raw_box = det["box_xyxy"]
        x1 = int(round(raw_box[0] * scale_x))
        y1 = int(round(raw_box[1] * scale_y))
        x2 = int(round(raw_box[2] * scale_x))
        y2 = int(round(raw_box[3] * scale_y))
        w = x2 - x1
        h = y2 - y1
        label = f"{det['class_name']} {det['confidence']:.2f} ({w}x{h})"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), baseline = cv2.getTextSize(label, font_face, font_scale, thickness)
        text_y = max(y1 - 5, text_h + 5)
        cv2.rectangle(
            annotated,
            (x1, text_y - text_h - 4),
            (x1 + text_w + 4, text_y + baseline),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 2, text_y - 2),
            font_face,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
    return annotated


def get_video_creation_time(video_path: str) -> Optional[datetime]:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        logger.error(f"Failed to probe video: {video_path}")
        return None

    try:
        data = json.loads(res.stdout)
    except Exception as exc:
        logger.error(f"Failed to parse ffprobe json: {exc}")
        return None

    fmt_tags = data.get("format", {}).get("tags", {})
    t_str = fmt_tags.get("creation_time")
    if not t_str:
        for s in data.get("streams", []):
            t_str = s.get("tags", {}).get("creation_time")
            if t_str:
                break

    if not t_str:
        return None

    if t_str.endswith("Z"):
        t_str = t_str[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(t_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.replace(tzinfo=None)
    except Exception as exc:
        logger.error(f"Failed to parse creation_time '{t_str}': {exc}")
        return None


def find_matching_gpx(
    video_path: str,
    video_creation_time: Optional[datetime],
    gpx_override: Optional[str] = None,
) -> Optional[str]:
    if gpx_override and os.path.exists(gpx_override):
        return gpx_override

    candidate_dirs = []
    video_dir = os.path.dirname(os.path.abspath(video_path))
    candidate_dirs.append(video_dir)

    real_path = os.path.realpath(video_path)
    real_dir = os.path.dirname(real_path)
    if real_dir not in candidate_dirs:
        candidate_dirs.append(real_dir)

    base_name = os.path.basename(real_dir)
    gopro_light_dir = os.path.join("/media/taiduc_orico_ssd/capstone/data/gopro_light", base_name)
    if os.path.exists(gopro_light_dir) and gopro_light_dir not in candidate_dirs:
        candidate_dirs.append(gopro_light_dir)

    all_gpx = []
    for d in candidate_dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.lower().endswith(".gpx"):
                    all_gpx.append(os.path.join(d, f))

    if not all_gpx:
        return None

    if len(all_gpx) == 1 or video_creation_time is None:
        return all_gpx[0]

    for gfile in all_gpx:
        try:
            tree = ET.parse(gfile)
            root = tree.getroot()
            time_elems = [e for e in root.iter() if e.tag.endswith("time") and e.text]
            if len(time_elems) >= 2:
                t0_str = time_elems[0].text
                t1_str = time_elems[-1].text
                for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
                    try:
                        t0 = datetime.strptime(t0_str, fmt)
                        t1 = datetime.strptime(t1_str, fmt)
                        if t0 <= video_creation_time <= t1:
                            return gfile
                    except ValueError:
                        continue
        except Exception:
            continue

    return all_gpx[0]


def load_gpx_points(gpx_path: str) -> List[Dict[str, Any]]:
    try:
        tree = ET.parse(gpx_path)
        root = tree.getroot()
    except Exception as exc:
        logger.error(f"Failed to parse GPX XML {gpx_path}: {exc}")
        return []

    points: List[Dict[str, Any]] = []
    for trkpt in root.iter():
        if trkpt.tag.endswith("trkpt"):
            lat = trkpt.attrib.get("lat")
            lon = trkpt.attrib.get("lon")
            if lat is None or lon is None:
                continue

            pt: Dict[str, Any] = {
                "lat": float(lat),
                "lon": float(lon),
                "ele": None,
                "time": None,
                "speed": 0.0,
            }

            for child in trkpt:
                ctag = child.tag.split("}")[-1]
                if ctag == "ele" and child.text:
                    try:
                        pt["ele"] = float(child.text)
                    except ValueError:
                        pass
                elif ctag == "time" and child.text:
                    t_str = child.text
                    for fmt in (
                        "%Y-%m-%dT%H:%M:%S.%fZ",
                        "%Y-%m-%dT%H:%M:%SZ",
                        "%Y-%m-%d %H:%M:%S",
                    ):
                        try:
                            pt["time"] = datetime.strptime(t_str, fmt)
                            break
                        except ValueError:
                            continue
                elif ctag == "extensions":
                    for ext_child in child:
                        if ext_child.tag.split("}")[-1] == "meta":
                            s_val = ext_child.attrib.get("s")
                            if s_val is not None:
                                try:
                                    pt["speed"] = float(s_val)
                                except ValueError:
                                    pass

            if pt["time"] is not None:
                points.append(pt)

    del tree, root
    gc.collect()

    points.sort(key=lambda p: p["time"])
    unique_points: List[Dict[str, Any]] = []
    seen = set()
    for p in points:
        t_ts = p["time"].timestamp()
        if t_ts not in seen:
            seen.add(t_ts)
            unique_points.append(p)

    logger.info(f"Loaded {len(unique_points)} unique GPX track points from {gpx_path}")
    return unique_points


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)
    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(delta_lambda)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def interpolate_gps(points: List[Dict[str, Any]], target_dt: datetime) -> Optional[Dict[str, Any]]:
    if not points:
        return None

    if target_dt <= points[0]["time"]:
        p = points[0]
        heading = (
            calculate_bearing(p["lat"], p["lon"], points[1]["lat"], points[1]["lon"])
            if len(points) > 1
            else 0.0
        )
        return {
            "lat": p["lat"],
            "lon": p["lon"],
            "ele": p["ele"],
            "speed_mps": p["speed"],
            "speed_kmh": p["speed"] * 3.6,
            "heading": heading,
            "time_utc": target_dt,
        }

    if target_dt >= points[-1]["time"]:
        p = points[-1]
        heading = (
            calculate_bearing(points[-2]["lat"], points[-2]["lon"], p["lat"], p["lon"])
            if len(points) > 1
            else 0.0
        )
        return {
            "lat": p["lat"],
            "lon": p["lon"],
            "ele": p["ele"],
            "speed_mps": p["speed"],
            "speed_kmh": p["speed"] * 3.6,
            "heading": heading,
            "time_utc": target_dt,
        }

    timestamps = [p["time"].timestamp() for p in points]
    target_ts = target_dt.timestamp()
    idx = bisect.bisect_right(timestamps, target_ts)
    p1 = points[idx - 1]
    p2 = points[idx]

    t1 = timestamps[idx - 1]
    t2 = timestamps[idx]
    factor = (target_ts - t1) / (t2 - t1) if t2 > t1 else 0.0

    lat = p1["lat"] + factor * (p2["lat"] - p1["lat"])
    lon = p1["lon"] + factor * (p2["lon"] - p1["lon"])
    ele = None
    if p1["ele"] is not None and p2["ele"] is not None:
        ele = p1["ele"] + factor * (p2["ele"] - p1["ele"])
    speed = p1["speed"] + factor * (p2["speed"] - p1["speed"])
    heading = calculate_bearing(p1["lat"], p1["lon"], p2["lat"], p2["lon"])

    return {
        "lat": lat,
        "lon": lon,
        "ele": ele,
        "speed_mps": speed,
        "speed_kmh": speed * 3.6,
        "heading": heading,
        "time_utc": target_dt,
    }


def compute_camera_intrinsics(
    image_width: int,
    image_height: int,
    hfov_deg: float = 95.8,
) -> Dict[str, Any]:
    focal_px = (image_width / 2.0) / math.tan(math.radians(hfov_deg / 2.0))
    focal_35mm = (36.0 / 2.0) / math.tan(math.radians(hfov_deg / 2.0))
    return {
        "focal_px": round(focal_px, 2),
        "focal_35mm": round(focal_35mm, 1),
        "hfov_deg": round(hfov_deg, 2),
        "image_width": image_width,
        "image_height": image_height,
    }


def embed_image_exif(image_path: str, gps_info: Dict[str, Any], intrinsics: Dict[str, Any]) -> bool:
    dt_utc = gps_info["time_utc"]
    dt_str = dt_utc.strftime("%Y:%m:%d %H:%M:%S")
    date_stamp = dt_utc.strftime("%Y:%m:%d")
    time_stamp = dt_utc.strftime("%H:%M:%S")

    desc_dict = {
        "focal_px": intrinsics["focal_px"],
        "hfov_deg": intrinsics["hfov_deg"],
        "heading_deg": round(gps_info["heading"], 2),
        "c_lat": round(gps_info["lat"], 7),
        "c_lon": round(gps_info["lon"], 7),
        "c_ele": round(gps_info["ele"], 2) if gps_info.get("ele") is not None else None,
        "speed_kmh": round(gps_info["speed_kmh"], 2),
    }

    lat_ref = "N" if gps_info["lat"] >= 0 else "S"
    lon_ref = "E" if gps_info["lon"] >= 0 else "W"

    cmd = [
        "exiftool",
        "-overwrite_original",
        f"-GPSLatitudeRef={lat_ref}",
        f"-GPSLatitude={abs(gps_info['lat']):.7f}",
        f"-GPSLongitudeRef={lon_ref}",
        f"-GPSLongitude={abs(gps_info['lon']):.7f}",
    ]
    if gps_info.get("ele") is not None:
        cmd.extend([
            "-GPSAltitudeRef=0",
            f"-GPSAltitude={gps_info['ele']:.2f}",
        ])
    cmd.extend([
        "-GPSImgDirectionRef=T",
        f"-GPSImgDirection={gps_info['heading']:.2f}",
        "-GPSTrackRef=T",
        f"-GPSTrack={gps_info['heading']:.2f}",
        "-GPSSpeedRef=K",
        f"-GPSSpeed={gps_info['speed_kmh']:.2f}",
        f"-GPSDateStamp={date_stamp}",
        f"-GPSTimeStamp={time_stamp}",
        f"-DateTimeOriginal={dt_str}",
        f"-FocalLengthIn35mmFormat={int(round(intrinsics['focal_35mm']))}",
        f"-ImageDescription={json.dumps(desc_dict)}",
        image_path,
    ])
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        logger.error(f"ExifTool failed on {image_path}: {res.stderr[:200]}")
        return False
    return True


def inject_camm_track(
    clip_path: str,
    gpx_points: List[Dict[str, Any]],
    start_utc: datetime,
    duration: float,
    step: float = 0.5,
) -> bool:
    if not gpx_points:
        return False

    temp_path = clip_path + ".camm_temp.mp4"
    num_steps = max(1, int(round(duration / step)))
    points = []

    for i in range(num_steps + 1):
        t_offset = min(duration, i * step)
        cur_utc = start_utc + timedelta(seconds=t_offset)
        res = interpolate_gps(gpx_points, cur_utc)
        if res is None:
            continue

        epoch_sec = cur_utc.replace(tzinfo=timezone.utc).timestamp()
        p = telemetry.CAMMGPSPoint(
            time=float(t_offset),
            lat=float(res["lat"]),
            lon=float(res["lon"]),
            alt=float(res["ele"]) if res.get("ele") is not None else 0.0,
            angle=float(res["heading"]),
            time_gps_epoch=epoch_sec,
            gps_fix_type=3,
            horizontal_accuracy=1.0,
            vertical_accuracy=1.0,
            velocity_east=0.0,
            velocity_north=0.0,
            velocity_up=0.0,
            speed_accuracy=0.1,
        )
        points.append(p)

    if not points:
        return False

    try:
        camm_info = camm_parser.CAMMInfo(make="GoPro", model="HERO", gps=points)
        camm_generator = camm_builder.camm_sample_generator2(camm_info)

        with open(clip_path, "rb") as src_fp:
            stream = simple_mp4_builder.transform_mp4(src_fp, camm_generator)
            with open(temp_path, "wb") as dst_fp:
                shutil.copyfileobj(stream, dst_fp, length=2 * 1024 * 1024)

        os.replace(temp_path, clip_path)
        return True
    except Exception as exc:
        logger.error(f"Failed to inject CAMM track to {clip_path}: {exc}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False


def crop_video_clip(
    video_path: str,
    center_time: float,
    duration: float,
    lead_ratio: float,
    output_clip_path: str,
    scale: Optional[str] = None,
    gpx_points: Optional[List[Dict[str, Any]]] = None,
    video_creation_time: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    start_time = max(0.0, center_time - duration * lead_ratio)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_time:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        video_path,
    ]

    iso_location = None
    start_utc = None
    if video_creation_time and gpx_points:
        start_utc = video_creation_time + timedelta(seconds=start_time)
        res_center = interpolate_gps(gpx_points, video_creation_time + timedelta(seconds=center_time))
        if res_center:
            ele_val = res_center["ele"] if res_center.get("ele") is not None else 0.0
            iso_location = f"{res_center['lat']:+.6f}{res_center['lon']:+.6f}{ele_val:+.1f}/"

    if iso_location:
        cmd.extend(["-metadata", f"location={iso_location}"])
    if start_utc:
        cmd.extend(["-metadata", f"creation_time={start_utc.strftime('%Y-%m-%dT%H:%M:%S.%fZ')}"])

    if scale and scale.lower() != "native":
        cmd.extend([
            "-vf", f"scale={scale}",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "20",
            "-an",
            output_clip_path,
        ])
    else:
        cmd.extend([
            "-c:v", "copy",
            "-an",
            output_clip_path,
        ])
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        logger.error(f"FFmpeg failed to crop clip {output_clip_path}: {res.stderr[:200]}")
        return None

    camm_injected = False
    if gpx_points and start_utc:
        camm_injected = inject_camm_track(
            clip_path=output_clip_path,
            gpx_points=gpx_points,
            start_utc=start_utc,
            duration=duration,
        )

    return {
        "clip_file": os.path.basename(output_clip_path),
        "start_sec": round(start_time, 3),
        "duration_sec": round(duration, 3),
        "camm_track": camm_injected,
        "location_tag": iso_location,
    }


def select_diverse_candidates(
    candidates: List[Dict[str, Any]],
    num_samples: int,
    min_interval: int,
    min_signs: int,
) -> List[Dict[str, Any]]:
    filtered = [c for c in candidates if c["sign_count"] >= min_signs]
    if len(filtered) < num_samples:
        filtered = candidates

    sorted_candidates = sorted(filtered, key=lambda c: c["score"], reverse=True)
    selected: List[Dict[str, Any]] = []

    for cand in sorted_candidates:
        if all(abs(cand["frame_idx"] - s["frame_idx"]) >= min_interval for s in selected):
            selected.append(cand)
            if len(selected) >= num_samples:
                break

    selected.sort(key=lambda c: c["frame_idx"])
    return selected


def scan_and_evaluate(args, video_path: str, total_frames: int, fps: float) -> List[Dict[str, Any]]:
    weights_path = args.weights
    if not os.path.exists(weights_path):
        logger.error(f"Weights file not found: {weights_path}")
        sys.exit(1)

    logger.info(f"Loading YOLO detector: {weights_path}")
    task_param = "detect" if weights_path.endswith(".engine") else None
    model = YOLO(weights_path, task=task_param)

    max_frames = args.max_scan_frames if args.max_scan_frames > 0 else total_frames
    logger.info(
        f"Starting memory-efficient scan (min_conf={args.min_conf}, min_area={args.min_area}, step={args.step})..."
    )

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    candidates: List[Dict[str, Any]] = []

    with torch.inference_mode():
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            logger.trace(f"Scanning frame {frame_idx}/{max_frames}")

            if frame_idx % args.step == 0:
                results = model.predict(
                    source=frame,
                    conf=args.min_conf,
                    device=args.device,
                    verbose=False,
                )

                if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                    boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    clss = results[0].boxes.cls.int().cpu().numpy()

                    valid_detections = []
                    for box, conf, cls_id in zip(boxes_xyxy, confs, clss):
                        w = float(box[2] - box[0])
                        h = float(box[3] - box[1])
                        area = w * h
                        if area >= args.min_area and conf >= args.min_conf:
                            valid_detections.append({
                                "class_id": int(cls_id),
                                "class_name": model.names[int(cls_id)],
                                "confidence": round(float(conf), 4),
                                "box_xyxy": [round(float(c), 2) for c in box],
                                "box_w": round(w, 2),
                                "box_h": round(h, 2),
                                "box_area": round(area, 2),
                            })

                    if len(valid_detections) > 0:
                        score = calculate_frame_score(valid_detections)
                        candidates.append({
                            "frame_idx": frame_idx,
                            "timestamp_sec": round(frame_idx / fps, 3),
                            "sign_count": len(valid_detections),
                            "score": score,
                            "detections": valid_detections,
                        })

                del results

            del frame
            frame_idx += 1

            if frame_idx % 100 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    cap.release()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Scan completed. Evaluated {frame_idx} frames, found {len(candidates)} candidate frames.")
    return candidates


def main():
    args = parse_args()
    frames_dir = os.path.join(args.output_dir, "frames")
    clips_dir = os.path.join(args.output_dir, "clips")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(clips_dir, exist_ok=True)

    orig_meta_w = 0
    orig_meta_h = 0

    if args.from_metadata and os.path.exists(args.from_metadata):
        logger.info(f"Loading existing metadata from: {args.from_metadata}")
        with open(args.from_metadata, "r", encoding="utf-8") as f:
            metadata_source = json.load(f)

        video_path = args.video if args.video != "data/GX011183.MP4" else metadata_source.get("video_path", args.video)
        total_frames = metadata_source.get("total_video_frames", 0)
        fps = metadata_source.get("video_fps", 24.0)
        orig_meta_w = metadata_source.get("video_width", 640)
        orig_meta_h = metadata_source.get("video_height", 360)
        selected = metadata_source.get("samples") or metadata_source.get("frames") or []
    else:
        video_path = args.video
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            sys.exit(1)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            sys.exit(1)

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_meta_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_meta_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        logger.info(
            f"Video: {video_path} ({orig_meta_w}x{orig_meta_h}, {fps:.2f} fps, {total_frames} total frames)"
        )
        candidates = scan_and_evaluate(args, video_path, total_frames, fps)
        if not candidates:
            logger.warning("No candidate frames with signs meeting criteria were found.")
            sys.exit(0)

        selected = select_diverse_candidates(
            candidates,
            num_samples=args.num_samples,
            min_interval=args.min_interval,
            min_signs=args.min_signs,
        )

    video_creation_time = get_video_creation_time(video_path)
    if video_creation_time:
        logger.info(f"Video creation time (UTC): {video_creation_time.isoformat()}")
    else:
        logger.warning("Could not determine video creation time from ffprobe tags.")

    matched_gpx = find_matching_gpx(video_path, video_creation_time, args.gpx)
    gpx_points = []
    if matched_gpx and os.path.exists(matched_gpx):
        logger.info(f"Matched GPX file: {matched_gpx}")
        gpx_points = load_gpx_points(matched_gpx)
    else:
        logger.warning(f"No matching GPX file found for video {video_path}")

    logger.info(f"Processing {len(selected)} selected test samples from: {video_path}")
    output_records = []
    cap = cv2.VideoCapture(video_path) if args.save_frames else None

    for item in selected:
        f_idx = item["frame_idx"]
        t_sec = item["timestamp_sec"]

        target_utc = (
            video_creation_time + timedelta(seconds=t_sec)
            if video_creation_time
            else None
        )
        gps_info = interpolate_gps(gpx_points, target_utc) if (gpx_points and target_utc) else None

        raw_name = None
        annotated_name = None

        if args.save_frames and cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if ret:
                cur_h, cur_w = frame.shape[:2]

                if args.scale and args.scale.lower() != "native":
                    if ":" in args.scale:
                        target_w_str, target_h_str = args.scale.split(":")
                        target_w = int(target_w_str)
                        target_h = int(round(cur_h * (target_w / cur_w))) if target_h_str.startswith("-") else int(target_h_str)
                    else:
                        target_w, target_h = 1920, 1080
                    frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                    cur_h, cur_w = frame.shape[:2]

                intrinsics = compute_camera_intrinsics(cur_w, cur_h, args.hfov)

                raw_name = f"frame_{f_idx:06d}.jpg"
                raw_path = os.path.join(frames_dir, raw_name)
                cv2.imwrite(raw_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                if gps_info and args.embed_metadata:
                    embed_image_exif(raw_path, gps_info, intrinsics)

                if args.save_annotated:
                    scale_x = cur_w / orig_meta_w if orig_meta_w > 0 else 1.0
                    scale_y = cur_h / orig_meta_h if orig_meta_h > 0 else 1.0
                    annotated_frame = draw_detections(frame, item.get("detections", []), scale_x, scale_y)
                    annotated_name = f"annotated_{f_idx:06d}.jpg"
                    annotated_path = os.path.join(frames_dir, annotated_name)
                    cv2.imwrite(annotated_path, annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    if gps_info and args.embed_metadata:
                        embed_image_exif(annotated_path, gps_info, intrinsics)
                    del annotated_frame

                del frame
            else:
                logger.error(f"Failed to read frame {f_idx} for extraction.")
        else:
            intrinsics = compute_camera_intrinsics(orig_meta_w, orig_meta_h, args.hfov)

        clip_info = None
        if args.save_clips:
            clip_name = f"clip_{f_idx:06d}.mp4"
            clip_path = os.path.join(clips_dir, clip_name)
            clip_info = crop_video_clip(
                video_path=video_path,
                center_time=t_sec,
                duration=args.clip_duration,
                lead_ratio=args.clip_lead_ratio,
                output_clip_path=clip_path,
                scale=args.scale,
                gpx_points=gpx_points,
                video_creation_time=video_creation_time,
            )

        record: Dict[str, Any] = {
            "frame_idx": f_idx,
            "timestamp_sec": t_sec,
            "score": item.get("score"),
            "sign_count": item.get("sign_count"),
            "raw_frame_file": raw_name,
            "annotated_frame_file": annotated_name,
            "clip": clip_info,
            "detections": item.get("detections", []),
        }

        if gps_info:
            record["camera_gps"] = {
                "timestamp_utc": target_utc.isoformat() if target_utc else None,
                "lat": round(gps_info["lat"], 7),
                "lon": round(gps_info["lon"], 7),
                "ele": round(gps_info["ele"], 2) if gps_info.get("ele") is not None else None,
                "heading": round(gps_info["heading"], 2),
                "speed_kmh": round(gps_info["speed_kmh"], 2),
            }
            record["camera_intrinsics"] = intrinsics

        output_records.append(record)

    if cap is not None:
        cap.release()

    meta_file = os.path.join(args.output_dir, "test_samples_metadata.json")
    metadata = {
        "video_path": video_path,
        "video_width": orig_meta_w,
        "video_height": orig_meta_h,
        "video_fps": round(fps, 2),
        "total_video_frames": total_frames,
        "extracted_samples_count": len(output_records),
        "gpx_source": matched_gpx,
        "samples": output_records,
    }

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Successfully saved {len(output_records)} test samples and metadata to {args.output_dir}")


if __name__ == "__main__":
    main()
