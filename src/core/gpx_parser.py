import os
import json
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
import numpy as np
from scipy.interpolate import CubicSpline
from loguru import logger

def parse_gpx_time(t_str):
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M:%S"):
        try:
            dt = datetime.strptime(t_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            local_dt = dt.astimezone(timezone(timedelta(hours=7)))
            return local_dt.replace(tzinfo=None)
        except ValueError:
            continue
    return None

def parse_gpx_points(gpx_path):
    logger.info(f"Parsing GPX points from file: {gpx_path}")
    points = []
    try:
        tree = ET.parse(gpx_path)
        root = tree.getroot()
        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"
            
        for trkpt in root.iter(f"{ns}trkpt"):
            lat = float(trkpt.attrib.get('lat', 0.0))
            lon = float(trkpt.attrib.get('lon', 0.0))
            ele = 0.0
            time_val = None
            speed = 0.0
            
            ele_node = trkpt.find(f"{ns}ele")
            if ele_node is not None and ele_node.text:
                ele = float(ele_node.text)
                
            time_node = trkpt.find(f"{ns}time")
            if time_node is not None and time_node.text:
                time_val = parse_gpx_time(time_node.text)
                
            ext_node = trkpt.find(f"{ns}extensions")
            if ext_node is not None:
                meta_node = ext_node.find(f"{ns}meta")
                if meta_node is not None:
                    s_val = meta_node.attrib.get('s')
                    if s_val is not None:
                        speed = float(s_val)
                        
            if time_val:
                logger.trace(f"Parsed GPX point: lat={lat}, lon={lon}, time={time_val}")
                points.append({
                    "time": time_val,
                    "lat": lat,
                    "lon": lon,
                    "ele": ele,
                    "speed": speed
                })
        logger.debug(f"Parsed {len(points)} point(s) from GPX.")
    except Exception as e:
        logger.error(f"Error parsing GPX file {gpx_path}: {e}")
    return points

def extract_gps_from_video(video_path):
    """
    Uses exiftool to extract GPS telemetry embedded in video file (GPMF).
    Reads the raw coordinate and timestamp fields output by exiftool in JSON.
    """
    logger.info(f"Extracting GPS telemetry from video file: {video_path}")
    cmd = [
        "exiftool",
        "-api", "LargeFileSupport=1",
        "-ee",
        "-json",
        "-GPSLatitude",
        "-GPSLongitude",
        "-GPSAltitude",
        "-GPSDateTime",
        video_path
    ]
    points = []
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        data = json.loads(res.stdout)
        
        for entry in data:
            lat = entry.get("GPSLatitude")
            lon = entry.get("GPSLongitude")
            alt = entry.get("GPSAltitude", 0.0)
            dt_str = entry.get("GPSDateTime")
            
            if lat is not None and lon is not None and dt_str is not None:
                try:
                    lat_val = float(lat)
                    lon_val = float(lon)
                    alt_val = float(alt)
                except ValueError:
                    continue
                    
                time_val = parse_gpx_time(dt_str)
                if time_val:
                    logger.trace(f"Extracted video GPS: lat={lat_val}, lon={lon_val}, time={time_val}")
                    points.append({
                        "time": time_val,
                        "lat": lat_val,
                        "lon": lon_val,
                        "ele": alt_val,
                        "speed": 0.0
                    })
        logger.debug(f"Extracted {len(points)} point(s) from video.")
    except Exception as e:
        logger.error(f"Error extracting GPS from video {video_path}: {e}")
    return points

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    d_lon = np.radians(lon2 - lon1)
    
    y = np.sin(d_lon) * np.cos(lat2_r)
    x = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(d_lon)
    
    bearing = np.arctan2(y, x)
    return float(np.degrees(bearing) % 360)

def get_interpolated_gps_track(points):
    if not points:
        return None
        
    logger.info(f"Building interpolated GPS track for {len(points)} points.")
    points = sorted(points, key=lambda x: x["time"])
    
    unique_points = []
    seen_timestamps = set()
    for pt in points:
        ts = pt["time"].timestamp()
        if ts not in seen_timestamps:
            seen_timestamps.add(ts)
            unique_points.append(pt)
            
    if len(unique_points) < 2:
        logger.warning("Fewer than 2 unique GPS points available. Interpolation aborted.")
        return None
        
    for i in range(len(unique_points)):
        next_idx = min(i + 2, len(unique_points) - 1)
        prev_idx = max(0, i - 2)
        if next_idx - prev_idx > 0:
            lat1, lon1 = unique_points[prev_idx]["lat"], unique_points[prev_idx]["lon"]
            lat2, lon2 = unique_points[next_idx]["lat"], unique_points[next_idx]["lon"]
            unique_points[i]["bearing"] = calculate_bearing(lat1, lon1, lat2, lon2)
        else:
            unique_points[i]["bearing"] = 0.0
            
    times = np.array([pt["time"].timestamp() for pt in unique_points])
    lats = np.array([pt["lat"] for pt in unique_points])
    lons = np.array([pt["lon"] for pt in unique_points])
    eles = np.array([pt["ele"] for pt in unique_points])
    speeds = np.array([pt["speed"] for pt in unique_points])
    
    cs_lat = CubicSpline(times, lats)
    cs_lon = CubicSpline(times, lons)
    cs_ele = CubicSpline(times, eles)
    cs_speed = CubicSpline(times, speeds)
    
    gpx_timestamps = [pt["time"].timestamp() for pt in unique_points]
    logger.debug(f"Interpolated GPS track covers bounds: {unique_points[0]['time']} to {unique_points[-1]['time']}.")
    
    return {
        "times": times,
        "lats": lats,
        "lons": lons,
        "eles": eles,
        "speeds": speeds,
        "points": unique_points,
        "gpx_timestamps": gpx_timestamps,
        "cs_lat": cs_lat,
        "cs_lon": cs_lon,
        "cs_ele": cs_ele,
        "cs_speed": cs_speed
    }

def interpolate_position(track_data, query_timestamp):
    if track_data is None:
        return None
        
    times = track_data["times"]
    if query_timestamp < times[0] - 10.0 or query_timestamp > times[-1] + 10.0:
        logger.warning(f"Query timestamp {query_timestamp} is out of GPS track bounds [{times[0]}, {times[-1]}].")
        return None
        
    lat = float(track_data["cs_lat"](query_timestamp))
    lon = float(track_data["cs_lon"](query_timestamp))
    ele = float(track_data["cs_ele"](query_timestamp))
    speed = float(track_data["cs_speed"](query_timestamp))
    
    gpx_timestamps = track_data["gpx_timestamps"]
    import bisect
    idx = bisect.bisect_left(gpx_timestamps, query_timestamp)
    if idx == 0:
        best_idx = 0
    elif idx >= len(gpx_timestamps):
        best_idx = len(gpx_timestamps) - 1
    else:
        if abs(gpx_timestamps[idx] - query_timestamp) < abs(gpx_timestamps[idx - 1] - query_timestamp):
            best_idx = idx
        else:
            best_idx = idx - 1
            
    bearing = track_data["points"][best_idx].get("bearing", 0.0)
    
    return {
        "lat": lat,
        "lon": lon,
        "ele": ele,
        "speed": speed,
        "bearing": bearing
    }

def extract_exif_from_image(image_path):
    logger.info(f"Extracting EXIF metadata from image: {image_path}")
    cmd = [
        "exiftool",
        "-json",
        "-GPSLatitude",
        "-GPSLongitude",
        "-GPSDateTime",
        "-GPSImgDirection",
        "-DateTimeOriginal",
        "-CreateDate",
        image_path
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        data = json.loads(res.stdout)
        if data and isinstance(data, list) and len(data) > 0:
            entry = data[0]
            lat = entry.get("GPSLatitude")
            lon = entry.get("GPSLongitude")
            
            lat_val = float(lat) if lat is not None else None
            lon_val = float(lon) if lon is not None else None
            
            dt_str = entry.get("DateTimeOriginal") or entry.get("CreateDate") or entry.get("GPSDateTime")
            
            bearing = entry.get("GPSImgDirection")
            bearing_val = float(bearing) if bearing is not None else 0.0
            
            return {
                "latitude": lat_val,
                "longitude": lon_val,
                "timestamp": dt_str,
                "camera_bearing": bearing_val
            }
    except Exception as e:
        logger.error(f"Error extracting EXIF from image {image_path}: {e}")
    return {}
