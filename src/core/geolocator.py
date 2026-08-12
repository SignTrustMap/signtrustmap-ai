import math
import numpy as np
from loguru import logger

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    d_lon = math.radians(lon2 - lon1)
    
    y = math.sin(d_lon) * math.cos(lat2_r)
    x = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(d_lon)
    
    bearing = math.atan2(y, x)
    return math.degrees(bearing) % 360

def geolocate_sign_group(valid_frames, fov=90.0):
    if not valid_frames:
        return None, None, None, None
        
    ref_lat = valid_frames[0]["camera_lat"]
    ref_lon = valid_frames[0]["camera_lon"]
    
    deg_to_meter_lon = 111320.0 * math.cos(math.radians(ref_lat))
    deg_to_meter_lat = 111320.0
    
    f = 640.0 / (2.0 * math.tan(math.radians(fov / 2.0)))
    
    As = []
    Bs = []
    Cs = []
    
    for det in valid_frames:
        c_lat = det["camera_lat"]
        c_lon = det["camera_lon"]
        c_bearing = det["camera_bearing"]
        box = det["box_xyxy"]
        
        x_center = (box[0] + box[2]) / 2.0
        bearing_offset_deg = math.degrees(math.atan((x_center - 320.0) / f))
        sign_bearing = (c_bearing + bearing_offset_deg) % 360.0
        
        cx = (c_lon - ref_lon) * deg_to_meter_lon
        cy = (c_lat - ref_lat) * deg_to_meter_lat
        
        u = math.sin(math.radians(sign_bearing))
        v = math.cos(math.radians(sign_bearing))
        
        As.append(v)
        Bs.append(u)
        Cs.append(v * cx - u * cy)
        
    use_triangulation = False
    sx, sy = 0.0, 0.0
    avg_height = 0.6
    avg_dist = 0.0
    
    if len(valid_frames) >= 2:
        try:
            M = np.array([[A, -B] for A, B in zip(As, Bs)])
            C_vec = np.array(Cs)
            res, _, _, _ = np.linalg.lstsq(M, C_vec, rcond=None)
            sx, sy = res[0], res[1]
            
            temp_heights = []
            temp_dists = []
            for det in valid_frames:
                c_lat = det["camera_lat"]
                c_lon = det["camera_lon"]
                cx = (c_lon - ref_lon) * deg_to_meter_lon
                cy = (c_lat - ref_lat) * deg_to_meter_lat
                
                dist = math.sqrt((sx - cx)**2 + (sy - cy)**2)
                box = det["box_xyxy"]
                h_pixel = box[3] - box[1]
                h_physical = (dist * h_pixel) / f
                temp_heights.append(h_physical)
                temp_dists.append(dist)
                
            avg_height = float(np.mean(temp_heights))
            avg_dist = float(np.mean(temp_dists))
            
            if 0.3 <= avg_height <= 1.2:
                use_triangulation = True
            else:
                logger.warning(f"Calculated height {avg_height:.3f}m from triangulation is outside valid boundaries [0.3, 1.2]m. Triangulation rejected.")
        except Exception as e:
            logger.warning(f"Triangulation matrix solver failed: {e}")
            use_triangulation = False
            
    if use_triangulation:
        sign_lon = ref_lon + (sx / deg_to_meter_lon)
        sign_lat = ref_lat + (sy / deg_to_meter_lat)
        logger.debug(f"Triangulation successful: lat={sign_lat:.8f}, lon={sign_lon:.8f}, height={avg_height:.3f}m, distance={avg_dist:.2f}m")
        return round(sign_lat, 8), round(sign_lon, 8), round(avg_dist, 2), round(avg_height, 3)
    else:
        fallback_lats = []
        fallback_lons = []
        fallback_dists = []
        
        for det in valid_frames:
            c_lat = det["camera_lat"]
            c_lon = det["camera_lon"]
            c_bearing = det["camera_bearing"]
            box = det["box_xyxy"]
            
            x_center = (box[0] + box[2]) / 2.0
            bearing_offset_deg = math.degrees(math.atan((x_center - 320.0) / f))
            sign_bearing = (c_bearing + bearing_offset_deg) % 360.0
            
            h_pixel = box[3] - box[1]
            dist = (f * 0.6) / h_pixel
            
            cx = (c_lon - ref_lon) * deg_to_meter_lon
            cy = (c_lat - ref_lat) * deg_to_meter_lat
            
            sx_fallback = cx + dist * math.sin(math.radians(sign_bearing))
            sy_fallback = cy + dist * math.cos(math.radians(sign_bearing))
            
            fallback_lons.append(ref_lon + (sx_fallback / deg_to_meter_lon))
            fallback_lats.append(ref_lat + (sy_fallback / deg_to_meter_lat))
            fallback_dists.append(dist)
            
        sign_lat = float(np.mean(fallback_lats))
        sign_lon = float(np.mean(fallback_lons))
        avg_dist = float(np.mean(fallback_dists))
        
        logger.warning(f"Using fallback geolocation logic (assumed height 0.6m). Result: lat={sign_lat:.8f}, lon={sign_lon:.8f}, distance={avg_dist:.2f}m")
        return round(sign_lat, 8), round(sign_lon, 8), round(avg_dist, 2), 0.6
