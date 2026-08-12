import os
import uuid
import shutil
import tempfile
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from loguru import logger
from src.core.detector import SignDetector
from src.core.gpx_parser import extract_gps_from_video, parse_gpx_points, extract_exif_from_image
from src.pipelines.image_pipeline import run_image_pipeline
from src.pipelines.video_pipeline import run_video_pipeline
from src.api.schemas import ImageProcessingResponse, VideoProcessingResponse
from src.infrastructure.health import check_exiftool, check_minio, check_ffprobe

router = APIRouter()

_detector_instance = None

def get_detector():
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = SignDetector()
    return _detector_instance

@router.post("/api/process/image", response_model=ImageProcessingResponse)
async def process_image(
    background_task: BackgroundTasks,
    image: UploadFile = File(...),
    task_id: str = Form(...),
    callback_url: str = Form(...),
    fov: float = Form(90.0),
    conf: float = Form(0.6)
):
    logger.info(f"Received image processing request for task ID {task_id}. Image: {image.filename}, Callback: {callback_url}")
    temp_dir = tempfile.gettempdir()
    temp_image_path = os.path.join(temp_dir, f"img_{uuid.uuid4().hex}_{image.filename}")
    
    with open(temp_image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
        
    metadata = extract_exif_from_image(temp_image_path)
    latitude = metadata.get("latitude")
    longitude = metadata.get("longitude")
    timestamp = metadata.get("timestamp")
    camera_bearing = metadata.get("camera_bearing", 0.0)
    
    if latitude is None or longitude is None or timestamp is None:
        logger.warning(f"Rejecting image request for task ID {task_id}: Missing GPS telemetry in EXIF metadata.")
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        raise HTTPException(
            status_code=400,
            detail="Missing embedded GPS telemetry (latitude, longitude, timestamp) in the image EXIF metadata."
        )
        
    logger.debug(f"Metadata validation successful. Queueing background task ID: {task_id}")
    
    detector = get_detector()
    background_task.add_task(
        run_image_pipeline,
        task_id=task_id,
        callback_url=callback_url,
        detector=detector,
        image_path=temp_image_path,
        timestamp=timestamp,
        latitude=latitude,
        longitude=longitude,
        camera_bearing=camera_bearing,
        fov=fov,
        conf=conf
    )
    
    return {
        "task_id": task_id,
        "status": "PROCESSING",
        "message": "Image accepted. Processing started."
    }

@router.post("/api/process/video", response_model=VideoProcessingResponse)
async def process_video(
    background_task: BackgroundTasks,
    video: UploadFile = File(...),
    task_id: str = Form(...),
    callback_url: str = Form(...),
    gpx: Optional[UploadFile] = File(None),
    fov: float = Form(90.0),
    conf: float = Form(0.6),
    device: int = Form(0)
):
    logger.info(f"Received video processing request for task ID {task_id}. Video: {video.filename}, Callback: {callback_url}")
    temp_dir = tempfile.gettempdir()
    video_suffix = os.path.splitext(video.filename)[1]
    temp_video_path = os.path.join(temp_dir, f"vid_{uuid.uuid4().hex}{video_suffix}")
    
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
        
    temp_gpx_path = None
    if gpx is not None:
        temp_gpx_path = os.path.join(temp_dir, f"gpx_{uuid.uuid4().hex}_{gpx.filename}")
        with open(temp_gpx_path, "wb") as buffer:
            shutil.copyfileobj(gpx.file, buffer)
            
    gps_points = []
    if temp_gpx_path:
        gps_points = parse_gpx_points(temp_gpx_path)
    else:
        gps_points = extract_gps_from_video(temp_video_path)
        
    if not gps_points:
        logger.warning(f"Rejecting video request for task ID {task_id}: No GPS telemetry found in video and no GPX file provided.")
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if temp_gpx_path and os.path.exists(temp_gpx_path):
            os.remove(temp_gpx_path)
        raise HTTPException(
            status_code=400,
            detail="No GPS telemetry found in the request. Upload a GPX file or provide a video with embedded GPS track."
        )
        
    logger.debug(f"Metadata validation successful. Queueing background task ID: {task_id}")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    static_dir = os.path.join(base_dir, "static", "extracted_frames")
    os.makedirs(static_dir, exist_ok=True)
    
    detector = get_detector()
    background_task.add_task(
        run_video_pipeline,
        task_id=task_id,
        callback_url=callback_url,
        detector=detector,
        video_path=temp_video_path,
        gpx_path=temp_gpx_path,
        static_dir=static_dir,
        fov=fov,
        conf=conf,
        device=device
    )
    
    return {
        "task_id": task_id,
        "status": "PROCESSING",
        "message": "Video accepted. Processing started."
    }

@router.get("/")
async def health_check():
    logger.info("Executing API dependency health check.")
    
    exif = check_exiftool()
    ffprobe = check_ffprobe()
    minio = check_minio()
    
    overall_status = "healthy"
    if exif["status"] == "unhealthy" or ffprobe["status"] == "unhealthy":
        overall_status = "unhealthy"
        
    logger.debug(f"Health check status: {overall_status} (Exiftool: {exif['status']}, FFprobe: {ffprobe['status']}, MinIO: {minio['status']})")
    
    return {
        "status": overall_status,
        "dependencies": {
            "exiftool": exif,
            "ffprobe": ffprobe,
            "minio": minio
        }
    }
