import os
import uuid
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException, status
from loguru import logger
from src.core.detector import SignDetector
from src.core.gpx_parser import extract_gps_from_video, parse_gpx_points, extract_exif_from_image
from src.pipelines.image_pipeline import run_image_pipeline
from src.pipelines.video_pipeline import run_video_pipeline
from src.pipelines.export_pipeline import run_export_pipeline, export_lock, export_tasks
from src.api.schemas import (
    ImageProcessingResponse,
    VideoProcessingResponse,
    ModelExportRequest,
    ModelExportResponse,
    ModelExportStatusResponse,
    ModelListResponse,
    ModelFileInfo,
)
from src.infrastructure.health import check_exiftool, check_minio, check_ffprobe

router = APIRouter()

_detector_instance = None

def get_detector():
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = SignDetector()
    return _detector_instance


def _format_size_human(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


@router.post("/api/models/export", response_model=ModelExportResponse, status_code=status.HTTP_202_ACCEPTED)
async def export_model_endpoint(
    request: ModelExportRequest,
    background_tasks: BackgroundTasks,
):
    logger.info(f"Received model export request for: {request.weights_name}, format: {request.format}")

    base_dir = Path(__file__).resolve().parent.parent.parent
    if os.path.isabs(request.weights_name):
        weights_path = Path(request.weights_name)
    else:
        weights_path = base_dir / "weights" / request.weights_name

    if not weights_path.exists():
        logger.warning(f"Weights file not found: {weights_path}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model weights file '{request.weights_name}' not found in weights directory.",
        )

    if export_lock.locked():
        logger.warning("Export request rejected: another model export task is currently in progress.")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Another model export process is currently running on the server. Please wait until it completes to protect system memory.",
        )

    await export_lock.acquire()
    task_id = f"export-{uuid.uuid4().hex[:8]}"

    background_tasks.add_task(
        run_export_pipeline,
        task_id=task_id,
        weights_path=str(weights_path),
        export_format=request.format,
        half=request.half,
        device=request.device,
        workspace=request.workspace,
        dynamic=request.dynamic,
        opset=request.opset,
        callback_url=request.callback_url,
    )

    logger.debug(f"Queued model export task {task_id} for {weights_path.name}")
    return {
        "task_id": task_id,
        "status": "PROCESSING",
        "message": f"Model export job queued for '{weights_path.name}' ({request.format}).",
    }


@router.get("/api/models/export/{task_id}", response_model=ModelExportStatusResponse)
async def get_export_task_status(task_id: str):
    if task_id not in export_tasks:
        logger.warning(f"Export task ID not found: {task_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Export task '{task_id}' not found.",
        )

    task_data = export_tasks[task_id]
    return {
        "task_id": task_data["task_id"],
        "status": task_data["status"],
        "weights_name": task_data["weights_name"],
        "format": task_data["format"],
        "exported_files": task_data.get("exported_files", []),
        "error": task_data.get("error"),
        "duration_seconds": task_data.get("duration_seconds"),
    }


@router.get("/api/models", response_model=ModelListResponse)
async def list_available_models():
    base_dir = Path(__file__).resolve().parent.parent.parent
    weights_dir = base_dir / "weights"

    models: List[ModelFileInfo] = []
    if weights_dir.exists() and weights_dir.is_dir():
        for file_path in sorted(weights_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in [".pt", ".onnx", ".engine"]:
                stat = file_path.stat()
                mod_time = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                fmt = file_path.suffix.lstrip(".").lower()
                models.append(
                    ModelFileInfo(
                        filename=file_path.name,
                        format=fmt,
                        size_bytes=stat.st_size,
                        size_human=_format_size_human(stat.st_size),
                        modified_at=mod_time,
                    )
                )

    logger.debug(f"Retrieved {len(models)} model file(s) from weights directory.")
    return {
        "total": len(models),
        "models": models,
    }


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

@router.get("/health")
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
