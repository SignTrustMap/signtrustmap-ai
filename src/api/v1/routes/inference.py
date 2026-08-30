import os
import uuid
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks
from loguru import logger
from src.core.config_manager import config_manager
from src.core.model_manager import model_manager
from src.pipelines.image_pipeline import run_image_pipeline
from src.pipelines.video_pipeline import run_video_pipeline
from src.api.schemas import ImageProcessingResponse, VideoProcessingResponse

router = APIRouter(tags=["Inference"])


@router.post("/image", response_model=ImageProcessingResponse)
async def process_image(
    background_task: BackgroundTasks,
    image: UploadFile = File(...),
    task_id: str = Form(...),
    callback_url: str = Form(...),
    conf: Optional[float] = Form(None),
    device: Optional[int] = Form(None),
):
    logger.info(f"Received image inference request for task ID {task_id}. Image: {image.filename}, Callback: {callback_url}")
    temp_dir = tempfile.gettempdir()
    temp_image_path = os.path.join(temp_dir, f"img_{uuid.uuid4().hex}_{image.filename}")

    with open(temp_image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    effective_conf = conf if conf is not None else config_manager.default_detector_conf
    effective_device = device if device is not None else config_manager.default_device

    logger.debug(f"Queueing YOLO background inference task ID: {task_id} (conf={effective_conf}, device={effective_device})")

    detector = model_manager.get_detector()
    classifier = model_manager.get_classifier()
    background_task.add_task(
        run_image_pipeline,
        task_id=task_id,
        callback_url=callback_url,
        detector=detector,
        image_path=temp_image_path,
        classifier=classifier,
        conf=effective_conf,
        device=effective_device,
    )

    return {
        "task_id": task_id,
        "status": "PROCESSING",
        "message": "Image accepted. YOLO inference started.",
    }


@router.post("/video", response_model=VideoProcessingResponse)
async def process_video(
    background_task: BackgroundTasks,
    video: UploadFile = File(...),
    task_id: str = Form(...),
    callback_url: str = Form(...),
    conf: Optional[float] = Form(None),
    device: Optional[int] = Form(None),
):
    logger.info(f"Received video inference request for task ID {task_id}. Video: {video.filename}, Callback: {callback_url}")
    temp_dir = tempfile.gettempdir()
    video_suffix = os.path.splitext(video.filename)[1]
    temp_video_path = os.path.join(temp_dir, f"vid_{uuid.uuid4().hex}{video_suffix}")

    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    effective_conf = conf if conf is not None else config_manager.default_detector_conf
    effective_device = device if device is not None else config_manager.default_device

    logger.debug(f"Queueing YOLO background video tracking task ID: {task_id} (conf={effective_conf}, device={effective_device})")

    base_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
    static_dir = os.path.join(str(base_dir), "static", "extracted_frames")
    os.makedirs(static_dir, exist_ok=True)

    detector = model_manager.get_detector()
    classifier = model_manager.get_classifier()
    background_task.add_task(
        run_video_pipeline,
        task_id=task_id,
        callback_url=callback_url,
        detector=detector,
        video_path=temp_video_path,
        static_dir=static_dir,
        classifier=classifier,
        conf=effective_conf,
        device=effective_device,
    )

    return {
        "task_id": task_id,
        "status": "PROCESSING",
        "message": "Video accepted. YOLO inference started.",
    }
