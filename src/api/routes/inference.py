import os
import uuid
import shutil
import tempfile
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks
from loguru import logger
from src.core.detector import SignDetector
from src.pipelines.image_pipeline import run_image_pipeline
from src.pipelines.video_pipeline import run_video_pipeline
from src.api.schemas import ImageProcessingResponse, VideoProcessingResponse

router = APIRouter(tags=["Inference"])

_detector_instance = None


def get_detector():
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = SignDetector()
    return _detector_instance


@router.post("/image", response_model=ImageProcessingResponse)
async def process_image(
    background_task: BackgroundTasks,
    image: UploadFile = File(...),
    task_id: str = Form(...),
    callback_url: str = Form(...),
    conf: float = Form(0.6),
    device: int = Form(0),
):
    logger.info(f"Received image inference request for task ID {task_id}. Image: {image.filename}, Callback: {callback_url}")
    temp_dir = tempfile.gettempdir()
    temp_image_path = os.path.join(temp_dir, f"img_{uuid.uuid4().hex}_{image.filename}")

    with open(temp_image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    logger.debug(f"Image received and saved. Queueing YOLO background inference task ID: {task_id}")

    detector = get_detector()
    background_task.add_task(
        run_image_pipeline,
        task_id=task_id,
        callback_url=callback_url,
        detector=detector,
        image_path=temp_image_path,
        conf=conf,
        device=device,
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
    conf: float = Form(0.6),
    device: int = Form(0),
):
    logger.info(f"Received video inference request for task ID {task_id}. Video: {video.filename}, Callback: {callback_url}")
    temp_dir = tempfile.gettempdir()
    video_suffix = os.path.splitext(video.filename)[1]
    temp_video_path = os.path.join(temp_dir, f"vid_{uuid.uuid4().hex}{video_suffix}")

    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    logger.debug(f"Video received and saved. Queueing YOLO background inference task ID: {task_id}")

    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    static_dir = os.path.join(str(base_dir), "static", "extracted_frames")
    os.makedirs(static_dir, exist_ok=True)

    detector = get_detector()
    background_task.add_task(
        run_video_pipeline,
        task_id=task_id,
        callback_url=callback_url,
        detector=detector,
        video_path=temp_video_path,
        static_dir=static_dir,
        conf=conf,
        device=device,
    )

    return {
        "task_id": task_id,
        "status": "PROCESSING",
        "message": "Video accepted. YOLO inference started.",
    }
