import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from loguru import logger
from src.core.model_manager import model_manager
from src.pipelines.export_pipeline import run_export_pipeline, export_lock, export_tasks
from src.api.schemas import (
    ModelExportRequest,
    ModelExportResponse,
    ModelExportStatusResponse,
    ModelListResponse,
    ModelFileInfo,
    ModelUnloadResponse,
)

router = APIRouter(tags=["Models"])


def _format_size_human(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


@router.post("/export", response_model=ModelExportResponse, status_code=status.HTTP_202_ACCEPTED)
async def export_model_endpoint(
    request: ModelExportRequest,
    background_tasks: BackgroundTasks,
):
    logger.info(f"Received model export request for: {request.weights_name}, format: {request.format}")

    base_dir = Path(__file__).resolve().parent.parent.parent.parent
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


@router.get("/export/{task_id}", response_model=ModelExportStatusResponse)
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


@router.post("/unload", response_model=ModelUnloadResponse)
async def unload_model_endpoint():
    logger.info("Received request to explicitly unload model from memory.")
    was_unloaded = model_manager.unload_model()
    if was_unloaded:
        return {
            "status": "UNLOADED",
            "message": "Model successfully unloaded and RAM/VRAM memory reclaimed.",
        }
    return {
        "status": "ALREADY_UNLOADED",
        "message": "No model is currently loaded in memory.",
    }


@router.get("", response_model=ModelListResponse)
async def list_available_models():
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    weights_dir = base_dir / "weights"

    loaded_name = model_manager.get_loaded_model_filename()
    models: List[ModelFileInfo] = []
    if weights_dir.exists() and weights_dir.is_dir():
        for file_path in sorted(weights_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in [".pt", ".onnx", ".engine"]:
                stat = file_path.stat()
                mod_time = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                fmt = file_path.suffix.lstrip(".").lower()
                is_curr_loaded = (loaded_name is not None and file_path.name == loaded_name)
                models.append(
                    ModelFileInfo(
                        filename=file_path.name,
                        format=fmt,
                        size_bytes=stat.st_size,
                        size_human=_format_size_human(stat.st_size),
                        modified_at=mod_time,
                        is_loaded=is_curr_loaded,
                    )
                )

    logger.debug(f"Retrieved {len(models)} model file(s) from weights directory. Loaded: {loaded_name}")
    return {
        "total": len(models),
        "loaded_model": loaded_name,
        "models": models,
    }
