import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from loguru import logger
from src.core.model_manager import model_manager
from src.pipelines.export_pipeline import run_export_pipeline, export_lock, export_tasks
from src.api.schemas import (
    DetectorLoadRequest,
    DetectorLoadResponse,
    ClassifierLoadRequest,
    ClassifierLoadResponse,
    DetectorUnloadResponse,
    ClassifierUnloadResponse,
    AllModelsUnloadResponse,
    DetectorExportRequest,
    ClassifierExportRequest,
    ModelExportResponse,
    ModelExportStatusResponse,
    ModelListResponse,
    DetectorFileInfo,
    ClassifierFileInfo,
)

router = APIRouter(tags=["Models"])


def _format_size_human(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def _scan_detectors(target_dir: Path, loaded_name: Optional[str]) -> List[DetectorFileInfo]:
    models: List[DetectorFileInfo] = []
    if target_dir.exists() and target_dir.is_dir():
        for file_path in sorted(target_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in [".pt", ".onnx", ".engine"]:
                stat = file_path.stat()
                mod_time = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                fmt = file_path.suffix.lstrip(".").lower()
                is_curr_loaded = (loaded_name is not None and file_path.name == loaded_name)
                models.append(
                    DetectorFileInfo(
                        filename=file_path.name,
                        format=fmt,
                        size_bytes=stat.st_size,
                        size_human=_format_size_human(stat.st_size),
                        modified_at=mod_time,
                        is_loaded=is_curr_loaded,
                    )
                )
    return models


def _scan_classifiers(target_dir: Path, loaded_name: Optional[str]) -> List[ClassifierFileInfo]:
    models: List[ClassifierFileInfo] = []
    if target_dir.exists() and target_dir.is_dir():
        for file_path in sorted(target_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in [".pt", ".onnx", ".engine"]:
                stat = file_path.stat()
                mod_time = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                fmt = file_path.suffix.lstrip(".").lower()
                is_curr_loaded = (loaded_name is not None and file_path.name == loaded_name)
                models.append(
                    ClassifierFileInfo(
                        filename=file_path.name,
                        format=fmt,
                        size_bytes=stat.st_size,
                        size_human=_format_size_human(stat.st_size),
                        modified_at=mod_time,
                        is_loaded=is_curr_loaded,
                    )
                )
    return models


@router.post("/export/detector", response_model=ModelExportResponse, status_code=status.HTTP_202_ACCEPTED)
async def export_detector_endpoint(
    request: DetectorExportRequest,
    background_tasks: BackgroundTasks,
):
    logger.info(f"Received detector export request for: {request.weights_name}, format: {request.format}")

    base_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
    if os.path.isabs(request.weights_name):
        weights_path = Path(request.weights_name)
    else:
        weights_path = base_dir / "weights" / "detectors" / request.weights_name

    if not weights_path.exists():
        logger.warning(f"Detector weights file not found: {weights_path}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Detector weights file '{request.weights_name}' not found in weights/detectors/.",
        )

    if export_lock.locked():
        logger.warning("Export request rejected: another export task is currently in progress.")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Another model export process is currently running on the server. Please wait until it completes to protect system memory.",
        )

    await export_lock.acquire()
    task_id = f"export-det-{uuid.uuid4().hex[:8]}"

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

    logger.debug(f"Queued detector export task {task_id} for {weights_path.name}")
    return {
        "task_id": task_id,
        "status": "PROCESSING",
        "message": f"Detector export job queued for '{weights_path.name}' ({request.format}).",
    }


@router.post("/export/classifier", response_model=ModelExportResponse, status_code=status.HTTP_202_ACCEPTED)
async def export_classifier_endpoint(
    request: ClassifierExportRequest,
    background_tasks: BackgroundTasks,
):
    logger.info(f"Received classifier export request for: {request.weights_name}, format: {request.format}")

    base_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
    if os.path.isabs(request.weights_name):
        weights_path = Path(request.weights_name)
    else:
        weights_path = base_dir / "weights" / "classifiers" / request.weights_name

    if not weights_path.exists():
        logger.warning(f"Classifier weights file not found: {weights_path}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Classifier weights file '{request.weights_name}' not found in weights/classifiers/.",
        )

    if export_lock.locked():
        logger.warning("Export request rejected: another export task is currently in progress.")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Another model export process is currently running on the server. Please wait until it completes to protect system memory.",
        )

    await export_lock.acquire()
    task_id = f"export-cls-{uuid.uuid4().hex[:8]}"

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

    logger.debug(f"Queued classifier export task {task_id} for {weights_path.name}")
    return {
        "task_id": task_id,
        "status": "PROCESSING",
        "message": f"Classifier export job queued for '{weights_path.name}' ({request.format}).",
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


@router.post("/load/detector", response_model=DetectorLoadResponse)
async def load_detector_endpoint(request: DetectorLoadRequest):
    logger.info(f"Received request to load detector model: {request.weights_name}")
    try:
        loaded_name = model_manager.load_detector(request.weights_name)
        fmt = os.path.splitext(loaded_name)[1].lstrip(".")
        return {
            "status": "LOADED",
            "loaded_detector": loaded_name,
            "format": fmt,
            "message": f"Detector model '{loaded_name}' successfully loaded into memory.",
        }
    except FileNotFoundError as e:
        logger.warning(f"Detector load failed: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error loading detector '{request.weights_name}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/load/classifier", response_model=ClassifierLoadResponse)
async def load_classifier_endpoint(request: ClassifierLoadRequest):
    logger.info(f"Received request to load classifier model: {request.weights_name}")
    try:
        loaded_name = model_manager.load_classifier(request.weights_name)
        fmt = os.path.splitext(loaded_name)[1].lstrip(".")
        return {
            "status": "LOADED",
            "loaded_classifier": loaded_name,
            "format": fmt,
            "message": f"Classifier model '{loaded_name}' successfully loaded into memory.",
        }
    except FileNotFoundError as e:
        logger.warning(f"Classifier load failed: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error loading classifier '{request.weights_name}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/unload/detector", response_model=DetectorUnloadResponse)
async def unload_detector_endpoint():
    logger.info("Received request to unload detector model.")
    was_unloaded = model_manager.unload_detector()
    return {
        "status": "UNLOADED" if was_unloaded else "ALREADY_UNLOADED",
        "message": "Detector model unloaded." if was_unloaded else "No detector model currently loaded in memory.",
    }


@router.post("/unload/classifier", response_model=ClassifierUnloadResponse)
async def unload_classifier_endpoint():
    logger.info("Received request to unload classifier model.")
    was_unloaded = model_manager.unload_classifier()
    return {
        "status": "UNLOADED" if was_unloaded else "ALREADY_UNLOADED",
        "message": "Classifier model unloaded." if was_unloaded else "No classifier model currently loaded in memory.",
    }


@router.post("/unload", response_model=AllModelsUnloadResponse)
async def unload_all_models_endpoint():
    logger.info("Received request to unload all models from memory.")
    was_unloaded = model_manager.unload_all()
    return {
        "status": "UNLOADED" if was_unloaded else "ALREADY_UNLOADED",
        "message": "All models unloaded and memory reclaimed." if was_unloaded else "No models currently loaded in memory.",
    }


@router.get("", response_model=ModelListResponse)
async def list_available_models():
    base_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
    detectors_dir = base_dir / "weights" / "detectors"
    classifiers_dir = base_dir / "weights" / "classifiers"

    loaded_det = model_manager.get_loaded_detector_filename()
    loaded_cls = model_manager.get_loaded_classifier_filename()

    detectors = _scan_detectors(detectors_dir, loaded_det)
    classifiers = _scan_classifiers(classifiers_dir, loaded_cls)

    logger.debug(f"Retrieved {len(detectors)} detector(s) and {len(classifiers)} classifier(s).")
    return {
        "total_detectors": len(detectors),
        "total_classifiers": len(classifiers),
        "loaded_detector": loaded_det,
        "loaded_classifier": loaded_cls,
        "detectors": detectors,
        "classifiers": classifiers,
    }

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from src.infrastructure.db import get_async_db
from src.infrastructure.models import AiModelVersion
from src.api.schemas.models import (
    AiModelVersionItem,
    AiModelVersionSyncRequest,
)


@router.get("/versions", response_model=List[AiModelVersionItem])
async def list_model_versions(db: AsyncSession = Depends(get_async_db)):
    """List all AI model versions stored in AI Database."""
    stmt = select(AiModelVersion).order_by(AiModelVersion.id.asc())
    result = await db.execute(stmt)
    records = result.scalars().all()
    return [
        AiModelVersionItem(
            id=r.id,
            model_type=r.model_type,
            version_tag=r.version_tag,
            artifact_url=r.artifact_url,
            trained_at=r.trained_at.isoformat() if r.trained_at else None,
            is_active=r.is_active,
            evaluation_metrics=r.evaluation_metrics,
            created_at=r.created_at.isoformat() if r.created_at else None,
        )
        for r in records
    ]


@router.post("/versions/sync")
async def sync_model_version(
    request: AiModelVersionSyncRequest,
    db: AsyncSession = Depends(get_async_db),
):
    """Synchronize a model version record from Backend into AI Database."""
    try:
        existing = await db.get(AiModelVersion, request.id)
        if existing:
            existing.model_type = request.model_type
            existing.version_tag = request.version_tag
            existing.artifact_url = request.artifact_url
            existing.is_active = request.is_active
            existing.evaluation_metrics = request.evaluation_metrics
        else:
            db_item = AiModelVersion(
                id=request.id,
                model_type=request.model_type,
                version_tag=request.version_tag,
                artifact_url=request.artifact_url,
                is_active=request.is_active,
                evaluation_metrics=request.evaluation_metrics,
            )
            db.add(db_item)
        await db.commit()
        return {"status": "SUCCESS", "id": request.id, "is_active": request.is_active}
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to sync model version: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to sync model version: {str(e)}",
        )


@router.patch("/versions/{version_id}/activate")
async def activate_model_version(
    version_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Activate a model version in AI Database, deactivating others of same type."""
    try:
        target = await db.get(AiModelVersion, version_id)
        if not target:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"AI model version #{version_id} not found in AI DB.",
            )
        await db.execute(
            update(AiModelVersion)
            .where(AiModelVersion.model_type == target.model_type)
            .values(is_active=False)
        )
        target.is_active = True
        await db.commit()
        return {
            "status": "SUCCESS",
            "version_id": version_id,
            "model_type": target.model_type,
            "is_active": True,
        }
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to activate model version #{version_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to activate model version: {str(e)}",
        )


@router.patch("/versions/{version_id}/rollback")
async def rollback_model_version(
    version_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Rollback an active model version to previous stable version in AI Database."""
    try:
        target = await db.get(AiModelVersion, version_id)
        if not target:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"AI model version #{version_id} not found in AI DB.",
            )
        target.is_active = False

        prev_stmt = (
            select(AiModelVersion)
            .where(AiModelVersion.model_type == target.model_type, AiModelVersion.id != version_id)
            .order_by(AiModelVersion.created_at.desc())
            .limit(1)
        )
        prev_result = await db.execute(prev_stmt)
        prev = prev_result.scalar_one_or_none()
        reactivated_id = None
        if prev:
            prev.is_active = True
            reactivated_id = prev.id

        await db.commit()
        return {
            "status": "SUCCESS",
            "version_id": version_id,
            "rolled_back": True,
            "reactivated_id": reactivated_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to rollback model version #{version_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to rollback model version: {str(e)}",
        )