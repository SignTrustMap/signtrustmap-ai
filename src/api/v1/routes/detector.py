from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas.models import (
    ModelDeployResponse,
    ModelEvaluationResponse,
    ModelMetricItem,
    ModelRollbackResponse,
    ModelSwitchResponse,
    ModelVersionItem,
)
from src.core.few_shot_manager import few_shot_manager
from src.infrastructure.db import get_async_db
from src.infrastructure.models import ModelEvaluationMetric, ModelVersion

router = APIRouter(tags=["Detector MLOps"])


@router.get("/versions", response_model=List[ModelVersionItem])
async def list_detector_versions(db: AsyncSession = Depends(get_async_db)):
    stmt = (
        select(ModelVersion)
        .where(ModelVersion.model_type == "DETECTOR")
        .order_by(ModelVersion.id.asc())
    )
    result = await db.execute(stmt)
    records = result.scalars().all()
    return [
        ModelVersionItem(
            id=r.id,
            model_type=r.model_type,
            version_tag=r.version_tag,
            is_active=r.is_active,
            note=r.note,
            created_at=r.created_at.isoformat() if r.created_at else None,
        )
        for r in records
    ]


@router.get("/versions/{version_id}/evaluation", response_model=ModelEvaluationResponse)
async def get_detector_evaluation(version_id: int, db: AsyncSession = Depends(get_async_db)):
    v_stmt = select(ModelVersion).where(
        ModelVersion.id == version_id,
        ModelVersion.model_type == "DETECTOR",
    )
    v_res = await db.execute(v_stmt)
    version = v_res.scalar_one_or_none()
    if not version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Detector version #{version_id} not found",
        )

    m_stmt = (
        select(ModelEvaluationMetric)
        .where(ModelEvaluationMetric.model_id == version_id)
        .order_by(ModelEvaluationMetric.id.asc())
    )
    m_res = await db.execute(m_stmt)
    metrics_records = m_res.scalars().all()

    metric_items = [
        ModelMetricItem(
            metric_name=r.metric_name,
            metric_value=r.metric_value,
            split=r.split,
        )
        for r in metrics_records
    ]

    return ModelEvaluationResponse(
        version_id=version.id,
        version_tag=version.version_tag,
        is_active=version.is_active,
        created_at=version.created_at.isoformat() if version.created_at else None,
        metrics=metric_items,
    )


@router.post("/versions/{version_id}/deploy", response_model=ModelDeployResponse)
async def deploy_detector_version_endpoint(version_id: int):
    logger.info(f"Deploying detector version #{version_id}")
    try:
        res = few_shot_manager.deploy_detector_version(version_id)
        return ModelDeployResponse(
            status=res["status"],
            message=res["message"],
            deployed_version_id=res["deployed_version_id"],
            version_tag=res["version_tag"],
            is_active=res["is_active"],
            deployed_at=res["deployed_at"],
            batch_run_id=f"detector_{res['version_tag']}",
        )
    except ValueError as e:
        logger.warning(f"Deployment rejected: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Deploy failed: {str(e)}")


@router.post("/rollback", response_model=ModelRollbackResponse)
async def rollback_detector_version_endpoint():
    logger.info("Executing detector version rollback")
    try:
        res = few_shot_manager.rollback_detector_version()
        return ModelRollbackResponse(**res)
    except ValueError as e:
        logger.warning(f"Rollback rejected: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Rollback failed: {str(e)}")


@router.post("/switch/{version_id}", response_model=ModelSwitchResponse)
async def switch_detector_version_endpoint(version_id: int):
    logger.info(f"Switching detector to version #{version_id}")
    try:
        res = few_shot_manager.switch_detector_version(version_id)
        return ModelSwitchResponse(**res)
    except ValueError as e:
        logger.warning(f"Switch rejected: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Switch failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Switch failed: {str(e)}")
