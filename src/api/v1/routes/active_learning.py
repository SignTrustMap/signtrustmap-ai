from typing import Optional, List
from uuid import UUID
from fastapi import APIRouter, Header, HTTPException, status, Depends
import torch
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.db import get_async_db
from src.infrastructure.models import ActiveLearningRecord, CandidateEmbedding
from src.core.config import settings
from src.core.model_manager import model_manager
from src.core.few_shot_manager import few_shot_manager
from src.core.active_learning.detection_scorer import al_engine
from src.core.active_learning.store import al_store
from src.core.active_learning.feedback_processor import feedback_processor
from src.core.active_learning.retrain import trigger_retrain
from src.api.schemas.active_learning import (
    SingleDetectionScoreRequest,
    SingleDetectionScoreResponse,
    BatchDetectionScoreRequest,
    BatchDetectionScoreResponse,
    StrategyInfoResponse,
    ALFeedbackRequest,
    ALFeedbackResponse,
    DatasetStatusResponse,
    TrainTriggerRequest,
    TrainTriggerResponse,
    ActiveLearningRecordItem,
    ActiveLearningRecordBatchRequest,
    ActiveLearningSelectionItem,
    ActiveLearningSelectionBatchRequest,
)

router = APIRouter(tags=["Active Learning"])


@router.post("/detection/score", response_model=SingleDetectionScoreResponse)
async def score_single_detection(request: SingleDetectionScoreRequest):
    effective_name = request.s3_key or request.object_name
    logger.info(f"Received single detection scoring request for '{effective_name}'")
    try:
        return al_engine.score_single(
            object_name=effective_name,
            strategy=request.strategy,
            aggregation_method=request.aggregation_method,
            conf_threshold=request.conf_threshold,
        )
    except FileNotFoundError as e:
        logger.warning(f"Image not found on MinIO: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Single detection scoring failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scoring failed: {str(e)}",
        )


@router.post("/detection/score/batch", response_model=BatchDetectionScoreResponse)
async def score_batch_detections(request: BatchDetectionScoreRequest):
    logger.info(f"Received batch detection scoring request for {len(request.object_names)} object(s)")
    if not request.object_names:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="object_names list must not be empty.",
        )
    try:
        return al_engine.score_batch(
            object_names=request.object_names,
            top_k=request.top_k,
            strategy=request.strategy,
            aggregation_method=request.aggregation_method,
            conf_threshold=request.conf_threshold,
        )
    except Exception as e:
        logger.error(f"Batch detection scoring failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch scoring failed: {str(e)}",
        )


@router.get("/strategies", response_model=StrategyInfoResponse)
async def get_supported_strategies():
    logger.debug("Retrieving supported Active Learning strategies and aggregation methods.")
    return al_engine.get_supported_strategies()


@router.post("/feedback", response_model=ALFeedbackResponse)
async def ingest_active_learning_feedback(request: ALFeedbackRequest):
    logger.info(f"Received AL feedback batch with {len(request.items)} item(s)")
    if not request.items:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Feedback items list must not be empty.",
        )
    try:
        items_raw = [item.model_dump() for item in request.items]
        return feedback_processor.process_feedback_batch(
            items=items_raw,
            auto_trigger=request.auto_trigger,
        )
    except Exception as e:
        logger.error(f"Failed to process Active Learning feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feedback processing failed: {str(e)}",
        )


@router.get("/dataset/status", response_model=DatasetStatusResponse)
async def get_dataset_status():
    logger.debug("Retrieving Active Learning dataset status.")
    try:
        counts = al_store.get_counts()
        return {
            "status": "SUCCESS",
            "dataset_counts": counts,
        }
    except Exception as e:
        logger.error(f"Failed to fetch dataset status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not retrieve dataset status: {str(e)}",
        )


@router.post("/train/trigger", response_model=TrainTriggerResponse)
async def trigger_model_retrain(request: TrainTriggerRequest):
    logger.info(f"Received explicit retraining trigger for target='{request.target}' (dry_run={request.dry_run})")
    try:
        result = await trigger_retrain(
            target=request.target,
            dry_run=bool(request.dry_run),
        )
        return result
    except Exception as e:
        logger.error(f"Retraining trigger failed for '{request.target}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retraining execution failed: {str(e)}",
        )


@router.post("/classifier/rebuild")
@router.post("/prototypes/rebuild")
async def rebuild_classifier_weights(
    weights_path: Optional[str] = None,
    internal_token: Optional[str] = Header(None, alias="X-Internal-Token"),
):
    if internal_token and internal_token != settings.INTERNAL_API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid internal authorization token",
        )
    try:
        classifier = model_manager.get_classifier(weights_path)
        if classifier is None:
            raise RuntimeError("Classifier model is not loaded or configured.")

        timo_weights, timo_bias = few_shot_manager.build_timo_classifier(
            model_id=None,
            save_cache=True,
        )
        classifier.hot_reload_timo(timo_weights, timo_bias)
        num_classes = timo_bias.shape[0] if timo_bias is not None else 0
        feature_dim = timo_weights.shape[0] if timo_weights is not None else 0
        return {
            "status": "SUCCESS",
            "message": "TIMO classifier weights successfully rebuilt and hot-reloaded into memory",
            "classes_count": num_classes,
            "feature_dim": feature_dim,
            "device": str(classifier.device),
        }
    except Exception as e:
        logger.error(f"TIMO classifier rebuild failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TIMO classifier rebuild failed: {str(e)}",
        )


@router.post("/records", status_code=status.HTTP_201_CREATED)
@router.post("/selections", status_code=status.HTTP_201_CREATED)
async def record_active_learning_records(
    request: ActiveLearningRecordBatchRequest,
    db: AsyncSession = Depends(get_async_db),
):
    try:
        count = 0
        for item in request.items:
            cid = UUID(item.candidate_id) if isinstance(item.candidate_id, str) else item.candidate_id
            rec = ActiveLearningRecord(
                candidate_id=cid,
                model_id=item.model_id,
                class_id=item.class_id,
                box_xyxy=item.box_xyxy,
                detect_confidence=item.detect_confidence,
                classify_confidence=item.classify_confidence,
                strategy=item.strategy,
                acquisition_score=item.acquisition_score,
                flip_count=item.flip_count,
                flip_classes=item.flip_classes,
                calibrated_entropy=item.calibrated_entropy,
            )
            db.add(rec)
            count += 1
        await db.commit()
        return {"status": "SUCCESS", "count": count}
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to record AL records: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record AL records: {str(e)}",
        )


@router.get("/records", response_model=List[ActiveLearningRecordItem])
@router.get("/selections", response_model=List[ActiveLearningRecordItem])
async def list_active_learning_records(
    limit: int = 50,
    offset: int = 0,
    strategy: Optional[str] = None,
    db: AsyncSession = Depends(get_async_db),
):
    stmt = select(ActiveLearningRecord).order_by(ActiveLearningRecord.id.desc())
    if strategy:
        stmt = stmt.where(ActiveLearningRecord.strategy == strategy)
    stmt = stmt.limit(min(limit, 500)).offset(offset)
    result = await db.execute(stmt)
    records = result.scalars().all()
    return [
        ActiveLearningRecordItem(
            id=r.id,
            candidate_id=str(r.candidate_id),
            model_id=r.model_id,
            class_id=r.class_id,
            box_xyxy=r.box_xyxy,
            detect_confidence=r.detect_confidence,
            classify_confidence=r.classify_confidence,
            strategy=r.strategy,
            acquisition_score=r.acquisition_score,
            flip_count=r.flip_count,
            flip_classes=r.flip_classes or [],
            calibrated_entropy=r.calibrated_entropy,
            created_at=r.created_at.isoformat() if r.created_at else None,
        )
        for r in records
    ]
