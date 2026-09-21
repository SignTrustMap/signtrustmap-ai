from fastapi import APIRouter, HTTPException, status
from loguru import logger
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
)

router = APIRouter(tags=["Active Learning"])


@router.post("/detection/score", response_model=SingleDetectionScoreResponse)
async def score_single_detection(request: SingleDetectionScoreRequest):
    logger.info(f"Received single detection scoring request for '{request.object_name}'")
    try:
        return al_engine.score_single(
            object_name=request.object_name,
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


@router.post("/prototypes/rebuild")
async def rebuild_protofuse_prototypes(
    x_internal_token: Optional[str] = Header(None, alias="X-Internal-Token"),
):
    """
    Recomputes ProtoFuse prototypes from the current support cache / database
    and hot-reloads the active classifier prototypes in memory without downtime.
    """
    from src.core.config import settings
    from src.core.model_manager import model_manager

    if settings.INTERNAL_API_TOKEN and x_internal_token != settings.INTERNAL_API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Internal-Token header",
        )

    logger.info("Received request to rebuild ProtoFuse prototypes...")
    try:
        classifier = model_manager.get_classifier()
        if classifier is None:
            raise RuntimeError("Classifier model is not loaded in model_manager.")

        # Re-load or build prototypes and update RAM
        reloaded_proto = classifier._load_or_build_prototypes()
        classifier.rebuild_prototypes(reloaded_proto)

        return {
            "status": "SUCCESS",
            "message": "ProtoFuse prototypes successfully rebuilt and hot-reloaded into RAM.",
            "prototypes_shape": list(classifier.prototypes.shape),
        }
    except Exception as e:
        logger.error(f"ProtoFuse rebuild failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ProtoFuse rebuild failed: {str(e)}",
        )



