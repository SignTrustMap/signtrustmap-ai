import hmac

from fastapi import APIRouter, Header, HTTPException, status
from starlette.concurrency import run_in_threadpool

from src.api.schemas.inference import (
    SubmissionInferenceRequest,
    SubmissionInferenceResponse,
)
from src.pipelines.inference_engine import inference_engine
from src.core.config import settings


router = APIRouter(tags=["Inference"])


@router.post(
    "/submissions/{submission_id}",
    response_model=SubmissionInferenceResponse,
    summary="Run inference for submission media stored in MinIO",
)
async def process_submission(
    submission_id: str,
    request: SubmissionInferenceRequest,
    internal_token: str = Header(alias="X-Internal-Token"),
) -> dict:
    """Execute inference synchronously; Celery owns retries and orchestration."""
    if not hmac.compare_digest(internal_token, settings.INTERNAL_API_TOKEN):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid internal token")
    media_objects = [item.model_dump() for item in request.media_objects]
    return await run_in_threadpool(
        inference_engine.process,
        submission_id,
        media_objects,
    )
