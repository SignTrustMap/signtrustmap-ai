import asyncio
import json
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from src.core.system_metrics import metrics_collector
from src.infrastructure.health import check_cuda, check_minio
from src.api.schemas.system import SystemMetricsResponse, HealthResponse

router = APIRouter(tags=["System"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    logger.debug("Executing system and dependency health check.")
    cuda = check_cuda()
    minio = check_minio()

    overall_status = "healthy"
    if cuda["status"] == "unhealthy":
        overall_status = "unhealthy"

    return {
        "status": overall_status,
        "dependencies": {
            "cuda": cuda,
            "minio": minio,
        },
    }


@router.get("/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics():
    logger.trace("Serving system metrics snapshot.")
    return metrics_collector.collect_metrics()


@router.get("/stream")
async def stream_system_metrics(request: Request):
    logger.info("Client connected to real-time system telemetry SSE stream (1.0s interval).")

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    logger.info("Client disconnected from system telemetry SSE stream.")
                    break

                metrics = metrics_collector.collect_metrics()
                json_data = metrics.model_dump_json()
                yield f"data: {json_data}\n\n"
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            logger.info("System telemetry SSE stream cancelled by client disconnect.")
        except Exception as e:
            logger.error(f"Error in system telemetry SSE stream: {e}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
