from fastapi import APIRouter
from loguru import logger
from src.infrastructure.health import check_cuda, check_minio

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check():
    logger.info("Executing API dependency health check.")

    cuda = check_cuda()
    minio = check_minio()

    overall_status = "healthy"
    if cuda["status"] == "unhealthy":
        overall_status = "unhealthy"

    logger.debug(f"Health check status: {overall_status} (CUDA: {cuda['status']}, MinIO: {minio['status']})")

    return {
        "status": overall_status,
        "dependencies": {
            "cuda": cuda,
            "minio": minio,
        },
    }
