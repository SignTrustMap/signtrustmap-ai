from fastapi import APIRouter
from loguru import logger
from src.infrastructure.health import check_exiftool, check_minio, check_ffprobe

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check():
    logger.info("Executing API dependency health check.")

    exif = check_exiftool()
    ffprobe = check_ffprobe()
    minio = check_minio()

    overall_status = "healthy"
    if exif["status"] == "unhealthy" or ffprobe["status"] == "unhealthy":
        overall_status = "unhealthy"

    logger.debug(f"Health check status: {overall_status} (Exiftool: {exif['status']}, FFprobe: {ffprobe['status']}, MinIO: {minio['status']})")

    return {
        "status": overall_status,
        "dependencies": {
            "exiftool": exif,
            "ffprobe": ffprobe,
            "minio": minio,
        },
    }
