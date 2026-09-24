import os
import shutil
import time
import boto3
from botocore.config import Config
import torch
from src.core.config import settings

_HEALTH_CACHE = {}
_LAST_HEALTH_CHECK = 0.0
_HEALTH_CACHE_TTL = 15.0  # Cache health check for 15 seconds to prevent event-loop choking


def check_cuda():
    try:
        available = torch.cuda.is_available()
        if available:
            device_name = torch.cuda.get_device_name(0)
            return {"status": "healthy", "available": True, "device": device_name, "error": None}
        return {"status": "healthy", "available": False, "device": "CPU", "error": None}
    except Exception as e:
        return {"status": "unhealthy", "available": False, "device": None, "error": str(e)}


def check_tensorrt():
    try:
        import tensorrt as trt
        version = trt.__version__
        trtexec_path = "/usr/src/tensorrt/bin/trtexec"
        has_trtexec = os.path.exists(trtexec_path) or (shutil.which("trtexec") is not None)
        return {
            "status": "healthy",
            "available": True,
            "version": version,
            "trtexec_available": has_trtexec,
            "error": None,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "available": False,
            "version": None,
            "trtexec_available": False,
            "error": str(e),
        }


def check_minio():
    endpoint = settings.MINIO_ENDPOINT
    access_key = settings.MINIO_ACCESS_KEY
    secret_key = settings.MINIO_SECRET_KEY
    if not (endpoint and access_key and secret_key):
        return {"status": "not_configured", "error": "Environment variables missing"}
    try:
        endpoint_url = endpoint
        if not endpoint_url.startswith(("http://", "https://")):
            scheme = "https" if settings.MINIO_SECURE else "http"
            endpoint_url = f"{scheme}://{endpoint_url}"

        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="us-east-1",
            config=Config(connect_timeout=2.0, read_timeout=2.0, retries={"max_attempts": 0})
        )
        bucket = settings.MINIO_OUTPUT_BUCKET or settings.MINIO_BUCKET or "stm-sign-crops"
        try:
            s3.head_bucket(Bucket=bucket)
        except Exception:
            s3.list_buckets()
        return {"status": "healthy", "error": None}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def get_health_status(force_refresh: bool = False):
    global _HEALTH_CACHE, _LAST_HEALTH_CHECK
    now = time.time()
    if not force_refresh and _HEALTH_CACHE and (now - _LAST_HEALTH_CHECK < _HEALTH_CACHE_TTL):
        return _HEALTH_CACHE

    dependencies = {
        "cuda": check_cuda(),
        "tensorrt": check_tensorrt(),
        "minio": check_minio(),
    }
    overall_status = "unhealthy" if any(d.get("status") == "unhealthy" for d in dependencies.values()) else "healthy"

    _HEALTH_CACHE = {
        "status": overall_status,
        "dependencies": dependencies,
    }
    _LAST_HEALTH_CHECK = now
    return _HEALTH_CACHE

