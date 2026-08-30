import os
import shutil
import boto3
import torch
from src.core.config import settings


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
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="us-east-1"
        )
        s3.list_buckets()
        return {"status": "healthy", "error": None}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
