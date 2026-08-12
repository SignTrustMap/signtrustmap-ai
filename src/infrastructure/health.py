import subprocess
import boto3
from src.core.config import settings

def check_exiftool():
    try:
        res = subprocess.run(["exiftool", "-ver"], capture_output=True, text=True, timeout=2)
        if res.returncode == 0:
            return {"status": "healthy", "version": res.stdout.strip(), "error": None}
    except Exception as e:
        return {"status": "unhealthy", "version": None, "error": str(e)}
    return {"status": "unhealthy", "version": None, "error": "Unknown error"}

def check_ffprobe():
    try:
        res = subprocess.run(["ffprobe", "-version"], capture_output=True, text=True, timeout=2)
        if res.returncode == 0:
            version_line = res.stdout.splitlines()[0]
            return {"status": "healthy", "version": version_line, "error": None}
    except Exception as e:
        return {"status": "unhealthy", "version": None, "error": str(e)}
    return {"status": "unhealthy", "version": None, "error": "Unknown error"}

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
