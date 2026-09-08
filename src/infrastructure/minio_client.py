from pathlib import Path
from typing import Optional, Tuple

import boto3
from botocore.client import Config
from loguru import logger

from src.core.config import settings


class MinIOClient:
    """Small S3-compatible adapter used by the inference engine."""

    def __init__(self) -> None:
        self._s3_client = None

    def is_configured(self) -> bool:
        return bool(settings.MINIO_ENDPOINT and settings.MINIO_ACCESS_KEY and settings.MINIO_SECRET_KEY)

    def _get_client(self):
        if not self.is_configured():
            raise RuntimeError("MinIO credentials and endpoint must be configured")

        if self._s3_client is None:
            endpoint = settings.MINIO_ENDPOINT or ""
            if not endpoint.startswith(("http://", "https://")):
                scheme = "https" if settings.MINIO_SECURE else "http"
                endpoint = f"{scheme}://{endpoint}"
            logger.debug(f"Initializing MinIO S3 client connection to {endpoint}")
            self._s3_client = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=settings.MINIO_ACCESS_KEY,
                aws_secret_access_key=settings.MINIO_SECRET_KEY,
                config=Config(signature_version="s3v4"),
                region_name="us-east-1",
            )
        return self._s3_client

    def upload_file(
        self,
        local_file_path: str,
        object_name: str,
        bucket_name: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        s3 = self._get_client()
        bucket = bucket_name or settings.MINIO_OUTPUT_BUCKET

        logger.info(f"Uploading file '{local_file_path}' to MinIO: '{bucket}/{object_name}'")

        try:
            extra_args = {"ContentType": content_type} if content_type else None
            kwargs = {"ExtraArgs": extra_args} if extra_args else {}
            s3.upload_file(local_file_path, bucket, object_name, **kwargs)
            storage_key = f"{bucket}/{object_name}"
            logger.info(f"Successfully uploaded to MinIO: {storage_key}")
            return True, storage_key
        except Exception as e:
            logger.warning(f"MinIO upload failed: {e}")
            return False, None

    def download_file(self, bucket_name: str, object_name: str, local_file_path: str) -> bool:
        s3 = self._get_client()
        logger.info(f"Downloading file '{bucket_name}/{object_name}' to '{local_file_path}'")

        try:
            Path(local_file_path).parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket_name, object_name, local_file_path)
            logger.debug(f"Successfully downloaded '{object_name}' to '{local_file_path}'")
            return True
        except Exception as e:
            logger.error(f"Failed to download '{bucket_name}/{object_name}' from MinIO: {e}")
            return False


minio_client = MinIOClient()
