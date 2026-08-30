import os
from typing import Optional, Tuple
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from loguru import logger
from src.core.config import settings


class MinIOClient:
    def __init__(self):
        self._s3_client = None

    def is_configured(self) -> bool:
        return bool(settings.MINIO_ENDPOINT and settings.MINIO_ACCESS_KEY and settings.MINIO_SECRET_KEY)

    def _get_client(self):
        if not self.is_configured():
            return None

        if self._s3_client is None:
            logger.debug(f"Initializing MinIO S3 client connection to {settings.MINIO_ENDPOINT}")
            self._s3_client = boto3.client(
                "s3",
                endpoint_url=settings.MINIO_ENDPOINT,
                aws_access_key_id=settings.MINIO_ACCESS_KEY,
                aws_secret_access_key=settings.MINIO_SECRET_KEY,
                config=boto3.session.Config(signature_version="s3v4")
            )
        return self._s3_client

    def upload_file(self, local_file_path: str, object_name: str, bucket_name: Optional[str] = None) -> Tuple[bool, str]:
        s3 = self._get_client()
        bucket = bucket_name or settings.MINIO_BUCKET

        if not s3:
            logger.warning("MinIO credentials or endpoint missing. Skipping upload.")
            return False, "MinIO storage not configured"

        logger.info(f"Uploading file '{local_file_path}' to MinIO: '{bucket}/{object_name}'")

        try:
            try:
                s3.head_bucket(Bucket=bucket)
            except Exception:
                logger.debug(f"Bucket '{bucket}' not found. Attempting to create bucket...")
                s3.create_bucket(Bucket=bucket)

            s3.upload_file(local_file_path, bucket, object_name)
            url = f"{settings.MINIO_ENDPOINT}/{bucket}/{object_name}"
            logger.info(f"Successfully uploaded to MinIO. URL: {url}")
            return True, url
        except (NoCredentialsError, PartialCredentialsError) as e:
            logger.warning(f"MinIO credentials missing: {e}")
            return False, "MinIO credentials missing"
        except Exception as e:
            logger.warning(f"MinIO upload failed: {e}")
            return False, f"MinIO upload failed ({str(e)})"

    def download_file(self, object_name: str, local_file_path: str, bucket_name: Optional[str] = None) -> bool:
        s3 = self._get_client()
        bucket = bucket_name or settings.MINIO_BUCKET

        if not s3:
            logger.error("MinIO credentials or endpoint missing from environment. Cannot download file.")
            return False

        logger.info(f"Downloading file '{bucket}/{object_name}' from MinIO to '{local_file_path}'")

        try:
            os.makedirs(os.path.dirname(os.path.abspath(local_file_path)), exist_ok=True)
            s3.download_file(bucket, object_name, local_file_path)
            logger.debug(f"Successfully downloaded '{object_name}' to '{local_file_path}'")
            return True
        except Exception as e:
            logger.error(f"Failed to download '{object_name}' from MinIO: {e}")
            return False


minio_client = MinIOClient()
