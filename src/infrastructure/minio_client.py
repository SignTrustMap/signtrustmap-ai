import os
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from loguru import logger
from src.core.config import settings

def is_minio_configured():
    endpoint = settings.MINIO_ENDPOINT
    access_key = settings.MINIO_ACCESS_KEY
    secret_key = settings.MINIO_SECRET_KEY
    return bool(endpoint and access_key and secret_key)

def upload_frame_to_minio(local_file_path, object_name):
    endpoint = settings.MINIO_ENDPOINT
    access_key = settings.MINIO_ACCESS_KEY
    secret_key = settings.MINIO_SECRET_KEY
    bucket_name = settings.MINIO_BUCKET
    
    if not (endpoint and access_key and secret_key):
        logger.warning("MinIO credentials or endpoint missing from environment. Skipping upload.")
        return False, "MinIO storage not hosted yet"
        
    logger.info(f"Uploading local frame {local_file_path} to MinIO: {object_name}")
    logger.debug(f"MinIO target - endpoint: {endpoint}, bucket: {bucket_name}")
    
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=boto3.session.Config(signature_version="s3v4")
        )
        
        try:
            s3.head_bucket(Bucket=bucket_name)
        except Exception:
            logger.debug(f"Bucket '{bucket_name}' not found. Attempting to create bucket...")
            s3.create_bucket(Bucket=bucket_name)
            
        s3.upload_file(local_file_path, bucket_name, object_name)
        url = f"{endpoint}/{bucket_name}/{object_name}"
        logger.info(f"Successfully uploaded frame to MinIO. URL: {url}")
        return True, url
        
    except (NoCredentialsError, PartialCredentialsError) as e:
        logger.warning(f"MinIO storage credentials missing: {e}")
        return False, "MinIO storage credentials missing"
    except Exception as e:
        logger.warning(f"MinIO connection failed: {e}")
        return False, f"MinIO storage not hosted yet ({str(e)})"

def download_model_from_minio(object_name: str, local_file_path: str) -> bool:
    endpoint = settings.MINIO_ENDPOINT
    access_key = settings.MINIO_ACCESS_KEY
    secret_key = settings.MINIO_SECRET_KEY
    bucket_name = settings.MINIO_BUCKET
    
    if not (endpoint and access_key and secret_key):
        logger.error("MinIO credentials or endpoint missing from environment. Cannot download model.")
        return False
        
    logger.info(f"Downloading model {object_name} from MinIO to local cache: {local_file_path}")
    
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=boto3.session.Config(signature_version="s3v4")
        )
        
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        s3.download_file(bucket_name, object_name, local_file_path)
        logger.info(f"Successfully downloaded model to: {local_file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download model {object_name} from MinIO: {e}")
        return False
