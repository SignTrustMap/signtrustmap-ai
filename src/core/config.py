from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    LOG_LEVEL: str = "INFO"
    LOG_TO_JSON: bool = False
    
    MINIO_ENDPOINT: Optional[str] = None
    MINIO_ACCESS_KEY: Optional[str] = None
    MINIO_SECRET_KEY: Optional[str] = None
    MINIO_BUCKET: str = "best-frames"
    
    MODEL_S3_KEY: Optional[str] = "models/yolo12n_VTSDB100_v1.0.pt"
    LOCAL_WEIGHTS_DIR: str = "weights"
    FALLBACK_MODEL_PATH: str = "weights/yolo12n_VTSDB100_v1.0.pt"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
