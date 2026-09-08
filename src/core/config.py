from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    LOG_LEVEL: str = "INFO"
    LOG_TO_JSON: bool = False
    
    MINIO_ENDPOINT: Optional[str] = None
    MINIO_ACCESS_KEY: Optional[str] = None
    MINIO_SECRET_KEY: Optional[str] = None
    MINIO_SECURE: bool = False
    MINIO_OUTPUT_BUCKET: str = "stm-sign-crops"
    # Compatibility alias for existing runtime configuration files.
    MINIO_BUCKET: str = "stm-sign-crops"
    INTERNAL_API_TOKEN: str = "stm_internal_secret_dev"
    
    DETECTOR_S3_KEY: Optional[str] = "models/detectors/yolo26n_vtsdb_mtsd_dfg.engine"
    CLASSIFIER_S3_KEY: Optional[str] = "models/classifiers/mobileclip_s2.pt"
    
    LOCAL_DETECTORS_DIR: str = "weights/detectors"
    LOCAL_CLASSIFIERS_DIR: str = "weights/classifiers"
    
    FALLBACK_DETECTOR_PATH: str = "weights/detectors/yolo26n_vtsdb_mtsd_dfg.engine"
    FALLBACK_CLASSIFIER_PATH: Optional[str] = "weights/classifiers/mobileclip_s2.engine"
    CLASSIFIER_PROMPT_TEMPLATE: str = "a photo of a {} traffic sign."
    
    HEALTH_CHECK_TIMEOUT: int = 2
    MODEL_IDLE_TIMEOUT_SECONDS: int = 1800
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
