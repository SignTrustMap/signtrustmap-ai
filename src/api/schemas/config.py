from typing import Optional
from pydantic import BaseModel, Field


class AppConfigResponse(BaseModel):
    active_detector_model: str = Field(..., description="Currently active detector model filename in weights/detectors/", examples=["yolo12n_VTSDB100_v1.0.pt"])
    active_classifier_model: Optional[str] = Field(None, description="Currently active classifier model filename in weights/classifiers/", examples=["yolo26n_classifier.engine"])
    default_detector_conf: float = Field(..., description="Default detector confidence threshold (0.0 - 1.0)", examples=[0.6])
    default_classifier_conf: float = Field(..., description="Default classifier confidence threshold (0.0 - 1.0)", examples=[0.6])
    model_idle_timeout_seconds: int = Field(..., description="Inactivity timeout in seconds before auto-unloading models from memory", examples=[1800])
    default_device: int = Field(..., description="Default GPU device index for inference", examples=[0])
    log_level: str = Field(..., description="System logging level (TRACE, DEBUG, INFO, WARNING, ERROR)", examples=["INFO"])
    minio_endpoint: Optional[str] = Field(None, description="Configured MinIO storage endpoint")
    minio_bucket: str = Field(..., description="MinIO storage bucket for frame uploads", examples=["best-frames"])


class AppConfigUpdateRequest(BaseModel):
    active_detector_model: Optional[str] = Field(None, description="Change active detector model filename", examples=["yolo26n_vtsdb_mtsd_dfg.engine"])
    active_classifier_model: Optional[str] = Field(None, description="Change active classifier model filename", examples=["yolo26n_classifier.engine"])
    default_detector_conf: Optional[float] = Field(None, description="Change default detector confidence threshold (0.0 - 1.0)", ge=0.01, le=1.0, examples=[0.65])
    default_classifier_conf: Optional[float] = Field(None, description="Change default classifier confidence threshold (0.0 - 1.0)", ge=0.01, le=1.0, examples=[0.65])
    model_idle_timeout_seconds: Optional[int] = Field(None, description="Change idle timeout in seconds before auto-unload (0 to disable)", ge=0, examples=[900])
    default_device: Optional[int] = Field(None, description="Change default GPU device index", ge=0, examples=[0])
    log_level: Optional[str] = Field(None, description="Change system log level", examples=["DEBUG"])


class ConfigResetResponse(BaseModel):
    status: str = Field(..., description="Reset status", examples=["RESET_SUCCESSFUL"])
    message: str = Field(..., description="Confirmation detail", examples=["Configuration restored to factory default settings."])
    config: AppConfigResponse = Field(..., description="Active configuration state after reset")
