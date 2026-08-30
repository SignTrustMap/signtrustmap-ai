from typing import Optional
from pydantic import BaseModel, Field


class ModelLoadRequest(BaseModel):
    weights_name: str = Field(..., description="Name of the weights file in the weights directory to load into memory", examples=["yolo26n_vtsdb_mtsd_dfg.engine"])


class ModelLoadResponse(BaseModel):
    status: str = Field(..., description="Status of the load operation", examples=["LOADED"])
    loaded_model: str = Field(..., description="Filename of the model currently loaded in memory", examples=["yolo26n_vtsdb_mtsd_dfg.engine"])
    format: str = Field(..., description="Format of the loaded model (pt, onnx, engine)", examples=["pt"])
    message: str = Field(..., description="Status description detail", examples=["Model successfully loaded and set as active model."])


class AppConfigResponse(BaseModel):
    active_model: str = Field(..., description="Currently active model weights file", examples=["yolo26n_vtsdb_mtsd_dfg.engine"])
    default_conf: float = Field(..., description="Default inference confidence threshold (0.0 - 1.0)", examples=[0.6])
    model_idle_timeout_seconds: int = Field(..., description="Inactivity timeout in seconds before auto-unloading from memory", examples=[1800])
    default_device: int = Field(..., description="Default GPU device ID for inference", examples=[0])
    log_level: str = Field(..., description="System logging level (TRACE, DEBUG, INFO, WARNING, ERROR)", examples=["INFO"])
    minio_endpoint: Optional[str] = Field(None, description="Configured MinIO storage endpoint")
    minio_bucket: str = Field(..., description="MinIO storage bucket for frame uploads", examples=["best-frames"])


class AppConfigUpdateRequest(BaseModel):
    active_model: Optional[str] = Field(None, description="Change active model weights filename", examples=["yolo26n_vtsdb_mtsd_dfg.engine"])
    default_conf: Optional[float] = Field(None, description="Change default confidence threshold (0.0 - 1.0)", ge=0.01, le=1.0, examples=[0.65])
    model_idle_timeout_seconds: Optional[int] = Field(None, description="Change idle timeout in seconds before auto-unload (0 to disable)", ge=0, examples=[900])
    default_device: Optional[int] = Field(None, description="Change default GPU device index", ge=0, examples=[0])
    log_level: Optional[str] = Field(None, description="Change system log level", examples=["DEBUG"])


class ConfigResetResponse(BaseModel):
    status: str = Field(..., description="Reset status", examples=["RESET_SUCCESSFUL"])
    message: str = Field(..., description="Confirmation detail", examples=["Configuration restored to factory default settings."])
    config: AppConfigResponse = Field(..., description="Active configuration state after reset")
