from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field


class DetectorConfigUpdate(BaseModel):
    conf_thresh: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum confidence threshold for detector")


class ClassifierConfigUpdate(BaseModel):
    conf_thresh: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum confidence threshold for classifier")
    prompt_template: Optional[str] = Field(None, description="CLIP text prompt template")
    temperature: Optional[float] = Field(None, gt=0.0, description="Temperature scaling factor for classifier logits")


class TimoConfigUpdate(BaseModel):
    alpha: Optional[float] = Field(None, gt=0.0, description="TiMO alpha parameter")
    beta: Optional[float] = Field(None, gt=0.0, le=1.0, description="TiMO beta parameter")


class FewShotSupportConfigUpdate(BaseModel):
    shots_per_class: Optional[int] = Field(None, ge=1, description="Maximum support shots per class")
    num_anchors: Optional[int] = Field(None, ge=0, description="Number of anchor samples")
    num_medoids: Optional[int] = Field(None, ge=0, description="Number of medoid samples")
    min_crop_size: Optional[int] = Field(None, ge=1, description="Minimum crop size in pixels")
    min_sharpness: Optional[float] = Field(None, ge=0.0, description="Minimum Laplacian variance sharpness")


class ActiveLearningConfigUpdate(BaseModel):
    strategy: Optional[Literal["r3_cec", "least_confidence"]] = Field(None, description="Active learning selection strategy")
    query_budget: Optional[int] = Field(None, ge=1, description="Query budget per round")
    aggregation: Optional[Literal["avg", "max", "min", "sum"]] = Field(None, description="Uncertainty score aggregation method")
    no_detection_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Default score for frames with no detections")
    min_samples_detector: Optional[int] = Field(None, ge=1, description="Minimum samples to trigger detector retrain")
    min_samples_classifier: Optional[int] = Field(None, ge=1, description="Minimum samples to trigger classifier retrain")
    discrepancy_thresh: Optional[float] = Field(None, ge=0.0, le=1.0, description="Discrepancy threshold between detector and classifier")
    auto_trigger_retrain: Optional[bool] = Field(None, description="Whether to automatically trigger retraining")
    r3_cec_k_neighbors: Optional[int] = Field(None, ge=1, description="k-NN parameter for R3-CEC")
    r3_cec_temperature: Optional[float] = Field(None, gt=0.0, description="Platt scaling temperature for R3-CEC")


class SystemConfigUpdate(BaseModel):
    model_idle_timeout: Optional[int] = Field(None, ge=0, description="Model idle timeout in seconds")
    cuda_device: Optional[int] = Field(None, ge=0, description="CUDA device index")


class BulkConfigUpdate(BaseModel):
    detector: Optional[DetectorConfigUpdate] = None
    classifier: Optional[ClassifierConfigUpdate] = None
    timo: Optional[TimoConfigUpdate] = None
    few_shot_support: Optional[FewShotSupportConfigUpdate] = None
    active_learning: Optional[ActiveLearningConfigUpdate] = None
    system: Optional[SystemConfigUpdate] = None


class ConfigResetResponse(BaseModel):
    status: str = Field(..., description="Reset status", examples=["RESET_SUCCESSFUL"])
    message: str = Field(..., description="Confirmation detail", examples=["Configuration restored to factory default settings."])
    config: Dict[str, Dict[str, Any]] = Field(..., description="Active configuration state after reset")
