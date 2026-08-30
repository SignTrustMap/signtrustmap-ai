from typing import List, Optional
from pydantic import BaseModel, Field


class DetectionBoxScore(BaseModel):
    class_id: int = Field(..., description="Traffic sign class ID index", examples=[14])
    class_name: str = Field(..., description="Traffic sign class name", examples=["Speed_limit_50"])
    confidence: float = Field(..., description="Detector confidence score (0.0 - 1.0)", examples=[0.615])
    score: float = Field(..., description="Active learning score for this box", examples=[0.385])
    box_xyxy: List[float] = Field(..., description="Bounding box coordinates [xmin, ymin, xmax, ymax]", examples=[[120.5, 200.1, 180.2, 260.8]])


class SingleDetectionScoreRequest(BaseModel):
    object_name: str = Field(..., description="Object name / path of the image in MinIO bucket", examples=["vid-task-111/sign_1.jpg"])
    strategy: Optional[str] = Field(None, description="Active learning scoring strategy", examples=["least_confidence"])
    aggregation_method: Optional[str] = Field("avg", description="Box score aggregation method (avg, max, min, sum)", examples=["avg"])
    conf_threshold: Optional[float] = Field(None, description="Detection confidence threshold", ge=0.01, le=1.0, examples=[0.6])


class SingleDetectionScoreResponse(BaseModel):
    object_name: str = Field(..., description="Object name in MinIO storage", examples=["vid-task-111/sign_1.jpg"])
    score: float = Field(..., description="Aggregated active learning score for the image", examples=[0.385])
    total_detections: int = Field(..., description="Total bounding boxes detected", examples=[1])
    detections: List[DetectionBoxScore] = Field(..., description="List of detected boxes with scores")
    strategy: str = Field(..., description="Strategy used for scoring", examples=["least_confidence"])
    aggregation_method: str = Field(..., description="Aggregation method applied", examples=["avg"])


class BatchDetectionScoreRequest(BaseModel):
    object_names: List[str] = Field(..., description="List of object names / paths in MinIO storage", examples=[["vid-task-111/sign_1.jpg", "vid-task-111/sign_2.jpg"]])
    top_k: Optional[int] = Field(None, description="Top K samples to return (-1 for all)", examples=[20])
    strategy: Optional[str] = Field(None, description="Active learning scoring strategy", examples=["least_confidence"])
    aggregation_method: Optional[str] = Field("avg", description="Box score aggregation method (avg, max, min, sum)", examples=["avg"])
    conf_threshold: Optional[float] = Field(None, description="Detection confidence threshold", ge=0.01, le=1.0, examples=[0.6])


class BatchSampleScore(BaseModel):
    object_name: str = Field(..., description="Object name in MinIO storage", examples=["vid-task-111/sign_1.jpg"])
    score: float = Field(..., description="Aggregated active learning score for the image", examples=[0.385])
    total_detections: int = Field(..., description="Total bounding boxes detected", examples=[1])
    detections: List[DetectionBoxScore] = Field(..., description="List of detected boxes with scores")


class BatchDetectionScoreResponse(BaseModel):
    total_evaluated: int = Field(..., description="Total images successfully evaluated", examples=[2])
    top_k_returned: int = Field(..., description="Number of top samples returned", examples=[2])
    strategy: str = Field(..., description="Strategy used for scoring", examples=["least_confidence"])
    aggregation_method: str = Field(..., description="Aggregation method applied", examples=["avg"])
    ranked_samples: List[BatchSampleScore] = Field(..., description="Ranked samples sorted descending by score")


class StrategyInfoResponse(BaseModel):
    strategies: List[str] = Field(..., description="List of supported scoring strategies", examples=[["least_confidence"]])
    aggregation_methods: List[str] = Field(..., description="List of supported aggregation methods", examples=[["avg", "max", "min", "sum"]])
    default_strategy: str = Field(..., description="Default scoring strategy configured", examples=["least_confidence"])
    default_aggregation: str = Field(..., description="Default aggregation method configured", examples=["avg"])
    default_top_k: int = Field(..., description="Default top K samples configured for batch scoring", examples=[50])
