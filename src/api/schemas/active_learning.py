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


class FeedbackItem(BaseModel):
    object_name: str = Field(..., description="Object name / path in MinIO storage", examples=["vid-task-111/frame_1.jpg"])
    box_xyxy: List[float] = Field(..., description="Bounding box coordinates [xmin, ymin, xmax, ymax]", examples=[[120.0, 200.0, 180.0, 260.0]])
    evaluation: str = Field(..., description="Reviewer evaluation status: TRUE (correct), FALSE (classify error), NOT_GIVEN (detector error / not a sign)", examples=["NOT_GIVEN"])
    pred_class_id: Optional[int] = Field(None, description="Original predicted class ID", examples=[14])
    pred_class_name: Optional[str] = Field(None, description="Original predicted class name", examples=["Speed_limit_50"])
    pred_confidence: Optional[float] = Field(None, description="Original predicted confidence", examples=[0.85])
    oracle_class_id: Optional[int] = Field(None, description="Aggregated ground truth class ID (for FALSE)", examples=[15])
    oracle_class_name: Optional[str] = Field(None, description="Aggregated ground truth class name (for FALSE)", examples=["Speed_limit_60"])
    is_clip: Optional[bool] = Field(False, description="Flag indicating if the classifier was CLIP/Zero-shot")


class ALFeedbackRequest(BaseModel):
    items: List[FeedbackItem] = Field(..., description="List of feedback items from Backend Aggregator")
    auto_trigger: Optional[bool] = Field(None, description="Whether to check and trigger retrain automatically")


class DatasetCountsGroup(BaseModel):
    pending: int = Field(..., description="Pending samples for training", examples=[10])
    trained: int = Field(..., description="Trained samples count", examples=[50])


class DatasetCounts(BaseModel):
    detector: DatasetCountsGroup
    classifier: DatasetCountsGroup
    clip: DatasetCountsGroup
    total_pending: int = Field(..., description="Total pending samples", examples=[20])
    total_trained: int = Field(..., description="Total trained samples", examples=[100])
    train_history_count: int = Field(..., description="Number of past training events", examples=[3])
    store_path: str = Field(..., description="Path to persistent JSON dataset file")


class ALFeedbackResponse(BaseModel):
    total_processed: int = Field(..., description="Total items processed", examples=[5])
    detector_added: int = Field(..., description="Samples categorized to detector hard negatives (NOT_GIVEN)", examples=[2])
    classifier_added: int = Field(..., description="Samples categorized to classifier corrections (FALSE)", examples=[2])
    clip_added: int = Field(..., description="Samples categorized to CLIP staging (FALSE with is_clip=true)", examples=[1])
    correct_count: int = Field(..., description="Samples verified as correct (TRUE)", examples=[0])
    dataset_counts: DatasetCounts
    triggered_retrain: List[str] = Field(..., description="List of models where retrain was triggered", examples=[["detector"]])


class DatasetStatusResponse(BaseModel):
    status: str = Field("SUCCESS", description="Operation status")
    dataset_counts: DatasetCounts


class TrainTriggerRequest(BaseModel):
    target: str = Field("classifier", description="Model target to retrain (detector, classifier, clip)", examples=["classifier"])
    dry_run: Optional[bool] = Field(False, description="Simulate retrain without moving samples to trained state")


class TrainTriggerResponse(BaseModel):
    status: str = Field(..., description="Retrain trigger status (SUCCESS, SKIPPED, FAILED)", examples=["SUCCESS"])
    target: str = Field(..., description="Target model retrained", examples=["classifier"])
    sample_count: int = Field(..., description="Number of samples included in retraining", examples=[100])
    new_weights: Optional[str] = Field(None, description="Generated updated weights identifier")
    dry_run: bool = Field(False, description="Whether this was a dry run")
    message: Optional[str] = Field(None, description="Optional message details")
    error: Optional[str] = Field(None, description="Error message if failed")