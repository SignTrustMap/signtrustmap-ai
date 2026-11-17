from typing import List, Optional
from pydantic import BaseModel, Field


class DetectorLoadRequest(BaseModel):
    weights_name: str = Field(..., description="Weights filename in weights/detectors/ to load into memory", examples=["yolo12n_VTSDB100_v1.0.pt"])


class DetectorLoadResponse(BaseModel):
    status: str = Field(..., description="Load status", examples=["LOADED"])
    loaded_detector: str = Field(..., description="Filename of detector currently loaded in memory", examples=["yolo12n_VTSDB100_v1.0.pt"])
    format: str = Field(..., description="Format of loaded detector (pt, onnx, engine)", examples=["pt"])
    message: str = Field(..., description="Status description detail", examples=["Detector model successfully loaded."])


class ClassifierLoadRequest(BaseModel):
    weights_name: str = Field(..., description="Weights filename in weights/classifiers/ to load into memory", examples=["yolo26n_classifier.engine"])


class ClassifierLoadResponse(BaseModel):
    status: str = Field(..., description="Load status", examples=["LOADED"])
    loaded_classifier: str = Field(..., description="Filename of classifier currently loaded in memory", examples=["yolo26n_classifier.engine"])
    format: str = Field(..., description="Format of loaded classifier (pt, onnx, engine)", examples=["engine"])
    message: str = Field(..., description="Status description detail", examples=["Classifier model successfully loaded."])


class DetectorUnloadResponse(BaseModel):
    status: str = Field(..., description="Unload status", examples=["UNLOADED", "ALREADY_UNLOADED"])
    message: str = Field(..., description="Status detail", examples=["Detector model successfully unloaded."])


class ClassifierUnloadResponse(BaseModel):
    status: str = Field(..., description="Unload status", examples=["UNLOADED", "ALREADY_UNLOADED"])
    message: str = Field(..., description="Status detail", examples=["Classifier model successfully unloaded."])


class AllModelsUnloadResponse(BaseModel):
    status: str = Field(..., description="Unload status", examples=["UNLOADED", "ALREADY_UNLOADED"])
    message: str = Field(..., description="Status detail", examples=["All models unloaded and memory reclaimed."])


class DetectorExportRequest(BaseModel):
    weights_name: str = Field(..., description="Detector weights filename in weights/detectors/", examples=["yolo26n_vtsdb_mtsd_dfg.pt"])
    format: str = Field("engine", description="Export target format: 'onnx', 'engine', or 'all'", examples=["engine"])
    half: bool = Field(True, description="Enable FP16 half precision")
    device: int = Field(0, description="GPU device index")
    workspace: int = Field(1, description="TensorRT workspace limit in GB", ge=1, le=4)
    dynamic: bool = Field(False, description="Enable dynamic axes for ONNX/Engine")
    opset: int = Field(18, description="ONNX opset version")
    callback_url: Optional[str] = Field(None, description="Optional webhook URL to receive callback upon completion", examples=["http://api-gateway:8000/webhook/model-exported"])


class ClassifierExportRequest(BaseModel):
    weights_name: str = Field(..., description="Classifier weights filename in weights/classifiers/", examples=["yolo26n_classifier.pt"])
    format: str = Field("engine", description="Export target format: 'onnx', 'engine', or 'all'", examples=["engine"])
    half: bool = Field(True, description="Enable FP16 half precision")
    device: int = Field(0, description="GPU device index")
    workspace: int = Field(1, description="TensorRT workspace limit in GB", ge=1, le=4)
    dynamic: bool = Field(False, description="Enable dynamic axes for ONNX/Engine")
    opset: int = Field(18, description="ONNX opset version")
    callback_url: Optional[str] = Field(None, description="Optional webhook URL to receive callback upon completion", examples=["http://api-gateway:8000/webhook/model-exported"])


class ModelExportResponse(BaseModel):
    task_id: str = Field(..., description="Unique task ID for tracking export progress", examples=["export-task-550e8400"])
    status: str = Field(..., description="Initial task status", examples=["PROCESSING"])
    message: str = Field(..., description="Status message detail", examples=["Model export job started."])


class ModelExportStatusResponse(BaseModel):
    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Export status: PROCESSING, COMPLETED, or FAILED")
    weights_name: str = Field(..., description="Source weights name")
    format: str = Field(..., description="Requested export format")
    exported_files: List[str] = Field(default_factory=list, description="List of exported file paths")
    error: Optional[str] = Field(None, description="Error detail if export failed")
    duration_seconds: Optional[float] = Field(None, description="Total execution duration in seconds")


class ModelExportWebhookPayload(BaseModel):
    task_id: str = Field(..., description="Unique identifier of the export task")
    status: str = Field(..., description="Execution completion status (COMPLETED or FAILED)")
    exported_files: Optional[List[str]] = Field(None, description="List of exported file paths")
    error: Optional[str] = Field(None, description="Error message if failed")
    duration_seconds: Optional[float] = Field(None, description="Duration in seconds")


class DetectorFileInfo(BaseModel):
    filename: str = Field(..., description="Detector filename in weights/detectors/")
    format: str = Field(..., description="Model format (pt, onnx, engine)")
    size_bytes: int = Field(..., description="Size of the file in bytes")
    size_human: str = Field(..., description="Human-readable file size")
    modified_at: str = Field(..., description="Last modified timestamp")
    is_loaded: bool = Field(False, description="Whether this detector is currently loaded in memory")


class ClassifierFileInfo(BaseModel):
    filename: str = Field(..., description="Classifier filename in weights/classifiers/")
    format: str = Field(..., description="Model format (pt, onnx, engine)")
    size_bytes: int = Field(..., description="Size of the file in bytes")
    size_human: str = Field(..., description="Human-readable file size")
    modified_at: str = Field(..., description="Last modified timestamp")
    is_loaded: bool = Field(False, description="Whether this classifier is currently loaded in memory")


class ModelListResponse(BaseModel):
    total_detectors: int = Field(..., description="Total count of detector models")
    total_classifiers: int = Field(..., description="Total count of classifier models")
    loaded_detector: Optional[str] = Field(None, description="Filename of currently loaded detector in memory")
    loaded_classifier: Optional[str] = Field(None, description="Filename of currently loaded classifier in memory")
    detectors: List[DetectorFileInfo] = Field(default_factory=list, description="List of detector models")
    classifiers: List[ClassifierFileInfo] = Field(default_factory=list, description="List of classifier models")

class ModelVersionItem(BaseModel):
    id: int = Field(..., description="Primary key identifier of the model version in database", examples=[2])
    model_type: str = Field(..., description="Type of the model: CLASSIFIER or DETECTOR", examples=["CLASSIFIER"])
    version_tag: str = Field(..., description="Unique version tag string", examples=["classifier-v2.1"])
    is_active: bool = Field(False, description="Whether this version is currently deployed and active")
    note: Optional[str] = Field(None, description="Descriptive audit note or training details", examples=["Auto-rebuild with verified samples"])
    created_at: Optional[str] = Field(None, description="ISO 8601 creation timestamp", examples=["2026-11-15T14:00:00Z"])


class ModelVersionSyncRequest(BaseModel):
    id: int = Field(..., description="Model version ID to sync", examples=[2])
    model_type: str = Field(..., description="Model type (CLASSIFIER or DETECTOR)", examples=["CLASSIFIER"])
    version_tag: str = Field(..., description="Version tag string", examples=["classifier-v2.1"])
    is_active: bool = Field(False, description="Active status flag")
    note: Optional[str] = Field(None, description="Optional note")


class ModelVersionActivateRequest(BaseModel):
    id: int = Field(..., description="Model version ID to activate/deploy", examples=[2])


class ClassifierPoolStatus(BaseModel):
    active_version_id: Optional[int] = Field(None, description="ID of currently active classifier version in database", examples=[2])
    active_version_tag: Optional[str] = Field(None, description="Tag of currently active classifier version", examples=["mobileclip_s2"])
    total_verified_crops_in_ai_db: int = Field(0, description="Total verified candidate crops stored in Jetson AI database", examples=[489])
    new_verified_crops_since_active_model: int = Field(0, description="Verified crops added since the active model version was built", examples=[489])
    affected_classes_count: int = Field(0, description="Number of distinct classes represented in the verified pool", examples=[42])
    min_samples_threshold: int = Field(100, description="Minimum recommended verified samples before triggering classifier rebuild", examples=[100])
    is_eligible_for_rebuild: bool = Field(False, description="True if verified pool size meets or exceeds the minimum threshold", examples=[True])
    current_hyperparams: dict = Field(default_factory=dict, description="Current hyperparameters for TIMO training (alpha, beta, shots_per_class, etc.)")


class DetectorPoolStatus(BaseModel):
    active_version_id: Optional[int] = Field(None, description="ID of currently active detector version", examples=[1])
    active_version_tag: Optional[str] = Field(None, description="Tag of currently active detector version", examples=["yolo26n_vtsdb_mtsd_dfg"])
    data_source: str = Field("backend_survey_submissions", description="Upstream data source for multi-sign full-frame detector training")
    distinct_full_frames_count: int = Field(0, description="Estimated count of distinct full-frame road images containing verified signs", examples=[493])
    total_bounding_box_annotations: int = Field(0, description="Total verified bounding box annotations across full frames", examples=[493])
    min_samples_threshold: int = Field(50, description="Minimum recommended full frames before triggering detector fine-tuning", examples=[50])
    is_eligible_for_retrain: bool = Field(False, description="True if distinct frames meet or exceed the detector threshold", examples=[True])
    current_hyperparams: dict = Field(default_factory=dict, description="Current hyperparameters for YOLO fine-tuning (epochs, imgsz, lr, batch)")


class TrainingDataStatusResponse(BaseModel):
    status: str = Field("SUCCESS", description="Operation execution status", examples=["SUCCESS"])
    total_verified_signs_platform: int = Field(0, description="Total verified signs confirmed across the platform", examples=[493])
    classifier_pool: ClassifierPoolStatus = Field(..., description="Classifier-specific candidate crop training pool status")
    detector_pool: DetectorPoolStatus = Field(..., description="Detector-specific full-frame training pool status")


class ClassDistributionItem(BaseModel):
    class_id: int = Field(..., description="Official catalog sign type ID from backend", examples=[3])
    sign_code: str = Field(..., description="Official traffic sign code under QCVN 41:2019/BGTVT", examples=["P.102"])
    name_vi: str = Field(..., description="Vietnamese name of the sign type", examples=["Đường cấm / Cấm đi ngược chiều"])
    name_en: str = Field(..., description="English name of the sign type", examples=["No entry"])
    category: str = Field(..., description="Sign classification category: PROHIBITORY, MANDATORY, WARNING, INFORMATION, PRIORITY", examples=["PROHIBITORY"])
    verified_count: int = Field(..., description="Count of verified signs for this class across the platform", examples=[23])


class ClassDistributionResponse(BaseModel):
    status: str = Field("SUCCESS", description="Operation execution status", examples=["SUCCESS"])
    total_catalog_classes: int = Field(..., description="Total official traffic sign classes in catalog", examples=[429])
    classes_with_verified_signs: int = Field(..., description="Number of catalog classes that currently have at least one verified sample", examples=[42])
    distribution: List[ClassDistributionItem] = Field(..., description="Complete list of all catalog classes with their verified sample counts")


class AddedClassDiffItem(BaseModel):
    sign_code: str = Field(..., description="Sign code of newly added class not present in the reference model", examples=["I.407b"])
    name_vi: str = Field(..., description="Vietnamese name of the newly added class", examples=["Đường một chiều"])
    name_en: str = Field(..., description="English name of the newly added class", examples=["One way street"])
    current_verified_count: int = Field(..., description="Number of verified samples available for this new class", examples=[5])


class RemovedClassDiffItem(BaseModel):
    sign_code: str = Field(..., description="Sign code of class present in reference model but missing in current pool", examples=["P.101"])
    name_vi: str = Field(..., description="Vietnamese name of the class", examples=["Đường cấm"])
    name_en: str = Field(..., description="English name of the class", examples=["No vehicles"])
    previous_samples_count: int = Field(..., description="Support shots count in the previous reference model", examples=[16])


class ChangedClassDiffItem(BaseModel):
    sign_code: str = Field(..., description="Sign code of class whose sample count differs from the reference model", examples=["P.102"])
    name_vi: str = Field(..., description="Vietnamese name of the sign type", examples=["Đường cấm"])
    name_en: str = Field(..., description="English name of the sign type", examples=["No entry"])
    previous_samples_count: int = Field(..., description="Sample count in the reference model", examples=[16])
    current_samples_count: int = Field(..., description="Available verified sample count in current pool", examples=[39])
    delta_samples: int = Field(..., description="Difference in sample count (current - previous)", examples=[23])


class ClassDistributionDiffResponse(BaseModel):
    status: str = "SUCCESS"
    compared_with_version_id: int = Field(..., description="ID of the model version compared against", examples=[2])
    compared_version_tag: str = Field(..., description="Tag of the model version compared against", examples=["mobileclip_s2"])
    target_version_id: Optional[int] = Field(None, description="ID of target model version compared against (None for current pool)", examples=[6])
    target_version_tag: Optional[str] = Field("current_pool", description="Tag of target model version or 'current_pool'", examples=["classifier-v2.2"])
    total_classes_in_compared_model: int = Field(..., description="Total classes trained in the comparison reference model", examples=[100])
    added_classes: List[AddedClassDiffItem] = Field(default_factory=list, description="Classes present in verified pool but absent in reference model")
    removed_classes: List[RemovedClassDiffItem] = Field(default_factory=list, description="Classes present in reference model but absent in verified pool")
    changed_classes: List[ChangedClassDiffItem] = Field(default_factory=list, description="Classes whose sample counts differ between reference and pool")
    unchanged_classes_count: int = Field(0, description="Count of classes with identical sample counts", examples=[30])


class ClassifierRebuildRequest(BaseModel):
    version_tag: str = Field(..., description="Unique tag for new model version", examples=["classifier-v2.1"])
    note: Optional[str] = Field(None, description="Audit note describing the update trigger")
    alpha: Optional[float] = Field(250.0, description="TIMO alpha fusion factor")
    beta: Optional[float] = Field(0.8, description="TIMO beta text factor")
    shots_per_class: Optional[int] = Field(16, description="Total support shots per class")
    num_anchors: Optional[int] = Field(4, description="Target anchor count per class")
    num_medoids: Optional[int] = Field(12, description="Target medoid count per class")
    min_sharpness: Optional[float] = Field(15.0, description="Laplacian sharpness threshold")
    force: Optional[bool] = Field(False, description="Bypass minimum sample requirement if true")
    sample_limit: Optional[int] = Field(None, description="Optional limit of verified samples to ingest")


class EvaluationMetricsComparison(BaseModel):
    active_version_id: Optional[int] = Field(None, description="Active reference version ID for delta calculation", examples=[2])
    delta_accuracy: Optional[str] = Field(None, description="Difference in Top-1 accuracy compared to active model", examples=["+0.52%"])
    delta_f1: Optional[str] = Field(None, description="Difference in Macro F1 score compared to active model", examples=["+0.0068"])


class EvaluationMetricsDetail(BaseModel):
    split: str = Field("TEST", description="Dataset split evaluated on", examples=["TEST"])
    sample_count: int = Field(5911, description="Number of evaluation samples in test split", examples=[5911])
    top1_accuracy: float = Field(..., description="Top-1 classification accuracy percentage", examples=[71.47])
    top5_accuracy: float = Field(..., description="Top-5 classification accuracy percentage", examples=[87.23])
    macro_f1: float = Field(..., description="Macro-averaged F1 score across all classes", examples=[0.7112])
    latency_ms: float = Field(..., description="Inference evaluation latency per sample in milliseconds", examples=[0.082])
    comparison_with_active: Optional[EvaluationMetricsComparison] = Field(None, description="Comparative delta metrics against currently active model")


class ClassifierRebuildResponse(BaseModel):
    status: str = Field("SUCCESS", description="Operation execution status", examples=["SUCCESS"])
    message: str = Field(..., description="Human-readable execution summary message", examples=["Classifier rebuilt and evaluated successfully."])
    new_model: dict = Field(..., description="Metadata dictionary of the newly created model version")
    evaluation_metrics: EvaluationMetricsDetail = Field(..., description="Automated evaluation metrics computed on the test set")


class ModelMetricItem(BaseModel):
    metric_name: str = Field(..., description="Standard metric name (e.g. top1_accuracy, top5_accuracy, macro_f1)", examples=["top1_accuracy"])
    metric_value: Optional[float] = Field(None, description="Numeric metric value", examples=[71.47])
    split: str = Field("test", description="Dataset split metric was evaluated against", examples=["test"])


class ModelEvaluationResponse(BaseModel):
    version_id: int = Field(..., description="Model version primary key ID", examples=[3])
    version_tag: str = Field(..., description="Version tag string", examples=["classifier-v2.1"])
    is_active: bool = Field(..., description="Whether this version is currently deployed in production", examples=[False])
    created_at: Optional[str] = Field(None, description="Timestamp when evaluation was executed", examples=["2026-11-15T14:30:00Z"])
    metrics: List[ModelMetricItem] = Field(default_factory=list, description="List of recorded evaluation metric key-value items")


class ModelDeployResponse(BaseModel):
    status: str = Field("SUCCESS", description="Operation status", examples=["SUCCESS"])
    message: str = Field(..., description="Deployment summary message", examples=["Model version 3 deployed successfully."])
    deployed_version_id: int = Field(..., description="ID of the model version that was activated", examples=[3])
    version_tag: str = Field(..., description="Version tag of the deployed model", examples=["classifier-v2.1"])
    is_active: bool = Field(..., description="New active status flag (True)", examples=[True])
    deployed_at: str = Field(..., description="ISO 8601 deployment timestamp", examples=["2026-11-15T14:35:00Z"])
    batch_run_id: str = Field(..., description="Deployment tracking batch UUID", examples=["deploy-20261115-3"])


class ModelRollbackResponse(BaseModel):
    status: str = Field("SUCCESS", description="Operation status", examples=["SUCCESS"])
    message: str = Field(..., description="Rollback summary message", examples=["Rolled back from version 4 to version 3."])
    rolled_back_from: int = Field(..., description="ID of the previous active model version that was deactivated", examples=[4])
    active_version_id: int = Field(..., description="ID of the restored active model version", examples=[3])
    version_tag: str = Field(..., description="Version tag of the restored active model", examples=["classifier-v2.1"])


class ModelSwitchResponse(BaseModel):
    status: str = Field("SUCCESS", description="Operation status", examples=["SUCCESS"])
    message: str = Field(..., description="Switch summary message", examples=["Switched active model to version 2."])
    switched_to_version_id: int = Field(..., description="ID of the target model version activated", examples=[2])
    version_tag: str = Field(..., description="Version tag of the activated model", examples=["mobileclip_s2"])
    is_active: bool = Field(..., description="Active status flag (True)", examples=[True])
