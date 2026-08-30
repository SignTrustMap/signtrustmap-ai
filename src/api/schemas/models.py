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
