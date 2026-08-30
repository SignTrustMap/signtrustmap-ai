from typing import List, Optional
from pydantic import BaseModel, Field


class ModelExportRequest(BaseModel):
    weights_name: str = Field(..., description="Name of the weights file in weights directory or absolute path", examples=["yolo26n_vtsdb_mtsd_dfg.pt"])
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


class ModelFileInfo(BaseModel):
    filename: str = Field(..., description="File name of the model")
    format: str = Field(..., description="Model format extension (pt, onnx, engine)")
    size_bytes: int = Field(..., description="Size of the file in bytes")
    size_human: str = Field(..., description="Human-readable file size")
    modified_at: str = Field(..., description="Last modified timestamp")


class ModelListResponse(BaseModel):
    total: int = Field(..., description="Total count of model files")
    models: List[ModelFileInfo] = Field(..., description="List of models in weights directory")
