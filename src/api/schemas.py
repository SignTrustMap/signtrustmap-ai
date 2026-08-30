from pydantic import BaseModel, Field
from typing import List, Optional

class SignGPSResult(BaseModel):
    latitude: float = Field(..., description="GPS latitude of the geolocated traffic sign", examples=[10.7756])
    longitude: float = Field(..., description="GPS longitude of the geolocated traffic sign", examples=[106.7019])
    distance_meters: float = Field(..., description="Calculated distance from the camera to the traffic sign in meters", examples=[4.25])
    height_meters: float = Field(..., description="Estimated height of the traffic sign relative to the road surface in meters", examples=[1.85])

class SignDetectionResult(BaseModel):
    class_name: str = Field(..., description="Traffic sign class name (e.g., speed limit class name)", examples=["speed_limit_50"])
    confidence: float = Field(..., description="Inference confidence score of the YOLO model (0.0 - 1.0)", examples=[0.9254])
    box_xyxy: List[float] = Field(..., description="Bounding box coordinates on the image frame [xmin, ymin, xmax, ymax]", examples=[[120.5, 200.1, 180.2, 260.8]])
    timestamp: str = Field(..., description="Timestamp when the image/frame was captured (format: YYYY-MM-DD HH:MM:SS.mmm)", examples=["2026-08-12 19:00:00.000"])
    gps: SignGPSResult = Field(..., description="Calculated GPS position of the detected sign")

class ImageProcessingResponse(BaseModel):
    task_id: str = Field(..., description="Unique identifier (Task ID) of the image processing job", examples=["img-task-999"])
    status: str = Field(..., description="Initialization processing status of the task", examples=["PROCESSING"])
    message: str = Field(..., description="System response message detail", examples=["Image accepted. Processing started."])

class VideoProcessingResponse(BaseModel):
    task_id: str = Field(..., description="Unique identifier (Task ID) of the video processing job", examples=["vid-task-111"])
    status: str = Field(..., description="Initialization processing status of the task", examples=["PROCESSING"])
    message: str = Field(..., description="System response message detail", examples=["Video accepted. Processing started."])

class VideoSignResult(BaseModel):
    id: int = Field(..., description="Unique tracking identifier (Tracking ID) of the sign within the video", examples=[1])
    class_name: str = Field(..., description="Traffic sign class name", examples=["speed_limit_50"])
    confidence: float = Field(..., description="Highest confidence score recorded for this tracked sign (0.0 - 1.0)", examples=[0.9452])
    timestamp: str = Field(..., description="Timestamp of the clearest frame containing this sign (format: YYYY-MM-DD HH:MM:SS.mmm)", examples=["2026-08-12 19:01:23.456"])
    box_xyxy: List[float] = Field(..., description="Bounding box coordinates on the clearest frame [xmin, ymin, xmax, ymax]", examples=[[300.2, 150.5, 360.8, 210.9]])
    frame_image_url: Optional[str] = Field(None, description="Public URL of the best sign image frame uploaded to MinIO", examples=["http://minio:9000/best-frames/vid-task-111/sign_1.jpg"])
    gps: SignGPSResult = Field(..., description="Calculated GPS position of the sign geolocated across tracking frames")

class VideoWebhookPayload(BaseModel):
    task_id: str = Field(..., description="Unique identifier of the processing task", examples=["vid-task-111"])
    status: str = Field(..., description="Execution task completion status", examples=["COMPLETED"])
    result: Optional[List[VideoSignResult]] = Field(None, description="List of geolocated traffic signs detected in the video (populated when status is COMPLETED)")
    error: Optional[str] = Field(None, description="Error detail message if the status is FAILED", examples=["Failed to process video frame 120"])

class ImageWebhookPayload(BaseModel):
    task_id: str = Field(..., description="Unique identifier of the processing task", examples=["img-task-999"])
    status: str = Field(..., description="Execution task completion status", examples=["COMPLETED"])
    result: Optional[List[SignDetectionResult]] = Field(None, description="List of geolocated traffic signs detected in the image (populated when status is COMPLETED)")
    error: Optional[str] = Field(None, description="Error detail message if the status is FAILED", examples=["Failed to process image EXIF metadata"])

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

