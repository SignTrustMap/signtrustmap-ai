from typing import List, Optional
from pydantic import BaseModel, Field


class SignDetectionResult(BaseModel):
    class_id: int = Field(..., description="Traffic sign class ID index", examples=[0])
    class_name: str = Field(..., description="Traffic sign class name", examples=["speed_limit_50"])
    confidence: float = Field(..., description="Overall inference confidence score (0.0 - 1.0)", examples=[0.9254])
    detect_confidence: Optional[float] = Field(None, description="Detector bounding box confidence score", examples=[0.9521])
    classify_confidence: Optional[float] = Field(None, description="Classifier recognition confidence score", examples=[0.9254])
    box_xyxy: List[float] = Field(..., description="Bounding box coordinates [xmin, ymin, xmax, ymax]", examples=[[120.5, 200.1, 180.2, 260.8]])


class ImageProcessingResponse(BaseModel):
    task_id: str = Field(..., description="Unique identifier of the image processing job", examples=["img-task-999"])
    status: str = Field(..., description="Initialization status of the task", examples=["PROCESSING"])
    message: str = Field(..., description="Response message detail", examples=["Image accepted. Processing started."])


class VideoProcessingResponse(BaseModel):
    task_id: str = Field(..., description="Unique identifier of the video processing job", examples=["vid-task-111"])
    status: str = Field(..., description="Initialization status of the task", examples=["PROCESSING"])
    message: str = Field(..., description="Response message detail", examples=["Video accepted. Processing started."])


class VideoSignResult(BaseModel):
    id: int = Field(..., description="Unique tracking identifier of the sign", examples=[1])
    class_id: int = Field(..., description="Traffic sign class ID index", examples=[0])
    class_name: str = Field(..., description="Traffic sign class name", examples=["speed_limit_50"])
    confidence: float = Field(..., description="Highest overall confidence score recorded for this tracked sign", examples=[0.9452])
    detect_confidence: Optional[float] = Field(None, description="Detector confidence score on the best frame", examples=[0.9612])
    classify_confidence: Optional[float] = Field(None, description="Classifier confidence score on the best frame", examples=[0.9452])
    best_frame_idx: int = Field(..., description="Frame index of the clearest frame containing this sign", examples=[142])
    best_timestamp_seconds: float = Field(..., description="Timestamp in seconds within the video of the best frame", examples=[4.73])
    box_xyxy: List[float] = Field(..., description="Bounding box coordinates on the clearest frame [xmin, ymin, xmax, ymax]", examples=[[300.2, 150.5, 360.8, 210.9]])
    occurrences: int = Field(..., description="Total number of frames this sign appeared in", examples=[18])
    frame_image_url: Optional[str] = Field(None, description="Public URL of the best sign image frame uploaded to MinIO", examples=["http://minio:9000/best-frames/vid-task-111/sign_1.jpg"])


class VideoWebhookPayload(BaseModel):
    task_id: str = Field(..., description="Unique identifier of the processing task", examples=["vid-task-111"])
    status: str = Field(..., description="Execution completion status", examples=["COMPLETED"])
    video_fps: Optional[float] = Field(None, description="Video framerate in FPS", examples=[30.0])
    total_frames: Optional[int] = Field(None, description="Total number of video frames processed", examples=[1250])
    total_signs: Optional[int] = Field(None, description="Total number of unique signs tracked", examples=[3])
    result: Optional[List[VideoSignResult]] = Field(None, description="List of tracked traffic signs")
    error: Optional[str] = Field(None, description="Error detail message if the status is FAILED")


class ImageWebhookPayload(BaseModel):
    task_id: str = Field(..., description="Unique identifier of the processing task", examples=["img-task-999"])
    status: str = Field(..., description="Execution completion status", examples=["COMPLETED"])
    image_width: Optional[int] = Field(None, description="Width of the processed image in pixels", examples=[1920])
    image_height: Optional[int] = Field(None, description="Height of the processed image in pixels", examples=[1080])
    total_detections: Optional[int] = Field(None, description="Total number of detected signs", examples=[2])
    result: Optional[List[SignDetectionResult]] = Field(None, description="List of detected signs")
    error: Optional[str] = Field(None, description="Error detail message if the status is FAILED")
