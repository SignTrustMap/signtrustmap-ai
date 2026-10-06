from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class MediaObjectReference(BaseModel):
    media_type: Literal["VIDEO", "GPX", "IMAGE"]
    bucket: str = Field(..., min_length=1, max_length=255)
    object_key: str = Field(..., min_length=1, max_length=1024)

    @field_validator("bucket", "object_key")
    @classmethod
    def reject_unsafe_references(cls, value: str) -> str:
        if value != value.strip() or "\x00" in value:
            raise ValueError("MinIO references must not contain surrounding whitespace or NUL")
        return value


class SubmissionInferenceRequest(BaseModel):
    media_objects: List[MediaObjectReference] = Field(..., min_length=1)


class SubmissionSignResult(BaseModel):
    track_id: Optional[int]
    class_id: int
    class_name: str
    confidence: float = Field(..., ge=0, le=1)
    detect_confidence: float = Field(..., ge=0, le=1)
    classify_confidence: Optional[float] = Field(None, ge=0, le=1)
    box_xyxy: tuple[float, float, float, float]
    best_frame_key: str = Field(..., min_length=1)
    sign_crop_key: str = Field(..., min_length=1)
    frame_index: int = Field(..., ge=0)
    timestamp_seconds: float = Field(..., ge=0)
    category: Optional[str] = None
    name_vi: Optional[str] = None
    name_en: Optional[str] = None
    standard_code: Optional[str] = None


class SubmissionInferenceResponse(BaseModel):
    submission_id: str
    media_type: Literal["VIDEO", "IMAGE"]
    total_frames: int = Field(..., ge=0)
    signs: List[SubmissionSignResult]
    video_fps: Optional[float] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None


class EarlyClassificationRequest(BaseModel):
    media_url: Optional[str] = Field(None, description="Direct URL to the image (MinIO or HTTP)")
    bucket: Optional[str] = Field(None, description="MinIO bucket name (default: stm-sign-crops)")
    object_key: Optional[str] = Field(None, description="MinIO object key")
    sign_id: Optional[str] = Field(None, description="Target verified sign ID if revalidating")
    task_id: Optional[str] = Field(None, description="Revalidation task ID")


class EarlyClassificationResponse(BaseModel):
    status: str
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    confidence: float = 0.0
    detect_confidence: Optional[float] = None
    classify_confidence: Optional[float] = None
    category: Optional[str] = None
    name_vi: Optional[str] = None
    name_en: Optional[str] = None
    standard_code: Optional[str] = None
    sign_id: Optional[str] = None
    task_id: Optional[str] = None
    media_url: Optional[str] = None
    signs: List[SubmissionSignResult] = Field(default_factory=list)

