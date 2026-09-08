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


class SubmissionInferenceResponse(BaseModel):
    submission_id: str
    media_type: Literal["VIDEO", "IMAGE"]
    total_frames: int = Field(..., ge=0)
    signs: List[SubmissionSignResult]
    video_fps: Optional[float] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
