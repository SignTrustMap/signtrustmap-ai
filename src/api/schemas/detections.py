from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel, Field


class DetectionMetadataItem(BaseModel):
    candidate_id: UUID
    frame_number: Optional[int] = None
    bbox_x: Optional[float] = None
    bbox_y: Optional[float] = None
    bbox_width: Optional[float] = None
    bbox_height: Optional[float] = None
    tracking_id: Optional[str] = None
    detector_model_version_id: Optional[int] = None


class BatchDetectionMetadataRequest(BaseModel):
    items: List[DetectionMetadataItem] = Field(..., min_length=1)


class DetectionMetadataResponse(BaseModel):
    status: str
    count: int
    items: Optional[List[DetectionMetadataItem]] = None