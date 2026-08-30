from typing import List, Optional
from pydantic import BaseModel, Field


class SignClassItem(BaseModel):
    class_id: int = Field(..., description="Unique numerical class ID", examples=[0])
    class_name: str = Field(..., description="Machine-readable class name", examples=["No_entry"])
    human_readable_name: str = Field(..., description="Human-friendly class name", examples=["No entry"])
    prompt: str = Field(..., description="Formatted text prompt used for zero-shot classification", examples=["a photo of a No entry traffic sign."])


class ClassifierClassesResponse(BaseModel):
    total_classes: int = Field(..., description="Total count of registered classifier classes", examples=[100])
    prompt_template: str = Field(..., description="Active prompt template used to format prompts", examples=["a photo of a {} traffic sign."])
    classes: List[SignClassItem] = Field(default_factory=list, description="List of classifier classes")


class SignClassDetailResponse(BaseModel):
    class_item: SignClassItem = Field(..., description="Detailed class information")


class CreateClassRequest(BaseModel):
    class_name: str = Field(..., min_length=1, max_length=100, description="Name of the new traffic sign class", examples=["Custom_Speed_Limit_150"])
    custom_id: Optional[int] = Field(None, ge=0, description="Optional custom integer ID. If not provided, next highest integer ID will be assigned.", examples=[100])


class UpdateClassRequest(BaseModel):
    class_name: str = Field(..., min_length=1, max_length=100, description="Updated name of the traffic sign class", examples=["Speed_limit_120_kmh"])


class ClassActionResponse(BaseModel):
    status: str = Field(..., description="Action outcome status", examples=["CREATED", "UPDATED", "DELETED", "RESET_SUCCESSFUL"])
    message: str = Field(..., description="Detailed outcome message", examples=["Class successfully created."])
    class_item: Optional[SignClassItem] = Field(None, description="Affected class details")


class ClassSyncRequest(BaseModel):
    model_name: Optional[str] = Field(None, description="Optional classifier weights filename (e.g. 'mobileclip_s2.engine', 'ViT-B-16.pt') to sync embeddings for. If omitted, syncs active/loaded classifier.", examples=["mobileclip_s2.engine"])
    prompt_template: Optional[str] = Field(None, description="Optional prompt template override to use for generating text embeddings", examples=["a photo of a {} traffic sign."])


class ClassSyncResponse(BaseModel):
    status: str = Field(..., description="Sync outcome status", examples=["SYNC_SUCCESSFUL"])
    target_model: str = Field(..., description="Filename of the classifier model for which embeddings were synced", examples=["mobileclip_s2.engine"])
    total_classes_synced: int = Field(..., description="Total number of classes synced", examples=[100])
    is_classifier_loaded: bool = Field(..., description="Whether this classifier model was currently loaded in RAM/GPU during sync", examples=[True])
    message: str = Field(..., description="Detailed outcome message", examples=["Text embeddings successfully rebuilt and saved."])

