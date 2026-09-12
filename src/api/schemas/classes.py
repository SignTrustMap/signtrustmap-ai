from typing import List, Optional
from pydantic import BaseModel, Field


class SignClassItem(BaseModel):
    class_id: int = Field(..., description="Unique numerical class ID", examples=[0])
    class_name: str = Field(..., description="Machine-readable class name", examples=["No_entry"])
    human_readable_name: str = Field(..., description="Human-friendly class name", examples=["No entry"])
    prompt: str = Field(..., description="Formatted text prompt used for zero-shot classification", examples=["a photo of a No entry traffic sign."])
    category: Optional[str] = Field(None, description="Category code (e.g. PROHIBITORY, WARNING, ...)", examples=["PROHIBITORY"])
    name_vi: Optional[str] = Field(None, description="Vietnamese name of the sign according to QCVN 41:2019", examples=["Đường cấm / Cấm đi ngược chiều"])
    name_en: Optional[str] = Field(None, description="English name of the sign", examples=["No entry"])
    standard_code: Optional[str] = Field(None, description="Traffic standard code (e.g. P.102)", examples=["P.102"])


class CategoryItem(BaseModel):
    id: int = Field(..., description="Unique category ID", examples=[6])
    code: str = Field(..., description="Category machine code", examples=["PROHIBITORY"])
    name_vi: str = Field(..., description="Vietnamese category name", examples=["Biển báo cấm"])
    name_en: str = Field(..., description="English category name", examples=["Prohibitory sign"])
    description: str = Field(..., description="Category description", examples=["Nhóm biển báo cấm (P) theo QCVN 41:2019"])


class CategoryListResponse(BaseModel):
    total_categories: int = Field(..., description="Total count of sign categories", examples=[7])
    categories: List[CategoryItem] = Field(default_factory=list, description="List of sign categories")


class ClassifierClassesResponse(BaseModel):
    total_classes: int = Field(..., description="Total count of registered classifier classes", examples=[100])
    prompt_template: str = Field(..., description="Active prompt template used to format prompts", examples=["a photo of a {} traffic sign."])
    classes: List[SignClassItem] = Field(default_factory=list, description="List of classifier classes")


class SignClassDetailResponse(BaseModel):
    class_item: SignClassItem = Field(..., description="Detailed class information")


class CreateClassRequest(BaseModel):
    class_name: str = Field(..., min_length=1, max_length=100, description="Name of the new traffic sign class", examples=["Custom_Speed_Limit_150"])
    custom_id: Optional[int] = Field(None, ge=0, description="Optional custom integer ID. If not provided, next highest integer ID will be assigned.", examples=[100])
    category: Optional[str] = Field(None, description="Category code", examples=["PROHIBITORY"])
    name_vi: Optional[str] = Field(None, description="Vietnamese name", examples=["Đường cấm"])
    name_en: Optional[str] = Field(None, description="English name", examples=["No entry"])
    standard_code: Optional[str] = Field(None, description="Standard code", examples=["P.102"])


class UpdateClassRequest(BaseModel):
    class_name: Optional[str] = Field(None, min_length=1, max_length=100, description="Updated name of the traffic sign class", examples=["Speed_limit_120_kmh"])
    category: Optional[str] = Field(None, description="Category code", examples=["PROHIBITORY"])
    name_vi: Optional[str] = Field(None, description="Vietnamese name", examples=["Đường cấm"])
    name_en: Optional[str] = Field(None, description="English name", examples=["No entry"])
    standard_code: Optional[str] = Field(None, description="Standard code", examples=["P.102"])


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

