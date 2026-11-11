from datetime import datetime
from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel, Field


class EmbeddingItem(BaseModel):
    candidate_id: UUID
    embedding: List[float] = Field(..., min_length=512, max_length=512)
    embedding_model_version_id: Optional[int] = None
    generated_at: Optional[datetime] = None


class EmbeddingBatchRequest(BaseModel):
    items: List[EmbeddingItem] = Field(..., min_length=1)


class EmbeddingSimilarityQuery(BaseModel):
    embedding: List[float] = Field(..., min_length=512, max_length=512)
    top_k: int = Field(default=5, ge=1, le=50)


class EmbeddingSimilarityResult(BaseModel):
    candidate_id: UUID
    similarity_score: float
    embedding_model_version_id: Optional[int] = None