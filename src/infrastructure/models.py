from datetime import datetime, timezone
import uuid
from typing import Optional, Any
from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    String,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Enum,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class AiModelVersion(Base):
    __tablename__ = "ai_model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_type = Column(
        Enum("YOLO26", "CLIP", "TRACKER", name="ai_model_type", create_type=False),
        nullable=False,
    )
    version_tag = Column(String(50), nullable=False)
    artifact_url = Column(String(500), nullable=True)
    trained_at = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, nullable=False, default=False)
    evaluation_metrics = Column(JSONB, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index(
            "ai_model_versions_model_type_version_tag_idx",
            "model_type",
            "version_tag",
            unique=True,
        ),
    )


class CandidateDetectionMetadata(Base):
    __tablename__ = "candidate_detection_metadata"

    candidate_id = Column(UUID(as_uuid=True), primary_key=True)
    frame_number = Column(Integer, nullable=True)
    bbox_x = Column(Float, nullable=True)
    bbox_y = Column(Float, nullable=True)
    bbox_width = Column(Float, nullable=True)
    bbox_height = Column(Float, nullable=True)
    tracking_id = Column(String(100), nullable=True)
    detector_model_version_id = Column(
        Integer,
        ForeignKey("ai_model_versions.id", deferrable=True),
        nullable=True,
    )


class CandidateEmbedding(Base):
    __tablename__ = "candidate_embeddings"

    candidate_id = Column(UUID(as_uuid=True), primary_key=True)
    clip_embedding = Column(Vector(512), nullable=False)
    embedding_model_version_id = Column(
        Integer,
        ForeignKey("ai_model_versions.id", deferrable=True),
        nullable=True,
    )
    generated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class ActiveLearningSelection(Base):
    __tablename__ = "active_learning_selections"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    candidate_id = Column(UUID(as_uuid=True), nullable=False)
    selection_reason = Column(
        Enum("UNCERTAIN", "HIGH_CONFIDENCE", name="al_selection_reason", create_type=False),
        nullable=False,
    )
    selected_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )