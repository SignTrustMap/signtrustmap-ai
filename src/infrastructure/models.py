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
    Text,
    ForeignKey,
    Index,
    Enum,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_type = Column(
        Enum("DETECTOR", "CLASSIFIER", name="model_type", create_type=False),
        nullable=False,
    )
    version_tag = Column(String(50), nullable=False)
    is_active = Column(Boolean, nullable=False, default=False)
    note = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index(
            "model_versions_model_type_version_tag_idx",
            "model_type",
            "version_tag",
            unique=True,
        ),
    )



class ModelArtifact(Base):
    __tablename__ = "model_artifacts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(
        Integer,
        ForeignKey("model_versions.id", ondelete="CASCADE"),
        nullable=False,
    )
    artifact_type = Column(String(20), nullable=False)
    s3_key = Column(String(500), nullable=False)
    artifact_url = Column(String(1000), nullable=False)
    file_size_bytes = Column(BigInteger, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index("model_artifacts_model_id_artifact_type_idx", "model_id", "artifact_type", unique=True),
    )


class ModelEvaluationMetric(Base):
    __tablename__ = "model_evaluation_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(
        Integer,
        ForeignKey("model_versions.id", ondelete="CASCADE"),
        nullable=False,
    )
    metric_name = Column(String(50), nullable=False)
    metric_value = Column(Float, nullable=True)
    split = Column(String(20), nullable=False, default="val")
    note = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index("model_evaluation_metrics_model_metric_split_idx", "model_id", "metric_name", "split", unique=True),
    )


class ModelClass(Base):
    __tablename__ = "model_classes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(
        Integer,
        ForeignKey("model_versions.id", ondelete="CASCADE"),
        nullable=False,
    )
    class_index = Column(Integer, nullable=False)
    class_name = Column(String(100), nullable=False)
    sign_code = Column(String(50), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index("model_classes_model_id_class_index_idx", "model_id", "class_index", unique=True),
        Index("model_classes_model_id_class_name_idx", "model_id", "class_name", unique=True),
    )


class AiSystemConfig(Base):
    __tablename__ = "ai_system_configs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    config_key = Column(String(100), nullable=False, unique=True)
    config_value = Column(JSONB, nullable=False)
    description = Column(Text, nullable=True)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class CandidateEmbedding(Base):
    __tablename__ = "candidate_embeddings"

    candidate_id = Column(UUID(as_uuid=True), primary_key=True)
    s3_key = Column(String(500), nullable=False)
    model_id = Column(Integer, ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True)
    embedding = Column(Vector(512), nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class ActiveLearningRecord(Base):
    __tablename__ = "active_learning_records"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    candidate_id = Column(
        UUID(as_uuid=True),
        ForeignKey("candidate_embeddings.candidate_id", ondelete="CASCADE"),
        nullable=False,
    )
    model_id = Column(Integer, ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True)
    class_id = Column(Integer, ForeignKey("model_classes.id", ondelete="SET NULL"), nullable=True)
    box_xyxy = Column(JSONB, nullable=True)
    detect_confidence = Column(Float, nullable=True)
    classify_confidence = Column(Float, nullable=True)
    strategy = Column(String(50), nullable=False)
    acquisition_score = Column(Float, nullable=False)
    flip_count = Column(Integer, nullable=False, default=0)
    flip_classes = Column(JSONB, nullable=False, default=list)
    calibrated_entropy = Column(Float, nullable=False, default=0.0)
    batch_run_id = Column(String(100), nullable=True)
    cluster_id = Column(Integer, nullable=True)
    tri_rank_res = Column(Integer, nullable=True)
    tri_rank_info = Column(Integer, nullable=True)
    tri_rank_rep = Column(Integer, nullable=True)
    aggregate_rank = Column(Integer, nullable=True)
    is_selected = Column(Boolean, nullable=False, default=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class ActiveLearningContextualPrior(Base):
    __tablename__ = "active_learning_contextual_priors"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    run_id = Column(String(100), nullable=False)
    model_id = Column(Integer, ForeignKey("model_versions.id", ondelete="CASCADE"), nullable=True)
    pool_size = Column(Integer, nullable=False)
    top_n_per_class = Column(Integer, nullable=False, default=5)
    p0_prior = Column(JSONB, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index("idx_al_priors_run_model", "run_id", "model_id", unique=True),
    )


class SupportShot(Base):
    __tablename__ = "support_shots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey("model_versions.id", ondelete="CASCADE"), nullable=False)
    class_id = Column(Integer, ForeignKey("model_classes.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20), nullable=False, default="MEDOID")
    s3_key = Column(String(255), nullable=False)
    embedding = Column(Vector(512), nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class TextEmbedding(Base):
    __tablename__ = "text_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey("model_versions.id", ondelete="CASCADE"), nullable=False)
    class_id = Column(Integer, ForeignKey("model_classes.id", ondelete="CASCADE"), nullable=False)
    embedding = Column(Vector(512), nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class EvaluationEmbedding(Base):
    __tablename__ = "evaluation_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey("model_versions.id", ondelete="CASCADE"), nullable=False)
    class_id = Column(Integer, ForeignKey("model_classes.id", ondelete="CASCADE"), nullable=False)
    split = Column(String(10), nullable=False)
    embedding = Column(Vector(512), nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


