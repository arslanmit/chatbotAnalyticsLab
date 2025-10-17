"""SQLAlchemy ORM models for persistent storage."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    JSON,
    UniqueConstraint,
    Text,
    Float,
    Boolean,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"
    __table_args__ = (UniqueConstraint("run_id", name="uq_experiment_run_id"),)

    id = Column(Integer, primary_key=True)
    run_id = Column(String(128), nullable=False, index=True)
    model_id = Column(String(128), nullable=False, index=True)
    dataset_type = Column(String(64), nullable=False)
    dataset_name = Column(String(128), nullable=True)
    dataset_path = Column(String(512), nullable=True)
    tags = Column(JSON, default=list)
    training_metrics = Column(JSON, default=dict)
    validation_metrics = Column(JSON, default=dict)
    test_metrics = Column(JSON, default=dict)
    artifacts = Column(JSON, default=dict)
    extra_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ModelArtifactEntry(Base):
    __tablename__ = "model_artifacts"
    __table_args__ = (UniqueConstraint("model_id", "version", name="uq_model_version"),)

    id = Column(Integer, primary_key=True)
    model_id = Column(String(128), index=True, nullable=False)
    version = Column(String(64), nullable=False)
    path = Column(String(512), nullable=False)
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class DatasetRecord(Base):
    __tablename__ = "datasets"
    __table_args__ = (UniqueConstraint("name", "dataset_type", name="uq_dataset_name_type"),)

    id = Column(Integer, primary_key=True)
    name = Column(String(128), nullable=False, index=True)
    dataset_type = Column(String(64), nullable=False, index=True)
    path = Column(String(512), nullable=True)
    metadata_json = Column(JSON, default=dict)
    conversations = relationship("ConversationRecord", back_populates="dataset", cascade="all, delete-orphan")


class ConversationRecord(Base):
    __tablename__ = "conversations"
    __table_args__ = (UniqueConstraint("conversation_id", name="uq_conversation_id"),)

    id = Column(Integer, primary_key=True)
    conversation_id = Column(String(128), nullable=False, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    source_dataset = Column(String(64), nullable=False)
    success = Column(Boolean, nullable=True)
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    dataset = relationship("DatasetRecord", back_populates="conversations")
    turns = relationship(
        "ConversationTurnRecord",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ConversationTurnRecord.order_index",
    )


class ConversationTurnRecord(Base):
    __tablename__ = "conversation_turns"

    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    order_index = Column(Integer, nullable=False)
    speaker = Column(String(32), nullable=False)
    text = Column(Text, nullable=False)
    timestamp_iso = Column(String(64), nullable=True)
    intent = Column(String(128), nullable=True)
    confidence = Column(Float, nullable=True)

    conversation = relationship("ConversationRecord", back_populates="turns")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
