"""Repositories for persisting experiments, datasets, and model artifacts."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
from datetime import datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.repositories.database import get_session
from src.repositories.orm import (
    ExperimentRun,
    ModelArtifactEntry,
    DatasetRecord,
    ConversationRecord,
    ConversationTurnRecord,
)
from src.utils.logging import get_logger
from src.models.core import Dataset, DatasetType, Conversation, ConversationTurn, Speaker

logger = get_logger(__name__)


class ExperimentRepository:
    """Persist experiment runs and fetch them for analytics."""

    def save_from_summary(self, summary: Dict[str, Any]) -> ExperimentRun:
        with get_session() as session:
            run = self._upsert(session, summary)
            session.flush()
            logger.debug("Persisted experiment run %s", run.run_id)
            return run

    def list_runs(self, limit: Optional[int] = None) -> List[ExperimentRun]:
        with get_session() as session:
            query = session.query(ExperimentRun).order_by(ExperimentRun.created_at.desc())
            if limit:
                query = query.limit(limit)
            return query.all()

    def _upsert(self, session: Session, summary: Dict[str, Any]) -> ExperimentRun:
        run_id = summary.get("run_id")
        run = session.query(ExperimentRun).filter_by(run_id=run_id).one_or_none()

        values = {
            "model_id": summary.get("pipeline_config", {}).get("model_id")
            or summary.get("model_artifact_path", "unknown"),
            "dataset_type": summary.get("dataset", {}).get("type", "unknown"),
            "dataset_name": summary.get("dataset", {}).get("name"),
            "dataset_path": summary.get("dataset", {}).get("path"),
            "training_metrics": summary.get("training_metrics"),
            "validation_metrics": summary.get("validation_metrics"),
            "test_metrics": summary.get("test_metrics"),
            "artifacts": {"model": summary.get("model_artifact_path")},
            "extra_metadata": summary.get("extra_metadata") or summary.get("evaluation_summary"),
            "tags": summary.get("tags", []),
        }

        if run is None:
            run = ExperimentRun(run_id=run_id, **values)
            session.add(run)
        else:
            for key, value in values.items():
                setattr(run, key, value)
        return run


class ModelArtifactRepository:
    """Persist model artifact metadata."""

    def save(self, model_id: str, version: str, path: str, metadata: Dict[str, Any]) -> ModelArtifactEntry:
        with get_session() as session:
            entry = ModelArtifactEntry(
                model_id=model_id,
                version=version,
                path=path,
                metadata_json=metadata,
            )
            try:
                session.add(entry)
                session.flush()
            except IntegrityError:
                session.rollback()
                entry = (
                    session.query(ModelArtifactEntry)
                    .filter_by(model_id=model_id, version=version)
                    .one()
                )
                entry.path = path
                entry.metadata_json = metadata
                session.add(entry)
                session.flush()
            logger.debug("Registered model artifact %s:%s", model_id, version)
            return entry


class DatasetRepository:
    """Persist dataset metadata to track processed data."""

    def save(self, name: str, dataset_type: str, path: Optional[str], metadata: Dict[str, Any]) -> DatasetRecord:
        with get_session() as session:
            record = (
                session.query(DatasetRecord)
                .filter_by(name=name, dataset_type=dataset_type)
                .one_or_none()
            )
            if record is None:
                record = DatasetRecord(
                    name=name,
                    dataset_type=dataset_type,
                    path=path,
                    metadata_json=metadata,
                )
                session.add(record)
            else:
                record.path = path
                record.metadata_json = metadata
            session.flush()
            logger.debug("Stored dataset record %s (%s)", name, dataset_type)
            return record

    def get(self, name: str, dataset_type: str) -> Optional[DatasetRecord]:
        with get_session() as session:
            return (
                session.query(DatasetRecord)
                .filter_by(name=name, dataset_type=dataset_type)
                .one_or_none()
            )


class ConversationRepository:
    """Persist conversation data and reconstruct domain models."""

    def __init__(self):
        self._dataset_repository = DatasetRepository()

    def save_dataset_conversations(self, dataset: Dataset) -> None:
        with get_session() as session:
            dataset_record = (
                session.query(DatasetRecord)
                .filter_by(name=dataset.name, dataset_type=dataset.dataset_type.value)
                .one_or_none()
            )
            if dataset_record is None:
                dataset_record = DatasetRecord(
                    name=dataset.name,
                    dataset_type=dataset.dataset_type.value,
                    path=dataset.metadata.get("path"),
                    metadata_json=dataset.metadata,
                )
                session.add(dataset_record)
                session.flush()

            for conversation in dataset.conversations:
                record = (
                    session.query(ConversationRecord)
                    .filter_by(conversation_id=conversation.id)
                    .one_or_none()
                )
                if record is None:
                    record = ConversationRecord(
                        conversation_id=conversation.id,
                        dataset=dataset_record,
                        source_dataset=dataset.dataset_type.value,
                        success=conversation.success,
                        metadata_json=_safe_json(conversation.metadata),
                    )
                    session.add(record)
                    session.flush()
                else:
                    record.dataset = dataset_record
                    record.source_dataset = dataset.dataset_type.value
                    record.success = conversation.success
                    record.metadata_json = _safe_json(conversation.metadata)

                session.query(ConversationTurnRecord).filter_by(conversation_id=record.id).delete()
                turns = [
                    ConversationTurnRecord(
                        conversation_id=record.id,
                        order_index=index,
                        speaker=_speaker_to_str(turn.speaker),
                        text=turn.text,
                        timestamp_iso=turn.timestamp.isoformat() if turn.timestamp else None,
                        intent=turn.intent,
                        confidence=turn.confidence,
                    )
                    for index, turn in enumerate(conversation.turns)
                ]
                session.add_all(turns)
            logger.debug(
                "Persisted %d conversations for dataset %s",
                len(dataset.conversations),
                dataset.name,
            )

    def load_conversations(
        self,
        dataset_name: str,
        dataset_type: DatasetType,
        limit: Optional[int] = None,
    ) -> List[Conversation]:
        with get_session() as session:
            dataset_record = (
                session.query(DatasetRecord)
                .filter_by(name=dataset_name, dataset_type=dataset_type.value)
                .one_or_none()
            )
            if dataset_record is None:
                return []
            query = (
                session.query(ConversationRecord)
                .filter_by(dataset_id=dataset_record.id)
                .order_by(ConversationRecord.created_at.desc())
            )
            if limit:
                query = query.limit(limit)
            records = query.all()
            return [self._record_to_conversation(record, dataset_type) for record in records]

    @staticmethod
    def _record_to_conversation(record: ConversationRecord, dataset_type: DatasetType) -> Conversation:
        turns = [
            ConversationTurn(
                speaker=_str_to_speaker(turn_record.speaker),
                text=turn_record.text,
                timestamp=datetime.fromisoformat(turn_record.timestamp_iso) if turn_record.timestamp_iso else None,
                intent=turn_record.intent,
                confidence=turn_record.confidence,
            )
            for turn_record in record.turns
        ]
        return Conversation(
            id=record.conversation_id,
            turns=turns,
            source_dataset=dataset_type,
            metadata=record.metadata_json or {},
            success=record.success,
        )


def _safe_json(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not metadata:
        return {}
    def convert(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    return {k: convert(v) for k, v in metadata.items()}


def _speaker_to_str(speaker: Speaker | str) -> str:
    if isinstance(speaker, Speaker):
        return speaker.value
    return speaker


def _str_to_speaker(value: str) -> Speaker:
    try:
        return Speaker(value)
    except ValueError:
        return Speaker.USER
