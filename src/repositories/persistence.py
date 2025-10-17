"""Repositories for persisting experiments, datasets, and model artifacts."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.repositories.database import get_session
from src.repositories.orm import ExperimentRun, ModelArtifactEntry, DatasetRecord
from src.utils.logging import get_logger

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
