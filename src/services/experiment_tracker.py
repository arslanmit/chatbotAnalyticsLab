"""
Experiment tracking utilities for logging training runs, comparing results, and managing artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExperimentRecord:
    """Represents a stored experiment run with metadata and metrics."""

    run_id: str
    created_at: str
    model_id: str
    dataset: Dict[str, Any]
    training_config: Dict[str, Any]
    training_metrics: Dict[str, Any]
    validation_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "model_id": self.model_id,
            "dataset": self.dataset,
            "training_config": self.training_config,
            "training_metrics": self.training_metrics,
            "validation_metrics": self.validation_metrics,
            "test_metrics": self.test_metrics,
            "tags": self.tags,
            "artifacts": self.artifacts,
            "extra_metadata": self.extra_metadata,
        }


class ExperimentTracker:
    """Persist experiment metadata and enable comparison/reporting."""

    def __init__(self, storage_path: str = "./experiments/experiments.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.storage_path.exists():
            self._save_records([])
        logger.info("ExperimentTracker initialized at %s", self.storage_path.resolve())

    # Public API -----------------------------------------------------------------

    def log_experiment(
        self,
        run_summary: Dict[str, Any],
        tags: Optional[List[str]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> ExperimentRecord:
        """Log a training run and persist experiment metadata."""
        if "run_id" not in run_summary:
            raise ValueError("run_summary must include a 'run_id'")

        pipeline_config = run_summary.get("pipeline_config", {}) or {}
        model_id = pipeline_config.get("model_id", run_summary.get("model_artifact_path", ""))

        artifacts: Dict[str, str] = {}
        model_artifact = run_summary.get("model_artifact_path")
        if model_artifact:
            artifacts["model"] = model_artifact

        record = ExperimentRecord(
            run_id=run_summary["run_id"],
            created_at=run_summary.get("timestamp", datetime.utcnow().isoformat()),
            model_id=str(model_id),
            dataset=run_summary.get("dataset", {}),
            training_config=run_summary.get("training_config", {}),
            training_metrics=run_summary.get("training_metrics", {}),
            validation_metrics=run_summary.get("validation_metrics", {}),
            test_metrics=run_summary.get("test_metrics", {}),
            tags=tags or [],
            artifacts=artifacts,
            extra_metadata=extra_metadata or {},
        )

        records = self._load_records()
        # Replace existing record with same run_id if present
        records = [r for r in records if r["run_id"] != record.run_id]
        records.append(record.to_dict())
        self._save_records(records)
        logger.info("Logged experiment run_id=%s", record.run_id)
        return record

    def list_experiments(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List experiments optionally filtered by metadata values."""
        records = self._load_records()
        if not filters:
            return records

        def matches(record: Dict[str, Any]) -> bool:
            for key, value in filters.items():
                if key not in record:
                    return False
                if isinstance(value, dict):
                    if not isinstance(record[key], dict):
                        return False
                    for sub_key, sub_value in value.items():
                        if record[key].get(sub_key) != sub_value:
                            return False
                else:
                    if record[key] != value:
                        return False
            return True

        return [record for record in records if matches(record)]

    def compare_experiments(
        self,
        metric_path: str,
        top_n: int = 5,
        greater_is_better: bool = True,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Compare experiments using a metric path (e.g., 'validation_metrics.macro_f1')."""
        records = self.list_experiments(filters=filters)
        metric_keys = metric_path.split(".")

        def extract_metric(record: Dict[str, Any]) -> Optional[float]:
            value: Any = record
            for key in metric_keys:
                if isinstance(value, dict):
                    value = value.get(key)
                else:
                    return None
            return float(value) if isinstance(value, (int, float)) else None

        scored_records = []
        for record in records:
            score = extract_metric(record)
            if score is not None:
                scored_records.append((record, score))

        scored_records.sort(key=lambda item: item[1], reverse=greater_is_better)
        return [
            {
                "run_id": record["run_id"],
                "score": score,
                "metric": metric_path,
                "model_artifact": record["artifacts"].get("model"),
            }
            for record, score in scored_records[:top_n]
        ]

    def register_artifact(self, run_id: str, artifact_type: str, path: str):
        """Attach an artifact to an existing experiment."""
        records = self._load_records()
        updated = False
        for record in records:
            if record["run_id"] == run_id:
                record.setdefault("artifacts", {})
                record["artifacts"][artifact_type] = path
                updated = True
                break
        if updated:
            self._save_records(records)
            logger.info("Registered artifact for run_id=%s type=%s", run_id, artifact_type)
        else:
            logger.warning("Attempted to register artifact for unknown run_id=%s", run_id)

    def get_experiment(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single experiment by run_id."""
        records = self._load_records()
        for record in records:
            if record["run_id"] == run_id:
                return record
        return None

    # Internal helpers -----------------------------------------------------------

    def _load_records(self) -> List[Dict[str, Any]]:
        if not self.storage_path.exists():
            return []
        with open(self.storage_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return data if isinstance(data, list) else []
            except json.JSONDecodeError:
                logger.error("Failed to decode experiment storage file; resetting records")
                return []

    def _save_records(self, records: List[Dict[str, Any]]):
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
