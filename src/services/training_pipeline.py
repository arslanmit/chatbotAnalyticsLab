"""
Automated training pipeline orchestrating dataset loading, preprocessing,
model training, evaluation, and versioned artifact storage.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from src.config.settings import settings
from src.models.core import Dataset, DatasetType, TrainingConfig, PerformanceMetrics
from src.models.intent_classifier import IntentClassifier
from src.repositories.dataset_loaders import DatasetLoaderFactory
from src.repositories.model_repository import ModelRepository
from src.services.data_preprocessor import DataPreprocessor
from src.services.model_evaluator import ModelEvaluator
from src.services.experiment_tracker import ExperimentTracker
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingPipelineConfig:
    """Configuration for orchestrating a single training pipeline run."""

    dataset_type: DatasetType
    dataset_path: Optional[str] = None
    model_id: str = "intent_classifier"
    preprocess: bool = True
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    random_seed: int = 42
    output_dir: str = "./pipeline_runs"
    save_evaluation_artifacts: bool = True
    normalize_text: bool = True

    def resolved_dataset_path(self) -> Path:
        if self.dataset_path:
            dataset_path = Path(self.dataset_path)
            if dataset_path.exists():
                return dataset_path
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        base_dir = Path(settings.data.dataset_dir)
        candidates = [
            base_dir / self.dataset_type.value,
            base_dir / self.dataset_type.value.lower(),
            base_dir / self.dataset_type.value.upper(),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Dataset for type '{self.dataset_type.value}' not found in {base_dir}"
        )


class TrainingPipeline:
    """High-level orchestrator for intent classifier training runs."""

    def __init__(
        self,
        pipeline_config: TrainingPipelineConfig,
        training_config: TrainingConfig,
        model_repository: Optional[ModelRepository] = None,
        experiment_tracker: Optional[ExperimentTracker] = None,
    ):
        self.pipeline_config = pipeline_config
        self.training_config = training_config
        self.model_repository = model_repository or ModelRepository()
        self.preprocessor = DataPreprocessor()
        self.evaluator = ModelEvaluator()
        self.experiment_tracker = experiment_tracker or ExperimentTracker()

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{pipeline_config.model_id}_{timestamp}"
        self.run_dir = Path(pipeline_config.output_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._configure_file_logging()

        logger.info("Initialized TrainingPipeline run_id=%s", self.run_id)
        logger.info("Training config: %s", self.training_config.to_dict())
        logger.info("Pipeline config: %s", asdict(self.pipeline_config))

    # Public API -----------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Execute the training workflow end-to-end."""
        dataset_path = self.pipeline_config.resolved_dataset_path()
        dataset = self._load_dataset(dataset_path)

        if self.pipeline_config.preprocess:
            dataset = self._preprocess_dataset(dataset, self.pipeline_config.normalize_text)

        train_ds, val_ds, test_ds = self._split_dataset(dataset)
        classifier = self._train_model(train_ds, val_ds)
        test_metrics = classifier.evaluate(test_ds)

        evaluation_summary = self._generate_evaluation_artifacts(
            classifier, test_ds
        )

        saved_model_path = self._persist_model(classifier, test_metrics, evaluation_summary)
        run_summary = self._build_run_summary(
            dataset,
            train_ds,
            val_ds,
            test_ds,
            test_metrics,
            evaluation_summary,
            saved_model_path,
        )

        summary_path = self._write_run_summary(run_summary)
        self._log_experiment(run_summary, summary_path, evaluation_summary)
        logger.info("Training pipeline completed successfully for run_id=%s", self.run_id)
        return run_summary

    # Internal helpers -----------------------------------------------------------

    def _configure_file_logging(self):
        """Attach a file handler for pipeline-specific logs."""
        log_file = self.run_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        pipeline_logger = logging.getLogger("chatbot_analytics.pipeline")
        pipeline_logger.setLevel(logging.INFO)
        pipeline_logger.addHandler(file_handler)
        pipeline_logger.propagate = True
        self.pipeline_logger = pipeline_logger

    def _load_dataset(self, dataset_path: Path) -> Dataset:
        loader = DatasetLoaderFactory.get_loader(self.pipeline_config.dataset_type)
        self.pipeline_logger.info("Loading dataset from %s", dataset_path)
        dataset = loader.load(dataset_path)
        self.pipeline_logger.info(
            "Loaded dataset %s (%d conversations, %d total turns)",
            dataset.name,
            dataset.size,
            dataset.total_turns,
        )
        return dataset

    def _preprocess_dataset(self, dataset: Dataset, normalize: bool) -> Dataset:
        self.pipeline_logger.info("Preprocessing dataset (normalize=%s)", normalize)
        preprocessed = self.preprocessor.preprocess_dataset(dataset, normalize=normalize)
        self.pipeline_logger.info("Preprocessing complete")
        return preprocessed

    def _split_dataset(self, dataset: Dataset) -> Tuple[Dataset, Dataset, Dataset]:
        self.pipeline_logger.info(
            "Splitting dataset (train=%.2f, val=%.2f, seed=%d)",
            self.pipeline_config.train_ratio,
            self.pipeline_config.val_ratio,
            self.pipeline_config.random_seed,
        )
        train_ds, val_ds, test_ds = self.preprocessor.create_train_test_split(
            dataset,
            train_ratio=self.pipeline_config.train_ratio,
            val_ratio=self.pipeline_config.val_ratio,
            random_seed=self.pipeline_config.random_seed,
        )
        self.pipeline_logger.info(
            "Split complete (train=%d, val=%d, test=%d)",
            train_ds.size,
            val_ds.size,
            test_ds.size,
        )
        return train_ds, val_ds, test_ds

    def _train_model(self, train_ds: Dataset, val_ds: Dataset) -> IntentClassifier:
        classifier = IntentClassifier(
            model_name=self.training_config.model_name,
            batch_size=self.training_config.batch_size,
        )
        self.pipeline_logger.info("Starting model training with config=%s", self.training_config.to_dict())
        training_result = classifier.train(train_ds, val_ds, self.training_config)
        self.pipeline_logger.info(
            "Training complete (train_acc=%.4f, val_acc=%.4f)",
            training_result.training_metrics.accuracy,
            training_result.validation_metrics.accuracy,
        )
        self.training_result = training_result
        return classifier

    def _generate_evaluation_artifacts(
        self, classifier: IntentClassifier, test_ds: Dataset
    ) -> Dict[str, Any]:
        if not self.pipeline_config.save_evaluation_artifacts:
            return {}

        evaluation_dir = self.run_dir / "evaluation"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator.output_dir = evaluation_dir

        self.pipeline_logger.info("Running detailed evaluation and saving artifacts")
        evaluation_summary = self.evaluator.evaluate_model(
            classifier=classifier,
            test_data=test_ds,
            save_results=True,
            generate_visualizations=False,
        )
        artifact_path = self._latest_file(evaluation_dir, suffix=".json")
        return {
            "metrics": evaluation_summary,
            "artifact_path": str(artifact_path) if artifact_path else None,
        }

    def _persist_model(
        self,
        classifier: IntentClassifier,
        test_metrics: PerformanceMetrics,
        evaluation_summary: Dict[str, Any],
    ) -> str:
        metadata = {
            "run_id": self.run_id,
            "model_name": self.training_config.model_name,
            "dataset_type": self.pipeline_config.dataset_type.value,
            "training_metrics": self._serialize_metrics(self.training_result.training_metrics),
            "validation_metrics": self._serialize_metrics(self.training_result.validation_metrics),
            "test_metrics": self._serialize_metrics(test_metrics),
            "training_config": self.training_config.to_dict(),
            "evaluation_summary_path": evaluation_summary.get("artifact_path") if evaluation_summary else None,
        }

        saved_path = self.model_repository.save_model(
            model=self.training_result.model_path,
            model_id=self.pipeline_config.model_id,
            metadata=metadata,
        )

        self.pipeline_logger.info("Persisted trained model to %s", saved_path)
        return saved_path

    def _build_run_summary(
        self,
        dataset: Dataset,
        train_ds: Dataset,
        val_ds: Dataset,
        test_ds: Dataset,
        test_metrics: PerformanceMetrics,
        evaluation_summary: Dict[str, Any],
        saved_model_path: str,
    ) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "dataset": {
                "name": dataset.name,
                "type": dataset.dataset_type.value,
                "total_conversations": dataset.size,
            },
            "splits": {
                "train": train_ds.size,
                "validation": val_ds.size,
                "test": test_ds.size,
            },
            "pipeline_config": asdict(self.pipeline_config),
            "training_config": self.training_config.to_dict(),
            "training_metrics": self._serialize_metrics(self.training_result.training_metrics),
            "validation_metrics": self._serialize_metrics(self.training_result.validation_metrics),
            "test_metrics": self._serialize_metrics(test_metrics),
            "evaluation_summary": evaluation_summary,
            "model_artifact_path": saved_model_path,
        }

    def _write_run_summary(self, summary: Dict[str, Any]) -> Path:
        summary_path = self.run_dir / "run_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        self.pipeline_logger.info("Persisted run summary to %s", summary_path)
        return summary_path

    @staticmethod
    def _serialize_metrics(metrics: PerformanceMetrics) -> Dict[str, Any]:
        return {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "macro_precision": metrics.macro_precision,
            "macro_recall": metrics.macro_recall,
            "macro_f1": metrics.macro_f1,
            "confusion_matrix": metrics.confusion_matrix.tolist()
            if metrics.confusion_matrix is not None
            else None,
        }

    @staticmethod
    def _latest_file(directory: Path, suffix: str) -> Optional[Path]:
        candidates = sorted(
            directory.glob(f"*{suffix}"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def _log_experiment(
        self,
        run_summary: Dict[str, Any],
        summary_path: Path,
        evaluation_summary: Dict[str, Any],
    ):
        if not self.experiment_tracker:
            return

        tags = [
            self.pipeline_config.dataset_type.value,
            self.training_config.model_name,
        ]
        extra_metadata = {
            "run_dir": str(self.run_dir),
            "training_time": getattr(self.training_result, "training_time", None),
            "evaluation_artifact": evaluation_summary.get("artifact_path") if evaluation_summary else None,
        }

        record = self.experiment_tracker.log_experiment(
            run_summary=run_summary,
            tags=tags,
            extra_metadata=extra_metadata,
        )

        self.experiment_tracker.register_artifact(
            run_id=record.run_id,
            artifact_type="run_summary",
            path=str(summary_path),
        )
        if evaluation_summary and evaluation_summary.get("artifact_path"):
            self.experiment_tracker.register_artifact(
                run_id=record.run_id,
                artifact_type="evaluation",
                path=evaluation_summary["artifact_path"],
            )
