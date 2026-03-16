"""
Shared dependency providers for the API layer.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict

from src.api.cache import SimpleResponseCache
from src.api.monitoring import RequestMetricsCollector
from src.utils.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from src.models.core import TrainingConfig
    from src.models.intent_classifier import IntentClassifier
    from src.repositories.model_repository import ModelRepository
    from src.services.conversation_analyzer import ConversationFlowAnalyzer
    from src.services.data_preprocessor import DataPreprocessor
    from src.services.experiment_tracker import ExperimentTracker
    from src.services.hyperparameter_optimizer import HyperparameterOptimizer
    from src.services.performance_analyzer import PerformanceAnalyzer
    from src.services.sentiment_analyzer import SentimentAnalyzer
    from src.services.training_pipeline import TrainingPipeline, TrainingPipelineConfig

_classifier_cache: Dict[str, Any] = {}
_metrics_collector = RequestMetricsCollector()
_response_cache = SimpleResponseCache(ttl_seconds=300)


class OptionalDependencyUnavailable(RuntimeError):
    """Raised when an optional runtime dependency is not installed."""


def _raise_optional_dependency_error(feature: str, exc: ModuleNotFoundError) -> None:
    package_name = exc.name or "unknown package"
    raise OptionalDependencyUnavailable(
        f"{feature} is unavailable because the optional package "
        f"'{package_name}' is not installed in this Docker runtime."
    ) from exc


@lru_cache()
def get_model_repository() -> "ModelRepository":
    from src.repositories.model_repository import ModelRepository

    return ModelRepository()


@lru_cache()
def get_experiment_tracker() -> "ExperimentTracker":
    from src.services.experiment_tracker import ExperimentTracker

    return ExperimentTracker()


@lru_cache()
def get_data_preprocessor() -> "DataPreprocessor":
    from src.services.data_preprocessor import DataPreprocessor

    return DataPreprocessor()


@lru_cache()
def get_conversation_flow_analyzer() -> "ConversationFlowAnalyzer":
    from src.services.conversation_analyzer import ConversationFlowAnalyzer

    return ConversationFlowAnalyzer()


@lru_cache()
def get_sentiment_analyzer() -> "SentimentAnalyzer":
    from src.services.sentiment_analyzer import SentimentAnalyzer

    return SentimentAnalyzer()


@lru_cache()
def get_performance_analyzer() -> "PerformanceAnalyzer":
    from src.services.performance_analyzer import PerformanceAnalyzer

    return PerformanceAnalyzer()


@lru_cache()
def get_response_cache() -> SimpleResponseCache:
    return _response_cache


@lru_cache()
def get_metrics_collector() -> RequestMetricsCollector:
    return _metrics_collector


def create_training_pipeline_config(
    *,
    dataset_type,
    dataset_path: str | None = None,
    model_id: str = "intent_classifier",
) -> "TrainingPipelineConfig":
    try:
        from src.services.training_pipeline import TrainingPipelineConfig
    except ModuleNotFoundError as exc:
        _raise_optional_dependency_error("Training endpoints", exc)

    return TrainingPipelineConfig(
        dataset_type=dataset_type,
        dataset_path=dataset_path,
        model_id=model_id,
    )


def build_training_pipeline(
    config: "TrainingPipelineConfig",
    training: "TrainingConfig",
) -> "TrainingPipeline":
    """
    Factory helper used by endpoints to create a training pipeline with shared services.
    """
    try:
        from src.services.training_pipeline import TrainingPipeline

        pipeline = TrainingPipeline(
            pipeline_config=config,
            training_config=training,
            model_repository=get_model_repository(),
            experiment_tracker=get_experiment_tracker(),
        )
    except ModuleNotFoundError as exc:
        _raise_optional_dependency_error("Training endpoints", exc)

    logger.debug("Created TrainingPipeline with run_id=%s", pipeline.run_id)
    return pipeline


def build_hyperparameter_optimizer(
    pipeline_config: "TrainingPipelineConfig",
    training: "TrainingConfig",
) -> "HyperparameterOptimizer":
    """
    Factory helper for hyperparameter optimizer instances.
    """
    try:
        from src.services.hyperparameter_optimizer import HyperparameterOptimizer

        optimizer = HyperparameterOptimizer(
            pipeline_config=pipeline_config,
            base_training_config=training,
            model_repository=get_model_repository(),
            experiment_tracker=get_experiment_tracker(),
        )
    except ModuleNotFoundError as exc:
        _raise_optional_dependency_error("Training endpoints", exc)

    logger.debug("Initialized HyperparameterOptimizer for model_id=%s", pipeline_config.model_id)
    return optimizer


def get_intent_classifier(model_id: str = "intent_classifier") -> "IntentClassifier":
    """
    Retrieve a cached intent classifier instance, loading from the model repository on-demand.
    """
    if model_id in _classifier_cache:
        return _classifier_cache[model_id]

    try:
        from src.models.intent_classifier import IntentClassifier

        artifact, _ = get_model_repository().load_model(model_id)
        classifier = IntentClassifier(model_path=str(artifact.model_path))
    except ModuleNotFoundError as exc:
        _raise_optional_dependency_error("Intent prediction", exc)

    _classifier_cache[model_id] = classifier
    logger.info("Loaded intent classifier model '%s' from %s", model_id, artifact.model_path)
    return classifier
