"""
Shared dependency providers for the API layer.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict

from src.api.cache import SimpleResponseCache
from src.api.monitoring import RequestMetricsCollector
from src.repositories.model_repository import ModelRepository
from src.services import (
    DataPreprocessor,
    ConversationFlowAnalyzer,
    SentimentAnalyzer,
    PerformanceAnalyzer,
    TrainingPipeline,
    TrainingPipelineConfig,
    HyperparameterOptimizer,
    ExperimentTracker,
)
from src.models.core import TrainingConfig
from src.models.intent_classifier import IntentClassifier
from src.utils.logging import get_logger

logger = get_logger(__name__)

_classifier_cache: Dict[str, IntentClassifier] = {}
_metrics_collector = RequestMetricsCollector()
_response_cache = SimpleResponseCache(ttl_seconds=300)


@lru_cache()
def get_model_repository() -> ModelRepository:
    return ModelRepository()


@lru_cache()
def get_experiment_tracker() -> ExperimentTracker:
    return ExperimentTracker()


@lru_cache()
def get_data_preprocessor() -> DataPreprocessor:
    return DataPreprocessor()


@lru_cache()
def get_conversation_flow_analyzer() -> ConversationFlowAnalyzer:
    return ConversationFlowAnalyzer()


@lru_cache()
def get_sentiment_analyzer() -> SentimentAnalyzer:
    return SentimentAnalyzer()


@lru_cache()
def get_performance_analyzer() -> PerformanceAnalyzer:
    return PerformanceAnalyzer()


@lru_cache()
def get_response_cache() -> SimpleResponseCache:
    return _response_cache


@lru_cache()
def get_metrics_collector() -> RequestMetricsCollector:
    return _metrics_collector


def build_training_pipeline(config: TrainingPipelineConfig, training: TrainingConfig) -> TrainingPipeline:
    """
    Factory helper used by endpoints to create a training pipeline with shared services.
    """
    pipeline = TrainingPipeline(
        pipeline_config=config,
        training_config=training,
        model_repository=get_model_repository(),
        experiment_tracker=get_experiment_tracker(),
    )
    logger.debug("Created TrainingPipeline with run_id=%s", pipeline.run_id)
    return pipeline


def build_hyperparameter_optimizer(
    pipeline_config: TrainingPipelineConfig,
    training: TrainingConfig,
) -> HyperparameterOptimizer:
    """
    Factory helper for hyperparameter optimizer instances.
    """
    optimizer = HyperparameterOptimizer(
        pipeline_config=pipeline_config,
        base_training_config=training,
        model_repository=get_model_repository(),
        experiment_tracker=get_experiment_tracker(),
    )
    logger.debug("Initialized HyperparameterOptimizer for model_id=%s", pipeline_config.model_id)
    return optimizer


def get_intent_classifier(model_id: str = "intent_classifier") -> IntentClassifier:
    """
    Retrieve a cached intent classifier instance, loading from the model repository on-demand.
    """
    if model_id in _classifier_cache:
        return _classifier_cache[model_id]

    artifact, _ = get_model_repository().load_model(model_id)
    classifier = IntentClassifier(model_path=str(artifact.model_path))
    _classifier_cache[model_id] = classifier
    logger.info("Loaded intent classifier model '%s' from %s", model_id, artifact.model_path)
    return classifier
