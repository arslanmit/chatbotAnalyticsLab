"""
Business logic and service layer.
"""

from importlib import import_module
from typing import Any


_EXPORTS = {
    "DataValidator": ("src.services.data_validator", "DataValidator"),
    "DataQualityAnalyzer": ("src.services.data_validator", "DataQualityAnalyzer"),
    "DataPreprocessor": ("src.services.data_preprocessor", "DataPreprocessor"),
    "ConversationExtractor": ("src.services.data_preprocessor", "ConversationExtractor"),
    "DataAugmentor": ("src.services.data_preprocessor", "DataAugmentor"),
    "ModelEvaluator": ("src.services.model_evaluator", "ModelEvaluator"),
    "ConversationFlowAnalyzer": ("src.services.conversation_analyzer", "ConversationFlowAnalyzer"),
    "SentimentAnalyzer": ("src.services.sentiment_analyzer", "SentimentAnalyzer"),
    "PerformanceAnalyzer": ("src.services.performance_analyzer", "PerformanceAnalyzer"),
    "TrainingPipeline": ("src.services.training_pipeline", "TrainingPipeline"),
    "TrainingPipelineConfig": ("src.services.training_pipeline", "TrainingPipelineConfig"),
    "HyperparameterOptimizer": ("src.services.hyperparameter_optimizer", "HyperparameterOptimizer"),
    "ExperimentTracker": ("src.services.experiment_tracker", "ExperimentTracker"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'src.services' has no attribute {name!r}")

    module_name, attribute_name = _EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attribute_name)


def __dir__() -> list[str]:
    return sorted(__all__)
