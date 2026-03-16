"""
Data models and core interfaces for the Chatbot Analytics System.
"""

from importlib import import_module
from typing import Any


_EXPORTS = {
    "DatasetType": ("src.models.core", "DatasetType"),
    "Speaker": ("src.models.core", "Speaker"),
    "ConversationTurn": ("src.models.core", "ConversationTurn"),
    "Conversation": ("src.models.core", "Conversation"),
    "IntentPrediction": ("src.models.core", "IntentPrediction"),
    "Dataset": ("src.models.core", "Dataset"),
    "ValidationResult": ("src.models.core", "ValidationResult"),
    "QualityReport": ("src.models.core", "QualityReport"),
    "PerformanceMetrics": ("src.models.core", "PerformanceMetrics"),
    "TrainingConfig": ("src.models.core", "TrainingConfig"),
    "TrainingResult": ("src.models.core", "TrainingResult"),
    "IntentClassifier": ("src.models.intent_classifier", "IntentClassifier"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'src.models' has no attribute {name!r}")

    module_name, attribute_name = _EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attribute_name)


def __dir__() -> list[str]:
    return sorted(__all__)
