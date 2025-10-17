"""Data access helpers for the Streamlit dashboard."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.api.dependencies import (
    get_data_preprocessor,
    get_conversation_flow_analyzer,
    get_sentiment_analyzer,
    get_experiment_tracker,
)
from src.config.settings import settings
from src.models.core import Dataset, DatasetType, IntentPrediction
from src.repositories.dataset_loaders import DatasetLoaderFactory


# ---------------------------------------------------------------------------
# Experiment utilities
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_experiments() -> List[Dict[str, Any]]:
    """Load experiments from the shared tracker."""
    tracker = get_experiment_tracker()
    return tracker.list_experiments() or []


def get_recent_experiments(limit: int = 10) -> List[Dict[str, Any]]:
    experiments = load_experiments()
    experiments_sorted = sorted(
        experiments,
        key=lambda exp: exp.get("created_at", ""),
        reverse=True,
    )
    return experiments_sorted[:limit]


def compute_overview_metrics() -> Dict[str, Any]:
    experiments = load_experiments()
    experiment_count = len(experiments)
    models = {exp.get("model_id") for exp in experiments if exp.get("model_id")}

    successful_runs = sum(
        1
        for exp in experiments
        if exp.get("validation_metrics", {}).get("accuracy", 0.0) >= 0.5
    )

    latest_accuracy = None
    if experiments:
        recent = get_recent_experiments(1)
        if recent:
            latest_accuracy = recent[0].get("validation_metrics", {}).get("accuracy")

    return {
        "experiment_count": experiment_count,
        "model_count": len(models),
        "successful_runs": successful_runs,
        "latest_validation_accuracy": latest_accuracy,
    }


# ---------------------------------------------------------------------------
# Dataset loading and analytics helpers
# ---------------------------------------------------------------------------


def _resolve_dataset_path(dataset_type: DatasetType, dataset_path: Optional[str]) -> Path:
    if dataset_path:
        path = Path(dataset_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dataset path not found: {path}")
        return path

    base_dir = Path(settings.data.dataset_dir)
    candidates = [
        base_dir / dataset_type.value,
        base_dir / dataset_type.value.lower(),
        base_dir / dataset_type.value.upper(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Dataset for '{dataset_type.value}' not found in {base_dir}")


@lru_cache(maxsize=8)
def load_dataset(
    dataset_type: DatasetType,
    dataset_path: Optional[str] = None,
    preprocess: bool = True,
    normalize_text: bool = True,
) -> Dataset:
    """Load and optionally preprocess a dataset for dashboard analytics."""
    path = _resolve_dataset_path(dataset_type, dataset_path)
    loader = DatasetLoaderFactory.get_loader(dataset_type)
    dataset = loader.load(path)

    if preprocess:
        preprocessor = get_data_preprocessor()
        dataset = preprocessor.preprocess_dataset(dataset, normalize=normalize_text)

    return dataset


def compute_intent_distribution(
    dataset: Dataset,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for conversation in dataset.conversations:
        for turn in conversation.turns:
            if turn.intent:
                counts[turn.intent] = counts.get(turn.intent, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


def compute_flow_summary(
    dataset: Dataset,
    conversation_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    flow_analyzer = get_conversation_flow_analyzer()
    conversations = _select_conversations(dataset, conversation_ids)
    return flow_analyzer.analyze_dialogue_flow(conversations)


def compute_sentiment_trend(
    dataset: Dataset,
    granularity: str = "daily",
) -> Dict[str, Any]:
    sentiment_analyzer = get_sentiment_analyzer()
    return sentiment_analyzer.calculate_sentiment_trend(
        dataset.conversations,
        granularity=granularity,
    )


def compute_sentiment_summary(dataset: Dataset) -> Dict[str, Any]:
    sentiment_analyzer = get_sentiment_analyzer()
    return sentiment_analyzer.analyze_conversations(dataset.conversations)


def _select_conversations(
    dataset: Dataset,
    conversation_ids: Optional[Iterable[str]] = None,
) -> List:
    if conversation_ids is None:
        return dataset.conversations
    indexed = {conv.id: conv for conv in dataset.conversations}
    return [indexed[cid] for cid in conversation_ids if cid in indexed]


def collect_intent_predictions(dataset: Dataset) -> List[IntentPrediction]:
    predictions: List[IntentPrediction] = []
    for conversation in dataset.conversations:
        for turn in conversation.turns:
            if turn.intent:
                predictions.append(
                    IntentPrediction(
                        intent=turn.intent,
                        confidence=turn.confidence or 0.0,
                        alternatives=[],
                    )
                )
    return predictions
