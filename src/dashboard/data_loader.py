"""Data access helpers for the Streamlit dashboard."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List

from src.api.dependencies import get_experiment_tracker


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
