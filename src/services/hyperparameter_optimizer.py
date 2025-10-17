"""
Hyperparameter optimization service supporting grid and random search strategies.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

from src.models.core import TrainingConfig
from src.services.training_pipeline import TrainingPipeline, TrainingPipelineConfig
from src.services.experiment_tracker import ExperimentTracker
from src.repositories.model_repository import ModelRepository
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HyperparameterTrialResult:
    """Container for individual hyperparameter search trial results."""

    params: Dict[str, Any]
    training_config: TrainingConfig
    run_summary: Dict[str, Any]
    score: Optional[float]
    model_artifact_path: Optional[str]
    run_id: str


class HyperparameterOptimizer:
    """
    Execute hyperparameter searches by orchestrating repeated training pipeline runs.
    """

    def __init__(
        self,
        pipeline_config: TrainingPipelineConfig,
        base_training_config: TrainingConfig,
        search_metric: str = "validation_metrics.macro_f1",
        greater_is_better: bool = True,
        model_repository: Optional[ModelRepository] = None,
        random_seed: int = 42,
        experiment_tracker: Optional[ExperimentTracker] = None,
    ):
        self.base_pipeline_config = pipeline_config
        self.base_training_config = base_training_config
        self.metric_path = search_metric.split(".")
        self.greater_is_better = greater_is_better
        self.model_repository = model_repository or ModelRepository()
        self.random = random.Random(random_seed)
        self._trial_counter = 0
        self.experiment_tracker = experiment_tracker or ExperimentTracker()

    # Public API -----------------------------------------------------------------

    def grid_search(self, search_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Perform exhaustive search over the provided hyperparameter grid.
        """
        trials: List[HyperparameterTrialResult] = []
        keys = list(search_space.keys())
        combinations = list(product(*[search_space[key] for key in keys]))

        logger.info("Starting grid search across %d combinations", len(combinations))

        for combo in combinations:
            params = dict(zip(keys, combo))
            trial = self._run_trial(params, strategy="grid")
            trials.append(trial)

        return self._build_report("grid", trials)

    def random_search(
        self,
        search_space: Dict[str, List[Any]],
        num_samples: int = 10,
    ) -> Dict[str, Any]:
        """
        Perform random search sampling hyperparameters uniformly from provided options.
        """
        trials: List[HyperparameterTrialResult] = []
        keys = list(search_space.keys())

        logger.info("Starting random search with %d samples", num_samples)

        for _ in range(num_samples):
            params = {
                key: self.random.choice(search_space[key])
                for key in keys
            }
            trial = self._run_trial(params, strategy="random")
            trials.append(trial)

        return self._build_report("random", trials)

    # Internal helpers -----------------------------------------------------------

    def _run_trial(
        self,
        params: Dict[str, Any],
        strategy: str,
    ) -> HyperparameterTrialResult:
        training_config = replace(self.base_training_config, **params)
        pipeline_config = self._build_pipeline_config(
            strategy=strategy,
            params=params,
        )

        logger.info(
            "Running %s search trial with params: %s",
            strategy,
            params,
        )

        pipeline = TrainingPipeline(
            pipeline_config=pipeline_config,
            training_config=training_config,
            model_repository=self.model_repository,
            experiment_tracker=self.experiment_tracker,
        )
        run_summary = pipeline.run()
        score = self._extract_metric(run_summary)
        artifact_path = run_summary.get("model_artifact_path")

        logger.info(
            "Trial complete (run_id=%s, score=%s)",
            run_summary.get("run_id"),
            f"{score:.4f}" if score is not None else "N/A",
        )

        return HyperparameterTrialResult(
            params=params,
            training_config=training_config,
            run_summary=run_summary,
            score=score,
            model_artifact_path=artifact_path,
            run_id=run_summary.get("run_id", ""),
        )

    def _build_pipeline_config(
        self,
        strategy: str,
        params: Dict[str, Any],
    ) -> TrainingPipelineConfig:
        trial_index = self._next_trial_index()
        model_suffix = f"{strategy}_trial{trial_index}"
        model_id = f"{self.base_pipeline_config.model_id}_{model_suffix}"
        return replace(self.base_pipeline_config, model_id=model_id)

    def _next_trial_index(self) -> int:
        self._trial_counter += 1
        return self._trial_counter

    def _extract_metric(self, run_summary: Dict[str, Any]) -> Optional[float]:
        value: Any = run_summary
        for key in self.metric_path:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                value = None
                break
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _build_report(
        self,
        strategy: str,
        trials: List[HyperparameterTrialResult],
    ) -> Dict[str, Any]:
        ordered_trials = sorted(
            trials,
            key=lambda t: (t.score is not None, t.score),
            reverse=self.greater_is_better,
        )

        best_trial = next((trial for trial in ordered_trials if trial.score is not None), None)

        return {
            "strategy": strategy,
            "metric": ".".join(self.metric_path),
            "greater_is_better": self.greater_is_better,
            "best_trial": self._serialize_trial(best_trial) if best_trial else None,
            "trials": [self._serialize_trial(trial) for trial in ordered_trials],
        }

    @staticmethod
    def _serialize_trial(trial: HyperparameterTrialResult) -> Dict[str, Any]:
        if trial is None:
            return {}
        return {
            "params": trial.params,
            "score": trial.score,
            "run_id": trial.run_id,
            "model_artifact_path": trial.model_artifact_path,
            "training_config": trial.training_config.to_dict(),
            "run_summary": {
                "test_metrics": trial.run_summary.get("test_metrics"),
                "validation_metrics": trial.run_summary.get("validation_metrics"),
                "training_metrics": trial.run_summary.get("training_metrics"),
            },
        }
