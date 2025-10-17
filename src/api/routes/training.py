"""
Model training and optimization endpoints.
"""

import uuid
from typing import Dict, Any

from fastapi import APIRouter, BackgroundTasks, Depends

from src.api.dependencies import build_training_pipeline, build_hyperparameter_optimizer
from src.api.schemas.training import (
    TrainingRunRequest,
    TrainingRunResponse,
    HyperparameterSearchRequest,
    HyperparameterSearchResponse,
)
from src.config.settings import settings
from src.models.core import TrainingConfig
from src.services.training_pipeline import TrainingPipelineConfig

router = APIRouter()


@router.post("/run", response_model=TrainingRunResponse)
async def trigger_training(
    request: TrainingRunRequest,
    background_tasks: BackgroundTasks,
) -> TrainingRunResponse:
    """
    Kick off an automated training pipeline run.
    """
    training_config = _build_training_config(request.training_overrides)
    pipeline_config = TrainingPipelineConfig(
        dataset_type=request.dataset_type,
        dataset_path=request.dataset_path,
        model_id=request.model_id or "intent_classifier",
    )
    pipeline = build_training_pipeline(pipeline_config, training_config)

    if request.async_run:
        background_tasks.add_task(_run_pipeline, pipeline)
        message = f"Training scheduled for model '{pipeline_config.model_id}'."
        status = "accepted"
    else:
        _run_pipeline(pipeline)
        message = f"Training completed for model '{pipeline_config.model_id}'."
        status = "completed"

    return TrainingRunResponse(
        status=status,
        run_id=pipeline.run_id,
        message=message,
    )


@router.post("/hyperparameter-search", response_model=HyperparameterSearchResponse)
async def start_hyperparameter_search(
    request: HyperparameterSearchRequest,
    background_tasks: BackgroundTasks,
) -> HyperparameterSearchResponse:
    """
    Launch a hyperparameter search. Executes in the background by default.
    """
    training_config = _build_training_config(request.training_overrides)
    pipeline_config = TrainingPipelineConfig(
        dataset_type=request.dataset_type,
        dataset_path=request.dataset_path,
        model_id=request.model_id or "intent_classifier",
    )
    optimizer = build_hyperparameter_optimizer(pipeline_config, training_config)
    job_id = f"hpo-{uuid.uuid4().hex[:8]}"

    if request.strategy == "random" and request.num_samples:
        job_callable = optimizer.random_search
        job_args = (request.search_space,)
        job_kwargs: Dict[str, Any] = {"num_samples": request.num_samples}
    else:
        job_callable = optimizer.grid_search
        job_args = (request.search_space,)
        job_kwargs = {}

    if request.async_run:
        background_tasks.add_task(job_callable, *job_args, **job_kwargs)
        status = "accepted"
        message = f"Hyperparameter search scheduled (job_id={job_id})."
        best_trial = None
    else:
        result = job_callable(*job_args, **job_kwargs)
        status = "completed"
        message = f"Hyperparameter search finished (job_id={job_id})."
        best_trial = result.get("best_trial")

    return HyperparameterSearchResponse(
        status=status,
        job_id=job_id,
        best_trial=best_trial,
        message=message,
    )


def _build_training_config(overrides: Dict[str, Any]) -> TrainingConfig:
    base_config = TrainingConfig(model_name=settings.model.default_model)
    config_data = base_config.to_dict()
    config_data.update(overrides or {})
    return TrainingConfig(**config_data)


def _run_pipeline(pipeline):
    pipeline.run()
