"""
Model training schemas.
"""

from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field

from src.models.core import DatasetType


class TrainingRunRequest(BaseModel):
    dataset_type: DatasetType
    dataset_path: Optional[str] = Field(None, description="Optional override to dataset location.")
    model_id: Optional[str] = Field(None, description="Identifier for persisting the trained model.")
    training_overrides: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    async_run: bool = Field(True, description="Whether to execute training asynchronously.")


class TrainingRunResponse(BaseModel):
    status: str
    run_id: str
    message: Optional[str] = None


class HyperparameterSearchRequest(BaseModel):
    dataset_type: DatasetType
    strategy: str = Field("grid", regex="^(grid|random)$")
    search_space: Dict[str, List[Any]] = Field(..., description="Grid of parameters to explore.")
    num_samples: Optional[int] = Field(None, ge=1, description="Samples for random search.")
    model_id: Optional[str] = Field(None, description="Identifier for persisting the best model.")
    training_overrides: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    async_run: bool = Field(True, description="Whether to execute the search in the background.")


class HyperparameterSearchResponse(BaseModel):
    status: str
    job_id: str
    best_trial: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
