"""
Common request and response models.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class StatusResponse(BaseModel):
    status: str
    message: Optional[str] = None


class OperationStatus(StatusResponse):
    pass


class HealthResponse(StatusResponse):
    version: str
    timestamp: datetime


class ServiceInfoResponse(BaseModel):
    version: str
    environment: str
    debug: bool
    dataset_dir: str
    model_registry: str
