"""
Common request and response models.
"""

from datetime import datetime
from typing import Dict, Optional, Any

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


class EndpointMetricsResponse(BaseModel):
    path: str
    count: int
    average_latency_ms: float
    errors: int


class MonitoringMetricsResponse(BaseModel):
    total_requests: int
    total_errors: int
    average_latency_ms: float
    endpoints: Dict[str, EndpointMetricsResponse]
    system: Dict[str, Any]
