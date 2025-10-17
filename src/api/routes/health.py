"""
Health and diagnostics endpoints.
"""

from datetime import datetime

from fastapi import APIRouter

from src.api.schemas.common import HealthResponse, ServiceInfoResponse
from src.config.settings import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return simple health information used by load balancers and monitors."""
    return HealthResponse(
        status="ok",
        message="Service operational",
        version="0.1.0",
        timestamp=datetime.utcnow(),
    )


@router.get("/health/info", response_model=ServiceInfoResponse)
def service_info() -> ServiceInfoResponse:
    """Expose basic service configuration used by monitoring dashboards."""
    return ServiceInfoResponse(
        version="0.1.0",
        environment=settings.environment,
        debug=settings.debug,
        dataset_dir=settings.data.dataset_dir,
        model_registry=str(settings.model.cache_dir),
    )
