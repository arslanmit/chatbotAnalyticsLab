"""
Health and diagnostics endpoints.
"""

from datetime import datetime

from fastapi import APIRouter, Depends

from src.api.dependencies import get_metrics_collector
from src.api.schemas.common import (
    HealthResponse,
    ServiceInfoResponse,
    MonitoringMetricsResponse,
    AlertCheckResponse,
    AlertDetailResponse,
)
from src.config.settings import settings
from src.monitoring import collect_system_metrics, evaluate_alerts, dispatch_alerts

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


@router.get("/health/metrics", response_model=MonitoringMetricsResponse)
async def service_metrics(collector=Depends(get_metrics_collector)) -> MonitoringMetricsResponse:
    stats = await collector.snapshot()
    endpoints = {
        path: {
            "path": path,
            **metrics,
        }
        for path, metrics in stats["endpoints"].items()
    }
    system_metrics = collect_system_metrics()
    return MonitoringMetricsResponse(
        total_requests=stats["total_requests"],
        total_errors=stats["total_errors"],
        average_latency_ms=stats["average_latency_ms"],
        endpoints=endpoints,
        system=system_metrics,
    )


@router.get("/health/alerts", response_model=AlertCheckResponse)
async def alert_check(trigger: bool = False, collector=Depends(get_metrics_collector)) -> AlertCheckResponse:
    stats = await collector.snapshot()
    system_metrics = collect_system_metrics()
    alerts = evaluate_alerts(stats, system_metrics)
    if trigger and alerts:
        dispatch_alerts(alerts)
    return AlertCheckResponse(
        alerts=[
            AlertDetailResponse(
                name=alert.name,
                severity=alert.severity,
                message=alert.message,
                details=alert.details,
            )
            for alert in alerts
        ]
    )
