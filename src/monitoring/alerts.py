"""Alerting utilities for monitoring thresholds and notification channels."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

import httpx

from src.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    name: str
    severity: str
    message: str
    details: Dict[str, Any]


class AlertChannel(Protocol):
    def send(self, alert: Alert) -> None: ...


class LoggingAlertChannel:
    def send(self, alert: Alert) -> None:
        logger.warning("ALERT [%s] %s - %s", alert.severity, alert.name, alert.message)


class WebhookAlertChannel:
    def __init__(self, url: str):
        self.url = url

    def send(self, alert: Alert) -> None:
        payload = {
            "title": alert.name,
            "severity": alert.severity,
            "message": alert.message,
            "details": alert.details,
        }
        try:
            httpx.post(self.url, json=payload, timeout=5.0)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Failed to send webhook alert: %s", exc)


def build_channels() -> List[AlertChannel]:
    channels: List[AlertChannel] = []
    for entry in settings.alerts.channels:
        if entry == "log":
            channels.append(LoggingAlertChannel())
        elif entry.startswith("webhook:" ):
            _, url = entry.split(":", 1)
            channels.append(WebhookAlertChannel(url))
    return channels or [LoggingAlertChannel()]


def evaluate_alerts(metrics: Dict[str, Any], system_metrics: Dict[str, Any]) -> List[Alert]:
    alerts: List[Alert] = []
    if metrics.get("average_latency_ms", 0) > settings.alerts.request_latency_threshold_ms:
        alerts.append(
            Alert(
                name="High Request Latency",
                severity="warning",
                message="Average latency exceeds configured threshold",
                details={
                    "average_latency_ms": metrics.get("average_latency_ms"),
                    "threshold": settings.alerts.request_latency_threshold_ms,
                },
            )
        )

    cpu_percent = system_metrics.get("system", {}).get("cpu_percent", 0)
    if cpu_percent > settings.alerts.cpu_threshold:
        alerts.append(
            Alert(
                name="High CPU Usage",
                severity="critical",
                message="System CPU usage above threshold",
                details={
                    "cpu_percent": cpu_percent,
                    "threshold": settings.alerts.cpu_threshold,
                },
            )
        )

    memory_percent = system_metrics.get("system", {}).get("memory_percent", 0)
    if memory_percent > settings.alerts.memory_threshold:
        alerts.append(
            Alert(
                name="High Memory Usage",
                severity="critical",
                message="System memory usage above threshold",
                details={
                    "memory_percent": memory_percent,
                    "threshold": settings.alerts.memory_threshold,
                },
            )
        )

    return alerts


def dispatch_alerts(alerts: List[Alert]) -> None:
    channels = build_channels()
    for alert in alerts:
        for channel in channels:
            channel.send(alert)
