"""
Monitoring utilities for API request metrics.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class EndpointMetrics:
    count: int = 0
    total_latency: float = 0.0
    errors: int = 0

    def snapshot(self) -> Dict[str, Any]:
        average_latency = (self.total_latency / self.count) if self.count else 0.0
        return {
            "count": self.count,
            "average_latency_ms": round(average_latency * 1000, 2),
            "errors": self.errors,
        }


class RequestMetricsCollector:
    """Thread-safe collector for request metrics."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self.total_requests: int = 0
        self.total_latency: float = 0.0
        self.total_errors: int = 0
        self._endpoint_stats: Dict[str, EndpointMetrics] = defaultdict(EndpointMetrics)

    async def record(self, endpoint: str, latency: float, status_code: int):
        async with self._lock:
            self.total_requests += 1
            self.total_latency += latency
            if status_code >= 400:
                self.total_errors += 1

            stats = self._endpoint_stats[endpoint]
            stats.count += 1
            stats.total_latency += latency
            if status_code >= 400:
                stats.errors += 1

    async def snapshot(self) -> Dict[str, Any]:
        async with self._lock:
            average_latency = (
                (self.total_latency / self.total_requests) if self.total_requests else 0.0
            )
            endpoint_stats = {
                endpoint: metrics.snapshot()
                for endpoint, metrics in self._endpoint_stats.items()
            }
            return {
                "total_requests": self.total_requests,
                "total_errors": self.total_errors,
                "average_latency_ms": round(average_latency * 1000, 2),
                "endpoints": endpoint_stats,
            }
