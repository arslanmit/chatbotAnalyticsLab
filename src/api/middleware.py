"""
Custom middleware for rate limiting and request metrics collection.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Deque, Dict

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from src.api.monitoring import RequestMetricsCollector


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple fixed-window rate limiter per client IP."""

    def __init__(self, app, limit: int = 120, window_seconds: int = 60):
        super().__init__(app)
        self.limit = limit
        self.window = window_seconds
        self._requests: Dict[str, Deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        client_ip = self._get_client_ip(request)
        now = time.time()
        timestamps = self._requests[client_ip]

        while timestamps and timestamps[0] <= now - self.window:
            timestamps.popleft()

        if len(timestamps) >= self.limit:
            retry_after = max(int(self.window - (now - timestamps[0])), 1)
            return JSONResponse(
                status_code=429,
                content={
                    "status": "error",
                    "message": "Rate limit exceeded. Try again later.",
                    "retry_after": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        timestamps.append(now)
        response = await call_next(request)
        return response

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"


class RequestMetricsMiddleware(BaseHTTPMiddleware):
    """Collect latency and status metrics for each request."""

    def __init__(self, app, collector: RequestMetricsCollector):
        super().__init__(app)
        self.collector = collector

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        try:
            response: Response = await call_next(request)
        except Exception as exc:  # pragma: no cover
            latency = time.perf_counter() - start
            endpoint = request.url.path
            await self.collector.record(endpoint, latency, status_code=500)
            raise exc
        latency = time.perf_counter() - start
        endpoint = request.url.path
        await self.collector.record(endpoint, latency, response.status_code)
        return response
