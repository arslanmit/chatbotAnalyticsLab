"""System monitoring utilities leveraging psutil."""

from __future__ import annotations

import logging
import os
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional

import psutil

logger = logging.getLogger(__name__)


def collect_system_metrics() -> Dict[str, Any]:
    """Collect system and process-level metrics."""

    process = psutil.Process(os.getpid())
    with process.oneshot():
        mem_info = process.memory_info()
        cpu_percent = process.cpu_percent(interval=0.0)
        threads = process.num_threads()
        open_files = len(process.open_files())

    system_cpu = psutil.cpu_percent(interval=0.0)
    virtual_mem = psutil.virtual_memory()
    disk_usage = psutil.disk_usage(os.getcwd())._asdict()

    return {
        "process": {
            "pid": process.pid,
            "cpu_percent": cpu_percent,
            "memory_rss_mb": round(mem_info.rss / (1024 * 1024), 2),
            "memory_vms_mb": round(mem_info.vms / (1024 * 1024), 2),
            "threads": threads,
            "open_files": open_files,
        },
        "system": {
            "cpu_percent": system_cpu,
            "memory_percent": virtual_mem.percent,
            "memory_total_gb": round(virtual_mem.total / (1024 ** 3), 2),
            "disk_usage": {
                key: round(value / (1024 ** 3), 2) if isinstance(value, (int, float)) else value
                for key, value in disk_usage.items()
            },
        },
    }


def timed(metric_name: Optional[str] = None) -> Callable:
    """Decorator to measure execution time of synchronous functions."""

    def decorator(func: Callable) -> Callable:
        name = metric_name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = (time.perf_counter() - start) * 1000
                logger.debug("timed metric=%s duration_ms=%.2f", name, duration)

        return wrapper

    return decorator
