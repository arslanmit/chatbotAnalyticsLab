"""
Simple in-memory cache for API responses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CacheEntry:
    value: Any
    expiry: float


class SimpleResponseCache:
    """Minimal TTL-based cache intended for lightweight response caching."""

    def __init__(self, ttl_seconds: int = 300, max_entries: int = 512):
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._store: Dict[str, CacheEntry] = {}

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if not entry:
            return None
        if entry.expiry < time.time():
            self._store.pop(key, None)
            return None
        return entry.value

    def set(self, key: str, value: Any):
        if len(self._store) >= self.max_entries:
            self._prune()
        self._store[key] = CacheEntry(value=value, expiry=time.time() + self.ttl)

    def _prune(self):
        now = time.time()
        expired = [k for k, v in self._store.items() if v.expiry < now]
        for key in expired:
            self._store.pop(key, None)
        if len(self._store) >= self.max_entries and self._store:
            oldest_key = min(self._store.items(), key=lambda item: item[1].expiry)[0]
            self._store.pop(oldest_key, None)
