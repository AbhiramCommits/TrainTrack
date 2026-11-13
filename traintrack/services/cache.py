"""Redis client helpers for caching, rate limiting and log streams."""

from __future__ import annotations

import json
from typing import Any

import redis

from traintrack.config import settings

# redis-py's generic command return types confuse mypy into an
# ``Awaitable[Any] | Any`` union; the sync client is deliberately typed as Any.
_client: Any = None


def get_redis() -> Any:
    """Return a process-wide Redis client (decoded strings)."""
    global _client
    if _client is None:
        _client = redis.Redis.from_url(
            settings().redis_url, decode_responses=True, socket_timeout=5
        )
    return _client


def reset_redis_client() -> None:
    """Drop the cached client (tests / worker restart)."""
    global _client
    _client = None


def cache_get_json(key: str) -> Any | None:
    """Read and decode a JSON cache entry, returning None on a miss."""
    raw = get_redis().get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def cache_set_json(key: str, value: Any, ttl_seconds: int) -> None:
    """Store a JSON value with an explicit TTL."""
    get_redis().set(key, json.dumps(value, default=str), ex=ttl_seconds)


def cache_delete(*keys: str) -> None:
    """Delete one or more cache keys."""
    if keys:
        get_redis().delete(*keys)
