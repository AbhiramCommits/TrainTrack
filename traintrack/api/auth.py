"""API-key authentication and per-key Redis rate limiting."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

from flask import g, jsonify, request
from redis.exceptions import RedisError

from traintrack.db import get_session
from traintrack.services.auth import verify_api_key
from traintrack.services.cache import get_redis

logger = logging.getLogger(__name__)
F = TypeVar("F", bound=Callable[..., Any])


def extract_api_key() -> str | None:
    """Read an API key from header, bearer token, or SSE query parameter."""
    header_key = request.headers.get("X-API-Key")
    if header_key:
        return header_key.strip()
    authorization = request.headers.get("Authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return request.args.get("api_key")


def rate_limit_exceeded(api_key_id: str, limit: int) -> tuple[bool, int]:
    """Fixed-window per-key rate limit in Redis.

    Returns (exceeded, retry_after_seconds). Redis failures fail open so an
    outage of the cache does not take the control plane down.
    """
    window = int(time.time() // 60)
    key = f"ratelimit:{api_key_id}:{window}"
    try:
        client = get_redis()
        pipeline = client.pipeline()
        pipeline.incr(key)
        pipeline.expire(key, 61)
        count, _ = pipeline.execute()
    except RedisError as exc:
        logger.warning("rate limiter unavailable, failing open: %s", exc)
        return False, 0
    if int(count) > limit:
        return True, 60 - int(time.time() % 60)
    return False, 0


def require_api_key(func: F) -> F:
    """Decorator enforcing a valid API key and per-key rate limit."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        raw_key = extract_api_key()
        if not raw_key:
            return (
                jsonify(
                    {
                        "error": "unauthorized",
                        "message": "Provide X-API-Key or Authorization: Bearer <key>",
                    }
                ),
                401,
            )
        session = get_session()
        try:
            record = verify_api_key(session, raw_key)
        finally:
            session.close()
        if record is None:
            return jsonify({"error": "unauthorized", "message": "Invalid API key"}), 401

        limit = record.rate_limit_per_min
        exceeded, retry_after = rate_limit_exceeded(str(record.id), limit)
        if exceeded:
            response = jsonify(
                {"error": "rate_limited", "message": "Too many requests"}
            )
            response.headers["Retry-After"] = str(retry_after)
            return response, 429

        g.api_key = record
        return func(*args, **kwargs)

    return cast(F, wrapper)

def optional_api_key(func: F) -> F:
    """Decorator attaching an API key when present without requiring one.

    Only used by health endpoints; every data endpoint requires a key.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        raw_key = extract_api_key()
        if raw_key:
            session = get_session()
            try:
                g.api_key = verify_api_key(session, raw_key)
            finally:
                session.close()
        return func(*args, **kwargs)

    return cast(F, wrapper)
