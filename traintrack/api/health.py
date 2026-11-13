"""Liveness and readiness endpoints."""

from __future__ import annotations

from flask import Blueprint, jsonify
from sqlalchemy import text

from traintrack.api.auth import optional_api_key
from traintrack.db import get_session
from traintrack.services.cache import get_redis

bp = Blueprint("health", __name__)


@bp.get("/healthz")
@optional_api_key
def healthz():
    """Process liveness: always returns ok when the app can serve traffic."""
    return jsonify({"status": "ok", "service": "traintrack-api"}), 200


@bp.get("/readyz")
@optional_api_key
def readyz():
    """Readiness: verifies Postgres and Redis connectivity."""
    checks: dict[str, str] = {}
    healthy = True

    try:
        session = get_session()
        try:
            session.execute(text("SELECT 1"))
        finally:
            session.close()
        checks["postgres"] = "ok"
    except Exception as exc:  # pragma: no cover - exercised by failure-path tests
        checks["postgres"] = f"error: {exc}"
        healthy = False

    try:
        get_redis().ping()
        checks["redis"] = "ok"
    except Exception as exc:  # pragma: no cover
        checks["redis"] = f"error: {exc}"
        healthy = False

    status = 200 if healthy else 503
    return jsonify({"status": "ok" if healthy else "degraded", "checks": checks}), status
