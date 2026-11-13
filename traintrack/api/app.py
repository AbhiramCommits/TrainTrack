"""Flask application factory."""

from __future__ import annotations

import logging
import os
from typing import Any

from flask import Flask, jsonify, request
from sqlalchemy import select

from traintrack.api.artifacts import bp as artifacts_bp
from traintrack.api.deps import close_db_session
from traintrack.api.errors import register_error_handlers
from traintrack.api.experiments import bp as experiments_bp
from traintrack.api.health import bp as health_bp
from traintrack.api.queue import bp as queue_bp
from traintrack.api.runs import bp as runs_bp
from traintrack.api.snapshots import bp as snapshots_bp
from traintrack.config import settings
from traintrack.db import get_session, init_db
from traintrack.models.api_key import ApiKey
from traintrack.services.auth import create_api_key, hash_api_key

logger = logging.getLogger(__name__)


def _bootstrap_dev_key(app: Flask) -> None:
    """Ensure the documented development API key exists for first-run UX."""
    if os.environ.get("TRAINTRACK_BOOTSTRAP_DEV_KEY", "1") == "0":
        return
    session = get_session()
    try:
        existing = session.execute(select(ApiKey).limit(1)).scalar_one_or_none()
        if existing is not None:
            return
        configured = settings().dev_api_key
        session.add(
            ApiKey(
                name="dev",
                key_hash=hash_api_key(configured),
                key_prefix=configured[:8],
                rate_limit_per_min=settings().rate_limit,
            )
        )
        session.commit()
        app.logger.info(
            "Bootstrapped development API key (prefix %s...)", configured[:8]
        )
    except Exception as exc:  # pragma: no cover - depends on DB being migrated
        session.rollback()
        logger.warning("Could not bootstrap development API key: %s", exc)
    finally:
        session.close()


def create_app(config: dict[str, Any] | None = None) -> Flask:
    """Create and configure the TrainTrack REST API."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = settings().secret_key
    app.config["JSON_SORT_KEYS"] = False
    if config:
        app.config.update(config)

    logging.basicConfig(
        level=os.environ.get("TRAINTRACK_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    init_db(app.config.get("DATABASE_URL"))
    _bootstrap_dev_key(app)

    register_error_handlers(app)
    app.teardown_appcontext(close_db_session)

    app.register_blueprint(health_bp)
    app.register_blueprint(experiments_bp)
    app.register_blueprint(runs_bp)
    app.register_blueprint(artifacts_bp)
    app.register_blueprint(snapshots_bp)
    app.register_blueprint(queue_bp)

    @app.before_request
    def _cors_preflight():
        if request.method == "OPTIONS":
            return ("", 204)

    @app.after_request
    def _cors_headers(response):
        origin = request.headers.get("Origin", "*")
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, X-API-Key, Authorization, Idempotency-Key, Last-Event-ID"
        )
        response.headers["Access-Control-Allow-Methods"] = (
            "GET, POST, PATCH, DELETE, OPTIONS"
        )
        response.headers["Vary"] = "Origin"
        return response

    @app.get("/")
    def index():
        return jsonify(
            {
                "service": "traintrack-api",
                "version": "0.1.0",
                "docs": "https://github.com/example/traintrack",
                "endpoints": [
                    "/healthz",
                    "/readyz",
                    "/api/experiments",
                    "/api/runs",
                    "/api/artifacts/<sha>/lineage",
                    "/api/snapshots",
                ],
            }
        )

    return app


def _seed_cli() -> None:
    """Create a developer API key and demo experiment (CLI helper)."""
    init_db()
    session = get_session()
    try:
        record, raw = create_api_key(session, "cli")
        session.commit()
        print(f"Created API key {record.key_prefix}...: {raw}")
    finally:
        session.close()


if __name__ == "__main__":  # pragma: no cover
    create_app().run(
        host=settings().api_host,
        port=settings().api_port,
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
