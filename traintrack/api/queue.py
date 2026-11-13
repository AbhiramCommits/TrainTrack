"""Queue health endpoint backed by RabbitMQ queue introspection."""

from __future__ import annotations

import logging

import pika
from flask import Blueprint, jsonify

from traintrack.api.auth import require_api_key
from traintrack.broker.publisher import queue_stats
from traintrack.broker.topology import RUNS_DEAD, RUNS_RETRY, RUNS_SUBMITTED

bp = Blueprint("queue", __name__, url_prefix="/api/queue")
logger = logging.getLogger(__name__)


@bp.get("/health")
@require_api_key
def queue_health():
    """Return work, retry and dead-letter queue depths."""
    try:
        stats = queue_stats()
    except (pika.exceptions.AMQPError, OSError) as exc:
        logger.warning("queue introspection failed: %s", exc)
        return (
            jsonify(
                {
                    "status": "unavailable",
                    "error": "broker_unavailable",
                    "message": str(exc),
                }
            ),
            503,
        )
    return jsonify(
        {
            "status": "ok",
            "depth": stats.get(RUNS_SUBMITTED, 0),
            "retry_depth": stats.get(RUNS_RETRY, 0),
            "dlq_messages": stats.get(RUNS_DEAD, 0),
            "queues": stats,
        }
    )
