"""Publisher with confirms and persistent delivery."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from typing import Any

import pika

from traintrack.broker.topology import (
    EXCHANGE,
    RK_DEAD,
    RK_RETRY,
    RK_SUBMITTED,
    RUNS_DEAD,
    RUNS_RETRY,
    RUNS_SUBMITTED,
    declare_topology,
    persistent_properties,
)
from traintrack.config import settings

logger = logging.getLogger(__name__)


def connect(url: str | None = None, timeout: int = 10) -> pika.BlockingConnection:
    """Open a blocking RabbitMQ connection with a heartbeat."""
    parameters = pika.URLParameters(url or settings().amqp_url)
    parameters.heartbeat = 30
    parameters.blocked_connection_timeout = 30
    parameters.socket_timeout = timeout
    return pika.BlockingConnection(parameters)


def publish_run(
    channel: Any,
    run_id: str | uuid.UUID,
    attempt: int,
    *,
    delay_seconds: float = 0,
) -> None:
    """Publish a run message, either immediately or via the TTL retry queue."""
    body = json.dumps({"run_id": str(run_id), "attempt": attempt}).encode()
    properties = persistent_properties(str(run_id), attempt)
    if delay_seconds > 0:
        properties.expiration = str(max(100, int(delay_seconds * 1000)))
        routing_key = RK_RETRY
    else:
        routing_key = RK_SUBMITTED
    channel.basic_publish(
        exchange=EXCHANGE,
        routing_key=routing_key,
        body=body,
        properties=properties,
        mandatory=True,
    )


def publish_dead(channel: Any, run_id: str | uuid.UUID, attempt: int, reason: str) -> None:
    """Publish a failure record to the dead-letter queue."""
    body = json.dumps(
        {
            "run_id": str(run_id),
            "attempts": attempt,
            "reason": reason,
            "run_dead_letter": True,
        }
    ).encode()
    properties = persistent_properties(str(run_id), attempt)
    channel.basic_publish(
        exchange=EXCHANGE,
        routing_key=RK_DEAD,
        body=body,
        properties=properties,
        mandatory=True,
    )


class Publisher:
    """Thread-safe singleton publisher using publisher confirms."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._connection: pika.BlockingConnection | None = None
        self._channel: Any = None

    def _ensure_channel(self) -> Any:
        if self._connection is not None and self._connection.is_open:
            return self._channel
        self._connection = connect()
        self._channel = self._connection.channel()
        declare_topology(self._channel)
        self._channel.confirm_delivery()
        return self._channel

    def publish(self, run_id: str | uuid.UUID, attempt: int, delay_seconds: float = 0) -> None:
        """Publish with confirms, reconnecting once on transport failure."""
        with self._lock:
            try:
                channel = self._ensure_channel()
                publish_run(channel, run_id, attempt, delay_seconds=delay_seconds)
            except (pika.exceptions.AMQPError, OSError):
                self.close()
                channel = self._ensure_channel()
                publish_run(channel, run_id, attempt, delay_seconds=delay_seconds)

    def dead_letter(self, run_id: str | uuid.UUID, attempt: int, reason: str) -> None:
        """Publish a DLQ record with confirms."""
        with self._lock:
            channel = self._ensure_channel()
            publish_dead(channel, run_id, attempt, reason)

    def close(self) -> None:
        """Close the underlying connection if open."""
        try:
            if self._connection is not None and self._connection.is_open:
                self._connection.close()
        except pika.exceptions.AMQPError:  # pragma: no cover
            logger.warning("error closing publisher connection", exc_info=True)
        self._connection = None
        self._channel = None


_publisher: Publisher | None = None
_publisher_lock = threading.Lock()


def get_publisher() -> Publisher:
    """Return the process-wide publisher singleton."""
    global _publisher
    with _publisher_lock:
        if _publisher is None:
            _publisher = Publisher()
        return _publisher


def reset_publisher() -> None:
    """Close and drop the singleton (tests and shutdown)."""
    global _publisher
    with _publisher_lock:
        if _publisher is not None:
            _publisher.close()
        _publisher = None


def queue_stats() -> dict[str, int]:
    """Return message counts for work, retry and dead queues."""
    connection = connect(timeout=3)
    try:
        channel = connection.channel()
        declare_topology(channel)
        stats: dict[str, int] = {}
        for queue in (RUNS_SUBMITTED, RUNS_RETRY, RUNS_DEAD):
            declared = channel.queue_declare(queue=queue, passive=True)
            stats[queue] = int(declared.method.message_count)
        return stats
    finally:
        if connection.is_open:
            connection.close()
