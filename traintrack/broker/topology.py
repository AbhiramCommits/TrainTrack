"""Durable RabbitMQ topology: work queue, delayed retry queue, and DLQ.

The retry queue is a classic durable queue. Messages are published to it with a
per-message TTL; when the TTL expires RabbitMQ dead-letters them back to the
main exchange with the ``run.submitted`` routing key.
"""

from __future__ import annotations

from typing import Any

import pika

EXCHANGE = "traintrack"
RUNS_SUBMITTED = "runs.submitted"
RUNS_RETRY = "runs.retry"
RUNS_DEAD = "runs.dead"

RK_SUBMITTED = "run.submitted"
RK_RETRY = "run.retry"
RK_DEAD = "run.dead"


def declare_topology(channel: Any) -> None:
    """Declare the exchange, work queue, retry queue and DLQ idempotently."""
    channel.exchange_declare(
        exchange=EXCHANGE, exchange_type="direct", durable=True
    )

    channel.queue_declare(
        queue=RUNS_SUBMITTED,
        durable=True,
        arguments={"x-queue-type": "quorum"},
    )
    channel.queue_bind(
        queue=RUNS_SUBMITTED, exchange=EXCHANGE, routing_key=RK_SUBMITTED
    )

    channel.queue_declare(
        queue=RUNS_RETRY,
        durable=True,
        arguments={
            "x-dead-letter-exchange": EXCHANGE,
            "x-dead-letter-routing-key": RK_SUBMITTED,
        },
    )
    channel.queue_bind(queue=RUNS_RETRY, exchange=EXCHANGE, routing_key=RK_RETRY)

    channel.queue_declare(queue=RUNS_DEAD, durable=True)
    channel.queue_bind(queue=RUNS_DEAD, exchange=EXCHANGE, routing_key=RK_DEAD)


def persistent_properties(
    run_id: str, attempt: int, *, message_id: str | None = None
) -> pika.BasicProperties:
    """Message properties for at-least-once, persistent delivery."""
    return pika.BasicProperties(
        content_type="application/json",
        delivery_mode=2,
        message_id=message_id or run_id,
        headers={"attempt": attempt, "run_id": run_id},
    )
