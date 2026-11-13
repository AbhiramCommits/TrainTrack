"""Durable RabbitMQ topology: work queue, delayed retry queue, and DLQ.

The retry queue is a classic durable queue. Messages are published to it with a
per-message TTL; when the TTL expires RabbitMQ dead-letters them back to the
main exchange with the ``run.submitted`` routing key.
"""

from __future__ import annotations

import os
from typing import Any

import pika

# A prefix lets tests (and parallel dev stacks) isolate their queues. The
# default is empty, so production queue names are exactly runs.submitted,
# runs.retry and runs.dead as documented.
_PREFIX = os.environ.get("TRAINTRACK_QUEUE_PREFIX", "")

EXCHANGE = f"{_PREFIX}traintrack"
RUNS_SUBMITTED = f"{_PREFIX}runs.submitted"
RUNS_RETRY = f"{_PREFIX}runs.retry"
RUNS_DEAD = f"{_PREFIX}runs.dead"

RK_SUBMITTED = f"{_PREFIX}run.submitted"
RK_RETRY = f"{_PREFIX}run.retry"
RK_DEAD = f"{_PREFIX}run.dead"


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
