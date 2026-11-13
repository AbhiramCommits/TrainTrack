"""Prometheus exporter backed by the run registry, Redis and RabbitMQ.

A custom collector is used so every scrape reflects live state rather than
per-process counters that would miss work done by worker processes. Terminally
completed runs are counted from the append-only transition audit, which makes
``traintrack_runs_total`` monotonic.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from typing import Any

from prometheus_client import generate_latest
from prometheus_client.core import (
    CounterMetricFamily,
    GaugeMetricFamily,
    HistogramMetricFamily,
)
from prometheus_client.registry import Collector, CollectorRegistry
from sqlalchemy import func, select

from traintrack.broker.publisher import queue_stats
from traintrack.db import get_session
from traintrack.models.metric import MetricPoint
from traintrack.models.run import Run, StatusTransition
from traintrack.models.state import RunStatus
from traintrack.services.cache import get_redis

logger = logging.getLogger(__name__)

DURATION_BUCKETS = [0.1, 1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600, 7200]
QUEUE_WAIT_BUCKETS = [0.01, 0.1, 0.5, 1, 2, 5, 10, 30, 60, 300, 900, 3600]
SCHEDULING_BUCKETS = [0.1, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600]
DLQ_CACHE_KEY = "metrics:dlq"
DLQ_CACHE_TTL = 10


def _histogram(name: str, documentation: str, values: list[float], buckets: list[float]):
    """Build a cumulative HistogramMetricFamily from raw observations."""
    bucket_counts: list[tuple[str, float]] = []
    for boundary in buckets:
        bucket_counts.append(
            (str(boundary), float(sum(1 for value in values if value <= boundary)))
        )
    bucket_counts.append(("+Inf", float(len(values))))
    return HistogramMetricFamily(
        name,
        documentation,
        buckets=bucket_counts,
        sum_value=sum(values) if values else 0.0,
    )


def _dlq_count() -> int | None:
    """Read the DLQ depth from a short-lived Redis cache."""
    redis_client = get_redis()
    try:
        cached = redis_client.get(DLQ_CACHE_KEY)
        if cached is not None:
            return int(cached)
        stats = queue_stats()
        from traintrack.broker.topology import RUNS_DEAD

        count = int(stats.get(RUNS_DEAD, 0))
        redis_client.set(DLQ_CACHE_KEY, count, ex=DLQ_CACHE_TTL)
        return count
    except Exception as exc:
        logger.debug("DLQ depth unavailable: %s", exc)
        return None


class TrainTrackCollector(Collector):
    """Collects TrainTrack metrics at scrape time."""

    def collect(self) -> Iterable[Any]:
        """Yield all metric families."""
        session = get_session()
        try:
            yield from self._collect_db(session)
        finally:
            session.close()
        yield from self._collect_resources()

    def _collect_db(self, session) -> Iterable[Any]:
        terminal_counts = dict(
            session.execute(
                select(StatusTransition.to_status, func.count(StatusTransition.id))
                .where(
                    StatusTransition.to_status.in_(
                        [
                            RunStatus.SUCCEEDED.value,
                            RunStatus.FAILED.value,
                            RunStatus.CANCELED.value,
                        ]
                    )
                )
                .group_by(StatusTransition.to_status)
            ).all()
        )
        runs_family = CounterMetricFamily(
            "traintrack_runs_total",
            "Terminal run outcomes recorded in the audit log",
            labels=["status"],
        )
        for status in ("SUCCEEDED", "FAILED", "CANCELED"):
            runs_family.add_metric([status], int(terminal_counts.get(status, 0)))
        yield runs_family

        rows = session.execute(
            select(Run.started_at, Run.finished_at, Run.status).where(
                Run.finished_at.is_not(None), Run.started_at.is_not(None)
            )
        ).all()
        durations = [
            (finished - started).total_seconds()
            for started, finished, _ in rows
            if finished and started
        ]
        yield _histogram(
            "traintrack_run_duration_seconds",
            "Wall-clock duration of finished runs",
            durations,
            DURATION_BUCKETS,
        )

        queued = session.execute(
            select(func.count(Run.id)).where(Run.status == RunStatus.QUEUED)
        ).scalar_one()
        yield GaugeMetricFamily(
            "traintrack_queue_depth", "Runs waiting in the QUEUED state", value=int(queued)
        )

        waits = [
            (scheduled - created).total_seconds()
            for created, scheduled in session.execute(
                select(Run.created_at, Run.scheduled_at).where(
                    Run.scheduled_at.is_not(None)
                )
            ).all()
            if scheduled and created
        ]
        yield _histogram(
            "traintrack_queue_wait_seconds",
            "Time between submission and scheduler claim",
            waits,
            QUEUE_WAIT_BUCKETS,
        )

        retries = session.execute(
            select(func.count(StatusTransition.id)).where(
                StatusTransition.from_status == RunStatus.FAILED.value,
                StatusTransition.to_status == RunStatus.QUEUED.value,
            )
        ).scalar_one()
        yield CounterMetricFamily(
            "traintrack_retries_total",
            "Retry transitions from FAILED back to QUEUED",
            value=int(retries),
        )

        latency_values = [
            float(value)
            for (value,) in session.execute(
                select(Run.scheduling_latency_seconds).where(
                    Run.scheduling_latency_seconds.is_not(None)
                )
            ).all()
            if value is not None
        ]
        yield _histogram(
            "traintrack_scheduling_latency_seconds",
            "Seconds from submission to a Kubernetes pod reaching Running",
            latency_values,
            SCHEDULING_BUCKETS,
        )

        dlq = _dlq_count()
        if dlq is not None:
            yield GaugeMetricFamily(
                "traintrack_dlq_messages",
                "Messages currently in the runs.dead queue",
                value=dlq,
            )

        metric_count = session.execute(
            select(func.count(MetricPoint.id))
        ).scalar_one()
        yield CounterMetricFamily(
            "traintrack_metric_points_total",
            "Metric points ingested",
            value=int(metric_count),
        )

    def _collect_resources(self) -> Iterable[Any]:
        """Latest device utilization and memory per run from Redis streams."""
        utilization = GaugeMetricFamily(
            "traintrack_run_device_utilization_percent",
            "Latest device utilization sample for a run",
            labels=["run_id", "device"],
        )
        memory = GaugeMetricFamily(
            "traintrack_run_memory_used_mb",
            "Latest device memory usage for a run",
            labels=["run_id", "device"],
        )
        found = False
        try:
            redis_client = get_redis()
            for key in redis_client.scan_iter(match="resources:*", count=100):
                entries = redis_client.xrevrange(key, max="+", min="-", count=1)
                if not entries:
                    continue
                try:
                    sample = json.loads(entries[0][1].get("data", "{}"))
                except (json.JSONDecodeError, AttributeError):
                    continue
                run_id = str(key).split(":", 1)[1]
                device = sample.get("device", "unknown")
                utilization.add_metric(
                    [run_id, device], float(sample.get("utilization_pct", 0.0))
                )
                memory.add_metric(
                    [run_id, device], float(sample.get("memory_used_mb", 0.0))
                )
                found = True
        except Exception as exc:
            logger.debug("per-run resource metrics unavailable: %s", exc)
        if found:
            yield utilization
            yield memory


def render_metrics() -> bytes:
    """Render the current /metrics exposition payload."""
    registry = CollectorRegistry()
    registry.register(TrainTrackCollector())
    return generate_latest(registry)


__all__ = ["TrainTrackCollector", "render_metrics"]
