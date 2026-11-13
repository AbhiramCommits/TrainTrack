"""Leaderboard aggregation with explicit Redis TTL and invalidation."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from traintrack.config import settings
from traintrack.models.metric import MetricPoint
from traintrack.models.run import Run, StatusTransition
from traintrack.models.state import RunStatus
from traintrack.services.cache import cache_delete, cache_get_json, cache_set_json

LEADERBOARD_METRIC = "val_accuracy"
FALLBACK_METRICS = ("val_accuracy", "test_accuracy", "train_accuracy")


def leaderboard_cache_key(experiment_id: uuid.UUID | str) -> str:
    """Cache key for an experiment leaderboard."""
    return f"leaderboard:{experiment_id}"


def invalidate_leaderboard(experiment_id: uuid.UUID | str) -> None:
    """Drop the cached leaderboard for an experiment."""
    cache_delete(leaderboard_cache_key(experiment_id))


def compute_leaderboard(session: Session, experiment_id: uuid.UUID) -> list[dict[str, Any]]:
    """Return each run's best score, ordered best-first."""
    best: dict[uuid.UUID, tuple[str, float]] = {}
    for metric_name in FALLBACK_METRICS:
        rows = session.execute(
            select(MetricPoint.run_id, func.max(MetricPoint.value))
            .join(Run, Run.id == MetricPoint.run_id)
            .where(
                Run.experiment_id == experiment_id,
                MetricPoint.name == metric_name,
            )
            .group_by(MetricPoint.run_id)
        ).all()
        for run_id, value in rows:
            if run_id not in best and value is not None:
                best[run_id] = (metric_name, float(value))

    entries: list[dict[str, Any]] = []
    if not best:
        return entries
    runs = {
        run.id: run
        for run in session.execute(
            select(Run).where(Run.id.in_(list(best.keys())))
        ).scalars()
    }
    for run_id, (metric_name, value) in best.items():
        run = runs.get(run_id)
        if run is None or run.status not in (
            RunStatus.SUCCEEDED,
            RunStatus.RUNNING,
        ):
            continue
        entries.append(
            {
                "run_id": str(run.id),
                "experiment_id": str(run.experiment_id),
                "status": run.status.value,
                "metric_name": metric_name,
                "best_value": round(value, 6),
                "hyperparams": run.hyperparams,
                "dataset_snapshot_id": (
                    str(run.dataset_snapshot_id) if run.dataset_snapshot_id else None
                ),
                "created_at": run.created_at.isoformat(),
                "finished_at": run.finished_at.isoformat() if run.finished_at else None,
            }
        )
    entries.sort(key=lambda entry: entry["best_value"], reverse=True)
    for rank, entry in enumerate(entries, start=1):
        entry["rank"] = rank
    return entries


def get_leaderboard(
    session: Session, experiment_id: uuid.UUID, *, use_cache: bool = True
) -> tuple[list[dict[str, Any]], bool]:
    """Return (entries, cache_hit) for an experiment."""
    key = leaderboard_cache_key(experiment_id)
    if use_cache:
        cached = cache_get_json(key)
        if isinstance(cached, list):
            return cached, True
    entries = compute_leaderboard(session, experiment_id)
    cache_set_json(key, entries, settings().leaderboard_ttl)
    return entries, False


def transition_reason(session: Session, run: Run, status: RunStatus) -> str | None:
    """Most recent audit reason for a given status (used in API detail views)."""
    reason = session.execute(
        select(StatusTransition.reason)
        .where(
            StatusTransition.run_id == run.id,
            StatusTransition.to_status == status.value,
        )
        .order_by(StatusTransition.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()
    return reason
