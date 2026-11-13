"""Run registry services: submission, transitions, claiming, queue position."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select, tuple_
from sqlalchemy.orm import Session

from traintrack.models.base import utcnow
from traintrack.models.run import Run, StatusTransition
from traintrack.models.snapshot import DatasetSnapshot
from traintrack.models.state import TERMINAL_STATUSES, RunStatus, ensure_transition


class RunNotFound(Exception):
    """No run with the requested id."""


class SnapshotRejected(Exception):
    """The requested dataset snapshot is missing or not validated."""


class InvalidStateChange(Exception):
    """A caller attempted an illegal transition."""

    def __init__(self, current: str, target: str) -> None:
        self.current = current
        self.target = target
        super().__init__(f"Cannot move run from {current} to {target}")


def canonical_request_hash(payload: dict[str, Any]) -> str:
    """Stable hash of a submission payload for idempotency conflict checks."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def get_run(session: Session, run_id: uuid.UUID | str) -> Run:
    """Fetch a run or raise :class:`RunNotFound`."""
    try:
        parsed = uuid.UUID(str(run_id))
    except ValueError as exc:
        raise RunNotFound(str(run_id)) from exc
    run = session.get(Run, parsed)
    if run is None:
        raise RunNotFound(str(run_id))
    return run


def transition_run(
    session: Session,
    run: Run,
    target: RunStatus | str,
    *,
    reason: str | None = None,
    actor: str | None = None,
    now: datetime | None = None,
) -> Run:
    """Apply a validated state transition and append an audit row."""
    timestamp = now or utcnow()
    current = run.status
    try:
        new_status = ensure_transition(current, target)
    except Exception as exc:
        raise InvalidStateChange(
            current.value, getattr(target, "value", str(target))
        ) from exc

    run.status = new_status
    if new_status is RunStatus.SCHEDULED:
        run.scheduled_at = timestamp
    if new_status is RunStatus.RUNNING:
        run.started_at = timestamp
        run.finished_at = None
    if new_status in TERMINAL_STATUSES:
        run.finished_at = timestamp
    if new_status is RunStatus.QUEUED:
        run.finished_at = None
        run.started_at = None
        run.scheduled_at = None
    if new_status is RunStatus.FAILED and reason:
        run.failure_reason = reason
    if new_status is RunStatus.SUCCEEDED:
        run.failure_reason = None
    session.add(
        StatusTransition(
            run_id=run.id,
            from_status=current.value,
            to_status=new_status.value,
            reason=reason,
            actor=actor,
            created_at=timestamp,
        )
    )
    session.flush()
    return run


def queue_position(session: Session, run: Run) -> int | None:
    """Return the 1-based position among QUEUED runs, or None when not queued."""
    if run.status is not RunStatus.QUEUED:
        return None
    ahead = session.execute(
        select(func.count(Run.id)).where(
            Run.status == RunStatus.QUEUED,
            tuple_(Run.priority, Run.created_at, Run.id)
            < (run.priority, run.created_at, run.id),
        )
    ).scalar_one()
    return int(ahead) + 1


def queued_runs(session: Session, limit: int = 100) -> list[Run]:
    """List queued runs in scheduling order."""
    return list(
        session.execute(
            select(Run)
            .where(Run.status == RunStatus.QUEUED)
            .order_by(Run.priority.asc(), Run.created_at.asc(), Run.id.asc())
            .limit(limit)
        ).scalars()
    )


def claim_run(
    session: Session, run_id: uuid.UUID, worker_id: str
) -> Run | None:
    """Atomically claim a QUEUED run for execution.

    Uses ``SELECT ... FOR UPDATE SKIP LOCKED`` so concurrent workers can never
    claim the same row. Returns None when another worker won the race.
    """
    run = session.execute(
        select(Run)
        .where(Run.id == run_id, Run.status == RunStatus.QUEUED)
        .with_for_update(skip_locked=True)
    ).scalar_one_or_none()
    if run is None:
        return None
    run.worker_id = worker_id
    run.attempt_count += 1
    run.executor = None
    transition_run(
        session, run, RunStatus.SCHEDULED, reason=f"claimed by {worker_id}", actor=worker_id
    )
    return run


def validate_snapshot(
    session: Session, snapshot_id: uuid.UUID | None
) -> DatasetSnapshot:
    """Ensure a snapshot exists and passed the validation gate."""
    if snapshot_id is None:
        raise SnapshotRejected("dataset_snapshot_id is required for training runs")
    snapshot = session.get(DatasetSnapshot, snapshot_id)
    if snapshot is None:
        raise SnapshotRejected(f"snapshot {snapshot_id} not found")
    if not snapshot.validated:
        raise SnapshotRejected(f"snapshot {snapshot.version} is not validated")
    return snapshot


def processing_timeout(run: Run) -> int:
    """Remaining wall-clock budget in seconds (never negative)."""
    if run.started_at is None:
        return run.timeout_seconds
    started = run.started_at
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    elapsed = (utcnow() - started).total_seconds()
    return max(0, int(run.timeout_seconds - elapsed))
