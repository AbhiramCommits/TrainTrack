"""JSON serialisation helpers for API responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from traintrack.models.artifact import Artifact
from traintrack.models.run import Run, StatusTransition
from traintrack.models.snapshot import DatasetSnapshot


def iso(value: datetime | None) -> str | None:
    """Serialise a datetime as ISO-8601, preserving None."""
    return value.isoformat() if value else None


def transition_to_dict(transition: StatusTransition) -> dict[str, Any]:
    """Serialise an audit row."""
    return {
        "id": transition.id,
        "from_status": transition.from_status,
        "to_status": transition.to_status,
        "reason": transition.reason,
        "actor": transition.actor,
        "created_at": iso(transition.created_at),
    }


def run_to_dict(
    run: Run,
    *,
    position: int | None = None,
    include_transitions: bool = False,
) -> dict[str, Any]:
    """Serialise a run, optionally including its transition timeline."""
    payload: dict[str, Any] = {
        "id": str(run.id),
        "experiment_id": str(run.experiment_id),
        "status": run.status.value,
        "job_kind": run.job_kind,
        "hyperparams": run.hyperparams,
        "git_sha": run.git_sha,
        "dataset_snapshot_id": (
            str(run.dataset_snapshot_id) if run.dataset_snapshot_id else None
        ),
        "queue_position": position,
        "requested_gpus": run.requested_gpus,
        "image": run.image,
        "command": run.command,
        "timeout_seconds": run.timeout_seconds,
        "priority": run.priority,
        "attempt_count": run.attempt_count,
        "max_attempts": run.max_attempts,
        "idempotency_key": run.idempotency_key,
        "failure_reason": run.failure_reason,
        "executor": run.executor,
        "worker_id": run.worker_id,
        "created_at": iso(run.created_at),
        "scheduled_at": iso(run.scheduled_at),
        "started_at": iso(run.started_at),
        "finished_at": iso(run.finished_at),
        "duration_seconds": (
            round((run.finished_at - run.started_at).total_seconds(), 3)
            if run.finished_at and run.started_at
            else None
        ),
    }
    if include_transitions:
        payload["transitions"] = [
            transition_to_dict(transition) for transition in run.transitions
        ]
    return payload


def artifact_to_dict(artifact: Artifact) -> dict[str, Any]:
    """Serialise artifact metadata."""
    return {
        "id": artifact.id,
        "sha256": artifact.sha256,
        "kind": artifact.kind,
        "size": artifact.size,
        "uri": artifact.uri,
        "run_id": str(artifact.run_id) if artifact.run_id else None,
        "dataset_snapshot_id": (
            str(artifact.dataset_snapshot_id)
            if artifact.dataset_snapshot_id
            else None
        ),
        "parent_sha": artifact.parent_sha,
        "ref_count": artifact.ref_count,
        "bytes_saved": artifact.bytes_saved,
        "created_at": iso(artifact.created_at),
    }


def snapshot_to_dict(snapshot: DatasetSnapshot) -> dict[str, Any]:
    """Serialise a dataset snapshot."""
    return {
        "id": str(snapshot.id),
        "version": snapshot.version,
        "content_hash": snapshot.content_hash,
        "row_count": snapshot.row_count,
        "manifest": snapshot.manifest,
        "validated": snapshot.validated,
        "path": snapshot.path,
        "created_at": iso(snapshot.created_at),
    }
