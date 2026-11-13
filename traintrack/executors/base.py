"""Executor interface shared by local subprocess and Kubernetes jobs."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class JobContext:
    """Everything an executor needs about a run, detached from the ORM."""

    run_id: uuid.UUID
    experiment_id: uuid.UUID
    job_kind: str
    hyperparams: dict[str, Any]
    dataset_snapshot_path: str | None
    dataset_snapshot_id: uuid.UUID | None
    requested_gpus: int
    image: str | None
    timeout_seconds: int
    git_sha: str | None = None
    dataset_snapshot_version: str | None = None
    dataset_snapshot_hash: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Terminal outcome reported by an executor."""

    exit_code: int
    canceled: bool = False
    reason: str | None = None
    artifact_sha: str | None = None


class BaseExecutor:
    """Abstract job executor."""

    name = "base"

    def run(self, job: JobContext) -> ExecutionResult:
        """Execute the job and return its terminal result."""
        raise NotImplementedError

    def cancel(self, run_id: uuid.UUID) -> None:
        """Best-effort cancellation hook (the DB flag is authoritative)."""
        return None
