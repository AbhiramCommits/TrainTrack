"""Run registry models: runs and the append-only transition audit."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from traintrack.models.base import Base, CreatedAtMixin, UpdatedAtMixin, new_uuid
from traintrack.models.state import RunStatus

if TYPE_CHECKING:
    from traintrack.models.artifact import Artifact
    from traintrack.models.experiment import Experiment
    from traintrack.models.metric import MetricPoint
    from traintrack.models.snapshot import DatasetSnapshot


def _status_column() -> Enum:
    return Enum(
        RunStatus,
        name="run_status",
        native_enum=False,
        length=20,
        validate_strings=True,
        values_callable=lambda enum: [member.value for member in enum],
    )


class Run(Base, CreatedAtMixin, UpdatedAtMixin):
    """A single submitted training or evaluation job."""

    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=new_uuid
    )
    experiment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[RunStatus] = mapped_column(
        _status_column(), nullable=False, default=RunStatus.QUEUED, index=True
    )
    job_kind: Mapped[str] = mapped_column(
        String(20), nullable=False, default="train"
    )
    hyperparams: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict
    )
    git_sha: Mapped[str | None] = mapped_column(String(64), nullable=True)
    dataset_snapshot_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("dataset_snapshots.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    queue_position: Mapped[int | None] = mapped_column(Integer, nullable=True)
    requested_gpus: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    image: Mapped[str | None] = mapped_column(String(300), nullable=True)
    command: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    timeout_seconds: Mapped[int] = mapped_column(
        Integer, nullable=False, default=3600
    )
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    idempotency_key: Mapped[str | None] = mapped_column(
        String(200), nullable=True, unique=True
    )
    request_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    executor: Mapped[str | None] = mapped_column(String(50), nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    pod_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    scheduling_latency_seconds: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    scheduled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    experiment: Mapped[Experiment] = relationship(back_populates="runs")
    dataset_snapshot: Mapped[DatasetSnapshot | None] = relationship(
        back_populates="runs"
    )
    transitions: Mapped[list[StatusTransition]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="StatusTransition.created_at",
    )
    metrics: Mapped[list[MetricPoint]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    artifacts: Mapped[list[Artifact]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_runs_queue_order", "status", "priority", "created_at"),
        Index("ix_runs_experiment_created", "experiment_id", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Run {self.id} {self.status.value}>"


class StatusTransition(Base):
    """Append-only audit row for a run status change."""

    __tablename__ = "status_transitions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    from_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    to_status: Mapped[str] = mapped_column(String(20), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    actor: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default="now()"
    )

    run: Mapped[Run] = relationship(back_populates="transitions")

    def __repr__(self) -> str:
        return f"<StatusTransition {self.from_status}->{self.to_status}>"
