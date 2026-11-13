"""Metric points emitted by training and evaluation jobs."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from traintrack.models.base import Base, utcnow

if TYPE_CHECKING:
    from traintrack.models.run import Run


class MetricPoint(Base):
    """A scalar metric sample: (run, name, step, value)."""

    __tablename__ = "metric_points"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    step: Mapped[int] = mapped_column(Integer, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    wall_time: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )

    run: Mapped[Run] = relationship(back_populates="metrics")

    __table_args__ = (Index("ix_metrics_run_name_step", "run_id", "name", "step"),)

    def __repr__(self) -> str:
        return f"<MetricPoint {self.name}[{self.step}]={self.value}>"
