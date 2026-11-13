"""Experiment model."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from traintrack.models.base import Base, CreatedAtMixin, new_uuid

if TYPE_CHECKING:
    from traintrack.models.run import Run


class Experiment(Base, CreatedAtMixin):
    """A named grouping of runs that share a hypothesis."""

    __tablename__ = "experiments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=new_uuid
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    runs: Mapped[list[Run]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Experiment {self.name}>"
