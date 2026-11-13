"""Dataset snapshot registry model."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from traintrack.models.base import Base, CreatedAtMixin, new_uuid

if TYPE_CHECKING:
    from traintrack.models.run import Run


class DatasetSnapshot(Base, CreatedAtMixin):
    """An immutable, content-hashed dataset version produced by the pipeline."""

    __tablename__ = "dataset_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=new_uuid
    )
    version: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    manifest: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    validated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)

    runs: Mapped[list[Run]] = relationship(back_populates="dataset_snapshot")

    def __repr__(self) -> str:
        return f"<DatasetSnapshot {self.version} {self.content_hash[:8]}>"
