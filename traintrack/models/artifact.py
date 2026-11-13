"""Content-addressed artifact metadata."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from traintrack.models.base import Base, CreatedAtMixin

if TYPE_CHECKING:
    from traintrack.models.run import Run
    from traintrack.models.snapshot import DatasetSnapshot


class Artifact(Base, CreatedAtMixin):
    """A stored file keyed by its sha256 digest.

    One row exists per unique digest; ``ref_count`` counts duplicate writes so
    dedupe savings can be reported honestly.
    """

    __tablename__ = "artifacts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    sha256: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    kind: Mapped[str] = mapped_column(String(50), nullable=False)
    size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    uri: Mapped[str] = mapped_column(Text, nullable=False)
    run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("runs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    dataset_snapshot_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("dataset_snapshots.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    parent_sha: Mapped[str | None] = mapped_column(String(64), nullable=True)
    ref_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    bytes_saved: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)

    run: Mapped[Run | None] = relationship(back_populates="artifacts")
    dataset_snapshot: Mapped[DatasetSnapshot | None] = relationship()

    def __repr__(self) -> str:
        return f"<Artifact {self.kind} {self.sha256[:8]}>"
