"""Declarative base and common column helpers."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    """Timezone-aware UTC now."""
    return datetime.now(UTC)


def new_uuid() -> uuid.UUID:
    """Generate a random UUID4."""
    return uuid.uuid4()


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class CreatedAtMixin:
    """Adds a timezone-aware ``created_at`` column."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class UpdatedAtMixin:
    """Adds a timezone-aware ``updated_at`` column updated on write."""

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )
