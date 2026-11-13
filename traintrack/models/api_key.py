"""API keys with hashed storage."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from traintrack.models.base import Base, CreatedAtMixin, new_uuid


class ApiKey(Base, CreatedAtMixin):
    """A hashed API key record. Plaintext keys are never stored."""

    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=new_uuid
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    key_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    key_prefix: Mapped[str] = mapped_column(String(12), nullable=False)
    rate_limit_per_min: Mapped[int] = mapped_column(
        Integer, nullable=False, default=120
    )
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    def __repr__(self) -> str:
        return f"<ApiKey {self.name} {self.key_prefix}...>"
