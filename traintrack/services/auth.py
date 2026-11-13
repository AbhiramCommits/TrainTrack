"""API key hashing, creation and constant-time verification."""

from __future__ import annotations

import hashlib
import hmac
import secrets

from sqlalchemy import select
from sqlalchemy.orm import Session

from traintrack.models.api_key import ApiKey

KEY_PREFIX = "tt_"
DISPLAY_PREFIX_LENGTH = 8


def hash_api_key(raw_key: str) -> str:
    """Hash an API key for storage and comparison (sha256 hex)."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_api_key() -> str:
    """Generate a new random plaintext API key."""
    return KEY_PREFIX + secrets.token_urlsafe(32)


def create_api_key(
    session: Session, name: str, rate_limit_per_min: int = 120
) -> tuple[ApiKey, str]:
    """Create and persist an API key, returning (record, plaintext key)."""
    raw = generate_api_key()
    record = ApiKey(
        name=name,
        key_hash=hash_api_key(raw),
        key_prefix=raw[:DISPLAY_PREFIX_LENGTH],
        rate_limit_per_min=rate_limit_per_min,
    )
    session.add(record)
    session.flush()
    return record, raw


def verify_api_key(session: Session, raw_key: str) -> ApiKey | None:
    """Look up a key by hash and compare in constant time.

    The lookup is by hash; the digest comparison still uses
    :func:`hmac.compare_digest` so a timing signal cannot leak a stored hash.
    """
    if not raw_key:
        return None
    digest = hash_api_key(raw_key)
    record = session.execute(
        select(ApiKey).where(ApiKey.key_hash == digest, ApiKey.active.is_(True))
    ).scalar_one_or_none()
    if record is None:
        # Compare against a dummy digest to keep the failure path cheap to
        # reason about without revealing whether the prefix existed.
        hmac.compare_digest(digest, hashlib.sha256(b"missing").hexdigest())
        return None
    if not hmac.compare_digest(record.key_hash, digest):
        return None
    return record
