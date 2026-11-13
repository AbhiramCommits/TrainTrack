"""Redis execution locks so two workers cannot recover the same run twice."""

from __future__ import annotations

import uuid

from traintrack.services.cache import get_redis


def execution_lock_key(run_id: uuid.UUID | str) -> str:
    """Redis key guarding execution of a run."""
    return f"runexec:{run_id}"


def acquire_execution_lock(
    run_id: uuid.UUID | str, worker_id: str, ttl_seconds: int = 30
) -> bool:
    """Try to take the per-run execution lock."""
    return bool(
        get_redis().set(
            execution_lock_key(run_id), worker_id, nx=True, ex=ttl_seconds
        )
    )


def refresh_execution_lock(
    run_id: uuid.UUID | str, worker_id: str, ttl_seconds: int = 30
) -> bool:
    """Extend a held lock, verifying ownership."""
    client = get_redis()
    key = execution_lock_key(run_id)
    if client.get(key) != worker_id:
        return False
    return bool(client.expire(key, ttl_seconds))


def release_execution_lock(run_id: uuid.UUID | str, worker_id: str) -> None:
    """Release a lock if this worker owns it."""
    client = get_redis()
    key = execution_lock_key(run_id)
    if client.get(key) == worker_id:
        client.delete(key)


def lock_holder(run_id: uuid.UUID | str) -> str | None:
    """Return the worker id currently holding a run lock, if any."""
    value = get_redis().get(execution_lock_key(run_id))
    return str(value) if value else None
