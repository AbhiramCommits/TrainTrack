"""Redis Stream live log storage for run output."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from traintrack.config import settings
from traintrack.services.cache import get_redis


def log_stream_key(run_id: uuid.UUID | str) -> str:
    """Redis stream key holding a run's log lines."""
    return f"logs:{run_id}"


def append_log(
    run_id: uuid.UUID | str, line: str, *, level: str = "info", maxlen: int | None = None
) -> str:
    """Append one log line to the run's capped Redis stream."""
    payload = json.dumps(
        {"ts": time.time(), "level": level, "line": line.rstrip("\n")}
    )
    key = log_stream_key(run_id)
    limit = maxlen or settings().log_maxlen
    return str(
        get_redis().xadd(key, {"data": payload}, maxlen=limit, approximate=True)
    )


def read_logs(
    run_id: uuid.UUID | str, *, tail: int = 200, after_id: str | None = None
) -> list[dict[str, Any]]:
    """Read log entries, either the last N or entries after a stream id."""
    key = log_stream_key(run_id)
    client = get_redis()
    if after_id:
        raw = client.xrange(key, min=f"({after_id}", max="+", count=tail)
    else:
        raw = client.xrevrange(key, max="+", min="-", count=tail)
        raw = list(reversed(raw))
    entries: list[dict[str, Any]] = []
    for entry_id, fields in raw:
        try:
            payload = json.loads(fields.get("data", "{}"))
        except json.JSONDecodeError:
            payload = {"line": fields.get("data", ""), "ts": 0.0, "level": "info"}
        entries.append({"id": entry_id, **payload})
    return entries


def latest_log_id(run_id: uuid.UUID | str) -> str | None:
    """Return the newest stream entry id, if any."""
    entries = get_redis().xrevrange(log_stream_key(run_id), max="+", min="-", count=1)
    return str(entries[0][0]) if entries else None


def wait_for_logs(
    run_id: uuid.UUID | str, last_id: str, *, block_ms: int = 15000, count: int = 100
) -> list[tuple[str, dict[str, str]]]:
    """Block for new log entries using XREAD."""
    result = get_redis().xread(
        {log_stream_key(run_id): last_id}, count=count, block=block_ms
    )
    if not result:
        return []
    _, entries = result[0]
    return [(str(entry_id), fields) for entry_id, fields in entries]
