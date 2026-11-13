"""Per-run device utilization samples stored in Redis."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from traintrack.services.cache import get_redis


def resource_stream_key(run_id: uuid.UUID | str) -> str:
    """Redis stream key for a run's device samples."""
    return f"resources:{run_id}"


def append_resources(
    run_id: uuid.UUID | str, samples: list[dict[str, Any]], maxlen: int = 2000
) -> int:
    """Append device samples to a capped stream."""
    client = get_redis()
    pipe = client.pipeline()
    for sample in samples:
        payload = dict(sample)
        payload.setdefault("wall_time", time.time())
        pipe.xadd(
            resource_stream_key(run_id),
            {"data": json.dumps(payload)},
            maxlen=maxlen,
            approximate=True,
        )
    pipe.execute()
    return len(samples)


def read_resources(run_id: uuid.UUID | str, limit: int = 1000) -> list[dict[str, Any]]:
    """Read the most recent device samples for a run."""
    raw = get_redis().xrevrange(
        resource_stream_key(run_id), max="+", min="-", count=limit
    )
    samples: list[dict[str, Any]] = []
    for entry_id, fields in reversed(raw):
        try:
            payload = json.loads(fields.get("data", "{}"))
        except json.JSONDecodeError:
            continue
        payload["id"] = entry_id
        samples.append(payload)
    return samples
