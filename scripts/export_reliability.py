"""Measure retry and dead-letter behavior with a deliberately failing job.

Creates a validated snapshot row that points at a missing directory, submits a
training run with max_attempts=2, runs one local worker, and records the
observed attempts, transition timeline, retry delay, and DLQ depth.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT_ROOT / "docs" / "results" / "reliability.json"
API = "http://localhost:8010"
KEY = "tt_dev_local_key_do_not_use_in_prod"
HEADERS = {"X-API-Key": KEY}


def api(method: str, path: str, payload: dict | None = None, idem: str | None = None) -> Any:
    headers = dict(HEADERS)
    if idem:
        headers["Idempotency-Key"] = idem
    response = requests.request(
        method, f"{API}{path}", json=payload, headers=headers, timeout=30
    )
    if response.status_code == 429:
        time.sleep(float(response.headers.get("Retry-After", "2")))
        return api(method, path, payload, idem)
    response.raise_for_status()
    return response.json()


def wait_for_worker(timeout: float = 30) -> None:
    from traintrack.broker.publisher import queue_stats

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            queue_stats()
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("broker unavailable")


def purge() -> None:
    from traintrack.broker.publisher import connect
    from traintrack.broker.topology import RUNS_DEAD, RUNS_RETRY, RUNS_SUBMITTED, declare_topology

    connection = connect(timeout=3)
    try:
        channel = connection.channel()
        declare_topology(channel)
        for queue in (RUNS_SUBMITTED, RUNS_RETRY, RUNS_DEAD):
            channel.queue_purge(queue=queue)
    finally:
        connection.close()


def main() -> int:
    from traintrack.broker.publisher import queue_stats
    from traintrack.broker.topology import RUNS_DEAD
    from traintrack.db import get_session, init_db
    from traintrack.models.experiment import Experiment
    from traintrack.models.run import StatusTransition
    from traintrack.models.snapshot import DatasetSnapshot

    init_db()
    wait_for_worker()
    purge()

    session = get_session()
    try:
        experiment = Experiment(
            name=f"reliability-{uuid.uuid4().hex[:8]}",
            description="Deliberate retry/DLQ measurement",
        )
        session.add(experiment)
        session.flush()
        snapshot = DatasetSnapshot(
            version=f"reliability-{uuid.uuid4().hex[:8]}",
            content_hash=uuid.uuid4().hex * 2,
            row_count=100,
            manifest={"seed": 1, "purpose": "retry measurement"},
            validated=True,
            path="/nonexistent/traintrack/reliability-snapshot",
        )
        session.add(snapshot)
        session.commit()
        experiment_id = str(experiment.id)
        snapshot_id = str(snapshot.id)
    finally:
        session.close()

    worker = subprocess.Popen(
        [sys.executable, "-m", "traintrack.worker"],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(3)
        submitted = api(
            "POST",
            "/api/runs",
            {
                "experiment_id": experiment_id,
                "job_kind": "train",
                "hyperparams": {"lr": 0.001, "batch_size": 8, "epochs": 1},
                "dataset_snapshot_id": snapshot_id,
                "max_attempts": 2,
                "timeout_seconds": 60,
            },
            idem=f"reliability-{uuid.uuid4().hex}",
        )
        run_id = submitted["id"]
        started = time.monotonic()
        detail: dict[str, Any] = {}
        while time.monotonic() - started < 180:
            detail = api("GET", f"/api/runs/{run_id}")
            if detail["status"] in ("SUCCEEDED", "FAILED", "CANCELED"):
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("failing run did not terminate")
        time.sleep(2)
        stats = queue_stats()

        session = get_session()
        try:
            transitions = list(
                session.execute(
                    select(StatusTransition)
                    .where(StatusTransition.run_id == uuid.UUID(run_id))
                    .order_by(StatusTransition.id)
                ).scalars()
            )
            timeline = [
                {
                    "from": transition.from_status,
                    "to": transition.to_status,
                    "reason": transition.reason,
                    "at": transition.created_at.isoformat(),
                }
                for transition in transitions
            ]
        finally:
            session.close()

        retries = [
            entry for entry in timeline if entry["from"] == "FAILED" and entry["to"] == "QUEUED"
        ]
        observed_retry_delay = None
        if retries:
            retry_queued_at = datetime.fromisoformat(retries[0]["at"])
            retry_index = next(
                index
                for index, entry in enumerate(timeline)
                if entry["from"] == "FAILED" and entry["to"] == "QUEUED"
            )
            later_scheduled = next(
                (
                    entry
                    for entry in timeline[retry_index + 1 :]
                    if entry["to"] == "SCHEDULED"
                ),
                None,
            )
            if later_scheduled:
                scheduled_at = datetime.fromisoformat(later_scheduled["at"])
                observed_retry_delay = round(
                    (scheduled_at - retry_queued_at).total_seconds(), 3
                )

        payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "generated_by": "scripts/export_reliability.py",
            "run_id": run_id,
            "experiment_id": experiment_id,
            "executor": detail.get("executor"),
            "status": detail["status"],
            "attempts_configured": 2,
            "attempts_observed": detail["attempt_count"],
            "failure_reason": detail.get("failure_reason"),
            "dlq_messages": int(stats.get(RUNS_DEAD, 0)),
            "retry_transitions": len(retries),
            "observed_retry_delay_seconds": observed_retry_delay,
            "timeline": timeline,
            "total_seconds": round(time.monotonic() - started, 3),
        }
    finally:
        worker.terminate()
        try:
            worker.wait(timeout=15)
        except subprocess.TimeoutExpired:
            worker.kill()
        purge()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
