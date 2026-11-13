"""Integration tests for the RabbitMQ work queue and scheduler workers.

These exercise real RabbitMQ, Postgres, Redis and subprocess execution.
"""

from __future__ import annotations

import time
import uuid

import pytest
from sqlalchemy import func, select
from traintrack.broker.publisher import get_publisher, queue_stats
from traintrack.db import get_session
from traintrack.models.base import utcnow
from traintrack.models.run import Run, StatusTransition
from traintrack.models.snapshot import DatasetSnapshot
from traintrack.models.state import RunStatus
from traintrack.worker import Worker

pytestmark = pytest.mark.integration


def _submit_noop(client, auth_headers, experiment, key: str) -> dict:
    response = client.post(
        "/api/runs",
        json={"experiment_id": experiment["id"], "job_kind": "noop", "hyperparams": {}},
        headers={**auth_headers, "Idempotency-Key": key},
    )
    assert response.status_code == 201, response.get_json()
    return response.get_json()


def _wait_for(predicate, timeout: float = 60.0, interval: float = 0.25) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("condition not met before timeout")


def _statuses(run_ids: list[uuid.UUID]) -> dict[uuid.UUID, RunStatus]:
    session = get_session()
    try:
        rows = session.execute(
            select(Run.id, Run.status).where(Run.id.in_(run_ids))
        ).all()
        return dict(rows)
    finally:
        session.close()


def test_two_workers_process_twenty_runs_exactly_once(
    app, client, auth_headers, experiment
) -> None:
    workers = [
        Worker(worker_id=f"test-worker-{index}", prefetch=1) for index in range(2)
    ]
    threads = [worker.run_in_thread() for worker in workers]
    try:
        submitted = [
            _submit_noop(client, auth_headers, experiment, f"concurrent-{index}")
            for index in range(20)
        ]
        run_ids = [uuid.UUID(run["id"]) for run in submitted]

        def all_terminal() -> bool:
            states = _statuses(run_ids)
            return all(state in (RunStatus.SUCCEEDED, RunStatus.FAILED) for state in states.values())

        _wait_for(all_terminal, timeout=90)
        states = _statuses(run_ids)
        assert all(state is RunStatus.SUCCEEDED for state in states.values()), states

        session = get_session()
        try:
            successes = dict(
                session.execute(
                    select(StatusTransition.run_id, func.count(StatusTransition.id))
                    .where(
                        StatusTransition.run_id.in_(run_ids),
                        StatusTransition.to_status == RunStatus.SUCCEEDED.value,
                    )
                    .group_by(StatusTransition.run_id)
                ).all()
            )
            attempts = dict(
                session.execute(
                    select(Run.id, Run.attempt_count).where(Run.id.in_(run_ids))
                ).all()
            )
        finally:
            session.close()

        assert set(successes) == set(run_ids)
        assert all(count == 1 for count in successes.values())
        assert all(count == 1 for count in attempts.values())
    finally:
        for worker in workers:
            worker.request_stop()
        for thread in threads:
            thread.join(timeout=5)
        assert all(not thread.is_alive() for thread in threads)


def test_failing_job_retries_then_lands_in_dlq(
    app, client, auth_headers, experiment
) -> None:
    session = get_session()
    try:
        snapshot = DatasetSnapshot(
            version=f"broken-{uuid.uuid4().hex[:8]}",
            content_hash=uuid.uuid4().hex * 2,
            row_count=10,
            manifest={"seed": 1},
            validated=True,
            path="/nonexistent/traintrack/snapshot",
        )
        session.add(snapshot)
        session.commit()
        snapshot_id = str(snapshot.id)
    finally:
        session.close()

    submitted = client.post(
        "/api/runs",
        json={
            "experiment_id": experiment["id"],
            "job_kind": "train",
            "hyperparams": {"lr": 0.01, "batch_size": 8, "epochs": 1},
            "dataset_snapshot_id": snapshot_id,
            "timeout_seconds": 30,
            "max_attempts": 2,
        },
        headers={**auth_headers, "Idempotency-Key": f"broken-{uuid.uuid4().hex}"},
    )
    assert submitted.status_code == 201, submitted.get_json()
    run_id = uuid.UUID(submitted.get_json()["id"])

    worker = Worker(worker_id="test-dlq-worker", prefetch=1)
    thread = worker.run_in_thread()
    try:
        def failed() -> bool:
            session = get_session()
            try:
                status = session.execute(
                    select(Run.status).where(Run.id == run_id)
                ).scalar_one()
                return status is RunStatus.FAILED
            finally:
                session.close()

        _wait_for(failed, timeout=90, interval=0.5)
        time.sleep(1.0)  # allow the DLQ publish to settle
        stats = queue_stats()
        assert stats["runs.dead"] >= 1

        session = get_session()
        try:
            model = session.get(Run, run_id)
            assert model is not None
            assert model.attempt_count == 2
            failures = session.execute(
                select(func.count(StatusTransition.id)).where(
                    StatusTransition.run_id == run_id,
                    StatusTransition.to_status == RunStatus.FAILED.value,
                )
            ).scalar_one()
            assert failures == 2
            queued = session.execute(
                select(func.count(StatusTransition.id)).where(
                    StatusTransition.run_id == run_id,
                    StatusTransition.to_status == RunStatus.QUEUED.value,
                )
            ).scalar_one()
            assert queued == 2  # initial submission + one retry
        finally:
            session.close()
    finally:
        worker.request_stop()
        thread.join(timeout=5)


def test_sse_stream_yields_live_logs(
    app, client, auth_headers, experiment
) -> None:
    worker = Worker(worker_id="test-sse-worker", prefetch=1)
    thread = worker.run_in_thread()
    try:
        run = _submit_noop(client, auth_headers, experiment, "sse-run")
        run_id = uuid.UUID(run["id"])

        def succeeded() -> bool:
            return _statuses([run_id]).get(run_id) is RunStatus.SUCCEEDED

        _wait_for(succeeded, timeout=30)

        logs = client.get(f"/api/runs/{run_id}/logs?tail=50", headers=auth_headers)
        assert logs.status_code == 200
        lines = [entry["line"] for entry in logs.get_json()["logs"]]
        assert any("noop" in line for line in lines)
        assert any("SUCCEEDED" in line for line in lines)

        stream = client.get(
            f"/api/runs/{run_id}/logs/stream?replay=50", headers=auth_headers
        )
        assert stream.status_code == 200
        body = b"".join(stream.response).decode()
        assert "[noop]" in body
        assert "run SUCCEEDED" in body
    finally:
        worker.request_stop()
        thread.join(timeout=5)


def test_queue_health_reports_depth(
    client, auth_headers, experiment
) -> None:
    _submit_noop(client, auth_headers, experiment, "queue-health")
    response = client.get("/api/queue/health", headers=auth_headers)
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["depth"] >= 1
    assert body["dlq_messages"] >= 0


def test_worker_recovers_orphaned_running_run(
    app, client, auth_headers, experiment
) -> None:
    """A message redelivered for a RUNNING run with no lock is re-executed."""
    run = _submit_noop(client, auth_headers, experiment, "orphan-run")
    run_id = uuid.UUID(run["id"])

    session = get_session()
    try:
        model = session.get(Run, run_id)
        assert model is not None
        model.status = RunStatus.RUNNING
        model.attempt_count = 1
        model.worker_id = "dead-worker"
        model.started_at = utcnow()
        model.updated_at = utcnow()
        session.commit()
    finally:
        session.close()

    get_publisher().publish(run_id, 1)
    worker = Worker(worker_id="recovery-worker", prefetch=1)
    thread = worker.run_in_thread()
    try:
        def terminal() -> bool:
            state = _statuses([run_id]).get(run_id)
            return state in (RunStatus.SUCCEEDED, RunStatus.FAILED)

        _wait_for(terminal, timeout=60, interval=0.5)
        assert _statuses([run_id])[run_id] is RunStatus.SUCCEEDED
    finally:
        worker.request_stop()
        thread.join(timeout=5)
