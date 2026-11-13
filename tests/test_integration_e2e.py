"""End-to-end integration and failure-path tests with real services."""

from __future__ import annotations

import threading
import time
import uuid

import pytest
from sqlalchemy import select
from traintrack.db import get_session
from traintrack.executors.base import BaseExecutor, ExecutionResult, JobContext
from traintrack.models.run import Run, StatusTransition
from traintrack.models.state import RunStatus
from traintrack.services.cache import get_redis
from traintrack.services.logs import append_log
from traintrack.worker import Worker

from snapshot_factory import register_synthetic_snapshot, write_synthetic_snapshot

pytestmark = pytest.mark.integration


def _wait_for(predicate, timeout: float = 120.0, interval: float = 0.25) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("condition not met before timeout")


def _status(run_id: str) -> RunStatus:
    session = get_session()
    try:
        return session.execute(select(Run.status).where(Run.id == uuid.UUID(run_id))).scalar_one()
    finally:
        session.close()


def test_real_training_metrics_artifacts_and_lineage(
    app, client, auth_headers, experiment, live_api_url, tmp_path
) -> None:
    directory, manifest = write_synthetic_snapshot(tmp_path / "snapshot")
    snapshot_id = register_synthetic_snapshot(directory, manifest)

    worker = Worker(worker_id="e2e-training-worker", prefetch=1)
    thread = worker.run_in_thread()
    try:
        response = client.post(
            "/api/runs",
            json={
                "experiment_id": experiment["id"],
                "job_kind": "train",
                "hyperparams": {
                    "lr": 0.01,
                    "batch_size": 128,
                    "epochs": 1,
                    "optimizer": "sgd",
                    "momentum": 0.9,
                    "seed": 7,
                },
                "dataset_snapshot_id": str(snapshot_id),
                "timeout_seconds": 300,
            },
            headers={**auth_headers, "Idempotency-Key": f"e2e-{uuid.uuid4().hex}"},
        )
        assert response.status_code == 201, response.get_json()
        run_id = response.get_json()["id"]

        _wait_for(
            lambda: _status(run_id) in (RunStatus.SUCCEEDED, RunStatus.FAILED),
            timeout=180,
        )
        assert _status(run_id) is RunStatus.SUCCEEDED

        metrics = client.get(
            f"/api/runs/{run_id}/metrics?names=val_accuracy,test_accuracy,train_loss_epoch",
            headers=auth_headers,
        ).get_json()["points"]
        names = {point["name"] for point in metrics}
        assert {"val_accuracy", "test_accuracy", "train_loss_epoch"} <= names

        artifacts = client.get(
            f"/api/artifacts?run_id={run_id}", headers=auth_headers
        ).get_json()["artifacts"]
        kinds = {artifact["kind"] for artifact in artifacts}
        assert {"checkpoint", "eval_report", "train_summary"} <= kinds
        checkpoint = next(artifact for artifact in artifacts if artifact["kind"] == "checkpoint")

        lineage = client.get(
            f"/api/artifacts/{checkpoint['sha256']}/lineage", headers=auth_headers
        ).get_json()
        assert lineage["run"]["id"] == run_id
        assert lineage["dataset_snapshot"]["id"] == str(snapshot_id)
        assert lineage["dataset_snapshot"]["content_hash"] == manifest["content_hash"]
        assert lineage["run"]["hyperparams"]["optimizer"] == "sgd"

        logs = client.get(f"/api/runs/{run_id}/logs?tail=100", headers=auth_headers)
        lines = "\n".join(entry["line"] for entry in logs.get_json()["logs"])
        assert "TRAIN_SUMMARY" in lines
        assert "snapshot" in lines

        resolved = client.get(f"/api/runs/{run_id}", headers=auth_headers).get_json()
        assert resolved["executor"] == "local"
        assert resolved["attempt_count"] == 1
    finally:
        worker.request_stop()
        thread.join(timeout=10)


class BlockingExecutor(BaseExecutor):
    """Waits for cancellation or the timeout, so a running run can be canceled."""

    name = "blocking"

    def __init__(self) -> None:
        self.started = threading.Event()
        self.closed = False

    def run(self, job: JobContext) -> ExecutionResult:
        self.started.set()
        append_log(job.run_id, "[blocking] started")
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if get_redis().get(f"cancel:{job.run_id}"):
                self.closed = True
                append_log(job.run_id, "[blocking] observed cancel")
                return ExecutionResult(exit_code=143, canceled=True, reason="canceled")
            time.sleep(0.05)
        return ExecutionResult(exit_code=0)


def test_cancel_of_running_job(
    app, client, auth_headers, experiment
) -> None:
    executor = BlockingExecutor()
    worker = Worker(worker_id="cancel-worker", executor=executor, prefetch=1)
    thread = worker.run_in_thread()
    try:
        response = client.post(
            "/api/runs",
            json={
                "experiment_id": experiment["id"],
                "job_kind": "noop",
                "hyperparams": {},
                "timeout_seconds": 120,
            },
            headers={**auth_headers, "Idempotency-Key": f"cancel-{uuid.uuid4().hex}"},
        )
        assert response.status_code == 201
        run_id = response.get_json()["id"]

        assert executor.started.wait(timeout=30), "executor did not start"
        canceled = client.post(f"/api/runs/{run_id}/cancel", headers=auth_headers)
        assert canceled.status_code == 200
        assert canceled.get_json()["status"] == "CANCELED"

        _wait_for(lambda: executor.closed, timeout=20)
        _wait_for(lambda: _status(run_id) is RunStatus.CANCELED, timeout=20)

        session = get_session()
        try:
            transitions = list(
                session.execute(
                    select(StatusTransition).where(
                        StatusTransition.run_id == uuid.UUID(run_id)
                    )
                ).scalars()
            )
            assert any(
                transition.from_status == "RUNNING"
                and transition.to_status == "CANCELED"
                for transition in transitions
            )
        finally:
            session.close()
    finally:
        worker.request_stop()
        thread.join(timeout=10)


def test_worker_reconnects_after_broker_disconnect(
    app, client, auth_headers, experiment
) -> None:
    worker = Worker(worker_id="reconnect-worker", prefetch=1)
    thread = worker.run_in_thread()
    try:
        first = client.post(
            "/api/runs",
            json={"experiment_id": experiment["id"], "job_kind": "noop", "hyperparams": {}},
            headers={**auth_headers, "Idempotency-Key": f"reconnect-1-{uuid.uuid4().hex}"},
        ).get_json()
        _wait_for(lambda: _status(first["id"]) is RunStatus.SUCCEEDED, timeout=30)

        connection = worker._connection
        assert connection is not None
        connection.add_callback_threadsafe(connection.close)
        time.sleep(4)

        second = client.post(
            "/api/runs",
            json={"experiment_id": experiment["id"], "job_kind": "noop", "hyperparams": {}},
            headers={**auth_headers, "Idempotency-Key": f"reconnect-2-{uuid.uuid4().hex}"},
        ).get_json()
        _wait_for(
            lambda: _status(second["id"]) in (RunStatus.SUCCEEDED, RunStatus.FAILED),
            timeout=60,
        )
        assert _status(second["id"]) is RunStatus.SUCCEEDED
    finally:
        worker.request_stop()
        thread.join(timeout=10)
