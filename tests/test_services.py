"""Focused tests for services, local executor, and API error paths."""

from __future__ import annotations

import sys
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pika
from traintrack.artifacts.store import ArtifactStore
from traintrack.config import reset_settings_cache
from traintrack.db import get_session
from traintrack.executors.base import JobContext
from traintrack.executors.local import LocalExecutor
from traintrack.models.experiment import Experiment
from traintrack.models.run import Run
from traintrack.models.state import RunStatus
from traintrack.services.cache import get_redis
from traintrack.services.locks import (
    acquire_execution_lock,
    execution_lock_key,
    lock_holder,
    refresh_execution_lock,
    release_execution_lock,
)
from traintrack.services.logs import read_logs
from traintrack.services.resources import append_resources, read_resources
from traintrack.worker import Worker, reconcile_queued_runs

from snapshot_factory import register_synthetic_snapshot, write_synthetic_snapshot


def _job(run_id: uuid.UUID | None = None, **overrides) -> JobContext:
    values = {
        "run_id": run_id or uuid.uuid4(),
        "experiment_id": uuid.uuid4(),
        "job_kind": "train",
        "hyperparams": {},
        "dataset_snapshot_path": None,
        "dataset_snapshot_id": None,
        "requested_gpus": 0,
        "image": None,
        "timeout_seconds": 10,
    }
    values.update(overrides)
    return JobContext(**values)


def test_resource_stream_roundtrip_and_cap() -> None:
    run_id = uuid.uuid4()
    sample = {
        "step": 1,
        "device": "cpu",
        "utilization_pct": 42.5,
        "memory_used_mb": 100.0,
        "cpu_pct": 42.5,
        "rss_mb": 100.0,
        "wall_time": time.time(),
    }
    append_resources(run_id, [sample, {**sample, "step": 2}])
    samples = read_resources(run_id)
    assert [entry["step"] for entry in samples] == [1, 2]
    assert samples[0]["device"] == "cpu"
    assert read_resources(uuid.uuid4()) == []


def test_execution_locks_are_owner_checked() -> None:
    run_id = uuid.uuid4()
    assert acquire_execution_lock(run_id, "worker-a", ttl_seconds=5) is True
    assert acquire_execution_lock(run_id, "worker-b", ttl_seconds=5) is False
    assert lock_holder(run_id) == "worker-a"
    assert refresh_execution_lock(run_id, "worker-b", ttl_seconds=5) is False
    assert refresh_execution_lock(run_id, "worker-a", ttl_seconds=5) is True
    release_execution_lock(run_id, "worker-b")
    assert get_redis().get(execution_lock_key(run_id)) == "worker-a"
    release_execution_lock(run_id, "worker-a")
    assert lock_holder(run_id) is None


def test_local_executor_failure_streams_logs(monkeypatch) -> None:
    executor = LocalExecutor()
    job = _job()
    monkeypatch.setattr(
        executor,
        "_command",
        lambda _job: [sys.executable, "-c", "print('hello'); raise SystemExit(3)"],
    )
    result = executor.run(job)
    assert result.exit_code == 3
    assert "process exited with 3" in (result.reason or "")
    lines = [entry["line"] for entry in read_logs(job.run_id, tail=20)]
    assert any("hello" in line for line in lines)


def test_local_executor_missing_binary_fails_cleanly(monkeypatch) -> None:
    executor = LocalExecutor()
    job = _job()
    monkeypatch.setattr(executor, "_command", lambda _job: ["/definitely/not/a/binary"])
    result = executor.run(job)
    assert result.exit_code == 127
    assert "failed to start" in (result.reason or "")


def test_local_executor_enforces_timeout(monkeypatch) -> None:
    executor = LocalExecutor()
    job = _job(timeout_seconds=1)
    monkeypatch.setattr(
        executor,
        "_command",
        lambda _job: [sys.executable, "-c", "import time; time.sleep(20)"],
    )
    started = time.monotonic()
    result = executor.run(job)
    assert time.monotonic() - started < 8
    assert "timeout" in (result.reason or "")


def test_local_executor_observes_cancel(monkeypatch) -> None:
    executor = LocalExecutor()
    job = _job(timeout_seconds=30)
    monkeypatch.setattr(
        executor,
        "_command",
        lambda _job: [sys.executable, "-c", "import time; time.sleep(20)"],
    )
    results = []
    thread = threading.Thread(target=lambda: results.append(executor.run(job)))
    thread.start()
    time.sleep(1)
    get_redis().set(f"cancel:{job.run_id}", "1", ex=30)
    thread.join(timeout=10)
    assert results and results[0].canceled is True
    assert "canceled" in (results[0].reason or "")


def test_local_executor_noop() -> None:
    executor = LocalExecutor()
    job = _job(job_kind="noop")
    result = executor.run(job)
    assert result.exit_code == 0
    lines = [entry["line"] for entry in read_logs(job.run_id, tail=10)]
    assert any("[noop]" in line for line in lines)


def test_worker_backoff_grows_and_caps(monkeypatch) -> None:
    monkeypatch.setenv("TRAINTRACK_RETRY_BASE_SECONDS", "2")
    monkeypatch.setenv("TRAINTRACK_RETRY_MAX_SECONDS", "10")
    reset_settings_cache()
    worker = Worker(connect=False)
    assert worker._backoff(1) == 2
    assert worker._backoff(2) == 4
    assert worker._backoff(3) == 8
    assert worker._backoff(5) == 10
    reset_settings_cache()


def test_worker_rejects_malformed_message(monkeypatch) -> None:
    worker = Worker(connect=False)
    acks: list[int] = []
    dead: list[tuple] = []

    class FakeChannel:
        def basic_ack(self, delivery_tag):
            acks.append(delivery_tag)

    worker._pub_channel = FakeChannel()
    monkeypatch.setattr(
        "traintrack.worker.publish_dead",
        lambda channel, run_id, attempt, reason: dead.append((run_id, attempt, reason)),
    )
    worker._on_message(
        FakeChannel(),
        SimpleNamespace(delivery_tag=99),
        SimpleNamespace(headers={}),
        b"not-json",
    )
    assert acks == [99]
    assert dead and "malformed" in dead[0][2]


def test_worker_context_uses_snapshot_metadata(app) -> None:
    directory, manifest = write_synthetic_snapshot(Path("/tmp") / f"ctx-{uuid.uuid4().hex[:6]}")
    snapshot_id = register_synthetic_snapshot(directory, manifest)
    session = get_session()
    try:
        experiment = Experiment(name=f"ctx-{uuid.uuid4().hex[:8]}")
        session.add(experiment)
        session.flush()
        run = Run(
            experiment_id=experiment.id,
            status=RunStatus.RUNNING,
            job_kind="train",
            hyperparams={"lr": 0.01, "batch_size": 8},
            dataset_snapshot_id=snapshot_id,
        )
        session.add(run)
        session.commit()
        from traintrack.models.snapshot import DatasetSnapshot

        snapshot = session.get(DatasetSnapshot, snapshot_id)
        assert snapshot is not None
        worker = Worker(connect=False)
        context = worker._job_context(session, run, snapshot)
        assert context.dataset_snapshot_path == str(directory)
        assert context.dataset_snapshot_version == manifest["version"]
        assert context.dataset_snapshot_hash == manifest["content_hash"]
    finally:
        session.close()


def test_reconcile_republishes_queued_runs(app) -> None:
    session = get_session()
    try:
        experiment = Experiment(name=f"reconcile-{uuid.uuid4().hex[:8]}")
        session.add(experiment)
        session.flush()
        run = Run(
            experiment_id=experiment.id,
            status=RunStatus.QUEUED,
            job_kind="noop",
            hyperparams={},
        )
        session.add(run)
        session.commit()
    finally:
        session.close()
    assert reconcile_queued_runs() == 1
    assert reconcile_queued_runs(limit=0) == 0


def test_artifact_store_file_roundtrip(tmp_path) -> None:
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"traintrack-artifact" * 100)
    store = ArtifactStore(tmp_path / "store")
    session = get_session()
    try:
        first = store.write_file(source, kind="report", session=session)
        session.commit()
        second = store.write_file(source, kind="report", session=session)
        session.commit()
        assert first.sha256 == second.sha256
        assert second.ref_count == 2
        assert store.read_bytes(first.sha256) == source.read_bytes()
        assert store.exists(first.sha256)
        assert store.stats()["blob_count"] == 1
        assert not store.exists("0" * 64)
    finally:
        session.close()


def test_snapshot_detail_endpoints(client, auth_headers) -> None:
    directory, manifest = write_synthetic_snapshot(Path("/tmp") / f"api-{uuid.uuid4().hex[:6]}")
    snapshot_id = register_synthetic_snapshot(directory, manifest)
    response = client.get(f"/api/snapshots/{snapshot_id}", headers=auth_headers)
    assert response.status_code == 200
    assert response.get_json()["version"] == manifest["version"]
    assert client.get("/api/snapshots/nope", headers=auth_headers).status_code == 400
    assert (
        client.get(f"/api/snapshots/{uuid.uuid4()}", headers=auth_headers).status_code
        == 404
    )


def test_artifact_listing_and_lineage_missing(client, auth_headers) -> None:
    listing = client.get("/api/artifacts?kind=checkpoint", headers=auth_headers)
    assert listing.status_code == 200
    assert listing.get_json()["artifacts"] == []
    missing = client.get(f"/api/artifacts/{'a' * 64}/lineage", headers=auth_headers)
    assert missing.status_code == 404


def test_queue_health_when_broker_unavailable(client, auth_headers, monkeypatch) -> None:
    def broken():
        raise pika.exceptions.AMQPConnectionError("down")

    monkeypatch.setattr("traintrack.api.queue.queue_stats", broken)
    response = client.get("/api/queue/health", headers=auth_headers)
    assert response.status_code == 503
    assert response.get_json()["error"] == "broker_unavailable"


def test_metrics_survive_dlq_introspection_failure(client, monkeypatch) -> None:
    get_redis().delete("metrics:dlq")

    def broken():
        raise pika.exceptions.AMQPConnectionError("down")

    monkeypatch.setattr("traintrack.metrics.exporter.queue_stats", broken)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "traintrack_queue_depth" in response.get_data(as_text=True)


def test_inactive_api_key_is_rejected(app, client) -> None:
    from traintrack.services.auth import create_api_key, verify_api_key

    session = get_session()
    try:
        record, raw = create_api_key(session, "inactive")
        record.active = False
        session.commit()
        assert verify_api_key(session, raw) is None
    finally:
        session.close()
    assert client.get("/api/runs", headers={"X-API-Key": raw}).status_code == 401
