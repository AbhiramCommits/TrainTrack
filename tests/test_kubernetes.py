"""Kubernetes manifest golden tests, executor unit tests, and metrics tests."""

from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from traintrack.db import get_session
from traintrack.executors.base import JobContext
from traintrack.executors.kubernetes import (
    K8sRenderOptions,
    KubernetesExecutor,
    render_job_manifest,
)
from traintrack.models.experiment import Experiment
from traintrack.models.run import Run
from traintrack.models.state import RunStatus
from traintrack.services.cache import get_redis
from traintrack.services.logs import read_logs
from traintrack.services.resources import append_resources
from traintrack.services.runs import transition_run
from traintrack.training.device import select_device

GOLDEN = Path(__file__).parent / "golden"
RUN_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")
EXPERIMENT_ID = uuid.UUID("22222222-2222-4222-8222-222222222222")

OPTIONS = K8sRenderOptions(
    namespace="traintrack",
    image="traintrack:dev",
    cpu_request="1",
    cpu_limit="2",
    memory_request="1Gi",
    memory_limit="2Gi",
    env={
        "DATABASE_URL": "postgresql+psycopg://traintrack:traintrack@postgres:5432/traintrack",
        "TRAINTRACK_API_URL": "http://traintrack-api:8010",
        "TRAINTRACK_API_KEY": "test-key",
    },
)


def _job(gpus: int = 0) -> JobContext:
    return JobContext(
        run_id=RUN_ID,
        experiment_id=EXPERIMENT_ID,
        job_kind="train",
        hyperparams={"lr": 0.001},
        dataset_snapshot_path="/data/snapshots/mnist-v1",
        dataset_snapshot_id=None,
        requested_gpus=gpus,
        image="traintrack:dev",
        timeout_seconds=1800,
        git_sha="abc123",
    )


@pytest.mark.parametrize(("gpus", "golden_file"), [(0, "job_cpu.yaml"), (2, "job_gpu.yaml")])
def test_rendered_manifest_matches_golden(gpus: int, golden_file: str) -> None:
    rendered = render_job_manifest(_job(gpus), OPTIONS)
    expected = yaml.safe_load((GOLDEN / golden_file).read_text())
    assert rendered == expected


def test_gpu_manifest_requests_nvidia_gpu_and_selects_gpu_nodes() -> None:
    manifest = render_job_manifest(_job(2), OPTIONS)
    pod_spec = manifest["spec"]["template"]["spec"]
    container = pod_spec["containers"][0]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 2
    assert pod_spec["nodeSelector"] == {"runpod.io/gpu": "true"}
    assert pod_spec["tolerations"][0]["key"] == "nvidia.com/gpu"
    assert manifest["spec"]["backoffLimit"] == 0
    assert manifest["spec"]["ttlSecondsAfterFinished"] == 600
    labels = manifest["metadata"]["labels"]
    assert labels["traintrack.run-id"] == str(RUN_ID)
    assert labels["traintrack.experiment-id"] == str(EXPERIMENT_ID)


def test_cpu_manifest_has_no_gpu_request() -> None:
    manifest = render_job_manifest(_job(0), OPTIONS)
    pod_spec = manifest["spec"]["template"]["spec"]
    container = pod_spec["containers"][0]
    assert "nvidia.com/gpu" not in container["resources"]["limits"]
    assert "nodeSelector" not in pod_spec
    assert "tolerations" not in pod_spec


def _seed_running_run() -> uuid.UUID:
    session = get_session()
    try:
        experiment = Experiment(name=f"k8s-{uuid.uuid4().hex[:8]}")
        session.add(experiment)
        session.flush()
        run = Run(
            experiment_id=experiment.id,
            status=RunStatus.RUNNING,
            job_kind="train",
            hyperparams={"lr": 0.001, "batch_size": 32},
            requested_gpus=0,
            timeout_seconds=60,
        )
        session.add(run)
        session.commit()
        return run.id
    finally:
        session.close()


class _FakeBatch:
    def __init__(self, status_sequence: list[tuple[int, int]]) -> None:
        self.status_sequence = status_sequence
        self.created: dict | None = None
        self.deleted: list[str] = []
        self.poll = 0

    def create_namespaced_job(self, namespace, body):
        self.created = body

    def read_namespaced_job_status(self, name, namespace):
        succeeded, failed = self.status_sequence[min(self.poll, len(self.status_sequence) - 1)]
        self.poll += 1
        return SimpleNamespace(
            status=SimpleNamespace(succeeded=succeeded, failed=failed)
        )

    def delete_namespaced_job(self, name, namespace, body):
        self.deleted.append(name)


class _FakeCore:
    def __init__(self, pods: list, logs: str = "") -> None:
        self.pods = pods
        self.logs = logs
        self.index = 0

    def list_namespaced_pod(self, namespace, label_selector):
        pod = self.pods[min(self.index, len(self.pods) - 1)]
        self.index += 1
        return SimpleNamespace(items=[pod])

    def read_namespaced_pod_log(self, name, namespace, container="job"):
        return self.logs


class _ConflictError(Exception):
    status = 409


class _ConflictBatch(_FakeBatch):
    """Fails the first create with 409, as a stale Job from a retry would."""

    def __init__(self, status_sequence: list[tuple[int, int]]) -> None:
        super().__init__(status_sequence)
        self.create_attempts = 0

    def create_namespaced_job(self, namespace, body):
        self.create_attempts += 1
        if self.create_attempts == 1:
            raise _ConflictError("job already exists")
        super().create_namespaced_job(namespace, body)


def _pod(name: str, phase: str, reason: str | None = None) -> SimpleNamespace:
    terminated = SimpleNamespace(reason=reason, message="boom") if reason else None
    statuses = (
        [SimpleNamespace(state=SimpleNamespace(terminated=terminated))]
        if reason
        else []
    )
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name),
        status=SimpleNamespace(phase=phase, container_statuses=statuses),
    )


def test_executor_success_streams_logs_and_records_pod_latency() -> None:
    run_id = _seed_running_run()
    job = JobContext(
        run_id=run_id,
        experiment_id=uuid.uuid4(),
        job_kind="train",
        hyperparams={"lr": 0.001},
        dataset_snapshot_path="/data/snapshots/mnist-v1",
        dataset_snapshot_id=None,
        requested_gpus=0,
        image="traintrack:dev",
        timeout_seconds=60,
    )
    batch = _FakeBatch([(0, 0), (1, 0)])
    core = _FakeCore([_pod("pod-a", "Running"), _pod("pod-a", "Succeeded")], "epoch 1\nTRAIN done\n")
    executor = KubernetesExecutor(
        batch, core, namespace="traintrack", image="traintrack:dev", poll_interval=0.01, options=OPTIONS
    )
    result = executor.run(job)
    assert result.exit_code == 0
    assert batch.created is not None
    assert batch.created["metadata"]["name"] == f"traintrack-run-{run_id}"
    lines = [entry["line"] for entry in read_logs(run_id, tail=50)]
    assert any("epoch 1" in line for line in lines)
    assert any("pod pod-a is RUNNING" in line for line in lines)

    session = get_session()
    try:
        run = session.get(Run, run_id)
        assert run is not None
        assert run.pod_name == "pod-a"
        assert run.scheduling_latency_seconds is not None
    finally:
        session.close()


def test_executor_maps_job_failure_to_execution_result() -> None:
    run_id = _seed_running_run()
    job = JobContext(
        run_id=run_id,
        experiment_id=uuid.uuid4(),
        job_kind="train",
        hyperparams={},
        dataset_snapshot_path="/data/snapshots/mnist-v1",
        dataset_snapshot_id=None,
        requested_gpus=0,
        image="traintrack:dev",
        timeout_seconds=60,
    )
    batch = _FakeBatch([(0, 1)])
    core = _FakeCore([_pod("pod-b", "Failed", reason="OOMKilled")], "ERROR out of memory\n")
    executor = KubernetesExecutor(
        batch, core, namespace="traintrack", image="traintrack:dev", poll_interval=0.01, options=OPTIONS
    )
    result = executor.run(job)
    assert result.exit_code == 1
    assert result.reason is not None and "OOMKilled" in result.reason


def test_executor_deletes_job_on_cancel() -> None:
    run_id = _seed_running_run()
    get_redis().set(f"cancel:{run_id}", "1", ex=60)
    job = JobContext(
        run_id=run_id,
        experiment_id=uuid.uuid4(),
        job_kind="train",
        hyperparams={},
        dataset_snapshot_path="/data/snapshots/mnist-v1",
        dataset_snapshot_id=None,
        requested_gpus=0,
        image="traintrack:dev",
        timeout_seconds=60,
    )
    batch = _FakeBatch([(0, 0)])
    core = _FakeCore([_pod("pod-c", "Running")])
    executor = KubernetesExecutor(
        batch, core, namespace="traintrack", image="traintrack:dev", poll_interval=0.01, options=OPTIONS
    )
    result = executor.run(job)
    assert result.canceled is True
    assert batch.deleted == [f"traintrack-run-{run_id}"]


def test_executor_replaces_stale_job_after_conflict() -> None:
    run_id = _seed_running_run()
    job = JobContext(
        run_id=run_id,
        experiment_id=uuid.uuid4(),
        job_kind="train",
        hyperparams={},
        dataset_snapshot_path="/data/snapshots/mnist-v1",
        dataset_snapshot_id=None,
        requested_gpus=0,
        image="traintrack:dev",
        timeout_seconds=60,
    )
    batch = _ConflictBatch([(1, 0)])
    core = _FakeCore([_pod("pod-d", "Succeeded")], "done\n")
    executor = KubernetesExecutor(
        batch, core, namespace="traintrack", image="traintrack:dev", poll_interval=0.01, options=OPTIONS
    )
    executor._delete_job = lambda name: batch.deleted.append(name)  # type: ignore[method-assign]
    result = executor.run(job)
    assert result.exit_code == 0
    assert batch.create_attempts == 2
    assert batch.deleted == [f"traintrack-run-{run_id}"]


def test_metrics_endpoint_exposes_run_and_resource_metrics(client, app) -> None:
    session = get_session()
    try:
        experiment = Experiment(name=f"metrics-{uuid.uuid4().hex[:8]}")
        session.add(experiment)
        session.flush()
        run = Run(
            experiment_id=experiment.id,
            status=RunStatus.QUEUED,
            job_kind="train",
            hyperparams={"lr": 0.01, "batch_size": 32},
        )
        session.add(run)
        session.commit()
        run_id = run.id
        transition_run(session, run, RunStatus.SCHEDULED, reason="test")
        transition_run(session, run, RunStatus.RUNNING, reason="test")
        transition_run(session, run, RunStatus.SUCCEEDED, reason="test")
        session.commit()
    finally:
        session.close()

    append_resources(
        run_id,
        [
            {
                "step": 1,
                "device": select_device("cpu").backend,
                "utilization_pct": 42.5,
                "memory_used_mb": 128.0,
                "cpu_pct": 42.5,
                "rss_mb": 128.0,
                "wall_time": 0.0,
            }
        ],
    )
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    for name in (
        "traintrack_runs_total",
        "traintrack_run_duration_seconds_bucket",
        "traintrack_queue_depth",
        "traintrack_queue_wait_seconds_bucket",
        "traintrack_retries_total",
        "traintrack_dlq_messages",
        "traintrack_scheduling_latency_seconds_bucket",
        "traintrack_run_device_utilization_percent",
        "traintrack_run_memory_used_mb",
    ):
        assert name in body
    assert 'traintrack_runs_total{status="SUCCEEDED"} 1.0' in body
    assert str(run_id) in body
