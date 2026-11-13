"""Kubernetes batch/v1 Job executor."""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from traintrack.config import PROJECT_ROOT, settings
from traintrack.db import get_session
from traintrack.executors.base import BaseExecutor, ExecutionResult, JobContext
from traintrack.models.base import utcnow
from traintrack.models.run import Run
from traintrack.services.cache import get_redis
from traintrack.services.logs import append_log

logger = logging.getLogger(__name__)

TEMPLATE_DIR = PROJECT_ROOT / "deploy" / "k8s"
TEMPLATE_NAME = "job-template.yaml.j2"


@dataclass
class K8sRenderOptions:
    """Values that shape a rendered Job manifest."""

    namespace: str = "traintrack"
    image: str = "traintrack:dev"
    cpu_request: str = "1"
    cpu_limit: str = "4"
    memory_request: str = "1Gi"
    memory_limit: str = "6Gi"
    gpu_node_selector_key: str = "runpod.io/gpu"
    gpu_node_selector_value: str = "true"
    gpu_toleration_key: str = "nvidia.com/gpu"
    service_account: str = "traintrack-job"
    data_host_path: str = "/data"
    artifacts_host_path: str = "/artifacts"
    ttl_seconds: int = 600
    image_pull_policy: str = "IfNotPresent"
    env: dict[str, str] = field(default_factory=dict)


def _environment() -> dict[str, str]:
    """Environment needed by the in-cluster training/eval process."""
    return {
        "DATABASE_URL": os.environ.get(
            "TRAINTRACK_K8S_DATABASE_URL",
            "postgresql+psycopg://traintrack:traintrack@postgres:5432/traintrack",
        ),
        "REDIS_URL": os.environ.get(
            "TRAINTRACK_K8S_REDIS_URL", "redis://redis:6379/0"
        ),
        "TRAINTRACK_API_URL": os.environ.get(
            "TRAINTRACK_K8S_API_URL", "http://traintrack-api:8010"
        ),
        "TRAINTRACK_API_KEY": os.environ.get(
            "TRAINTRACK_K8S_API_KEY", settings().dev_api_key
        ),
        "TRAINTRACK_ARTIFACT_ROOT": "/artifacts",
        "TRAINTRACK_DATA_ROOT": "/data",
        "PYTHONUNBUFFERED": "1",
        "HOME": "/tmp",
    }


def render_job_manifest(
    job: JobContext,
    options: K8sRenderOptions | None = None,
) -> dict[str, Any]:
    """Render deploy/k8s/job-template.yaml.j2 into a Job manifest dict."""
    opts = options if options is not None else K8sRenderOptions(
        namespace=settings().k8s_namespace,
        image=job.image or settings().k8s_image,
        cpu_request=settings().k8s_cpu_request,
        cpu_limit=settings().k8s_cpu_limit,
        memory_request=settings().k8s_memory_request,
        memory_limit=settings().k8s_memory_limit,
        gpu_node_selector_key=settings().k8s_gpu_node_selector_key,
        env=_environment(),
    )
    if job.job_kind == "train":
        command = [
            "python",
            "-m",
            "traintrack.training.train",
            "--run-id",
            str(job.run_id),
        ]
    elif job.job_kind == "eval":
        command = [
            "python",
            "-m",
            "traintrack.training.eval",
            "--run-id",
            str(job.run_id),
        ]
    else:
        raise ValueError(f"KubernetesExecutor cannot run job kind {job.job_kind!r}")

    environment = dict(opts.env or _environment())
    environment.update(
        {
            "TRAINTRACK_RUN_ID": str(job.run_id),
            "TRAINTRACK_EXPERIMENT_ID": str(job.experiment_id),
            "TRAINTRACK_JOB_KIND": job.job_kind,
        }
    )
    context = {
        "job_name": f"traintrack-run-{job.run_id}",
        "namespace": opts.namespace,
        "image": opts.image,
        "command": command,
        "env": [
            {"name": name, "value": value} for name, value in sorted(environment.items())
        ],
        "cpu_request": opts.cpu_request,
        "cpu_limit": opts.cpu_limit,
        "memory_request": opts.memory_request,
        "memory_limit": opts.memory_limit,
        "gpus": job.requested_gpus,
        "gpu_node_selector_key": opts.gpu_node_selector_key,
        "gpu_node_selector_value": opts.gpu_node_selector_value,
        "gpu_toleration_key": opts.gpu_toleration_key,
        "run_id": str(job.run_id),
        "experiment_id": str(job.experiment_id),
        "ttl_seconds": opts.ttl_seconds,
        "active_deadline_seconds": job.timeout_seconds + 120,
        "service_account": opts.service_account,
        "data_host_path": opts.data_host_path,
        "artifacts_host_path": opts.artifacts_host_path,
        "image_pull_policy": opts.image_pull_policy,
    }
    environment_jinja = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        undefined=StrictUndefined,
        autoescape=False,
    )
    template = environment_jinja.get_template(TEMPLATE_NAME)
    rendered = template.render(**context)
    manifest = yaml.safe_load(rendered)
    if not isinstance(manifest, dict):
        raise ValueError("rendered job template did not produce a mapping")
    return manifest


class KubernetesExecutor(BaseExecutor):
    """Create one batch/v1 Job per run and follow it to completion."""

    name = "kubernetes"

    def __init__(
        self,
        batch_api: Any | None = None,
        core_api: Any | None = None,
        *,
        namespace: str | None = None,
        image: str | None = None,
        poll_interval: float = 1.0,
        options: K8sRenderOptions | None = None,
    ) -> None:
        self.namespace = namespace or settings().k8s_namespace
        self.image = image or settings().k8s_image
        self.poll_interval = poll_interval
        self.options = options
        if batch_api is None or core_api is None:
            from kubernetes import client, config

            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()
            self.batch_api = batch_api or client.BatchV1Api()
            self.core_api = core_api or client.CoreV1Api()
        else:
            self.batch_api = batch_api
            self.core_api = core_api

    def _record_running(self, job: JobContext, pod_name: str) -> None:
        session = get_session()
        try:
            run = session.get(Run, job.run_id)
            if run is None:
                return
            created = run.created_at
            if created is not None and created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            latency = (utcnow() - created).total_seconds() if created else None
            run.pod_name = pod_name
            if run.scheduling_latency_seconds is None:
                run.scheduling_latency_seconds = latency
            session.commit()
        finally:
            session.close()

    def _append_new_logs(
        self, run_id: uuid.UUID, pod_name: str, seen: int
    ) -> tuple[int, str]:
        try:
            raw = self.core_api.read_namespaced_pod_log(
                name=pod_name, namespace=self.namespace, container="job"
            )
        except Exception as exc:
            logger.debug("pod logs not ready for %s: %s", pod_name, exc)
            return seen, ""
        if not raw:
            return seen, ""
        lines = str(raw).splitlines()
        new_lines = lines[seen:]
        for line in new_lines:
            level = (
                "error"
                if any(token in line for token in ("ERROR", "Traceback", "FAILED"))
                else "info"
            )
            append_log(
                run_id,
                line[:4000],
                level=level,
            )
        return len(lines), "\n".join(new_lines)

    def _options_for(self, job: JobContext) -> K8sRenderOptions:
        if self.options is not None:
            return self.options
        config = settings()
        return K8sRenderOptions(
            namespace=self.namespace,
            image=job.image or self.image,
            cpu_request=config.k8s_cpu_request,
            cpu_limit=config.k8s_cpu_limit,
            memory_request=config.k8s_memory_request,
            memory_limit=config.k8s_memory_limit,
            gpu_node_selector_key=config.k8s_gpu_node_selector_key,
            env=_environment(),
        )

    def _pod_failure_reason(self, pod: Any) -> str:
        try:
            statuses = pod.status.container_statuses or []
            for status in statuses:
                terminated = getattr(status.state, "terminated", None)
                if terminated is not None:
                    return (
                        f"pod {pod.metadata.name} terminated: "
                        f"{terminated.reason or 'error'} {terminated.message or ''}".strip()
                    )
            return f"pod {pod.metadata.name} phase {pod.status.phase}"
        except Exception:  # pragma: no cover - defensive
            return "pod failed"

    def run(self, job: JobContext) -> ExecutionResult:
        """Create the Job, stream logs, and map its outcome to the run."""
        manifest = render_job_manifest(job, self._options_for(job))
        job_name = manifest["metadata"]["name"]
        append_log(job.run_id, f"[k8s] creating job {job_name}")
        if not self._create_job(job.run_id, job_name, manifest):
            return ExecutionResult(
                exit_code=1, reason=f"job creation failed for {job_name}"
            )

        deadline = time.monotonic() + job.timeout_seconds
        seen_lines = 0
        pod_name: str | None = None
        running_recorded = False
        last_phase: str | None = None
        try:
            while True:
                if get_redis().get(f"cancel:{job.run_id}"):
                    self._delete_job(job_name)
                    append_log(job.run_id, f"[k8s] job {job_name} deleted on cancel")
                    return ExecutionResult(
                        exit_code=143, canceled=True, reason="canceled by operator"
                    )
                if time.monotonic() > deadline:
                    self._delete_job(job_name)
                    append_log(
                        job.run_id,
                        f"[k8s] job {job_name} exceeded timeout",
                        level="error",
                    )
                    return ExecutionResult(
                        exit_code=124, reason=f"timeout after {job.timeout_seconds}s"
                    )

                pods = self._list_pods(job_name)
                if pods:
                    pod = pods[0]
                    pod_name = str(pod.metadata.name)
                    phase = pod.status.phase
                    if phase != last_phase:
                        append_log(job.run_id, f"[k8s] pod {pod_name} phase {phase}")
                        last_phase = phase
                    if phase == "Running" and not running_recorded:
                        running_recorded = True
                        self._record_running(job, pod_name)
                        append_log(job.run_id, f"[k8s] pod {pod_name} is RUNNING")
                    seen_lines, _ = self._append_new_logs(job.run_id, pod_name, seen_lines)

                status = self.batch_api.read_namespaced_job_status(
                    name=job_name, namespace=self.namespace
                )
                succeeded = int(getattr(status.status, "succeeded", 0) or 0)
                failed = int(getattr(status.status, "failed", 0) or 0)
                if succeeded > 0:
                    if pod_name:
                        seen_lines, _ = self._append_new_logs(
                            job.run_id, pod_name, seen_lines
                        )
                    append_log(job.run_id, f"[k8s] job {job_name} succeeded")
                    return ExecutionResult(exit_code=0)
                if failed > 0:
                    if pod_name:
                        seen_lines, _ = self._append_new_logs(
                            job.run_id, pod_name, seen_lines
                        )
                    reason = (
                        self._pod_failure_reason(pods[0])
                        if pods
                        else f"job {job_name} failed"
                    )
                    append_log(job.run_id, f"[k8s] {reason}", level="error")
                    return ExecutionResult(exit_code=1, reason=reason)
                time.sleep(self.poll_interval)
        finally:
            if pod_name:
                append_log(job.run_id, f"[k8s] stopped watching pod {pod_name}")

    def _create_job(
        self, run_id: uuid.UUID, job_name: str, manifest: dict[str, Any]
    ) -> bool:
        """Create the Job, clearing a stale Job left by a failed attempt."""
        for attempt in range(1, 4):
            try:
                self.batch_api.create_namespaced_job(
                    namespace=self.namespace, body=manifest
                )
                return True
            except Exception as exc:
                status = getattr(exc, "status", None)
                if status == 409 and attempt < 3:
                    append_log(
                        run_id,
                        f"[k8s] stale job {job_name} exists; deleting before retry",
                        level="error",
                    )
                    self._delete_job(job_name)
                    time.sleep(1)
                    continue
                append_log(
                    run_id,
                    f"[k8s] job creation failed: {exc}",
                    level="error",
                )
                return False
        return False

    def _list_pods(self, job_name: str) -> list[Any]:
        try:
            pods = self.core_api.list_namespaced_pod(
                namespace=self.namespace, label_selector=f"job-name={job_name}"
            )
            return list(pods.items)
        except Exception as exc:  # pragma: no cover - transient API errors
            logger.debug("pod listing failed: %s", exc)
            return []

    def _delete_job(self, job_name: str) -> None:
        try:
            from kubernetes.client import V1DeleteOptions

            self.batch_api.delete_namespaced_job(
                name=job_name,
                namespace=self.namespace,
                body=V1DeleteOptions(propagation_policy="Background"),
            )
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("could not delete job %s: %s", job_name, exc)

    def cancel(self, run_id: uuid.UUID) -> None:
        """Delete the Job for a canceled run."""
        self._delete_job(f"traintrack-run-{run_id}")
