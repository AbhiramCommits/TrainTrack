"""Local subprocess executor with live log streaming and timeouts."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from itertools import count
from pathlib import Path
from typing import Any

from traintrack.config import settings
from traintrack.executors.base import BaseExecutor, ExecutionResult, JobContext
from traintrack.services.cache import get_redis
from traintrack.services.logs import append_log

logger = logging.getLogger(__name__)


class LocalExecutor(BaseExecutor):
    """Run a job as a subprocess and stream stdout/stderr to Redis."""

    name = "local"

    def __init__(self, python_executable: str | None = None) -> None:
        self.python = python_executable or sys.executable
        self._counter = count(1)

    def _command(self, job: JobContext) -> list[str]:
        if job.job_kind == "train":
            return [self.python, "-m", "traintrack.training.train", "--run-id", str(job.run_id)]
        if job.job_kind == "eval":
            return [self.python, "-m", "traintrack.training.eval", "--run-id", str(job.run_id)]
        raise ValueError(f"LocalExecutor cannot run job kind {job.job_kind!r}")

    def _environment(self, job: JobContext) -> dict[str, str]:
        env = os.environ.copy()
        config = settings()
        env.update(
            {
                "DATABASE_URL": config.database_url,
                "REDIS_URL": config.redis_url,
                "TRAINTRACK_API_URL": config.api_url,
                "TRAINTRACK_API_KEY": config.dev_api_key,
                "TRAINTRACK_ARTIFACT_ROOT": str(config.artifact_root),
                "TRAINTRACK_DATA_ROOT": str(config.data_root),
                "PYTHONUNBUFFERED": "1",
            }
        )
        return env

    def run(self, job: JobContext) -> ExecutionResult:
        """Execute the job, enforcing the run timeout and cancel flag."""
        if job.job_kind == "noop":
            return self._run_noop(job)

        command = self._command(job)
        append_log(job.run_id, f"[executor] $ {' '.join(command)}")
        startup = time.monotonic()
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=self._environment(job),
                cwd=str(Path(__file__).resolve().parents[2]),
            )
        except OSError as exc:
            append_log(job.run_id, f"[executor] failed to start: {exc}", level="error")
            return ExecutionResult(exit_code=127, reason=f"failed to start job: {exc}")

        state: dict[str, Any] = {
            "canceled": False,
            "timed_out": False,
            "reason": None,
        }
        stop_monitor = threading.Event()

        def monitor() -> None:
            client = get_redis()
            while not stop_monitor.wait(0.5):
                if client.get(f"cancel:{job.run_id}"):
                    state["canceled"] = True
                    state["reason"] = "canceled by operator"
                    process.kill()
                    return
                if time.monotonic() - startup > job.timeout_seconds:
                    state["timed_out"] = True
                    state["reason"] = (
                        f"timeout after {job.timeout_seconds}s"
                    )
                    process.kill()
                    return

        monitor_thread = threading.Thread(
            target=monitor, name=f"tt-monitor-{job.run_id}", daemon=True
        )
        monitor_thread.start()
        try:
            assert process.stdout is not None
            for line in process.stdout:
                level = (
                    "error"
                    if any(token in line for token in ("ERROR", "Traceback", "FAILED"))
                    else "info"
                )
                append_log(job.run_id, line.rstrip("\n")[:4000], level=level)
            exit_code = process.wait()
        finally:
            stop_monitor.set()
            monitor_thread.join(timeout=2)

        if state["canceled"]:
            return ExecutionResult(exit_code=exit_code, canceled=True, reason=state["reason"])
        if state["timed_out"]:
            return ExecutionResult(exit_code=exit_code, reason=state["reason"])
        if exit_code != 0:
            return ExecutionResult(exit_code=exit_code, reason=f"process exited with {exit_code}")
        return ExecutionResult(exit_code=0)

    def _run_noop(self, job: JobContext) -> ExecutionResult:
        """Fast no-op used by throughput benchmarks and scheduler tests."""
        append_log(job.run_id, f"[noop] run {job.run_id} started")
        time.sleep(min(0.02, max(0.0, job.timeout_seconds)))
        if get_redis().get(f"cancel:{job.run_id}"):
            append_log(job.run_id, "[noop] canceled", level="error")
            return ExecutionResult(exit_code=143, canceled=True, reason="canceled")
        append_log(job.run_id, "[noop] run completed")
        return ExecutionResult(exit_code=0)
