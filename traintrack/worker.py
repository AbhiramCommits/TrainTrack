"""Scheduler worker: consume run messages, execute, retry and dead-letter."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import socket
import threading
import time
import uuid
from typing import Any

import pika
from sqlalchemy import select

from traintrack.broker.publisher import connect as broker_connect
from traintrack.broker.publisher import publish_dead, publish_run
from traintrack.broker.topology import (
    RUNS_SUBMITTED,
    declare_topology,
)
from traintrack.config import settings
from traintrack.db import get_session, init_db
from traintrack.executors.base import BaseExecutor, JobContext
from traintrack.executors.local import LocalExecutor
from traintrack.models.base import utcnow
from traintrack.models.run import Run
from traintrack.models.snapshot import DatasetSnapshot
from traintrack.models.state import TERMINAL_STATUSES, RunStatus
from traintrack.services.locks import (
    acquire_execution_lock,
    refresh_execution_lock,
    release_execution_lock,
)
from traintrack.services.logs import append_log
from traintrack.services.runs import claim_run, transition_run

logger = logging.getLogger(__name__)


def build_executor(name: str | None = None) -> BaseExecutor:
    """Select an executor from configuration."""
    executor_name = (name or settings().executor).lower()
    if executor_name == "local":
        return LocalExecutor()
    if executor_name == "kubernetes":
        from traintrack.executors.kubernetes import KubernetesExecutor

        return KubernetesExecutor()
    raise ValueError(f"Unknown executor: {executor_name}")


class Heartbeat:
    """Renew the execution lock and touch the run mutably while a job runs."""

    def __init__(self, run_id: uuid.UUID, worker_id: str, ttl: int = 30) -> None:
        self.run_id = run_id
        self.worker_id = worker_id
        self.ttl = ttl
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name=f"tt-heartbeat-{run_id}", daemon=True
        )

    def start(self) -> None:
        """Start the heartbeat thread."""
        self._thread.start()

    def stop(self) -> None:
        """Stop and join the heartbeat thread."""
        self._stop.set()
        self._thread.join(timeout=3)

    def _loop(self) -> None:
        while not self._stop.wait(max(1.0, self.ttl / 3)):
            try:
                refresh_execution_lock(self.run_id, self.worker_id, self.ttl)
                session = get_session()
                try:
                    run = session.get(Run, self.run_id)
                    if run is not None and run.status is RunStatus.RUNNING:
                        run.updated_at = utcnow()
                        session.commit()
                finally:
                    session.close()
            except Exception:  # pragma: no cover - heartbeat is best effort
                logger.warning("heartbeat failed for %s", self.run_id, exc_info=True)


class Worker:
    """A single scheduler consumer. Multiple instances are safe concurrently."""

    def __init__(
        self,
        worker_id: str | None = None,
        executor: BaseExecutor | None = None,
        *,
        publisher_channel: Any = None,
        prefetch: int | None = None,
        connect: bool = True,
    ) -> None:
        self.worker_id = worker_id or (
            f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        )
        self.executor = executor or build_executor()
        self.prefetch = prefetch or settings().worker_prefetch
        self.lock_ttl = int(os.environ.get("TRAINTRACK_EXEC_LOCK_TTL", "30"))
        self._pub_connection: pika.BlockingConnection | None = None
        self._pub_channel = publisher_channel
        self._connection: pika.BlockingConnection | None = None
        self._channel: Any = None
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self.processed = 0
        if connect:
            self._connect()

    # ---- connection management -------------------------------------------------
    def _connect(self, attempts: int = 60, delay: float = 2.0) -> None:
        """Connect to RabbitMQ, waiting for a broker that is still starting."""
        config = settings()
        parameters = pika.URLParameters(config.amqp_url)
        parameters.heartbeat = 30
        parameters.blocked_connection_timeout = 30
        parameters.socket_timeout = 30
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            if self._stopping.is_set():
                raise pika.exceptions.AMQPConnectionError("worker stopping")
            try:
                self._connection = pika.BlockingConnection(parameters)
                break
            except (pika.exceptions.AMQPError, OSError) as exc:
                last_error = exc
                logger.warning(
                    "RabbitMQ connect attempt %s/%s failed: %s", attempt, attempts, exc
                )
                time.sleep(delay)
        if self._connection is None or not self._connection.is_open:
            raise pika.exceptions.AMQPConnectionError(
                f"could not connect to RabbitMQ: {last_error}"
            )
        self._channel = self._connection.channel()
        declare_topology(self._channel)
        self._channel.basic_qos(prefetch_count=self.prefetch)

    def _publish_channel(self) -> Any:
        if self._pub_channel is None:
            parameters = pika.URLParameters(settings().amqp_url)
            parameters.heartbeat = 30
            self._pub_connection = pika.BlockingConnection(parameters)
            channel = self._pub_connection.channel()
            declare_topology(channel)
            channel.confirm_delivery()
            self._pub_channel = channel
        return self._pub_channel

    # ---- lifecycle -------------------------------------------------------------
    def start_consuming(self) -> None:
        """Block consuming messages until :meth:`request_stop` is called."""
        while not self._stopping.is_set():
            try:
                if self._connection is None or not self._connection.is_open:
                    self._connect()
                assert self._channel is not None
                self._channel.basic_consume(
                    queue=RUNS_SUBMITTED,
                    on_message_callback=self._on_message,
                    auto_ack=False,
                )
                logger.info("worker %s consuming %s", self.worker_id, RUNS_SUBMITTED)
                self._channel.start_consuming()
                if self._stopping.is_set():
                    break
                # A closed/lost connection can make start_consuming return
                # without raising; treat it as a disconnect and reconnect.
                logger.warning("consumer stopped unexpectedly; reconnecting")
                self._close()
                time.sleep(2)
            except (pika.exceptions.AMQPError, OSError) as exc:
                self._close()
                if self._stopping.is_set():
                    break
                logger.warning("RabbitMQ connection lost (%s); reconnecting", exc)
                time.sleep(2)
        self._close()

    def run_in_thread(self) -> threading.Thread:
        """Start the consumer on a background thread (used by tests)."""
        self._thread = threading.Thread(
            target=self.start_consuming, name=f"tt-worker-{self.worker_id}", daemon=True
        )
        self._thread.start()
        return self._thread

    def request_stop(self, timeout: float = 10.0) -> None:
        """Stop consuming and join the thread if running."""
        self._stopping.set()
        if self._connection is not None and self._connection.is_open:
            with contextlib.suppress(pika.exceptions.AMQPError, RuntimeError):
                self._connection.add_callback_threadsafe(self._stop_consuming)
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _stop_consuming(self) -> None:
        if self._channel is not None and self._channel.is_open:
            self._channel.stop_consuming()

    def _close(self) -> None:
        for connection in (self._pub_connection, self._connection):
            try:
                if connection is not None and connection.is_open:
                    connection.close()
            except pika.exceptions.AMQPError:  # pragma: no cover
                pass
        self._pub_connection = None
        self._pub_channel = None
        self._connection = None
        self._channel = None

    # ---- message handling ------------------------------------------------------
    def _on_message(self, channel: Any, method: Any, properties: Any, body: bytes) -> None:
        try:
            payload = json.loads(body)
            run_id = uuid.UUID(str(payload["run_id"]))
        except (ValueError, KeyError, TypeError):
            logger.error("malformed run message; sending to DLQ: %r", body[:500])
            try:
                publish_dead(
                    self._publish_channel(), "unknown", 0, f"malformed message: {body[:200]!r}"
                )
            finally:
                channel.basic_ack(method.delivery_tag)
            return

        headers = properties.headers or {}
        attempt = int(headers.get("attempt", payload.get("attempt", 1)))
        try:
            self._handle_run(channel, method, run_id, attempt)
        except Exception as exc:  # pragma: no cover - defensive requeue path
            logger.exception("worker %s failed handling run %s", self.worker_id, run_id)
            with contextlib.suppress(Exception):
                append_log(run_id, f"[worker] internal error: {exc}", level="error")
            with contextlib.suppress(pika.exceptions.AMQPError):
                channel.basic_nack(method.delivery_tag, requeue=True)
            time.sleep(0.5)

    def _handle_run(self, channel: Any, method: Any, run_id: uuid.UUID, attempt: int) -> None:
        session = get_session()
        run: Run | None = None
        lock_acquired = False
        heartbeat: Heartbeat | None = None
        try:
            run = session.get(Run, run_id)
            if run is None:
                logger.warning("message for unknown run %s; acking", run_id)
                channel.basic_ack(method.delivery_tag)
                return
            if run.status in TERMINAL_STATUSES:
                channel.basic_ack(method.delivery_tag)
                return

            if run.status is RunStatus.QUEUED:
                claimed = claim_run(session, run.id, self.worker_id)
                session.commit()
                if claimed is None:
                    channel.basic_ack(method.delivery_tag)
                    return
                run = session.get(Run, run_id)
                assert run is not None
            else:
                # Redelivery after a consumer crash: the previous worker is gone
                # once its execution lock expires.
                if not acquire_execution_lock(run.id, self.worker_id, self.lock_ttl):
                    logger.info(
                        "run %s is owned by %s; deferring redelivery",
                        run.id,
                        "another worker",
                    )
                    session.rollback()
                    self._defer(channel, method, run.id, attempt, delay=5.0)
                    return
                lock_acquired = True
                try:
                    transition_run(
                        session,
                        run,
                        RunStatus.FAILED,
                        reason="worker lost ownership (crash or connection loss)",
                        actor=self.worker_id,
                    )
                except Exception:
                    session.rollback()
                    release_execution_lock(run.id, self.worker_id)
                    channel.basic_ack(method.delivery_tag)
                    return
                session.commit()
                if not self._retry_or_dead(session, channel, method, run, attempt):
                    return
                # The delayed retry message now owns the next attempt; ack the
                # redelivered message instead of executing twice.
                return

            snapshot: DatasetSnapshot | None = None
            if run.job_kind in ("train", "eval") and run.dataset_snapshot_id:
                snapshot = session.get(DatasetSnapshot, run.dataset_snapshot_id)
            if run.job_kind == "train" and (snapshot is None or not snapshot.validated):
                reason = (
                    f"snapshot {run.dataset_snapshot_id} is missing or not validated"
                )
                append_log(run.id, f"[worker] refusing run: {reason}", level="error")
                transition_run(
                    session, run, RunStatus.FAILED, reason=reason, actor=self.worker_id
                )
                session.commit()
                publish_dead(self._publish_channel(), run.id, run.attempt_count, reason)
                channel.basic_ack(method.delivery_tag)
                return

            if not lock_acquired:
                if not acquire_execution_lock(run.id, self.worker_id, self.lock_ttl):
                    self._defer(channel, method, run.id, attempt, delay=5.0)
                    return
                lock_acquired = True

            run.executor = self.executor.name
            run.worker_id = self.worker_id
            run.failure_reason = None
            transition_run(
                session, run, RunStatus.RUNNING, reason=f"executor {self.executor.name}",
                actor=self.worker_id,
            )
            session.commit()
            run_id_value = run.id
            job = self._job_context(session, run, snapshot)
            if run.requested_gpus and self.executor.name == "local":
                append_log(
                    run_id_value,
                    f"[worker] run requested {run.requested_gpus} GPU(s); the local "
                    "executor has no NVIDIA device, so training selects MPS/CPU",
                    level="info",
                )
            append_log(
                run_id_value,
                f"[worker] {self.worker_id} executing {run.job_kind} attempt {run.attempt_count}",
            )

            heartbeat = Heartbeat(run_id_value, self.worker_id, ttl=self.lock_ttl)
            heartbeat.start()
            try:
                result = self.executor.run(job)
            finally:
                heartbeat.stop()
                heartbeat = None

            session.expire_all()
            run = session.get(Run, run_id_value)
            if run is None:  # pragma: no cover
                channel.basic_ack(method.delivery_tag)
                return
            if run.status is RunStatus.CANCELED or result.canceled:
                if run.status is not RunStatus.CANCELED:
                    transition_run(
                        session, run, RunStatus.CANCELED, reason="executor reported cancel",
                        actor=self.worker_id,
                    )
                    session.commit()
                append_log(run_id_value, "[worker] run canceled")
                channel.basic_ack(method.delivery_tag)
                return
            if result.exit_code == 0:
                transition_run(
                    session, run, RunStatus.SUCCEEDED, reason="job completed",
                    actor=self.worker_id,
                )
                session.commit()
                append_log(run_id_value, "[worker] run SUCCEEDED")
                self.processed += 1
                channel.basic_ack(method.delivery_tag)
                return

            reason = result.reason or f"exit code {result.exit_code}"
            transition_run(
                session, run, RunStatus.FAILED, reason=reason, actor=self.worker_id
            )
            append_log(run_id_value, f"[worker] run FAILED: {reason}", level="error")
            self._retry_or_dead(session, channel, method, run, attempt, reason=reason)
        finally:
            if heartbeat is not None:
                heartbeat.stop()
            if lock_acquired and run_id is not None:
                release_execution_lock(run_id, self.worker_id)
            session.close()

    def _job_context(
        self, session: Any, run: Run, snapshot: DatasetSnapshot | None
    ) -> JobContext:
        return JobContext(
            run_id=run.id,
            experiment_id=run.experiment_id,
            job_kind=run.job_kind,
            hyperparams=dict(run.hyperparams or {}),
            dataset_snapshot_path=snapshot.path if snapshot else None,
            dataset_snapshot_id=run.dataset_snapshot_id,
            dataset_snapshot_version=snapshot.version if snapshot else None,
            dataset_snapshot_hash=snapshot.content_hash if snapshot else None,
            requested_gpus=run.requested_gpus,
            image=run.image,
            timeout_seconds=run.timeout_seconds,
            git_sha=run.git_sha,
        )

    def _backoff(self, attempt: int) -> float:
        base = settings().retry_base_seconds
        cap = settings().retry_max_seconds
        return min(cap, base * (2 ** max(0, attempt - 1)))

    def _retry_or_dead(
        self,
        session: Any,
        channel: Any,
        method: Any,
        run: Run,
        attempt: int,
        *,
        reason: str | None = None,
    ) -> bool:
        """Either requeue with backoff or dead-letter; always acks.

        Returns True when the run was requeued (caller may continue), False when
        it reached the DLQ.
        """
        reason = reason or run.failure_reason or "run failed"
        if run.attempt_count < run.max_attempts:
            try:
                transition_run(
                    session,
                    run,
                    RunStatus.QUEUED,
                    reason=f"retry {run.attempt_count}/{run.max_attempts}: {reason}",
                    actor=self.worker_id,
                )
            except Exception:
                session.rollback()
                channel.basic_ack(method.delivery_tag)
                return False
            session.commit()
            delay = self._backoff(run.attempt_count)
            publish_run(
                self._publish_channel(),
                run.id,
                run.attempt_count + 1,
                delay_seconds=delay,
            )
            append_log(
                run.id,
                f"[worker] retrying in {delay:.1f}s (attempt {run.attempt_count}/{run.max_attempts})",
                level="error",
            )
            channel.basic_ack(method.delivery_tag)
            return True

        session.commit()
        publish_dead(
            self._publish_channel(), run.id, run.attempt_count, reason
        )
        append_log(
            run.id,
            f"[worker] attempts exhausted ({run.attempt_count}/{run.max_attempts}); sent to DLQ",
            level="error",
        )
        channel.basic_ack(method.delivery_tag)
        return False

    def _defer(
        self, channel: Any, method: Any, run_id: uuid.UUID, attempt: int, delay: float
    ) -> None:
        """Republish a message later without losing it."""
        publish_run(
            self._publish_channel(),
            run_id,
            max(1, attempt),
            delay_seconds=delay,
        )
        channel.basic_ack(method.delivery_tag)


def reconcile_queued_runs(limit: int = 1000) -> int:
    """Republish durable QUEUED runs so broker outages self-heal on startup."""
    session = get_session()
    try:
        queued = list(
            session.execute(
                select(Run)
                .where(Run.status == RunStatus.QUEUED)
                .order_by(Run.created_at.asc())
                .limit(limit)
            ).scalars()
        )
    finally:
        session.close()
    if not queued:
        return 0
    connection = broker_connect()
    try:
        channel = connection.channel()
        declare_topology(channel)
        channel.confirm_delivery()
        for run in queued:
            publish_run(channel, run.id, max(1, run.attempt_count))
        return len(queued)
    finally:
        connection.close()


def main() -> None:
    """CLI entry point for ``python -m traintrack.worker``."""
    logging.basicConfig(
        level=os.environ.get("TRAINTRACK_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    init_db()
    worker = Worker()
    try:
        reconciled = reconcile_queued_runs()
        if reconciled:
            logger.info("republished %s queued runs on startup", reconciled)
    except Exception:
        logger.warning("startup reconciliation failed", exc_info=True)
    signal.signal(signal.SIGINT, lambda *_: worker.request_stop())
    signal.signal(signal.SIGTERM, lambda *_: worker.request_stop())
    logger.info("worker %s ready (executor=%s)", worker.worker_id, worker.executor.name)
    worker.start_consuming()


if __name__ == "__main__":
    main()
