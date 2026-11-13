"""Run lifecycle endpoints: submit, list, detail, cancel, metrics, logs."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator

from flask import Blueprint, Response, g, jsonify, request, stream_with_context
from sqlalchemy import select, tuple_
from sqlalchemy.exc import IntegrityError

from traintrack.api.auth import require_api_key
from traintrack.api.deps import db_session
from traintrack.api.errors import ApiError, decode_cursor, encode_cursor, json_body
from traintrack.api.schemas import MetricBatch, ResourceBatch, RunCreate
from traintrack.api.serializers import run_to_dict
from traintrack.models.experiment import Experiment
from traintrack.models.metric import MetricPoint
from traintrack.models.run import Run, StatusTransition
from traintrack.models.state import TERMINAL_STATUSES, RunStatus, can_transition
from traintrack.services.cache import get_redis
from traintrack.services.leaderboard import invalidate_leaderboard
from traintrack.services.logs import read_logs, wait_for_logs
from traintrack.services.resources import append_resources, read_resources
from traintrack.services.runs import (
    InvalidStateChange,
    RunNotFound,
    SnapshotRejected,
    canonical_request_hash,
    get_run,
    queue_position,
    transition_run,
    validate_snapshot,
)

bp = Blueprint("runs", __name__, url_prefix="/api/runs")
logger = logging.getLogger(__name__)


def _parse_status(value: str | None) -> RunStatus | None:
    if not value:
        return None
    try:
        return RunStatus(value.upper())
    except ValueError as exc:
        raise ApiError(f"Invalid status filter: {value}", 400, "invalid_status") from exc


@bp.post("")
@require_api_key
def submit_run():
    """Submit a run. Requires an Idempotency-Key; replays return 200."""
    idempotency_key = (request.headers.get("Idempotency-Key") or "").strip()
    if not idempotency_key:
        raise ApiError(
            "Idempotency-Key header is required", 400, "missing_idempotency_key"
        )
    if len(idempotency_key) > 200:
        raise ApiError("Idempotency-Key is too long", 400, "invalid_idempotency_key")

    raw = json_body()
    payload = RunCreate.model_validate(raw)
    request_hash = canonical_request_hash(
        payload.model_dump(mode="json", exclude_none=False)
    )
    session = db_session()

    existing = session.execute(
        select(Run).where(Run.idempotency_key == idempotency_key)
    ).scalar_one_or_none()
    if existing is not None:
        if existing.request_hash not in (None, request_hash):
            raise ApiError(
                "Idempotency-Key was already used with a different payload",
                409,
                "idempotency_conflict",
            ) from None
        return jsonify(run_to_dict(existing, position=queue_position(session, existing))), 200

    experiment = session.get(Experiment, payload.experiment_id)
    if experiment is None:
        raise ApiError("Experiment not found", 404, "not_found")

    if payload.job_kind == "train":
        try:
            validate_snapshot(session, payload.dataset_snapshot_id)
        except SnapshotRejected as exc:
            raise ApiError(str(exc), 422, "snapshot_rejected") from exc

    run = Run(
        experiment_id=payload.experiment_id,
        status=RunStatus.QUEUED,
        job_kind=payload.job_kind,
        hyperparams=payload.hyperparams,
        git_sha=payload.git_sha,
        dataset_snapshot_id=payload.dataset_snapshot_id,
        requested_gpus=payload.requested_gpus,
        priority=payload.priority,
        image=payload.image,
        timeout_seconds=payload.timeout_seconds,
        max_attempts=payload.max_attempts,
        idempotency_key=idempotency_key,
        request_hash=request_hash,
    )
    session.add(run)
    try:
        session.flush()
        actor = f"api-key:{g.api_key.name}" if getattr(g, "api_key", None) else "api"
        session.add(
            StatusTransition(
                run_id=run.id,
                from_status=None,
                to_status=RunStatus.QUEUED.value,
                reason="submitted",
                actor=actor,
            )
        )
        session.commit()
    except IntegrityError:
        session.rollback()
        existing = session.execute(
            select(Run).where(Run.idempotency_key == idempotency_key)
        ).scalar_one_or_none()
        if existing is None:
            raise
        if existing.request_hash not in (None, request_hash):
            raise ApiError(
                "Idempotency-Key was already used with a different payload",
                409,
                "idempotency_conflict",
            ) from None
        return jsonify(run_to_dict(existing)), 200

    logger.info("submitted run %s (%s)", run.id, payload.job_kind)
    return jsonify(run_to_dict(run, position=queue_position(session, run))), 201


@bp.get("")
@require_api_key
def list_runs():
    """List runs with experiment/status filters and keyset pagination."""
    session = db_session()
    limit_raw = request.args.get("limit", "50")
    try:
        limit = max(1, min(200, int(limit_raw)))
    except ValueError as exc:
        raise ApiError("limit must be an integer", 400, "invalid_limit") from exc

    query = select(Run)
    experiment_id = request.args.get("experiment_id")
    if experiment_id:
        try:
            query = query.where(Run.experiment_id == uuid.UUID(experiment_id))
        except ValueError as exc:
            raise ApiError("Invalid experiment_id", 400, "invalid_id") from exc
    status = _parse_status(request.args.get("status"))
    if status is not None:
        query = query.where(Run.status == status)
    job_kind = request.args.get("job_kind")
    if job_kind:
        query = query.where(Run.job_kind == job_kind)

    cursor = request.args.get("cursor")
    if cursor:
        created_at, run_id = decode_cursor(cursor)
        query = query.where(
            tuple_(Run.created_at, Run.id) < (created_at, run_id)
        )

    query = query.order_by(Run.created_at.desc(), Run.id.desc()).limit(limit + 1)
    runs = list(session.execute(query).scalars())
    has_more = len(runs) > limit
    runs = runs[:limit]
    next_cursor = (
        encode_cursor(runs[-1].created_at, runs[-1].id)
        if has_more and runs
        else None
    )
    return jsonify(
        {
            "runs": [
                run_to_dict(run, position=queue_position(session, run))
                for run in runs
            ],
            "next_cursor": next_cursor,
            "count": len(runs),
        }
    )


@bp.get("/<run_id>")
@require_api_key
def get_run_detail(run_id: str):
    """Fetch a run with its full status timeline and current queue position."""
    session = db_session()
    try:
        run = get_run(session, run_id)
    except RunNotFound as exc:
        raise ApiError("Run not found", 404, "not_found") from exc
    return jsonify(
        run_to_dict(
            run, position=queue_position(session, run), include_transitions=True
        )
    )


@bp.post("/<run_id>/cancel")
@require_api_key
def cancel_run(run_id: str):
    """Cancel a queued, scheduled, or running run."""
    session = db_session()
    try:
        run = get_run(session, run_id)
    except RunNotFound as exc:
        raise ApiError("Run not found", 404, "not_found") from exc
    if run.status in TERMINAL_STATUSES:
        raise ApiError(
            f"Run is already {run.status.value}", 409, "not_cancelable"
        )
    if not can_transition(run.status, RunStatus.CANCELED):
        raise ApiError("Run cannot be canceled from its current state", 409, "not_cancelable")
    run.cancel_requested = True
    try:
        transition_run(
            session,
            run,
            RunStatus.CANCELED,
            reason="cancel requested via API",
            actor="api",
        )
    except InvalidStateChange as exc:
        session.rollback()
        raise ApiError(str(exc), 409, "not_cancelable") from exc
    session.commit()
    try:
        get_redis().set(f"cancel:{run.id}", "1", ex=3600)
    except Exception:  # pragma: no cover - cancellation still recorded in DB
        logger.warning("could not publish cancel flag for %s", run.id, exc_info=True)
    return jsonify(run_to_dict(run)), 200


@bp.post("/<run_id>/metrics")
@require_api_key
def ingest_metrics(run_id: str):
    """Batch-ingest metric points for a run and invalidate its leaderboard."""
    session = db_session()
    try:
        run = get_run(session, run_id)
    except RunNotFound as exc:
        raise ApiError("Run not found", 404, "not_found") from exc
    if run.status not in (RunStatus.RUNNING, RunStatus.SUCCEEDED):
        raise ApiError(
            f"Cannot ingest metrics for a {run.status.value} run",
            409,
            "invalid_state",
        )
    batch = MetricBatch.model_validate(json_body())
    now = time.time()
    session.add_all(
        [
            MetricPoint(
                run_id=run.id,
                name=point.name,
                step=point.step,
                value=point.value,
                wall_time=point.wall_time if point.wall_time is not None else now,
            )
            for point in batch.metrics
        ]
    )
    session.commit()
    invalidate_leaderboard(run.experiment_id)
    return jsonify({"ingested": len(batch.metrics), "run_id": str(run.id)}), 201


@bp.get("/<run_id>/metrics")
@require_api_key
def get_metrics(run_id: str):
    """Read metric points for a run, optionally filtered by name."""
    session = db_session()
    try:
        run = get_run(session, run_id)
    except RunNotFound as exc:
        raise ApiError("Run not found", 404, "not_found") from exc
    query = select(MetricPoint).where(MetricPoint.run_id == run.id)
    names = request.args.get("names") or request.args.get("name")
    if names:
        wanted = [name.strip() for name in names.split(",") if name.strip()]
        query = query.where(MetricPoint.name.in_(wanted))
    try:
        limit = max(1, min(20000, int(request.args.get("limit", "5000"))))
    except ValueError as exc:
        raise ApiError("limit must be an integer", 400, "invalid_limit") from exc
    points = list(
        session.execute(
            query.order_by(MetricPoint.step.asc(), MetricPoint.id.asc()).limit(limit)
        ).scalars()
    )
    return jsonify(
        {
            "run_id": str(run.id),
            "points": [
                {
                    "name": point.name,
                    "step": point.step,
                    "value": point.value,
                    "wall_time": point.wall_time,
                }
                for point in points
            ],
            "count": len(points),
        }
    )


@bp.post("/<run_id>/resources")
@require_api_key
def ingest_resources(run_id: str):
    """Batch-ingest device utilization samples for a running job."""
    session = db_session()
    try:
        run = get_run(session, run_id)
    except RunNotFound as exc:
        raise ApiError("Run not found", 404, "not_found") from exc
    if run.status not in (RunStatus.RUNNING, RunStatus.SUCCEEDED):
        raise ApiError(
            f"Cannot ingest resources for a {run.status.value} run",
            409,
            "invalid_state",
        )
    batch = ResourceBatch.model_validate(json_body())
    count = append_resources(run.id, [s.model_dump() for s in batch.samples])
    return jsonify({"ingested": count, "run_id": str(run.id)}), 201


@bp.get("/<run_id>/resources")
@require_api_key
def get_resources(run_id: str):
    """Return device utilization samples for a run."""
    session = db_session()
    try:
        run = get_run(session, run_id)
    except RunNotFound as exc:
        raise ApiError("Run not found", 404, "not_found") from exc
    try:
        limit = max(1, min(5000, int(request.args.get("limit", "1000"))))
    except ValueError:
        limit = 1000
    samples = read_resources(run.id, limit=limit)
    return jsonify(
        {
            "run_id": str(run.id),
            "samples": samples,
            "count": len(samples),
            "device": samples[-1]["device"] if samples else None,
        }
    )


@bp.get("/<run_id>/logs")
@require_api_key
def get_logs(run_id: str):
    """Return the last N log lines for a run."""
    session = db_session()
    try:
        get_run(session, run_id)
    except RunNotFound as exc:
        raise ApiError("Run not found", 404, "not_found") from exc
    try:
        tail = max(1, min(5000, int(request.args.get("tail", "200"))))
    except ValueError as exc:
        raise ApiError("tail must be an integer", 400, "invalid_tail") from exc
    entries = read_logs(run_id, tail=tail)
    return jsonify({"run_id": str(run_id), "logs": entries, "count": len(entries)})


def _sse_event(entry_id: str, data: str) -> str:
    return f"id: {entry_id}\ndata: {data}\n\n"


@bp.get("/<run_id>/logs/stream")
@require_api_key
def stream_logs(run_id: str):
    """Server-Sent Events: replay recent log history then tail live output."""
    session = db_session()
    try:
        run = get_run(session, run_id)
        run_uuid = run.id
    except RunNotFound as exc:
        raise ApiError("Run not found", 404, "not_found") from exc

    try:
        replay_count = max(0, min(5000, int(request.args.get("replay", "200"))))
    except ValueError:
        replay_count = 200
    after_id = request.headers.get("Last-Event-ID") or request.args.get("after")
    if after_id == "0-0":
        after_id = None

    def generate() -> Iterator[str]:
        from traintrack.db import get_session as new_session

        last_id = after_id
        if last_id is None:
            history = read_logs(run_uuid, tail=replay_count or 1)
            for entry in history:
                last_id = entry["id"]
                yield _sse_event(
                    entry["id"],
                    json.dumps(
                        {
                            "ts": entry.get("ts", 0.0),
                            "level": entry.get("level", "info"),
                            "line": entry.get("line", ""),
                        }
                    ),
                )
            if last_id is None:
                last_id = "0-0"

        idle_loops = 0
        while True:
            entries = wait_for_logs(run_uuid, last_id, block_ms=2000)
            if entries:
                idle_loops = 0
                for entry_id, fields in entries:
                    last_id = entry_id
                    yield _sse_event(entry_id, fields.get("data", "{}"))
            else:
                idle_loops += 1
                yield ": keepalive\n\n"

            check = new_session()
            try:
                status = check.execute(
                    select(Run.status).where(Run.id == run_uuid)
                ).scalar_one_or_none()
            finally:
                check.close()
            if status is not None and status in TERMINAL_STATUSES and not entries:
                break
            if idle_loops > 900:  # 30 minutes with no logs and a live run
                break

    response = Response(stream_with_context(generate()), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Connection"] = "keep-alive"
    return response
