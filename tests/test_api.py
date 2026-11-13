"""API tests against real Postgres and Redis."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from traintrack.db import get_session
from traintrack.models.run import Run
from traintrack.services.logs import append_log
from traintrack.services.runs import transition_run


def _snapshot(client, auth_headers) -> dict:
    """Create a validated snapshot directly in the DB and return its payload."""
    from traintrack.models.snapshot import DatasetSnapshot

    session = get_session()
    try:
        snapshot = DatasetSnapshot(
            version=f"test-{uuid.uuid4().hex[:8]}",
            content_hash=uuid.uuid4().hex * 2,
            row_count=100,
            manifest={"seed": 42, "splits": {"train": 80, "val": 10, "test": 10}},
            validated=True,
            path="/tmp/snapshot",
        )
        session.add(snapshot)
        session.commit()
        return {"id": str(snapshot.id), "content_hash": snapshot.content_hash}
    finally:
        session.close()


def _run_payload(experiment: dict, snapshot: dict, **overrides) -> dict:
    payload = {
        "experiment_id": experiment["id"],
        "job_kind": "train",
        "hyperparams": {"lr": 0.001, "batch_size": 64, "epochs": 2},
        "dataset_snapshot_id": snapshot["id"],
        "requested_gpus": 0,
    }
    payload.update(overrides)
    return payload


def test_healthz(client) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_readyz_checks_dependencies(client) -> None:
    response = client.get("/readyz")
    assert response.status_code == 200
    checks = response.get_json()["checks"]
    assert checks["postgres"] == "ok"
    assert checks["redis"] == "ok"


def test_auth_is_required(client) -> None:
    response = client.get("/api/runs")
    assert response.status_code == 401
    assert response.get_json()["error"] == "unauthorized"


def test_auth_rejects_bad_key(client) -> None:
    response = client.get("/api/runs", headers={"X-API-Key": "nope"})
    assert response.status_code == 401


def test_auth_accepts_bearer_token(client, api_key) -> None:
    response = client.get("/api/runs", headers={"Authorization": f"Bearer {api_key}"})
    assert response.status_code == 200


def test_create_and_list_experiments(client, auth_headers) -> None:
    created = client.post(
        "/api/experiments", json={"name": "exp-a", "description": "first"}, headers=auth_headers
    )
    assert created.status_code == 201
    experiment = created.get_json()
    assert experiment["name"] == "exp-a"

    duplicate = client.post(
        "/api/experiments", json={"name": "exp-a"}, headers=auth_headers
    )
    assert duplicate.status_code == 409

    listing = client.get("/api/experiments", headers=auth_headers)
    assert listing.status_code == 200
    assert len(listing.get_json()["experiments"]) == 1

    fetched = client.get(f"/api/experiments/{experiment['id']}", headers=auth_headers)
    assert fetched.status_code == 200
    assert fetched.get_json()["name"] == "exp-a"


def test_run_submission_validation(client, auth_headers, experiment, idempotency_headers) -> None:
    snapshot = _snapshot(client, auth_headers)
    bad = client.post(
        "/api/runs",
        json=_run_payload(experiment, snapshot, hyperparams={"lr": -1, "batch_size": 0}),
        headers=idempotency_headers,
    )
    assert bad.status_code == 422

    missing_snapshot = client.post(
        "/api/runs",
        json=_run_payload(experiment, snapshot, dataset_snapshot_id=None),
        headers=idempotency_headers,
    )
    assert missing_snapshot.status_code == 422

    unknown_experiment = client.post(
        "/api/runs",
        json=_run_payload({"id": str(uuid.uuid4())}, snapshot),
        headers=idempotency_headers,
    )
    assert unknown_experiment.status_code == 404


def test_unvalidated_snapshot_is_rejected(client, auth_headers, experiment, idempotency_headers) -> None:
    from traintrack.models.snapshot import DatasetSnapshot

    session = get_session()
    try:
        snapshot = DatasetSnapshot(
            version=f"bad-{uuid.uuid4().hex[:8]}",
            content_hash=uuid.uuid4().hex * 2,
            row_count=10,
            manifest={},
            validated=False,
            path="/tmp/nope",
        )
        session.add(snapshot)
        session.commit()
        snapshot_id = str(snapshot.id)
    finally:
        session.close()

    response = client.post(
        "/api/runs",
        json={
            "experiment_id": experiment["id"],
            "hyperparams": {"lr": 0.01, "batch_size": 32},
            "dataset_snapshot_id": snapshot_id,
        },
        headers=idempotency_headers,
    )
    assert response.status_code == 422
    assert response.get_json()["error"] == "snapshot_rejected"


def test_submit_run_and_fetch_detail(
    client, auth_headers, experiment, idempotency_headers
) -> None:
    snapshot = _snapshot(client, auth_headers)
    response = client.post(
        "/api/runs",
        json=_run_payload(experiment, snapshot, hyperparams={"lr": 0.01, "batch_size": 32, "epochs": 2}),
        headers=idempotency_headers,
    )
    assert response.status_code == 201, response.get_json()
    run = response.get_json()
    assert run["status"] == "QUEUED"
    assert run["queue_position"] == 1
    assert run["hyperparams"]["optimizer"] == "adam"

    detail = client.get(f"/api/runs/{run['id']}", headers=auth_headers)
    assert detail.status_code == 200
    body = detail.get_json()
    assert body["transitions"][0]["to_status"] == "QUEUED"
    assert body["transitions"][0]["reason"] == "submitted"

    missing = client.get(f"/api/runs/{uuid.uuid4()}", headers=auth_headers)
    assert missing.status_code == 404


def test_run_filters_and_keyset_pagination(
    client, auth_headers, experiment, idempotency_headers
) -> None:
    snapshot = _snapshot(client, auth_headers)
    ids = []
    for index in range(5):
        response = client.post(
            "/api/runs",
            json=_run_payload(
                experiment,
                snapshot,
                hyperparams={"lr": 0.001 * (index + 1), "batch_size": 16 * (index + 1)},
            ),
            headers={**idempotency_headers, "Idempotency-Key": f"page-{index}"},
        )
        assert response.status_code == 201
        ids.append(response.get_json()["id"])

    page_one = client.get(
        f"/api/runs?experiment_id={experiment['id']}&limit=2", headers=auth_headers
    )
    assert page_one.status_code == 200
    body = page_one.get_json()
    assert len(body["runs"]) == 2
    assert body["next_cursor"]
    page_two = client.get(
        f"/api/runs?experiment_id={experiment['id']}&limit=2&cursor={body['next_cursor']}",
        headers=auth_headers,
    )
    assert len(page_two.get_json()["runs"]) == 2
    assert {run["id"] for run in page_two.get_json()["runs"]}.isdisjoint(
        {run["id"] for run in body["runs"]}
    )

    queued = client.get("/api/runs?status=QUEUED", headers=auth_headers)
    assert queued.get_json()["count"] == 5
    running = client.get("/api/runs?status=RUNNING", headers=auth_headers)
    assert running.get_json()["count"] == 0

    bad_status = client.get("/api/runs?status=NOPE", headers=auth_headers)
    assert bad_status.status_code == 400
    bad_cursor = client.get("/api/runs?cursor=***", headers=auth_headers)
    assert bad_cursor.status_code == 400


def test_queue_position_orders_by_priority(
    client, auth_headers, experiment, idempotency_headers
) -> None:
    snapshot = _snapshot(client, auth_headers)
    low = client.post(
        "/api/runs",
        json=_run_payload(experiment, snapshot, priority=1),
        headers={**idempotency_headers, "Idempotency-Key": "low-priority"},
    ).get_json()
    high = client.post(
        "/api/runs",
        json=_run_payload(experiment, snapshot, priority=9),
        headers={**idempotency_headers, "Idempotency-Key": "high-priority"},
    ).get_json()
    assert low["queue_position"] == 1
    assert high["queue_position"] == 2


def test_metrics_ingest_and_leaderboard_cache(
    client, auth_headers, experiment, idempotency_headers
) -> None:
    snapshot = _snapshot(client, auth_headers)
    run = client.post(
        "/api/runs",
        json=_run_payload(experiment, snapshot),
        headers=idempotency_headers,
    ).get_json()

    rejected = client.post(
        f"/api/runs/{run['id']}/metrics",
        json={"metrics": [{"name": "val_accuracy", "step": 1, "value": 0.5}]},
        headers=auth_headers,
    )
    assert rejected.status_code == 409

    session = get_session()
    try:
        model = session.get(Run, uuid.UUID(run["id"]))
        assert model is not None
        transition_run(session, model, "SCHEDULED", reason="test")
        transition_run(session, model, "RUNNING", reason="test")
        session.commit()
    finally:
        session.close()

    ingest = client.post(
        f"/api/runs/{run['id']}/metrics",
        json={
            "metrics": [
                {"name": "loss", "step": 1, "value": 0.9},
                {"name": "val_accuracy", "step": 1, "value": 0.8},
                {"name": "val_accuracy", "step": 2, "value": 0.85},
            ]
        },
        headers=auth_headers,
    )
    assert ingest.status_code == 201
    assert ingest.get_json()["ingested"] == 3

    points = client.get(
        f"/api/runs/{run['id']}/metrics?name=val_accuracy", headers=auth_headers
    )
    assert points.get_json()["count"] == 2

    first = client.get(
        f"/api/experiments/{experiment['id']}/leaderboard", headers=auth_headers
    ).get_json()
    assert first["cache_hit"] is False
    assert first["entries"][0]["best_value"] == pytest.approx(0.85)

    second = client.get(
        f"/api/experiments/{experiment['id']}/leaderboard", headers=auth_headers
    ).get_json()
    assert second["cache_hit"] is True

    # A new metric write invalidates the cached leaderboard.
    client.post(
        f"/api/runs/{run['id']}/metrics",
        json={"metrics": [{"name": "val_accuracy", "step": 3, "value": 0.9}]},
        headers=auth_headers,
    )
    third = client.get(
        f"/api/experiments/{experiment['id']}/leaderboard", headers=auth_headers
    ).get_json()
    assert third["cache_hit"] is False
    assert third["entries"][0]["best_value"] == pytest.approx(0.9)


def test_cancel_run_and_terminal_guard(
    client, auth_headers, experiment, idempotency_headers
) -> None:
    snapshot = _snapshot(client, auth_headers)
    run = client.post(
        "/api/runs",
        json=_run_payload(experiment, snapshot),
        headers=idempotency_headers,
    ).get_json()
    canceled = client.post(f"/api/runs/{run['id']}/cancel", headers=auth_headers)
    assert canceled.status_code == 200
    assert canceled.get_json()["status"] == "CANCELED"
    again = client.post(f"/api/runs/{run['id']}/cancel", headers=auth_headers)
    assert again.status_code == 409


def test_logs_endpoints(client, auth_headers, experiment, idempotency_headers) -> None:
    snapshot = _snapshot(client, auth_headers)
    run = client.post(
        "/api/runs",
        json=_run_payload(experiment, snapshot),
        headers=idempotency_headers,
    ).get_json()
    for index in range(5):
        append_log(run["id"], f"line {index}")

    logs = client.get(f"/api/runs/{run['id']}/logs?tail=3", headers=auth_headers)
    assert logs.status_code == 200
    lines = [entry["line"] for entry in logs.get_json()["logs"]]
    assert lines == ["line 2", "line 3", "line 4"]

    # Cancel to make the run terminal so the SSE stream terminates after replay.
    client.post(f"/api/runs/{run['id']}/cancel", headers=auth_headers)
    stream = client.get(
        f"/api/runs/{run['id']}/logs/stream?replay=5", headers=auth_headers
    )
    assert stream.status_code == 200
    assert stream.mimetype == "text/event-stream"
    body = b"".join(stream.response).decode()
    assert "line 4" in body
    assert "data:" in body


def test_logs_require_auth(client, auth_headers, experiment, idempotency_headers) -> None:
    snapshot = _snapshot(client, auth_headers)
    run = client.post(
        "/api/runs",
        json=_run_payload(experiment, snapshot),
        headers=idempotency_headers,
    ).get_json()
    assert client.get(f"/api/runs/{run['id']}/logs").status_code == 401
    assert (
        client.get(f"/api/runs/{run['id']}/logs?api_key=nope").status_code == 401
    )


def test_snapshots_endpoint(client, auth_headers) -> None:
    snapshot = _snapshot(client, auth_headers)
    response = client.get("/api/snapshots", headers=auth_headers)
    assert response.status_code == 200
    assert response.get_json()["snapshots"][0]["id"] == snapshot["id"]


def test_unknown_route_returns_json_404(client) -> None:
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert response.is_json


def test_per_key_rate_limit(client, app) -> None:
    from traintrack.db import get_session
    from traintrack.services.auth import create_api_key

    session = get_session()
    try:
        record, raw = create_api_key(session, "limited", rate_limit_per_min=1)
        session.commit()
    finally:
        session.close()

    headers = {"X-API-Key": raw}
    first = client.get("/api/runs", headers=headers)
    assert first.status_code == 200
    second = client.get("/api/runs", headers=headers)
    assert second.status_code == 429
    assert second.headers["Retry-After"]
    assert second.get_json()["error"] == "rate_limited"


def test_queued_run_timestamps(client, auth_headers, experiment, idempotency_headers) -> None:
    snapshot = _snapshot(client, auth_headers)
    run = client.post(
        "/api/runs",
        json=_run_payload(experiment, snapshot),
        headers=idempotency_headers,
    ).get_json()
    created = datetime.fromisoformat(run["created_at"])
    assert created.tzinfo is not None
    assert created.replace(tzinfo=UTC).timestamp() <= datetime.now(
        UTC
    ).timestamp() + 5
