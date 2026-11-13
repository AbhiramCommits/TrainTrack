"""Idempotency semantics for run submission."""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor

from traintrack.db import get_session
from traintrack.models.snapshot import DatasetSnapshot


def _snapshot() -> str:
    session = get_session()
    try:
        snapshot = DatasetSnapshot(
            version=f"idem-{uuid.uuid4().hex[:8]}",
            content_hash=uuid.uuid4().hex * 2,
            row_count=10,
            manifest={"seed": 1},
            validated=True,
            path="/tmp/idem",
        )
        session.add(snapshot)
        session.commit()
        return str(snapshot.id)
    finally:
        session.close()


def _payload(experiment_id: str, snapshot_id: str, lr: float = 0.001) -> dict:
    return {
        "experiment_id": experiment_id,
        "hyperparams": {"lr": lr, "batch_size": 32},
        "dataset_snapshot_id": snapshot_id,
    }


def test_replay_returns_same_run_with_200(
    client, auth_headers, experiment
) -> None:
    snapshot_id = _snapshot()
    headers = {**auth_headers, "Idempotency-Key": "replay-key-1"}
    first = client.post("/api/runs", json=_payload(experiment["id"], snapshot_id), headers=headers)
    second = client.post("/api/runs", json=_payload(experiment["id"], snapshot_id), headers=headers)
    assert first.status_code == 201
    assert second.status_code == 200
    assert first.get_json()["id"] == second.get_json()["id"]


def test_same_key_different_payload_conflicts(
    client, auth_headers, experiment
) -> None:
    snapshot_id = _snapshot()
    headers = {**auth_headers, "Idempotency-Key": "conflict-key"}
    first = client.post(
        "/api/runs", json=_payload(experiment["id"], snapshot_id, lr=0.01), headers=headers
    )
    assert first.status_code == 201
    conflict = client.post(
        "/api/runs", json=_payload(experiment["id"], snapshot_id, lr=0.02), headers=headers
    )
    assert conflict.status_code == 409
    assert conflict.get_json()["error"] == "idempotency_conflict"


def test_missing_idempotency_key_is_rejected(client, auth_headers, experiment) -> None:
    snapshot_id = _snapshot()
    response = client.post(
        "/api/runs", json=_payload(experiment["id"], snapshot_id), headers=auth_headers
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "missing_idempotency_key"


def test_concurrent_replays_create_one_run(client, auth_headers, experiment) -> None:
    snapshot_id = _snapshot()
    headers = {**auth_headers, "Idempotency-Key": "concurrent-key"}
    payload = _payload(experiment["id"], snapshot_id)

    def submit() -> tuple[int, str]:
        response = client.post("/api/runs", json=payload, headers=headers)
        return response.status_code, response.get_json().get("id", "")

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: submit(), range(4)))

    ids = {run_id for _, run_id in results}
    assert len(ids) == 1
    statuses = sorted(status for status, _ in results)
    assert statuses.count(201) == 1
    assert all(status in (200, 201) for status in statuses)


def test_distinct_keys_create_distinct_runs(client, auth_headers, experiment) -> None:
    snapshot_id = _snapshot()
    payload = _payload(experiment["id"], snapshot_id)
    first = client.post(
        "/api/runs", json=payload, headers={**auth_headers, "Idempotency-Key": "k1"}
    )
    second = client.post(
        "/api/runs", json=payload, headers={**auth_headers, "Idempotency-Key": "k2"}
    )
    assert first.get_json()["id"] != second.get_json()["id"]
