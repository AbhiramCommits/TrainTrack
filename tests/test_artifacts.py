"""Content-addressed artifact store and lineage tests."""

from __future__ import annotations

import uuid

from traintrack.artifacts.store import ArtifactStore
from traintrack.db import get_session
from traintrack.models.experiment import Experiment
from traintrack.models.run import Run
from traintrack.models.snapshot import DatasetSnapshot
from traintrack.models.state import RunStatus
from traintrack.training.metrics import classification_report


def _seed_run() -> tuple[uuid.UUID, uuid.UUID]:
    session = get_session()
    try:
        experiment = Experiment(name=f"artifacts-{uuid.uuid4().hex[:8]}")
        session.add(experiment)
        session.flush()
        snapshot = DatasetSnapshot(
            version=f"artifacts-{uuid.uuid4().hex[:8]}",
            content_hash=uuid.uuid4().hex * 2,
            row_count=100,
            manifest={"seed": 1},
            validated=True,
            path="/tmp/snapshot",
        )
        session.add(snapshot)
        session.flush()
        run = Run(
            experiment_id=experiment.id,
            status=RunStatus.RUNNING,
            job_kind="train",
            hyperparams={"lr": 0.01, "batch_size": 32},
            dataset_snapshot_id=snapshot.id,
            git_sha="deadbeef",
        )
        session.add(run)
        session.commit()
        return run.id, snapshot.id
    finally:
        session.close()


def test_content_addressed_write_and_dedupe(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    run_id, snapshot_id = _seed_run()
    payload = b"same checkpoint bytes" * 100
    session = get_session()
    try:
        first = store.write_bytes(
            payload,
            kind="checkpoint",
            session=session,
            run_id=run_id,
            dataset_snapshot_id=snapshot_id,
        )
        session.commit()
        second = store.write_bytes(
            payload,
            kind="checkpoint",
            session=session,
            run_id=run_id,
            dataset_snapshot_id=snapshot_id,
        )
        session.commit()
        assert first.sha256 == second.sha256
        assert first.id == second.id
        assert second.ref_count == 2
        assert second.bytes_saved == len(payload)
        assert store.path_for(first.sha256).read_bytes() == payload
        assert store.path_for(first.sha256).parent.name == first.sha256[:2]
        assert store.stats()["blob_count"] == 1
    finally:
        session.close()


def test_artifact_lineage_endpoint(
    client, auth_headers
) -> None:
    run_id, snapshot_id = _seed_run()
    store = ArtifactStore()
    checkpoint_bytes = b"checkpoint-for-lineage-" + uuid.uuid4().bytes
    report_bytes = b"eval-report-for-lineage-" + uuid.uuid4().bytes
    session = get_session()
    try:
        checkpoint = store.write_bytes(
            checkpoint_bytes,
            kind="checkpoint",
            session=session,
            run_id=run_id,
            dataset_snapshot_id=snapshot_id,
        )
        session.commit()
        report = store.write_bytes(
            report_bytes,
            kind="eval_report",
            session=session,
            run_id=run_id,
            dataset_snapshot_id=snapshot_id,
            parent_sha=checkpoint.sha256,
        )
        session.commit()
        checkpoint_sha, report_sha = checkpoint.sha256, report.sha256
    finally:
        session.close()

    response = client.get(
        f"/api/artifacts/{report_sha}/lineage", headers=auth_headers
    )
    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["artifact"]["kind"] == "eval_report"
    assert len(body["chain"]) == 2
    assert body["chain"][1]["artifact"]["sha256"] == checkpoint_sha
    assert body["run"]["hyperparams"]["lr"] == 0.01
    assert body["run"]["git_sha"] == "deadbeef"
    assert body["dataset_snapshot"]["id"] == str(snapshot_id)

    missing = client.get(f"/api/artifacts/{'0' * 64}/lineage", headers=auth_headers)
    assert missing.status_code == 404


def test_classification_report_macro_metrics() -> None:
    import numpy as np

    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 2, 2, 2])
    report = classification_report(y_true, y_pred, num_classes=3)
    assert report["accuracy"] == 0.833333
    assert report["macro_f1"] == 0.822222
    assert report["confusion_matrix"] == [[2, 0, 0], [0, 1, 1], [0, 0, 2]]
    assert report["per_class"]["2"]["recall"] == 1.0
