"""Standalone evaluation job for a stored checkpoint artifact."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
import uuid
from pathlib import Path

import torch
from sqlalchemy import select

from traintrack.artifacts.store import ArtifactStore
from traintrack.db import get_session, init_db
from traintrack.models.artifact import Artifact
from traintrack.models.run import Run
from traintrack.models.snapshot import DatasetSnapshot
from traintrack.models.state import RunStatus
from traintrack.pipeline.validate import SnapshotValidationError, load_validated_snapshot
from traintrack.training.checkpoint import load_checkpoint
from traintrack.training.client import ApiClient
from traintrack.training.device import select_device
from traintrack.training.evaluation import evaluate_model, make_loader
from traintrack.training.model import build_model

logger = logging.getLogger("traintrack.eval")


def main(argv: list[str] | None = None) -> int:
    """Score a checkpoint against the held-out test split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level="INFO", format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    init_db()
    run_id = uuid.UUID(args.run_id)
    session = get_session()
    try:
        run = session.get(Run, run_id)
        if run is None:
            logger.error("run %s not found", run_id)
            return 2
        if run.job_kind != "eval":
            logger.error("run %s is not an eval job", run_id)
            return 2
        if run.status is not RunStatus.RUNNING:
            logger.error("run %s is %s, expected RUNNING", run_id, run.status.value)
            return 2
        checkpoint_sha = str(run.hyperparams.get("checkpoint_sha", ""))
        checkpoint_artifact = session.execute(
            select(Artifact).where(Artifact.sha256 == checkpoint_sha)
        ).scalar_one_or_none()
        if checkpoint_artifact is None:
            logger.error("checkpoint artifact %s not found", checkpoint_sha)
            return 2
        if checkpoint_artifact.kind != "checkpoint":
            logger.error(
                "artifact %s is %s, not a checkpoint", checkpoint_sha, checkpoint_artifact.kind
            )
            return 2
        snapshot = (
            session.get(DatasetSnapshot, checkpoint_artifact.dataset_snapshot_id)
            if checkpoint_artifact.dataset_snapshot_id
            else None
        )
        if snapshot is None or not snapshot.validated:
            logger.error("checkpoint has no validated dataset snapshot")
            return 2
        batch_size = int(run.hyperparams.get("batch_size", 256))
        snapshot_id = snapshot.id
        snapshot_path = snapshot.path
        snapshot_version = snapshot.version
        snapshot_hash = snapshot.content_hash
        checkpoint_parent = checkpoint_artifact.parent_sha
        checkpoint_run_id = checkpoint_artifact.run_id
    finally:
        session.close()

    store = ArtifactStore()
    if not store.exists(checkpoint_sha):
        logger.error("checkpoint bytes for %s are missing from the artifact store", checkpoint_sha)
        return 2
    try:
        manifest, splits = load_validated_snapshot(snapshot_path)
    except SnapshotValidationError as exc:
        logger.error("refusing to evaluate: %s", exc)
        return 2

    checkpoint = load_checkpoint(store.path_for(checkpoint_sha))
    if checkpoint.get("snapshot_hash") != snapshot_hash:
        logger.error("checkpoint was trained on a different snapshot hash; refusing")
        return 2

    device_info = select_device(args.device)
    class_names = list(checkpoint.get("class_names", manifest.get("class_names", [])))
    model_config = checkpoint.get("model_config", {})
    model = build_model(
        dropout=float(model_config.get("dropout", 0.0)),
        num_classes=int(model_config.get("num_classes", 10)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device_info.device)

    test_loader = make_loader(
        splits["test"][0],
        splits["test"][1],
        batch_size=batch_size,
        max_batches=args.max_batches,
    )
    started = time.monotonic()
    output = evaluate_model(
        model,
        test_loader,
        device_info.device,
        criterion=torch.nn.CrossEntropyLoss(),
        class_names=class_names,
        measure_latency=True,
        max_batches=args.max_batches,
    )
    eval_seconds = time.monotonic() - started

    report = {
        "run_id": str(run_id),
        "job_kind": "eval",
        "checkpoint_sha": checkpoint_sha,
        "checkpoint_parent_sha": checkpoint_parent,
        "checkpoint_run_id": str(checkpoint_run_id) if checkpoint_run_id else None,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "dataset_snapshot_id": str(snapshot_id),
        "dataset_snapshot_version": snapshot_version,
        "dataset_snapshot_hash": snapshot_hash,
        "device": device_info.backend,
        "device_name": device_info.name,
        "batch_size": batch_size,
        "accuracy": output.report["accuracy"],
        "macro_precision": output.report["macro_precision"],
        "macro_recall": output.report["macro_recall"],
        "macro_f1": output.report["macro_f1"],
        "test_loss": output.loss,
        "confusion_matrix": output.report["confusion_matrix"],
        "per_class_accuracy": {
            name: values["accuracy"]
            for name, values in output.report["per_class"].items()
        },
        "per_class": output.report["per_class"],
        "inference_p50_ms": output.latency["p50_ms"],
        "inference_p95_ms": output.latency["p95_ms"],
        "num_test_samples": output.report["num_samples"],
        "eval_seconds": round(eval_seconds, 3),
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    report_dir = Path(tempfile.gettempdir()) / "traintrack-eval" / str(run_id)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    write_session = get_session()
    try:
        artifact = store.write_file(
            report_path,
            kind="eval_report",
            session=write_session,
            run_id=run_id,
            dataset_snapshot_id=snapshot_id,
            parent_sha=checkpoint_sha,
        )
        write_session.commit()
        report["eval_report_sha"] = artifact.sha256
    finally:
        write_session.close()
    report_path.write_text(json.dumps(report, indent=2))

    client = ApiClient(args.api_url, args.api_key)
    wall = time.time()
    client.post_metrics(
        str(run_id),
        [
            {"name": "test_accuracy", "step": 0, "value": report["accuracy"], "wall_time": wall},
            {"name": "test_macro_f1", "step": 0, "value": report["macro_f1"], "wall_time": wall},
            {"name": "test_loss", "step": 0, "value": report["test_loss"], "wall_time": wall},
            {
                "name": "inference_p50_ms",
                "step": 0,
                "value": report["inference_p50_ms"],
                "wall_time": wall,
            },
            {
                "name": "inference_p95_ms",
                "step": 0,
                "value": report["inference_p95_ms"],
                "wall_time": wall,
            },
            {"name": "eval_seconds", "step": 0, "value": eval_seconds, "wall_time": wall},
        ],
    )
    print("EVAL_REPORT " + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
