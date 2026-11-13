"""Generate docs/results/training_runs.json from the live registry.

Every value in the output is read from Postgres, Redis-backed run timing, the
content-addressed artifact store, and the real training/eval summaries written
by the jobs. No numbers are typed by hand.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from sqlalchemy import select
from traintrack.artifacts.store import ArtifactStore
from traintrack.db import get_session, init_db
from traintrack.models.artifact import Artifact
from traintrack.models.experiment import Experiment
from traintrack.models.metric import MetricPoint
from traintrack.models.run import Run
from traintrack.models.snapshot import DatasetSnapshot
from traintrack.models.state import RunStatus

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "docs" / "results" / "run_manifest.json"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "results" / "training_runs.json"
EXPERIMENT_NAME = "mnist-sweep-v1"
MIN_COMPLETED_RUNS = 5

METRIC_NAMES = (
    "val_accuracy",
    "test_accuracy",
    "test_macro_f1",
    "train_time_seconds",
    "inference_p50_ms",
    "inference_p95_ms",
    "best_val_accuracy",
    "epochs_completed",
)


def _metrics_for(session, run_id) -> dict[str, float]:
    rows = session.execute(
        select(MetricPoint.name, MetricPoint.value)
        .where(
            MetricPoint.run_id == run_id,
            MetricPoint.name.in_(METRIC_NAMES),
        )
        .order_by(MetricPoint.step.asc())
    ).all()
    metrics: dict[str, float] = {}
    for name, value in rows:
        current = metrics.get(name)
        if current is None or (name in ("val_accuracy", "test_accuracy") and value > current):
            metrics[name] = float(value)
    return metrics


def _artifact_json(session, store: ArtifactStore, run_id, kind: str) -> dict[str, Any] | None:
    artifact = session.execute(
        select(Artifact)
        .where(Artifact.run_id == run_id, Artifact.kind == kind)
        .order_by(Artifact.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()
    if artifact is None or not store.exists(artifact.sha256):
        return None
    return json.loads(store.read_bytes(artifact.sha256))


def _lineage_length(session, sha256: str | None) -> int:
    length = 0
    current = sha256
    seen: set[str] = set()
    while current and current not in seen and length < 20:
        seen.add(current)
        length += 1
        parent = session.execute(
            select(Artifact.parent_sha).where(Artifact.sha256 == current)
        ).scalar_one_or_none()
        current = parent
    return length


def main() -> int:
    """Export the training results JSON."""
    init_db()
    store = ArtifactStore()
    manifest = (
        json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}
    )
    experiment_name = manifest.get("experiment", {}).get("name", EXPERIMENT_NAME)
    session = get_session()
    try:
        experiment = session.execute(
            select(Experiment).where(Experiment.name == experiment_name)
        ).scalar_one_or_none()
        if experiment is None:
            raise SystemExit(f"experiment {experiment_name!r} not found")

        runs = list(
            session.execute(
                select(Run)
                .where(Run.experiment_id == experiment.id, Run.job_kind == "train")
                .order_by(Run.created_at.asc())
            ).scalars()
        )
        completed = [run for run in runs if run.status is RunStatus.SUCCEEDED]
        if len(completed) < MIN_COMPLETED_RUNS:
            raise SystemExit(
                f"need at least {MIN_COMPLETED_RUNS} completed training runs, "
                f"found {len(completed)}"
            )

        snapshot = session.get(DatasetSnapshot, completed[0].dataset_snapshot_id)
        if snapshot is None:
            raise SystemExit("dataset snapshot missing")

        run_records: list[dict[str, Any]] = []
        for run in completed:
            metrics = _metrics_for(session, run.id)
            summary = _artifact_json(session, store, run.id, "train_summary") or {}
            artifacts = list(
                session.execute(
                    select(Artifact).where(Artifact.run_id == run.id)
                ).scalars()
            )
            wall_clock = (
                round((run.finished_at - run.started_at).total_seconds(), 3)
                if run.started_at and run.finished_at
                else None
            )
            run_records.append(
                {
                    "run_id": str(run.id),
                    "status": run.status.value,
                    "hyperparams": run.hyperparams,
                    "git_sha": run.git_sha,
                    "requested_gpus": run.requested_gpus,
                    "val_accuracy": metrics.get("val_accuracy"),
                    "test_accuracy": metrics.get("test_accuracy"),
                    "test_macro_f1": metrics.get("test_macro_f1"),
                    "inference_p50_ms": metrics.get("inference_p50_ms"),
                    "inference_p95_ms": metrics.get("inference_p95_ms"),
                    "epochs": summary.get("epochs", metrics.get("epochs_completed")),
                    "device": summary.get("device"),
                    "device_name": summary.get("device_name"),
                    "train_seconds": summary.get("train_seconds"),
                    "wall_clock_seconds": wall_clock,
                    "checkpoint_sha": summary.get("checkpoint_sha"),
                    "attempt_count": run.attempt_count,
                    "created_at": run.created_at.isoformat(),
                    "started_at": run.started_at.isoformat() if run.started_at else None,
                    "finished_at": run.finished_at.isoformat() if run.finished_at else None,
                    "artifact_count": len(artifacts),
                    "artifact_bytes": sum(artifact.size for artifact in artifacts),
                }
            )

        run_records.sort(key=lambda record: record["val_accuracy"] or 0, reverse=True)
        best = run_records[0]
        best_run_id = best["run_id"]

        eval_runs = list(
            session.execute(
                select(Run)
                .where(
                    Run.experiment_id == experiment.id,
                    Run.job_kind == "eval",
                    Run.status == RunStatus.SUCCEEDED,
                )
                .order_by(Run.created_at.asc())
            ).scalars()
        )
        eval_run = eval_runs[-1] if eval_runs else None
        eval_report = (
            _artifact_json(session, store, eval_run.id, "eval_report")
            if eval_run is not None
            else None
        )
        eval_summary: dict[str, Any] | None = None
        if eval_run is not None and eval_report is not None:
            eval_summary = {
                "run_id": str(eval_run.id),
                "wall_clock_seconds": round(
                    (eval_run.finished_at - eval_run.started_at).total_seconds(), 3
                )
                if eval_run.started_at and eval_run.finished_at
                else None,
                "checkpoint_sha": eval_report.get("checkpoint_sha"),
                "checkpoint_run_id": eval_report.get("checkpoint_run_id"),
                "accuracy": eval_report.get("accuracy"),
                "macro_precision": eval_report.get("macro_precision"),
                "macro_recall": eval_report.get("macro_recall"),
                "macro_f1": eval_report.get("macro_f1"),
                "test_loss": eval_report.get("test_loss"),
                "inference_p50_ms": eval_report.get("inference_p50_ms"),
                "inference_p95_ms": eval_report.get("inference_p95_ms"),
                "num_test_samples": eval_report.get("num_test_samples"),
                "batch_size": eval_report.get("batch_size"),
                "device": eval_report.get("device"),
                "per_class": eval_report.get("per_class"),
                "confusion_matrix": eval_report.get("confusion_matrix"),
                "eval_seconds": eval_report.get("eval_seconds"),
            }

        all_run_ids = [run.id for run in completed] + (
            [eval_run.id] if eval_run is not None else []
        )
        artifacts = list(
            session.execute(
                select(Artifact).where(Artifact.run_id.in_(all_run_ids))
            ).scalars()
        )
        by_kind: dict[str, int] = {}
        for artifact in artifacts:
            by_kind[artifact.kind] = by_kind.get(artifact.kind, 0) + 1
        stored_bytes = sum(artifact.size for artifact in artifacts)
        attempted_bytes = sum(artifact.size * artifact.ref_count for artifact in artifacts)
        bytes_saved = sum(artifact.bytes_saved for artifact in artifacts)
        lineage_sha = best["checkpoint_sha"]

        payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "generated_by": "scripts/export_results.py",
            "build_git_sha": manifest.get("git_sha"),
            "experiment": {
                "id": str(experiment.id),
                "name": experiment.name,
                "description": experiment.description,
            },
            "environment": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "mps_available": torch.backends.mps.is_available(),
                "devices_used": sorted(
                    {record["device"] for record in run_records if record["device"]}
                ),
            },
            "dataset_snapshot": {
                "id": str(snapshot.id),
                "version": snapshot.version,
                "content_hash": snapshot.content_hash,
                "row_count": snapshot.row_count,
                "validated": snapshot.validated,
                "seed": snapshot.manifest.get("seed"),
                "mean": snapshot.manifest.get("mean"),
                "std": snapshot.manifest.get("std"),
                "splits": snapshot.manifest.get("splits"),
                "class_histogram": snapshot.manifest.get("class_histogram"),
            },
            "runs": run_records,
            "completed_runs": len(completed),
            "best_run": {
                "run_id": best_run_id,
                "val_accuracy": best["val_accuracy"],
                "test_accuracy": best["test_accuracy"],
                "checkpoint_sha": best["checkpoint_sha"],
            },
            "eval_on_best": eval_summary,
            "artifacts": {
                "unique_artifacts": len(artifacts),
                "stored_bytes": stored_bytes,
                "attempted_bytes": attempted_bytes,
                "bytes_saved": bytes_saved,
                "dedupe_savings_mb": round(bytes_saved / (1024 * 1024), 6),
                "dedupe_ratio": round(bytes_saved / attempted_bytes, 6)
                if attempted_bytes
                else 0.0,
                "by_kind": by_kind,
                "store_stats": store.stats(),
            },
            "lineage": {
                "best_checkpoint_sha": lineage_sha,
                "best_checkpoint_chain_length": _lineage_length(session, lineage_sha),
                "eval_report_parents_checkpoint": (
                    eval_summary is not None
                    and eval_summary.get("checkpoint_sha") == lineage_sha
                ),
            },
        }
    finally:
        session.close()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    print(f"wrote {OUTPUT_PATH}")
    print(
        f"runs={payload['completed_runs']} best_val="
        f"{payload['best_run']['val_accuracy']} artifacts="
        f"{payload['artifacts']['unique_artifacts']} saved_bytes="
        f"{payload['artifacts']['bytes_saved']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
