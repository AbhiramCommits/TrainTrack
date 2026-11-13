"""Real PyTorch MNIST training job driven by a registered run.

The job loads its hyperparameters and dataset snapshot from the run registry,
refuses unvalidated or hash-mismatched snapshots, trains a small CNN, posts
per-step loss and per-epoch accuracy to the API, stores each best-validation
checkpoint in the content-addressed artifact store, and evaluates the held-out
test split.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import torch

from traintrack.artifacts.store import ArtifactStore
from traintrack.db import get_session, init_db
from traintrack.models.artifact import Artifact
from traintrack.models.run import Run
from traintrack.models.snapshot import DatasetSnapshot
from traintrack.models.state import RunStatus
from traintrack.pipeline.validate import SnapshotValidationError, load_validated_snapshot
from traintrack.training.checkpoint import save_checkpoint
from traintrack.training.client import ApiClient
from traintrack.training.device import DeviceInfo, sample_resources, seed_everything, select_device
from traintrack.training.evaluation import evaluate_model, make_loader
from traintrack.training.model import build_model

logger = logging.getLogger("traintrack.training")
METRIC_FLUSH_EVERY = 40
RESOURCE_FLUSH_EVERY = 8


class Reporter:
    """Buffers metric and resource samples and posts them in batches."""

    def __init__(self, client: ApiClient, run_id: str) -> None:
        self.client = client
        self.run_id = run_id
        self.metrics: list[dict[str, Any]] = []
        self.resources: list[dict[str, Any]] = []

    def metric(self, name: str, value: float, step: int) -> None:
        self.metrics.append(
            {"name": name, "step": int(step), "value": float(value), "wall_time": time.time()}
        )
        if len(self.metrics) >= METRIC_FLUSH_EVERY:
            self.flush_metrics()

    def resource(self, sample: dict[str, Any]) -> None:
        self.resources.append(sample)
        if len(self.resources) >= RESOURCE_FLUSH_EVERY:
            self.flush_resources()

    def flush_metrics(self) -> None:
        if self.metrics:
            self.client.post_metrics(self.run_id, self.metrics)
            self.metrics = []

    def flush_resources(self) -> None:
        if self.resources:
            self.client.post_resources(self.run_id, self.resources)
            self.resources = []

    def flush(self) -> None:
        self.flush_metrics()
        self.flush_resources()


def _load_run(run_id: uuid.UUID) -> tuple[Run, DatasetSnapshot | None]:
    session = get_session()
    try:
        run = session.get(Run, run_id)
        if run is None:
            raise SystemExit(f"run {run_id} not found")
        if run.status is not RunStatus.RUNNING:
            raise SystemExit(f"run {run_id} is {run.status.value}, expected RUNNING")
        snapshot = (
            session.get(DatasetSnapshot, run.dataset_snapshot_id)
            if run.dataset_snapshot_id
            else None
        )
        if run.job_kind != "train":
            raise SystemExit(f"run {run_id} has job kind {run.job_kind}, expected train")
        if snapshot is None:
            raise SnapshotValidationError("training run has no dataset snapshot")
        if not snapshot.validated:
            raise SnapshotValidationError(
                f"snapshot {snapshot.version} is not validated; refusing to train"
            )
        return run, snapshot
    finally:
        session.close()


def _store_artifact(
    path: Path,
    *,
    kind: str,
    run_id: uuid.UUID,
    snapshot_id: uuid.UUID | None,
    parent_sha: str | None,
) -> Artifact:
    session = get_session()
    try:
        artifact = ArtifactStore().write_file(
            path,
            kind=kind,
            session=session,
            run_id=run_id,
            dataset_snapshot_id=snapshot_id,
            parent_sha=parent_sha,
        )
        session.commit()
        return artifact
    finally:
        session.close()


def _save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    """Write a deterministic checkpoint so identical states dedupe exactly."""
    save_checkpoint(path, payload)


def _checkpoint_payload(
    model: torch.nn.Module,
    *,
    run: Run,
    snapshot: DatasetSnapshot,
    hyperparams: dict[str, Any],
    device: DeviceInfo,
    epoch: int,
    val_accuracy: float,
    class_names: list[str],
) -> dict[str, Any]:
    return {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_config": {"dropout": float(hyperparams.get("dropout", 0.0)), "num_classes": 10},
        "hyperparams": hyperparams,
        "run_id": str(run.id),
        "experiment_id": str(run.experiment_id),
        "snapshot_id": str(snapshot.id),
        "snapshot_version": snapshot.version,
        "snapshot_hash": snapshot.content_hash,
        "class_names": class_names,
        "device": device.backend,
        "device_name": device.name,
        "epoch": epoch,
        "val_accuracy": val_accuracy,
        "git_sha": run.git_sha,
    }


def main(argv: list[str] | None = None) -> int:
    """Train one run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--metric-every", type=int, default=20)
    parser.add_argument("--resource-every", type=int, default=40)
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level="INFO", format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    init_db()
    run_id = uuid.UUID(args.run_id)
    try:
        run, snapshot = _load_run(run_id)
        assert snapshot is not None
        manifest, splits = load_validated_snapshot(snapshot.path)
    except (SnapshotValidationError, SystemExit) as exc:
        logger.error("refusing to train: %s", exc)
        return 2

    hyperparams = dict(run.hyperparams)
    hyperparams.setdefault("epochs", 3)
    hyperparams.setdefault("dropout", 0.0)
    hyperparams.setdefault("optimizer", "adam")
    hyperparams.setdefault("seed", 42)
    hyperparams.setdefault("momentum", 0.0)
    hyperparams.setdefault("weight_decay", 0.0)
    hyperparams.setdefault("label_smoothing", 0.0)
    class_names = list(manifest.get("class_names", [str(i) for i in range(10)]))

    device_info = select_device(args.device)
    seed_everything(int(hyperparams["seed"]))
    logger.info(
        "run %s: device=%s (%s) lr=%s batch=%s epochs=%s optimizer=%s",
        run_id,
        device_info.backend,
        device_info.name,
        hyperparams["lr"],
        hyperparams["batch_size"],
        hyperparams["epochs"],
        hyperparams["optimizer"],
    )

    batch_size = int(hyperparams["batch_size"])
    train_loader = make_loader(
        splits["train"][0],
        splits["train"][1],
        batch_size=batch_size,
        shuffle=True,
        seed=int(hyperparams["seed"]),
        max_batches=args.max_batches,
    )
    val_loader = make_loader(
        splits["val"][0],
        splits["val"][1],
        batch_size=batch_size,
        max_batches=args.max_batches,
    )
    test_loader = make_loader(
        splits["test"][0],
        splits["test"][1],
        batch_size=batch_size,
        max_batches=args.max_batches,
    )

    torch_device = device_info.device
    model = build_model(
        dropout=float(hyperparams["dropout"]), num_classes=len(class_names)
    ).to(torch_device)
    optimizer_name = str(hyperparams["optimizer"])
    optimizer: torch.optim.Optimizer
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(hyperparams["lr"]),
            momentum=float(hyperparams["momentum"]),
            weight_decay=float(hyperparams["weight_decay"]),
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(hyperparams["lr"]),
            weight_decay=float(hyperparams["weight_decay"]),
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(hyperparams["lr"]),
            weight_decay=float(hyperparams["weight_decay"]),
        )
    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=float(hyperparams["label_smoothing"])
    )

    client = ApiClient(args.api_url, args.api_key)
    reporter = Reporter(client, str(run_id))
    checkpoint_dir = Path(tempfile.gettempdir()) / "traintrack-checkpoints" / str(run_id)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_accuracy = -1.0
    best_sha: str | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    last_val_accuracy = 0.0
    global_step = 0
    epochs_completed = 0
    started = time.monotonic()
    logger.info("snapshot %s hash=%s", snapshot.version, snapshot.content_hash)

    for epoch in range(1, int(hyperparams["epochs"]) + 1):
        model.train()
        epoch_started = time.monotonic()
        running_loss = 0.0
        correct = 0
        seen = 0
        for images, labels in train_loader:
            global_step += 1
            images = images.to(torch_device)
            labels = labels.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            step_loss = float(loss.detach().cpu())
            running_loss += step_loss * len(labels)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            seen += len(labels)

            if global_step % args.metric_every == 0:
                reporter.metric("train_loss", step_loss, global_step)
            if global_step % args.resource_every == 0:
                reporter.resource(sample_resources(device_info, global_step))

        train_accuracy = correct / max(1, seen)
        val_output = evaluate_model(
            model,
            val_loader,
            torch_device,
            criterion=torch.nn.CrossEntropyLoss(),
            class_names=class_names,
            max_batches=args.max_batches,
        )
        val_accuracy = float(val_output.report["accuracy"])
        last_val_accuracy = val_accuracy
        epoch_seconds = time.monotonic() - epoch_started
        epochs_completed = epoch
        reporter.metric("train_accuracy", train_accuracy, epoch)
        reporter.metric("val_accuracy", val_accuracy, epoch)
        reporter.metric("val_loss", val_output.loss, epoch)
        reporter.metric("train_loss_epoch", running_loss / max(1, seen), epoch)
        reporter.metric("lr", float(hyperparams["lr"]), epoch)
        reporter.metric("epoch_seconds", epoch_seconds, epoch)
        reporter.resource(sample_resources(device_info, global_step))
        reporter.flush()

        logger.info(
            "epoch %d/%d train_acc=%.4f val_acc=%.4f val_loss=%.4f %.1fs",
            epoch,
            int(hyperparams["epochs"]),
            train_accuracy,
            val_accuracy,
            val_output.loss,
            epoch_seconds,
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            payload = _checkpoint_payload(
                model,
                run=run,
                snapshot=snapshot,
                hyperparams=hyperparams,
                device=device_info,
                epoch=epoch,
                val_accuracy=val_accuracy,
                class_names=class_names,
            )
            local_checkpoint = checkpoint_dir / f"epoch-{epoch:03d}.pt"
            _save_checkpoint(payload, local_checkpoint)
            artifact = _store_artifact(
                local_checkpoint,
                kind="checkpoint",
                run_id=run_id,
                snapshot_id=snapshot.id,
                parent_sha=best_sha,
            )
            best_sha = artifact.sha256
            local_checkpoint.unlink(missing_ok=True)
            logger.info(
                "new best checkpoint epoch=%d val_acc=%.4f sha=%s",
                epoch,
                val_accuracy,
                best_sha,
            )

    train_seconds = time.monotonic() - started
    if best_state is None or best_sha is None:
        logger.error("no checkpoint was produced")
        return 1

    # Persist the final-epoch checkpoint as well. When the last epoch was also
    # the best epoch this write is byte-identical and is deduped by the store.
    last_payload = _checkpoint_payload(
        model,
        run=run,
        snapshot=snapshot,
        hyperparams=hyperparams,
        device=device_info,
        epoch=epochs_completed,
        val_accuracy=last_val_accuracy,
        class_names=class_names,
    )
    last_checkpoint = checkpoint_dir / f"last-epoch-{epochs_completed:03d}.pt"
    _save_checkpoint(last_payload, last_checkpoint)
    _store_artifact(
        last_checkpoint,
        kind="checkpoint_last",
        run_id=run_id,
        snapshot_id=snapshot.id,
        parent_sha=best_sha,
    )
    last_checkpoint.unlink(missing_ok=True)

    model.load_state_dict(best_state)
    test_output = evaluate_model(
        model,
        test_loader,
        torch_device,
        criterion=torch.nn.CrossEntropyLoss(),
        class_names=class_names,
        measure_latency=True,
        max_batches=args.max_batches,
    )

    eval_report = {
        "run_id": str(run_id),
        "job_kind": "train",
        "checkpoint_sha": best_sha,
        "checkpoint_epoch": best_epoch,
        "dataset_snapshot_id": str(snapshot.id),
        "dataset_snapshot_version": snapshot.version,
        "dataset_snapshot_hash": snapshot.content_hash,
        "device": device_info.backend,
        "device_name": device_info.name,
        "hyperparams": hyperparams,
        "best_val_accuracy": best_val_accuracy,
        "test_loss": test_output.loss,
        "accuracy": test_output.report["accuracy"],
        "macro_precision": test_output.report["macro_precision"],
        "macro_recall": test_output.report["macro_recall"],
        "macro_f1": test_output.report["macro_f1"],
        "confusion_matrix": test_output.report["confusion_matrix"],
        "per_class": test_output.report["per_class"],
        "inference_p50_ms": test_output.latency["p50_ms"],
        "inference_p95_ms": test_output.latency["p95_ms"],
        "num_test_samples": test_output.report["num_samples"],
        "train_seconds": round(train_seconds, 3),
        "epochs": epochs_completed,
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    report_path = checkpoint_dir / "eval_report.json"
    report_path.write_text(json.dumps(eval_report, indent=2))
    report_artifact = _store_artifact(
        report_path,
        kind="eval_report",
        run_id=run_id,
        snapshot_id=snapshot.id,
        parent_sha=best_sha,
    )

    summary = {
        "run_id": str(run_id),
        "device": device_info.backend,
        "device_name": device_info.name,
        "epochs": epochs_completed,
        "train_seconds": round(train_seconds, 3),
        "best_val_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "checkpoint_sha": best_sha,
        "eval_report_sha": report_artifact.sha256,
        "test_accuracy": test_output.report["accuracy"],
        "test_macro_f1": test_output.report["macro_f1"],
        "inference_p50_ms": test_output.latency["p50_ms"],
        "inference_p95_ms": test_output.latency["p95_ms"],
        "snapshot_id": str(snapshot.id),
        "snapshot_version": snapshot.version,
        "snapshot_hash": snapshot.content_hash,
        "hyperparams": hyperparams,
        "git_sha": run.git_sha,
    }
    summary_path = checkpoint_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    _store_artifact(
        summary_path,
        kind="train_summary",
        run_id=run_id,
        snapshot_id=snapshot.id,
        parent_sha=best_sha,
    )

    reporter.metric("test_accuracy", test_output.report["accuracy"], epochs_completed)
    reporter.metric("test_macro_f1", test_output.report["macro_f1"], epochs_completed)
    reporter.metric(
        "inference_p50_ms", test_output.latency["p50_ms"], epochs_completed
    )
    reporter.metric(
        "inference_p95_ms", test_output.latency["p95_ms"], epochs_completed
    )
    reporter.metric("best_val_accuracy", best_val_accuracy, epochs_completed)
    reporter.metric("train_time_seconds", train_seconds, epochs_completed)
    reporter.metric("epochs_completed", float(epochs_completed), epochs_completed)
    reporter.flush()

    for path in checkpoint_dir.glob("*.json"):
        path.unlink(missing_ok=True)
    print("TRAIN_SUMMARY " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
