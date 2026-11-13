"""Dataset snapshot validation and training refusal tests."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import numpy as np
import pytest
from traintrack.db import get_session
from traintrack.models.run import Run
from traintrack.models.snapshot import DatasetSnapshot
from traintrack.models.state import RunStatus
from traintrack.pipeline.preprocess import content_hash, sha256_file
from traintrack.pipeline.validate import (
    SnapshotValidationError,
    load_validated_snapshot,
    validate_snapshot_dir,
)
from traintrack.training.train import _load_run

FILE_NAMES = [
    "train_images.npy",
    "train_labels.npy",
    "val_images.npy",
    "val_labels.npy",
    "test_images.npy",
    "test_labels.npy",
    "class_names.json",
]


def _write_snapshot(directory: Path, *, rows=(1000, 200, 200), labels_all_zero=False) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    splits = {}
    for name, count in zip(("train", "val", "test"), rows, strict=True):
        images = rng.normal(size=(count, 1, 28, 28)).astype(np.float32)
        labels = (
            np.zeros(count, dtype=np.int64)
            if labels_all_zero
            else np.arange(count, dtype=np.int64) % 10
        )
        np.save(directory / f"{name}_images.npy", images)
        np.save(directory / f"{name}_labels.npy", labels)
        splits[name] = (images, labels)
    (directory / "class_names.json").write_text(json.dumps([str(i) for i in range(10)]))

    def histogram(labels):
        counts = np.bincount(labels, minlength=10)
        return {str(i): int(c) for i, c in enumerate(counts)}

    manifest = {
        "version": "synthetic-v1",
        "seed": 7,
        "content_hash": "",
        "mean": float(np.mean([s[0].mean() for s in splits.values()])),
        "std": float(np.mean([s[0].std() for s in splits.values()])),
        "row_count": sum(len(s[1]) for s in splits.values()),
        "normalization": {"mean": 0.0, "std": 1.0},
        "class_names": [str(i) for i in range(10)],
        "splits": {
            name: {
                "rows": int(len(labels)),
                "shape": list(images.shape),
                "dtype": str(images.dtype),
                "class_histogram": histogram(labels),
            }
            for name, (images, labels) in splits.items()
        },
        "files": {
            name: {"sha256": sha256_file(directory / name), "bytes": (directory / name).stat().st_size}
            for name in FILE_NAMES
        },
    }
    manifest["content_hash"] = content_hash(directory)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def test_valid_synthetic_snapshot_passes(tmp_path: Path) -> None:
    directory = _write_snapshot(tmp_path / "snap")
    report = validate_snapshot_dir(directory)
    assert report["valid"] is True
    assert report["row_count"] == 1400
    assert report["splits"]["train"]["rows"] == 1000
    manifest, splits = load_validated_snapshot(directory)
    assert manifest["version"] == "synthetic-v1"
    assert splits["test"][0].shape == (200, 1, 28, 28)


def test_tampered_file_is_rejected(tmp_path: Path) -> None:
    directory = _write_snapshot(tmp_path / "snap")
    target = directory / "test_images.npy"
    data = bytearray(target.read_bytes())
    data[100] = data[100] ^ 0xFF
    target.write_bytes(bytes(data))
    with pytest.raises(SnapshotValidationError):
        validate_snapshot_dir(directory)
    with pytest.raises(SnapshotValidationError):
        load_validated_snapshot(directory)


def test_manifest_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    directory = _write_snapshot(tmp_path / "snap")
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["content_hash"] = hashlib.sha256(b"wrong").hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SnapshotValidationError, match="content hash"):
        validate_snapshot_dir(directory)


def test_nan_values_are_rejected(tmp_path: Path) -> None:
    directory = _write_snapshot(tmp_path / "snap")
    images = np.load(directory / "train_images.npy")
    images[0, 0, 0, 0] = np.nan
    np.save(directory / "train_images.npy", images)
    manifest = json.loads((directory / "manifest.json").read_text())
    name = "train_images.npy"
    manifest["files"][name] = {
        "sha256": sha256_file(directory / name),
        "bytes": (directory / name).stat().st_size,
    }
    manifest["content_hash"] = content_hash(directory)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SnapshotValidationError, match="NaN"):
        validate_snapshot_dir(directory)


def test_class_imbalance_is_rejected(tmp_path: Path) -> None:
    directory = _write_snapshot(tmp_path / "snap", labels_all_zero=True)
    with pytest.raises(SnapshotValidationError, match="balance"):
        validate_snapshot_dir(directory)


def test_training_refuses_unvalidated_snapshot(app) -> None:
    session = get_session()
    try:
        from traintrack.models.experiment import Experiment

        experiment = Experiment(name=f"refuse-{uuid.uuid4().hex[:8]}")
        session.add(experiment)
        session.flush()
        snapshot = DatasetSnapshot(
            version=f"unvalidated-{uuid.uuid4().hex[:8]}",
            content_hash="a" * 64,
            row_count=10,
            manifest={},
            validated=False,
            path="/tmp/none",
        )
        session.add(snapshot)
        session.flush()
        run = Run(
            experiment_id=experiment.id,
            status=RunStatus.RUNNING,
            job_kind="train",
            hyperparams={"lr": 0.01, "batch_size": 8},
            dataset_snapshot_id=snapshot.id,
        )
        session.add(run)
        session.commit()
        run_id = run.id
    finally:
        session.close()

    with pytest.raises(SnapshotValidationError, match="not validated"):
        _load_run(run_id)


def test_training_refuses_hash_mismatch(tmp_path: Path) -> None:
    directory = _write_snapshot(tmp_path / "snap")
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["content_hash"] = "b" * 64
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SnapshotValidationError):
        load_validated_snapshot(directory)
