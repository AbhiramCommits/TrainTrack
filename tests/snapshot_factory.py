"""Synthetic snapshot factory for fast end-to-end training tests."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
from traintrack.db import get_session
from traintrack.models.snapshot import DatasetSnapshot
from traintrack.pipeline.preprocess import content_hash, sha256_file

FILE_NAMES = [
    "train_images.npy",
    "train_labels.npy",
    "val_images.npy",
    "val_labels.npy",
    "test_images.npy",
    "test_labels.npy",
    "class_names.json",
]


def write_synthetic_snapshot(
    directory: Path, *, rows: tuple[int, int, int] = (1000, 200, 200), seed: int = 11
) -> tuple[Path, dict]:
    """Write a valid, content-hashed snapshot with random MNIST-shaped tensors."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    splits = {}
    for name, count in zip(("train", "val", "test"), rows, strict=True):
        images = rng.normal(size=(count, 1, 28, 28)).astype(np.float32)
        labels = np.arange(count, dtype=np.int64) % 10
        np.save(directory / f"{name}_images.npy", images)
        np.save(directory / f"{name}_labels.npy", labels)
        splits[name] = (images, labels)
    (directory / "class_names.json").write_text(json.dumps([str(index) for index in range(10)]))

    def histogram(labels: np.ndarray) -> dict[str, int]:
        counts = np.bincount(labels, minlength=10)
        return {str(index): int(count) for index, count in enumerate(counts)}

    train_mean = float(splits["train"][0].mean())
    train_std = float(splits["train"][0].std())
    manifest = {
        "version": f"synthetic-{uuid.uuid4().hex[:8]}",
        "seed": seed,
        "content_hash": "",
        "mean": train_mean,
        "std": train_std,
        "normalization": {"mean": train_mean, "std": train_std},
        "row_count": sum(len(split[1]) for split in splits.values()),
        "class_names": [str(index) for index in range(10)],
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
            name: {
                "sha256": sha256_file(directory / name),
                "bytes": (directory / name).stat().st_size,
            }
            for name in FILE_NAMES
        },
        "validated": True,
    }
    manifest["content_hash"] = content_hash(directory)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory, manifest


def register_synthetic_snapshot(directory: Path, manifest: dict) -> uuid.UUID:
    """Insert a validated snapshot row and return its id."""
    session = get_session()
    try:
        snapshot = DatasetSnapshot(
            version=manifest["version"],
            content_hash=manifest["content_hash"],
            row_count=int(manifest["row_count"]),
            manifest=manifest,
            validated=True,
            path=str(directory),
        )
        session.add(snapshot)
        session.commit()
        return snapshot.id
    finally:
        session.close()
