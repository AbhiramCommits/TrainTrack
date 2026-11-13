"""Build a deterministic, content-hashed MNIST snapshot.

Output layout (``data/snapshots/<version>/``): six ``.npy`` shards plus
``class_names.json`` and ``manifest.json``. The content hash is sha256 over the
sorted ``filename:digest`` lines of those shards and class names, so it changes
if any byte of the dataset changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select

from traintrack.config import settings
from traintrack.db import get_session, init_db
from traintrack.models.snapshot import DatasetSnapshot
from traintrack.pipeline.validate import (
    SnapshotValidationError,
    validate_snapshot_dir,
)

SOURCE = "torchvision MNIST (https://ossci-datasets.s3.amazonaws.com/mnist/)"
CLASS_NAMES = [str(index) for index in range(10)]
DEFAULT_VERSION = "mnist-v1"
FILE_NAMES = [
    "train_images.npy",
    "train_labels.npy",
    "val_images.npy",
    "val_labels.npy",
    "test_images.npy",
    "test_labels.npy",
    "class_names.json",
]


def sha256_file(path: Path) -> str:
    """Return a file's sha256 hex digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash(directory: Path, file_names: list[str] | None = None) -> str:
    """Hash the sorted ``name:digest`` lines of the snapshot's data files."""
    names = file_names or FILE_NAMES
    digest = hashlib.sha256()
    for name in sorted(names):
        digest.update(f"{name}:{sha256_file(directory / name)}\n".encode())
    return digest.hexdigest()


def _class_histogram(labels: np.ndarray) -> dict[str, int]:
    counts = np.bincount(labels.astype(np.int64), minlength=10)
    return {str(index): int(count) for index, count in enumerate(counts)}


def _load_mnist(data_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from torchvision.datasets import MNIST

    raw_root = data_root / "raw"
    train = MNIST(root=str(raw_root), train=True, download=True)
    test = MNIST(root=str(raw_root), train=False, download=True)
    train_images = train.data.numpy().astype(np.float32) / 255.0
    train_labels = train.targets.numpy().astype(np.int64)
    test_images = test.data.numpy().astype(np.float32) / 255.0
    test_labels = test.targets.numpy().astype(np.int64)
    return train_images, train_labels, test_images, test_labels


def build_snapshot(
    version: str = DEFAULT_VERSION,
    *,
    seed: int = 42,
    val_fraction: float = 0.1,
    data_root: Path | None = None,
    force: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Build the snapshot files and return (directory, manifest)."""
    root = Path(data_root or settings().data_root)
    snapshot_dir = root / "snapshots" / version
    manifest_path = snapshot_dir / "manifest.json"
    if snapshot_dir.exists() and manifest_path.exists() and not force:
        existing_manifest = json.loads(manifest_path.read_text())
        actual = content_hash(snapshot_dir)
        if actual != existing_manifest.get("content_hash"):
            raise SnapshotValidationError(
                f"existing snapshot {version} has a hash mismatch; use --force or a new version"
            )
        return snapshot_dir, existing_manifest

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    images, labels, test_images, test_labels = _load_mnist(root)

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(images))
    val_size = int(len(images) * val_fraction)
    val_indices = permutation[:val_size]
    train_indices = permutation[val_size:]

    train_images, train_labels = images[train_indices], labels[train_indices]
    val_images, val_labels = images[val_indices], labels[val_indices]

    mean = float(train_images.mean())
    std = float(train_images.std())
    if std <= 0:
        raise SnapshotValidationError("degenerate dataset: zero pixel standard deviation")

    def normalize(array: np.ndarray) -> np.ndarray:
        return ((array - mean) / std).astype(np.float32)

    train_images = normalize(train_images)[:, None, :, :]
    val_images = normalize(val_images)[:, None, :, :]
    test_images = normalize(test_images)[:, None, :, :]

    splits = {
        "train": (train_images, train_labels),
        "val": (val_images, val_labels),
        "test": (test_images, test_labels),
    }
    for split_name, (split_images, split_labels) in splits.items():
        np.save(snapshot_dir / f"{split_name}_images.npy", split_images)
        np.save(snapshot_dir / f"{split_name}_labels.npy", split_labels)

    (snapshot_dir / "class_names.json").write_text(json.dumps(CLASS_NAMES) + "\n")

    files = {
        name: {
            "sha256": sha256_file(snapshot_dir / name),
            "bytes": (snapshot_dir / name).stat().st_size,
        }
        for name in FILE_NAMES
    }
    digest = content_hash(snapshot_dir)

    manifest: dict[str, Any] = {
        "version": version,
        "format": "numpy-npy-v1",
        "source": SOURCE,
        "license": "MNIST is distributed under the terms of the original NIST dataset",
        "seed": seed,
        "val_fraction": val_fraction,
        "created_at": datetime.now(UTC).isoformat(),
        "content_hash": digest,
        "normalization": {"mean": mean, "std": std},
        "mean": mean,
        "std": std,
        "class_names": CLASS_NAMES,
        "row_count": int(
            len(train_labels) + len(val_labels) + len(test_labels)
        ),
        "splits": {
            name: {
                "rows": int(len(split_labels)),
                "shape": list(split_images.shape),
                "dtype": str(split_images.dtype),
                "class_histogram": _class_histogram(split_labels),
            }
            for name, (split_images, split_labels) in splits.items()
        },
        "class_histogram": _class_histogram(labels),
        "files": files,
        "validated": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return snapshot_dir, manifest


def register_snapshot(
    snapshot_dir: Path, manifest: dict[str, Any], *, validated: bool
) -> DatasetSnapshot:
    """Insert or update the registry row for a built snapshot."""
    session = get_session()
    try:
        existing = session.execute(
            select(DatasetSnapshot).where(
                DatasetSnapshot.version == manifest["version"]
            )
        ).scalar_one_or_none()
        if existing is not None:
            if existing.content_hash != manifest["content_hash"]:
                raise SnapshotValidationError(
                    f"snapshot version {manifest['version']} is immutable and already "
                    f"registered with hash {existing.content_hash[:12]}"
                )
            existing.validated = validated
            existing.manifest = manifest
            existing.path = str(snapshot_dir)
            session.commit()
            return existing
        snapshot = DatasetSnapshot(
            version=manifest["version"],
            content_hash=manifest["content_hash"],
            row_count=int(manifest["row_count"]),
            manifest=manifest,
            validated=validated,
            path=str(snapshot_dir),
        )
        session.add(snapshot)
        session.commit()
        return snapshot
    finally:
        session.close()


def build_and_validate(
    version: str = DEFAULT_VERSION, *, force: bool = False
) -> dict[str, Any]:
    """Build the snapshot, run the validation gate, and register it."""
    snapshot_dir, manifest = build_snapshot(version, force=force)
    report = validate_snapshot_dir(snapshot_dir)
    manifest["validated"] = True
    manifest["validation"] = report
    (snapshot_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (snapshot_dir / "validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    snapshot = register_snapshot(snapshot_dir, manifest, validated=True)
    return {
        "id": str(snapshot.id),
        "version": snapshot.version,
        "content_hash": snapshot.content_hash,
        "row_count": snapshot.row_count,
        "path": snapshot.path,
        "validated": snapshot.validated,
        "splits": manifest["splits"],
        "content_hash_recomputed": content_hash(snapshot_dir),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m traintrack.pipeline.preprocess``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    init_db()
    result = build_and_validate(args.version, force=args.force)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
