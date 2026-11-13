"""Snapshot validation gate.

Training calls :func:`load_validated_snapshot`, which refuses to return data
when the manifest, checksum, shapes, value ranges or class balance are wrong.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

MIN_ROWS = {"train": 1000, "val": 100, "test": 100}
CLASS_BALANCE_TOLERANCE = 0.5
EXPECTED_DTYPE = np.float32
IMAGE_SHAPE = (28, 28)


class SnapshotValidationError(Exception):
    """Raised when a snapshot fails integrity or schema checks."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_content_hash(directory: Path, files: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(f"{name}:{_sha256_file(directory / name)}\n".encode())
    return digest.hexdigest()


def validate_snapshot_dir(directory: Path | str) -> dict[str, Any]:
    """Validate a snapshot directory and return a report.

    Raises :class:`SnapshotValidationError` on any integrity failure.
    """
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise SnapshotValidationError(f"manifest.json missing in {directory}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise SnapshotValidationError(f"manifest is not valid JSON: {exc}") from exc

    required_keys = {
        "version",
        "content_hash",
        "seed",
        "splits",
        "files",
        "mean",
        "std",
        "row_count",
    }
    missing = required_keys - set(manifest)
    if missing:
        raise SnapshotValidationError(f"manifest missing keys: {sorted(missing)}")

    files = manifest["files"]
    for name, metadata in files.items():
        path = directory / name
        if not path.exists():
            raise SnapshotValidationError(f"missing data file: {name}")
        if path.stat().st_size != int(metadata["bytes"]):
            raise SnapshotValidationError(f"size mismatch for {name}")
        actual = _sha256_file(path)
        if actual != metadata["sha256"]:
            raise SnapshotValidationError(f"sha256 mismatch for {name}")
    recomputed = _expected_content_hash(directory, files)
    if recomputed != manifest["content_hash"]:
        raise SnapshotValidationError("dataset content hash does not match manifest")

    split_reports: dict[str, Any] = {}
    total_rows = 0
    for split_name in ("train", "val", "test"):
        split_meta = manifest["splits"].get(split_name)
        if not split_meta:
            raise SnapshotValidationError(f"missing split: {split_name}")
        images_path = directory / f"{split_name}_images.npy"
        labels_path = directory / f"{split_name}_labels.npy"
        if not images_path.exists() or not labels_path.exists():
            raise SnapshotValidationError(f"missing arrays for split {split_name}")
        images = np.load(images_path, mmap_mode="r")
        labels = np.load(labels_path, mmap_mode="r")
        if images.dtype != EXPECTED_DTYPE:
            raise SnapshotValidationError(
                f"{split_name} images dtype is {images.dtype}, expected {EXPECTED_DTYPE}"
            )
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, *IMAGE_SHAPE):
            raise SnapshotValidationError(
                f"{split_name} image shape {images.shape} is not (N, 1, 28, 28)"
            )
        if labels.ndim != 1 or len(labels) != len(images):
            raise SnapshotValidationError(f"{split_name} label/image length mismatch")
        if len(labels) < MIN_ROWS[split_name]:
            raise SnapshotValidationError(
                f"{split_name} split has {len(labels)} rows, below floor "
                f"{MIN_ROWS[split_name]}"
            )
        if not np.isfinite(images).all():
            raise SnapshotValidationError(f"{split_name} images contain NaN or Inf")
        if labels.min() < 0 or labels.max() > 9:
            raise SnapshotValidationError(f"{split_name} labels outside [0, 9]")

        counts = np.bincount(labels.astype(np.int64), minlength=10)
        histogram = {str(index): int(count) for index, count in enumerate(counts)}
        expected_histogram = split_meta.get("class_histogram")
        if expected_histogram != histogram:
            raise SnapshotValidationError(
                f"{split_name} class histogram does not match manifest"
            )
        expected = len(labels) / 10.0
        lowest, highest = int(counts.min()), int(counts.max())
        if lowest < expected * (1 - CLASS_BALANCE_TOLERANCE) or highest > expected * (
            1 + CLASS_BALANCE_TOLERANCE
        ):
            raise SnapshotValidationError(
                f"{split_name} class balance outside tolerance: min={lowest}, max={highest}"
            )
        if int(split_meta["rows"]) != len(labels):
            raise SnapshotValidationError(f"{split_name} row count mismatch")
        total_rows += len(labels)
        split_reports[split_name] = {
            "rows": int(len(labels)),
            "shape": list(images.shape),
            "class_histogram": histogram,
        }

    if total_rows != int(manifest["row_count"]):
        raise SnapshotValidationError("total row count does not match manifest")
    if not np.isfinite([manifest["mean"], manifest["std"]]).all():
        raise SnapshotValidationError("non-finite normalization statistics")

    return {
        "valid": True,
        "version": manifest["version"],
        "content_hash": manifest["content_hash"],
        "row_count": total_rows,
        "splits": split_reports,
        "mean": manifest["mean"],
        "std": manifest["std"],
        "seed": manifest["seed"],
    }


def load_validated_snapshot(
    directory: Path | str,
) -> tuple[dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Validate and load all split tensors; raises on any mismatch."""
    directory = Path(directory)
    report = validate_snapshot_dir(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if report["content_hash"] != manifest["content_hash"]:
        raise SnapshotValidationError("hash changed during validation")
    splits = {
        name: (
            np.load(directory / f"{name}_images.npy"),
            np.load(directory / f"{name}_labels.npy"),
        )
        for name in ("train", "val", "test")
    }
    return manifest, splits
