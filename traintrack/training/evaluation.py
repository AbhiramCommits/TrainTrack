"""Shared evaluation loop used by training and standalone eval jobs."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from traintrack.training.metrics import classification_report, latency_percentiles


@dataclass
class EvaluationOutput:
    """Predictions, loss, report and latency measured over a loader."""

    loss: float
    report: dict[str, Any]
    latency: dict[str, float]


def make_loader(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool = False,
    seed: int | None = None,
    max_batches: int | None = None,
) -> DataLoader:
    """Wrap numpy arrays in a TensorDataset/DataLoader."""
    image_tensor = torch.from_numpy(np.ascontiguousarray(images))
    label_tensor = torch.from_numpy(np.ascontiguousarray(labels)).long()
    dataset = torch.utils.data.TensorDataset(image_tensor, label_tensor)
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
        generator=generator,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    criterion: nn.Module | None = None,
    class_names: list[str] | None = None,
    measure_latency: bool = False,
    max_batches: int | None = None,
) -> EvaluationOutput:
    """Run the model over a loader and compute metrics and latency."""
    model.eval()
    loss_fn = criterion or nn.CrossEntropyLoss()
    losses: list[float] = []
    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    latency_samples: list[float] = []
    total = 0

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device)
            labels = labels.to(device)
            started = time.perf_counter()
            logits = model(images)
            _synchronize(device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            loss = loss_fn(logits, labels)
            losses.append(float(loss.detach().cpu()))
            predicted = logits.argmax(dim=1)
            truths.append(labels.detach().cpu().numpy())
            predictions.append(predicted.detach().cpu().numpy())
            total += len(labels)
            if measure_latency and batch_index >= 2:
                latency_samples.append(elapsed_ms / max(1, len(labels)))

    y_true = np.concatenate(truths) if truths else np.array([], dtype=np.int64)
    y_pred = (
        np.concatenate(predictions) if predictions else np.array([], dtype=np.int64)
    )
    report = classification_report(y_true, y_pred, class_names=class_names)
    latency = latency_percentiles(latency_samples)
    return EvaluationOutput(
        loss=float(np.mean(losses)) if losses else 0.0,
        report=report,
        latency=latency,
    )
