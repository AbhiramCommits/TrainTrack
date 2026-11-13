"""Classification metrics and inference latency statistics."""

from __future__ import annotations

from typing import Any

import numpy as np


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10
) -> np.ndarray:
    """Build a (num_classes, num_classes) confusion matrix (rows=true)."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (y_true.astype(np.int64), y_pred.astype(np.int64)), 1)
    return matrix


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int = 10,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute accuracy, macro precision/recall/F1 and per-class figures."""
    names = class_names or [str(index) for index in range(num_classes)]
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    per_class = {
        names[index]: {
            "precision": round(float(precision[index]), 6),
            "recall": round(float(recall[index]), 6),
            "f1": round(float(f1[index]), 6),
            "accuracy": round(float(recall[index]), 6),
            "support": int(support[index]),
        }
        for index in range(num_classes)
    }
    total = float(matrix.sum())
    return {
        "accuracy": round(float(tp.sum() / total), 6) if total else 0.0,
        "macro_precision": round(float(precision.mean()), 6),
        "macro_recall": round(float(recall.mean()), 6),
        "macro_f1": round(float(f1.mean()), 6),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
        "num_samples": int(total),
    }


def latency_percentiles(samples_ms: list[float]) -> dict[str, float]:
    """Return p50/p95 per-sample inference latency in milliseconds."""
    if not samples_ms:
        return {"p50_ms": 0.0, "p95_ms": 0.0}
    values = np.asarray(samples_ms, dtype=np.float64)
    return {
        "p50_ms": round(float(np.percentile(values, 50)), 6),
        "p95_ms": round(float(np.percentile(values, 95)), 6),
    }
