"""Device selection and per-run resource sampling.

CUDA exposes vendor utilization counters through torch. MPS does not: the
``utilization_pct`` reported for MPS is the training process's CPU utilization
(a documented proxy), while memory comes from torch.mps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import psutil
import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    """The selected compute device and how to sample it."""

    device: torch.device
    backend: str
    name: str


def select_device(preference: str = "auto") -> DeviceInfo:
    """Pick cuda > mps > cpu unless an explicit preference is requested."""
    preference = preference.lower()
    if preference in ("auto", "cuda") and torch.cuda.is_available():
        return DeviceInfo(
            device=torch.device("cuda"),
            backend="cuda",
            name=torch.cuda.get_device_name(0),
        )
    if preference in ("auto", "mps") and torch.backends.mps.is_available():
        return DeviceInfo(
            device=torch.device("mps"), backend="mps", name="Apple Silicon MPS"
        )
    if preference == "cuda":
        logger.warning("CUDA requested but unavailable; falling back to CPU")
    return DeviceInfo(device=torch.device("cpu"), backend="cpu", name="CPU")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs for a reproducible run."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


_process = psutil.Process()


def sample_resources(info: DeviceInfo, step: int) -> dict[str, Any]:
    """Sample utilization and memory for the current step."""
    cpu_pct = float(_process.cpu_percent(interval=None))
    rss_mb = float(_process.memory_info().rss / (1024 * 1024))
    utilization = cpu_pct
    memory_used_mb = rss_mb

    if info.backend == "cuda":
        try:
            utilization = float(torch.cuda.utilization())
        except Exception:  # pragma: no cover - depends on driver tooling
            utilization = 0.0
        try:
            memory_used_mb = float(
                torch.cuda.memory_allocated() / (1024 * 1024)
            )
        except Exception:  # pragma: no cover
            memory_used_mb = 0.0
    elif info.backend == "mps":
        try:
            memory_used_mb = float(
                torch.mps.driver_allocated_memory() / (1024 * 1024)
            )
        except Exception:  # pragma: no cover - older torch builds
            try:
                memory_used_mb = float(
                    torch.mps.current_allocated_memory() / (1024 * 1024)
                )
            except Exception:
                memory_used_mb = rss_mb

    return {
        "step": int(step),
        "device": info.backend,
        "utilization_pct": round(max(0.0, min(100.0, utilization)), 3),
        "memory_used_mb": round(max(0.0, memory_used_mb), 3),
        "cpu_pct": round(max(0.0, cpu_pct), 3),
        "rss_mb": round(max(0.0, rss_mb), 3),
    }
