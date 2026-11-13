"""Deterministic checkpoint serialization.

``torch.save`` embeds storage addresses (and zip timestamps) that change between
otherwise identical saves. Checkpoints here are ``.npz`` archives of tensor
bytes plus canonical JSON metadata, which makes byte-identical states hash
identically and lets the artifact store dedupe them for real.

Files remain fully usable: :func:`load_checkpoint` reconstructs a state dict of
PyTorch tensors ready for ``model.load_state_dict``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

META_KEY = "__meta__"


def save_checkpoint(path: Path | str, payload: dict[str, Any]) -> None:
    """Write a deterministic checkpoint from a payload containing state_dict."""
    state_dict = payload["state_dict"]
    metadata = {key: value for key, value in payload.items() if key != "state_dict"}
    arrays = {
        key: value.detach().cpu().numpy() for key, value in state_dict.items()
    }
    arrays[META_KEY] = np.frombuffer(
        json.dumps(metadata, sort_keys=True, default=str).encode("utf-8"),
        dtype=np.uint8,
    )
    with open(path, "wb") as handle:
        np.savez(handle, **arrays)


def load_checkpoint(path: Path | str) -> dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint`."""
    with np.load(path, allow_pickle=False) as data:
        if META_KEY not in data.files:
            raise ValueError("checkpoint is missing metadata")
        metadata = json.loads(bytes(data[META_KEY]).decode("utf-8"))
        state_dict = {
            key: torch.from_numpy(np.array(data[key]))
            for key in data.files
            if key != META_KEY
        }
    metadata["state_dict"] = state_dict
    return metadata
