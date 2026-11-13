"""Checkpoint serialization determinism and round-trip tests."""

from __future__ import annotations

import hashlib

import torch
from traintrack.training.checkpoint import load_checkpoint, save_checkpoint
from traintrack.training.model import build_model


def test_checkpoint_roundtrip_and_determinism(tmp_path) -> None:
    model = build_model(dropout=0.1)
    payload = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "epoch": 2,
        "val_accuracy": 0.98,
        "hyperparams": {"lr": 0.001, "batch_size": 64},
        "class_names": [str(index) for index in range(10)],
    }
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    save_checkpoint(first, payload)
    save_checkpoint(second, payload)

    first_hash = hashlib.sha256(first.read_bytes()).hexdigest()
    second_hash = hashlib.sha256(second.read_bytes()).hexdigest()
    assert first_hash == second_hash

    loaded = load_checkpoint(first)
    assert loaded["epoch"] == 2
    assert loaded["val_accuracy"] == 0.98
    assert loaded["hyperparams"]["lr"] == 0.001
    for key, tensor in model.state_dict().items():
        assert torch.equal(loaded["state_dict"][key], tensor)

    restored = build_model(dropout=0.1)
    restored.load_state_dict(loaded["state_dict"])
    for key, tensor in restored.state_dict().items():
        assert torch.equal(tensor, model.state_dict()[key])


def test_checkpoint_metadata_is_canonical(tmp_path) -> None:
    first = tmp_path / "a.npz"
    second = tmp_path / "b.npz"
    model = build_model()
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    save_checkpoint(first, {"state_dict": state, "a": 1, "b": 2})
    save_checkpoint(second, {"b": 2, "a": 1, "state_dict": state})
    assert hashlib.sha256(first.read_bytes()).hexdigest() == hashlib.sha256(
        second.read_bytes()
    ).hexdigest()
