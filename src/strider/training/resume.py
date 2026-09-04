"""Save and restore an epoch-boundary training state exactly.

The state contains model and optimizer values plus Python, NumPy and Torch
random-generator states. Resumption restarts after the last completed epoch;
an interrupted partial epoch is deliberately repeated from its beginning.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def capture_random_state() -> dict[str, Any]:
    numpy_name, numpy_keys, numpy_position, numpy_gaussian, numpy_cached = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "name": numpy_name,
            # PyTorch 2.5 can create uint32 tensors but cannot reload them from
            # a checkpoint. Int64 preserves every MT19937 key on all supported versions.
            "keys": torch.from_numpy(numpy_keys.astype(np.int64, copy=True)),
            "position": int(numpy_position),
            "has_gaussian": int(numpy_gaussian),
            "cached_gaussian": float(numpy_cached),
        },
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
        state["torch_mps"] = torch.mps.get_rng_state()
    return state


def restore_random_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            str(numpy_state["name"]),
            numpy_state["keys"].cpu().numpy().astype(np.uint32, copy=False),
            int(numpy_state["position"]),
            int(numpy_state["has_gaussian"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if (
        "torch_mps" in state
        and torch.backends.mps.is_available()
        and hasattr(torch.mps, "set_rng_state")
    ):
        torch.mps.set_rng_state(state["torch_mps"])


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Replace a checkpoint only after the complete temporary file is written."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    # The directory sync makes the rename durable across a node or filesystem
    # interruption, not merely visible to the current process.
    if hasattr(os, "O_DIRECTORY"):
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def load_training_state(path: Path) -> dict[str, Any]:
    """Load random-generator byte tensors on CPU before restoring a run."""
    return torch.load(path, map_location="cpu", weights_only=False)
