"""Choose the fastest available PyTorch device with a predictable fallback."""

from __future__ import annotations

import os

import torch


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if os.environ.get("STRIDER_REQUIRE_CUDA") == "1":
        raise RuntimeError(
            "This job requires CUDA, but PyTorch cannot see a CUDA device. "
            "Check the Slurm GPU request and the installed PyTorch build."
        )
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
