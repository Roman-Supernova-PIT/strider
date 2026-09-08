"""Numerical precision used by training and its timing comparison."""

from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager

import torch


def autocast_context(
    device: torch.device,
    precision: str,
) -> ContextManager:
    """Use A100 bfloat16 when requested and ordinary precision elsewhere."""
    if precision == "float32":
        return nullcontext()
    if precision != "bfloat16":
        raise ValueError("mixed_precision must be 'float32' or 'bfloat16'")
    if device.type != "cuda":
        # Local CPU and MPS checks keep their native numerical path.
        return nullcontext()
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("bfloat16 was requested but this CUDA device does not support it")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
