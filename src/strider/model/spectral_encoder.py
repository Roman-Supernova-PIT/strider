"""Phase-neutral convolutional encoder for one rest-aligned spectral visit."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn


class SpectralEncoder(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(32, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, flux: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Leading dimensions are combined temporarily; the visit and trial-z
        # axes are restored by the caller.
        leading_shape = flux.shape[:-1]
        values = flux.reshape(-1, 1, flux.shape[-1])
        valid = mask.reshape(-1, 1, mask.shape[-1])
        encoded = self.layers(values * valid)
        for _ in range(3):
            valid = functional.avg_pool1d(valid, kernel_size=3, stride=2, padding=1)
        valid = (valid > 0.25).to(encoded.dtype)
        pooled = (encoded * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)
        pooled = self.output_norm(pooled)
        return pooled.reshape(*leading_shape, pooled.shape[-1])
