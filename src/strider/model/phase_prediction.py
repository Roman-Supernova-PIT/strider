"""Predict rest-frame phase from encoded spectral visits.

Input features have shape ``(B, V, Z, D)``. The output has shape
``(B, V, C, Z, P)`` for class, trial redshift and phase bin. Simulation phase
is a training target outside this module and is never passed to ``forward``.
"""

from __future__ import annotations

import torch
from torch import nn


class PhasePredictionHead(nn.Module):
    """Return a class-conditioned phase distribution for every visit and z."""

    def __init__(
        self,
        hidden_dim: int,
        class_count: int,
        phase_bins: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.class_count = int(class_count)
        self.phase_bins = int(phase_bins)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.class_count * self.phase_bins),
        )

    def forward(self, visit_features: torch.Tensor) -> torch.Tensor:
        batch, visits, redshifts, _ = visit_features.shape
        logits = self.net(visit_features)  # (B,V,Z,C*P)
        logits = logits.reshape(
            batch,
            visits,
            redshifts,
            self.class_count,
            self.phase_bins,
        )
        return logits.permute(0, 1, 3, 2, 4).contiguous()  # (B,V,C,Z,P)
