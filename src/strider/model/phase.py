"""Candidate-dependent rest-frame phase features from observer-frame times."""

from __future__ import annotations

import math

import torch
from torch import nn


class CandidatePhaseEmbedding(nn.Module):
    def __init__(self, feature_count: int, hidden_dim: int) -> None:
        super().__init__()
        if feature_count % 2:
            raise ValueError("phase_features must be even")
        frequencies = torch.logspace(-2.0, -0.15, feature_count // 2)
        self.register_buffer("frequencies", frequencies)
        self.project = nn.Sequential(
            nn.Linear(feature_count, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, observer_days: torch.Tensor, redshift_grid: torch.Tensor) -> torch.Tensor:
        rest_days = observer_days[:, :, None] / (1.0 + redshift_grid[None, None, :])
        angles = 2.0 * math.pi * rest_days[..., None] * self.frequencies
        features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.project(features)
