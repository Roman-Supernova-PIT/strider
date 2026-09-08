"""Temporal evidence that can only act through measured spectral change."""

from __future__ import annotations

import torch
from torch import nn


class SpectralEvolutionEvidence(nn.Module):
    """Score whether spectral changes are compatible with candidate rest-frame times.

    Dates determine how an observed interval maps into rest-frame time. Every
    temporal feature is multiplied by the change between two measured spectral
    representations. Identical representations therefore give zero
    time-dependent output. Noise, masks, and visit selection can still create
    nonzero representation changes and must be checked empirically.
    """

    def __init__(
        self,
        hidden_dim: int,
        class_count: int,
        dropout: float,
        initial_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.time_modulation = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.class_score = nn.Linear(hidden_dim, class_count, bias=False)
        self.scale = nn.Parameter(torch.tensor(float(initial_scale)))

    def forward(
        self,
        spectral: torch.Tensor,
        observer_days: torch.Tensor,
        visit_mask: torch.Tensor,
        redshift_grid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return scaled and unscaled logits with shape ``[B, C, Z]``."""
        batch, visits, redshift_count, hidden_dim = spectral.shape
        if visits < 2:
            raw = spectral.new_zeros((batch, self.class_score.out_features, redshift_count))
            return raw, raw

        spectral_change = spectral[:, 1:] - spectral[:, :-1]
        observed_interval = observer_days[:, 1:] - observer_days[:, :-1]
        rest_interval = observed_interval[:, :, None] / (1.0 + redshift_grid[None, None, :])
        interval_scale = torch.log1p(rest_interval.abs()) / torch.log(
            spectral.new_tensor(101.0)
        )
        time_features = torch.stack(
            [
                interval_scale,
                torch.sign(rest_interval),
                torch.tanh(rest_interval / 30.0),
            ],
            dim=-1,
        )
        modulation = self.time_modulation(time_features)
        if modulation.shape != (batch, visits - 1, redshift_count, hidden_dim):
            raise RuntimeError("Temporal modulation shape is inconsistent with spectral input")
        pair_features = _bounded_change(spectral_change * modulation)
        pair_logits = self.class_score(pair_features)
        pair_valid = (visit_mask[:, 1:] * visit_mask[:, :-1])[:, :, None, None]
        raw = (pair_logits * pair_valid).sum(dim=1) / pair_valid.sum(dim=1).clamp_min(1.0)
        raw = raw.permute(0, 2, 1).contiguous()
        scaled = torch.tanh(self.scale) * raw
        return scaled, raw


def _bounded_change(change: torch.Tensor) -> torch.Tensor:
    """Limit large changes without promoting small fluctuations to unit scale."""
    mean_square = change.square().mean(dim=-1, keepdim=True)
    return change / torch.sqrt(1.0 + mean_square)
