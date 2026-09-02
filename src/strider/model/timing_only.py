"""Diagnostic model that receives observation times but no spectral values."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .redshift_scan import build_redshift_grid, redshift_cell_widths


class TimingOnlyModel(nn.Module):
    """Measure redshift information available from the observation schedule."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        redshift_grid = build_redshift_grid(
            float(model["redshift_min"]),
            float(model["redshift_max"]),
            int(model["redshift_bins"]),
            str(model.get("redshift_spacing", "linear")),
        )
        self.register_buffer("redshift_grid", torch.from_numpy(redshift_grid))
        self.register_buffer(
            "redshift_cell_width",
            torch.from_numpy(redshift_cell_widths(redshift_grid)),
        )
        self.redshift_prior = str(model.get("redshift_prior", "flat_z"))
        self.class_names = tuple(str(name) for name in model["classes"])
        hidden = int(
            model.get("hidden_dim", config.get("onir", {}).get("token_dim", 32))
        )
        output_count = len(self.class_names) * int(model["redshift_bins"])
        self.head = nn.Sequential(
            nn.Linear(8, hidden),
            nn.GELU(),
            nn.Dropout(float(model["dropout"])),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, output_count),
        )
        self.class_count = len(self.class_names)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        features = self._time_features(batch["observer_days"], batch["visit_mask"])
        logits = self.head(features).reshape(
            len(features), self.class_count, len(self.redshift_grid)
        )
        return {
            "joint_logits": logits,
            # This timing-only comparison does not estimate evidence sufficiency.
            "evidence_sufficiency_logit": torch.zeros(len(features), device=features.device),
        }

    @staticmethod
    def _time_features(times: torch.Tensor, visit_mask: torch.Tensor) -> torch.Tensor:
        valid = visit_mask > 0
        count = visit_mask.sum(dim=1).clamp_min(1.0)
        masked_times = times * visit_mask
        mean = masked_times.sum(dim=1) / count
        variance = ((times - mean[:, None]).square() * visit_mask).sum(dim=1) / count
        maximum = times.masked_fill(~valid, -torch.inf).amax(dim=1)
        pair_valid = valid[:, 1:] & valid[:, :-1]
        gaps = (times[:, 1:] - times[:, :-1]).masked_fill(~pair_valid, 0.0)
        gap_count = pair_valid.sum(dim=1).clamp_min(1)
        mean_gap = gaps.sum(dim=1) / gap_count
        gap_variance = (
            (gaps - mean_gap[:, None]).square() * pair_valid
        ).sum(dim=1) / gap_count
        minimum_gap = gaps.masked_fill(~pair_valid, torch.inf).amin(dim=1)
        maximum_gap = gaps.masked_fill(~pair_valid, -torch.inf).amax(dim=1)
        has_gap = pair_valid.any(dim=1)
        minimum_gap = torch.where(has_gap, minimum_gap, torch.zeros_like(minimum_gap))
        maximum_gap = torch.where(has_gap, maximum_gap, torch.zeros_like(maximum_gap))
        return torch.stack(
            [
                count / 32.0,
                maximum / 100.0,
                mean / 100.0,
                torch.sqrt(variance.clamp_min(0.0)) / 100.0,
                mean_gap / 30.0,
                torch.sqrt(gap_variance.clamp_min(0.0)) / 30.0,
                minimum_gap / 30.0,
                maximum_gap / 30.0,
            ],
            dim=-1,
        )
