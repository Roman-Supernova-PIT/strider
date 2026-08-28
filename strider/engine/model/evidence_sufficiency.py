"""Estimate whether observations support interpreting the class-redshift result.

Inputs are flux ``(B,V,L)``, wavelength and visit masks, and observer days.
The output is one logit per object. Dates are used only through observed span;
truth redshift, truth phase and clean simulation flux are prohibited.
"""

from __future__ import annotations

import torch
from torch import nn


class EvidenceSufficiencyHead(nn.Module):
    """Map transparent object-level measurement summaries to one logit."""

    def __init__(
        self,
        dropout: float,
        visit_count_reference: int,
        use_visit_count_and_span: bool = True,
    ) -> None:
        super().__init__()
        if visit_count_reference < 1:
            raise ValueError("visit_count_reference must be at least one")
        self.visit_count_reference = int(visit_count_reference)
        self.use_visit_count_and_span = bool(use_visit_count_and_span)
        feature_count = 8 if self.use_visit_count_and_span else 6
        self.network = nn.Sequential(
            nn.Linear(feature_count, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = observation_summary_features(
            batch,
            self.visit_count_reference,
            use_visit_count_and_span=self.use_visit_count_and_span,
        )  # (B,F)
        return self.network(features).squeeze(-1)  # (B,)


def observation_summary_features(
    batch: dict[str, torch.Tensor],
    visit_count_reference: int,
    use_visit_count_and_span: bool = True,
) -> torch.Tensor:
    """Return bounded measurement summaries with shape ``(B, 6 or 8)``."""
    mask = batch["wavelength_mask"] * batch["visit_mask"][:, :, None]  # (B,V,L)
    count = mask.sum(dim=(1, 2)).clamp_min(1.0)  # (B,)
    flux = batch["flux"] * mask  # (B,V,L)
    mean = flux.sum(dim=(1, 2)) / count  # (B,)
    mean_abs = flux.abs().sum(dim=(1, 2)) / count  # (B,)
    root_mean_square = torch.sqrt(
        (flux.square().sum(dim=(1, 2)) / count).clamp_min(1e-8)
    )  # (B,)
    visits_per_wavelength = mask.sum(dim=1).clamp_min(1.0)  # (B,L)
    mean_spectrum = flux.sum(dim=1) / visits_per_wavelength  # (B,L)
    wavelength_present = (mask.sum(dim=1) > 0).to(flux.dtype)  # (B,L)
    wavelength_count = wavelength_present.sum(dim=1).clamp_min(1.0)  # (B,)
    coherent_root_mean_square = torch.sqrt(
        (mean_spectrum.square() * wavelength_present).sum(dim=1) / wavelength_count
    ).clamp_min(1e-8)  # (B,)
    residual = (flux - mean_spectrum[:, None, :]) * mask  # (B,V,L)
    fluctuation_root_mean_square = torch.sqrt(
        (residual.square().sum(dim=(1, 2)) / count).clamp_min(1e-8)
    )  # (B,)
    coherence_ratio = coherent_root_mean_square / fluctuation_root_mean_square.clamp_min(1e-4)

    # Bounded transforms keep detector faults from dominating the small head.
    features = [
        torch.tanh(mean / 5.0),
        torch.log1p(mean_abs.clamp_max(100.0)),
        torch.log1p(root_mean_square.clamp_max(100.0)),
        torch.log1p(coherent_root_mean_square.clamp_max(100.0)),
        torch.log1p(fluctuation_root_mean_square.clamp_max(100.0)),
        torch.log(coherence_ratio.clamp(1e-4, 1e4)) / 5.0,
    ]
    if use_visit_count_and_span:
        visit_valid = batch["visit_mask"] > 0  # (B,V)
        maximum_time = batch["observer_days"].masked_fill(
            ~visit_valid, -torch.inf
        ).amax(dim=1)
        minimum_time = batch["observer_days"].masked_fill(
            ~visit_valid, torch.inf
        ).amin(dim=1)
        time_span = (maximum_time - minimum_time) / 100.0  # (B,)
        visit_count = batch["visit_mask"].sum(dim=1) / float(
            visit_count_reference
        )  # (B,)
        features.extend(
            [
                torch.log1p(time_span.clamp(0.0, 100.0)),
                visit_count,
            ]
        )
    return torch.stack(features, dim=-1)  # (B,F)
