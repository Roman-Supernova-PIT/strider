"""Multi-epoch evidence combiner for STRIDER.

Default is SNR-weighted SUM, which preserves the √N benefit of observing the
same transient repeatedly. Mean is supported only as an explicit ablation —
concurring epochs should sharpen evidence, not be averaged away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


CombinerMode = Literal['snr_sum', 'mean']


@dataclass(frozen=True)
class CombinerConfig:
    mode: CombinerMode = 'snr_sum'
    snr_clip: float = 1.0   # floor for SNR weight, so noisy epochs still contribute slightly


class MultiEpochCombiner(nn.Module):
    """Combine per-epoch (B, T, C, Z) evidence into object-level (B, C, Z).

    Epoch weight is `epoch_mask * mean_p(ivar_per_patch).clamp(min=snr_clip)`,
    falling back to `epoch_mask` alone when no inverse-variance is supplied.
    """

    def __init__(self, config: CombinerConfig | None = None):
        super().__init__()
        self.config = config or CombinerConfig()
        if self.config.mode not in ('snr_sum', 'mean'):
            raise ValueError(
                f"combiner mode must be 'snr_sum' or 'mean', got {self.config.mode!r}"
            )

    def forward(
        self,
        epoch_surface: torch.Tensor,                # (B, T, C, Z)
        epoch_mask: torch.Tensor | None = None,     # (B, T) bool or float
        ivar_per_patch_te: torch.Tensor | None = None,  # (B, T, P) float
    ) -> torch.Tensor:                              # (B, C, Z)
        B, T, C, Z = epoch_surface.shape
        weights = self._compute_weights(epoch_mask, ivar_per_patch_te, B, T,
                                         epoch_surface.dtype, epoch_surface.device)
        # weights: (B, T) → broadcast to (B, T, 1, 1)
        weighted = epoch_surface * weights.unsqueeze(-1).unsqueeze(-1)
        if self.config.mode == 'mean':
            denom = weights.sum(dim=1).clamp(min=1e-6).unsqueeze(-1).unsqueeze(-1)
            return weighted.sum(dim=1) / denom
        return weighted.sum(dim=1)

    def _compute_weights(
        self,
        epoch_mask: torch.Tensor | None,
        ivar_per_patch_te: torch.Tensor | None,
        B: int,
        T: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        mask = (
            epoch_mask.to(dtype) if epoch_mask is not None
            else torch.ones(B, T, dtype=dtype, device=device)
        )
        if self.config.mode == 'snr_sum' and ivar_per_patch_te is not None:
            # Mean ivar per epoch (across patches), clipped to a floor, then masked.
            snr = ivar_per_patch_te.to(dtype).mean(dim=-1).clamp(min=self.config.snr_clip)
            return mask * snr
        return mask
