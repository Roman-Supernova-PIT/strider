"""Window-gathering and phase-blending primitives for the evidence detector.

Pure functions on tensors. The matcher holds no state and knows nothing
about classes, phases, or templates beyond the shapes it's handed.
"""

from __future__ import annotations

import torch


def gather_windows(
    energy: torch.Tensor,        # (B, P, S)
    gather_idx: torch.Tensor,    # (Z, nF, W) long
    gather_valid: torch.Tensor,  # (Z, nF, W) bool
) -> torch.Tensor:               # (B, Z, nF, W, S)
    """Gather S-dim energy vectors at per-(z, feature) window positions.

    Invalid window positions (gather_valid=False) are zeroed in the output.
    Out-of-range indices are clamped to [0, P-1] before lookup; the
    `gather_valid` mask then zeros the gathered values, so callers may
    pass any index value at positions marked invalid.
    """
    B, P, S = energy.shape
    Z, nF, W = gather_idx.shape
    safe_idx = gather_idx.clamp(0, P - 1).reshape(-1)
    gathered = energy.index_select(1, safe_idx)
    gathered = gathered.reshape(B, Z, nF, W, S)
    gathered = gathered * gather_valid.unsqueeze(0).unsqueeze(-1).to(energy.dtype)
    return gathered


def phase_interpolate_weights(
    phase_days: torch.Tensor,         # (...,) phase per epoch in days
    phase_bin_centers: torch.Tensor,  # (P_bin,) bin centers in days
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Linear-blend weights for the two phase bins adjacent to each `phase_days`.

    Returns (lo_idx, hi_idx, alpha) where:
      score(phase) ≈ (1 - alpha) * score[lo_idx] + alpha * score[hi_idx]
    Phase outside [centers[0], centers[-1]] clamps to nearest edge (alpha = 0 or 1).
    """
    centers = phase_bin_centers.to(phase_days.dtype).to(phase_days.device)
    P = centers.shape[0]
    if P < 2:
        # Degenerate single-bin case: all weight on the sole bin.
        zeros = torch.zeros_like(phase_days, dtype=torch.long)
        return zeros, zeros, torch.zeros_like(phase_days)
    hi = torch.searchsorted(centers, phase_days, right=False).clamp(min=1, max=P - 1)
    lo = (hi - 1).clamp(min=0, max=P - 1)
    span = (centers[hi] - centers[lo]).clamp(min=1e-6)
    alpha = ((phase_days - centers[lo]) / span).clamp(min=0.0, max=1.0)
    return lo, hi, alpha
