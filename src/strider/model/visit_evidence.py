"""Combine visit-level class-redshift scores with an explicit count response.

Inputs are scores and validity masks with the same shape ``(B,V,Z,C)``.
The exponent controls how evidence scale grows with valid visit count. This
module receives neither dates nor simulation labels.
"""

from __future__ import annotations

import torch


def combine_visit_scores(
    scores: torch.Tensor,
    valid: torch.Tensor,
    exponent: float,
) -> torch.Tensor:
    """Return object scores with shape ``(B,Z,C)``."""
    if scores.shape != valid.shape:
        raise ValueError("visit scores and validity mask must have the same shape")
    if not 0.0 <= float(exponent) <= 1.0:
        raise ValueError("visit evidence exponent must lie in [0, 1]")
    count = valid.sum(dim=1).clamp_min(1).to(scores.dtype)  # (B,Z,C)
    total = (scores * valid.to(scores.dtype)).sum(dim=1)  # (B,Z,C)
    # exponent=0 is a mean; exponent=1 is a sum. Intermediate values let the
    # selection split measure how correlated visits change evidence growth.
    return total / count.pow(1.0 - float(exponent))
