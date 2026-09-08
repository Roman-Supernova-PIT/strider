"""Convert class-redshift evidence into posterior probability mass.

Evidence has shape ``(B, C, Z)`` for batch, class and candidate redshift.
Redshift cell widths are dimensionless. This module never receives simulation
truth; it applies only the declared inference prior.
"""

from __future__ import annotations

import torch


def posterior_logits(
    evidence_logits: torch.Tensor,
    redshift_cell_width: torch.Tensor,
    prior: str = "flat_z",
) -> torch.Tensor:
    """Add the grid measure required by the declared redshift prior."""
    if evidence_logits.ndim != 3:
        raise ValueError("evidence_logits must have shape (batch, class, redshift)")
    if redshift_cell_width.shape != (evidence_logits.shape[-1],):
        raise ValueError("redshift_cell_width must match the redshift dimension")
    if prior == "flat_z":
        log_measure = torch.log(redshift_cell_width.clamp_min(1e-12))
    elif prior == "flat_log1p":
        log_measure = torch.zeros_like(redshift_cell_width)
    else:
        raise ValueError("redshift_prior must be 'flat_z' or 'flat_log1p'")
    return evidence_logits + log_measure[None, None, :]


def joint_probability(
    evidence_logits: torch.Tensor,
    redshift_cell_width: torch.Tensor,
    prior: str = "flat_z",
) -> torch.Tensor:
    """Return normalized probability mass with shape ``(B, C, Z)``."""
    measured_logits = posterior_logits(evidence_logits, redshift_cell_width, prior)
    batch, classes, redshifts = measured_logits.shape
    probability = torch.softmax(measured_logits.reshape(batch, -1), dim=-1)
    return probability.reshape(batch, classes, redshifts)
