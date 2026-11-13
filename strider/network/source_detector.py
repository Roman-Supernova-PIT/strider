"""The source detector: do an object's visits hold a transient at all?

A small network maps six summaries of the measured flux (with ``evidence_use_visit_count_and_span`` in
``model_config.json``, eight: the visits' span and count too) to one logit. It never sees a redshift, a class or a
date beyond the span. The calibration turns the logit into the source probability (`strider.posterior`).
"""

import torch
from torch import nn

__all__ = ["SourceDetector", "summaries"]


def summaries(flux, mask, days, visit_count_reference=None):
    """Bounded summaries of one object's flux, (6,), or (8,) with the visits' span in days and their count over
    ``visit_count_reference``.

    Parameters
    ----------
    flux, mask : torch.Tensor
        (visits, pixels): each visit's flux over its scale, and 1 where it is measured.
    days : torch.Tensor
        (visits,) observer-frame days.
    """
    mask = mask.to(flux.dtype)
    count = mask.sum().clamp_min(1.0)
    flux = flux * mask
    mean = flux.sum() / count
    mean_abs = flux.abs().sum() / count
    root_mean_square = torch.sqrt((flux.square().sum() / count).clamp_min(1e-8))
    mean_spectrum = flux.sum(0) / mask.sum(0).clamp_min(1.0)
    present = (mask.sum(0) > 0).to(flux.dtype)
    coherent = torch.sqrt((mean_spectrum.square() * present).sum() / present.sum().clamp_min(1.0)).clamp_min(1e-8)
    fluctuation = torch.sqrt((((flux - mean_spectrum) * mask).square().sum() / count).clamp_min(1e-8))
    ratio = coherent / fluctuation.clamp_min(1e-4)
    # bounded transforms keep detector faults from dominating the small network
    features = [
        torch.tanh(mean / 5.0),
        torch.log1p(mean_abs.clamp_max(100.0)),
        torch.log1p(root_mean_square.clamp_max(100.0)),
        torch.log1p(coherent.clamp_max(100.0)),
        torch.log1p(fluctuation.clamp_max(100.0)),
        torch.log(ratio.clamp(1e-4, 1e4)) / 5.0,
    ]
    if visit_count_reference is not None:
        span = (days.max() - days.min()) / 100.0
        features += [torch.log1p(span.clamp(0.0, 100.0)), days.new_tensor(float(len(days))) / visit_count_reference]
    return torch.stack(features)


class SourceDetector(nn.Module):
    """The source detector of a model package.

    Parameters
    ----------
    dropout : float
        Dropout probability (training only).
    visit_count_reference : int, optional
        With it, the visits' span and count are summaries too.
    """

    def __init__(self, dropout, visit_count_reference=None):
        super().__init__()
        self.visit_count_reference = visit_count_reference
        inputs = 6 if visit_count_reference is None else 8
        self.network = nn.Sequential(nn.Linear(inputs, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, flux, mask, days):
        """The logit that the visits hold a transient."""
        return self.network(summaries(flux, mask, days, self.visit_count_reference))[0]
