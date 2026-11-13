"""The spectral model, which gives each class's rest-frame spectrum at each phase, and the reference spectra.

The spectral model is a small network whose output is a set of coefficients on a fixed cubic B-spline basis in
ln λ, so its spectra carry no structure finer than the knot spacing. The reference spectra, the spectral model
averaged over each phase bin, are used only where enough training spectra reach; ``weights.pt`` records the masks
that say where.
"""

import numpy as np
import torch
from torch import nn

__all__ = ["SpectralModel", "bspline_basis", "node_spectra", "reference_spectra"]


def bspline_basis(rest_wavelength, count):
    """Cubic B-splines on uniform knots in ln λ, (count, pixels), summing to one everywhere on the grid."""
    x = np.log(np.asarray(rest_wavelength, dtype=np.float64))
    spacing = (x[-1] - x[0]) / (count - 3)
    distance = np.abs(x[None, :] - (x[0] + (np.arange(count)[:, None] - 1) * spacing)) / spacing
    return np.where(
        distance < 1, 2 / 3 - distance**2 + distance**3 / 2, np.where(distance < 2, (2 - distance) ** 3 / 6, 0.0)
    )


class SpectralModel(nn.Module):
    """The spectral model: the rest-frame spectrum of a class at a phase in days.

    The inputs are a learned class embedding and Fourier features of the phase rescaled to t in [0, 1] over
    ``phase_range``: sin and cos of π 2^k t for k < ``phase_frequencies``.
    """

    def __init__(
        self,
        rest_wavelength,
        classes,
        phase_range,
        basis_functions,
        class_embedding_dim,
        phase_frequencies,
        hidden_dim,
        layers,
    ):
        super().__init__()
        self.phase_range = tuple(float(value) for value in phase_range)
        self.register_buffer("basis", torch.from_numpy(bspline_basis(rest_wavelength, basis_functions)).float())
        self.register_buffer("frequencies", torch.pi * 2.0 ** torch.arange(phase_frequencies).float())
        self.class_embedding = nn.Embedding(classes, class_embedding_dim)
        body, width = [], class_embedding_dim + 2 * phase_frequencies
        for _ in range(layers):
            body += [nn.Linear(width, hidden_dim), nn.GELU()]
            width = hidden_dim
        self.body = nn.Sequential(*body)
        self.coefficients = nn.Linear(width, basis_functions)

    def forward(self, class_index, phase_days):
        low, high = self.phase_range
        angle = ((phase_days - low) / (high - low))[..., None] * self.frequencies
        phase = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
        return self.coefficients(self.body(torch.cat([self.class_embedding(class_index), phase], dim=-1))) @ self.basis


def node_spectra(spectral_model, nodes, classes):
    """The spectral model at its phase nodes for every class, (classes, nodes, rest pixels), clipped at zero: it
    dips below zero in a few pixels, where flux cannot."""
    class_index = torch.arange(classes, device=nodes.device).repeat_interleave(len(nodes))
    return spectral_model(class_index, nodes.repeat(classes)).reshape(classes, len(nodes), -1).clamp_min(0.0)


def reference_spectra(spectral_model, nodes, window_weight, mask, continuum_removal):
    """The reference spectra: the spectral model averaged over each phase bin with ``window_weight``
    (classes, phase bins, nodes), as (classes, phase bins, rest pixels), zero outside ``mask``, in the full and
    continuum-removed views."""
    full = window_weight @ node_spectra(spectral_model, nodes, window_weight.shape[0])
    mask = mask.to(full.dtype)
    return {"full": full * mask, "continuum_removed": continuum_removal(full, mask)}
