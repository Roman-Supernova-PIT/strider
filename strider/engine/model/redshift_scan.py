"""Candidate-redshift grids and observer-to-rest-frame alignment.

Inputs use wavelength in Angstrom and redshift as a dimensionless quantity.
The scan never receives truth redshift. Read ``posterior.py`` next for the
conversion from evidence sampled on this grid to probability mass.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def build_redshift_grid(
    minimum: float,
    maximum: float,
    bins: int,
    spacing: str,
) -> np.ndarray:
    """Return candidate redshifts with linear or constant-log-shift spacing."""
    if bins < 2:
        raise ValueError("redshift_bins must be at least two")
    if minimum < 0.0 or maximum <= minimum:
        raise ValueError("redshift bounds must satisfy 0 <= minimum < maximum")
    if spacing == "linear":
        return np.linspace(minimum, maximum, bins, dtype=np.float32)
    if spacing == "log1p":
        log_shift = np.linspace(np.log1p(minimum), np.log1p(maximum), bins)
        return np.expm1(log_shift).astype(np.float32)
    raise ValueError("redshift_spacing must be 'linear' or 'log1p'")


def redshift_cell_widths(redshift_grid: np.ndarray) -> np.ndarray:
    """Return positive redshift widths for cells centred on the scan points."""
    grid = np.asarray(redshift_grid, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2 or np.any(np.diff(grid) <= 0.0):
        raise ValueError("redshift_grid must be a strictly increasing one-dimensional array")
    edges = np.empty(len(grid) + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (grid[:-1] + grid[1:])
    edges[0] = grid[0]
    edges[-1] = grid[-1]
    return np.diff(edges).astype(np.float32)


class RestFrameScan(nn.Module):
    def __init__(
        self,
        observed_wavelength: np.ndarray,
        rest_wavelength: np.ndarray,
        redshift_grid: np.ndarray,
    ) -> None:
        super().__init__()
        observed = np.asarray(observed_wavelength, dtype=np.float64)
        target = (1.0 + redshift_grid[:, None]) * rest_wavelength[None, :]
        upper = np.searchsorted(observed, target, side="left")
        valid = (upper > 0) & (upper < len(observed))
        upper = np.clip(upper, 1, len(observed) - 1)
        lower = upper - 1
        denominator = observed[upper] - observed[lower]
        weight = np.divide(
            target - observed[lower],
            denominator,
            out=np.zeros_like(target),
            where=denominator > 0,
        )
        self.register_buffer("lower_index", torch.from_numpy(lower.astype(np.int64)))
        self.register_buffer("upper_index", torch.from_numpy(upper.astype(np.int64)))
        self.register_buffer("upper_weight", torch.from_numpy(weight.astype(np.float32)))
        self.register_buffer("scan_valid", torch.from_numpy(valid.astype(np.float32)))

    def forward(
        self,
        flux: torch.Tensor,
        wavelength_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, visits, wavelength_bins = flux.shape
        flat_flux = flux.reshape(batch * visits, wavelength_bins)
        flat_mask = wavelength_mask.reshape(batch * visits, wavelength_bins)
        lower_flux = flat_flux[:, self.lower_index]
        upper_flux = flat_flux[:, self.upper_index]
        weight = self.upper_weight.unsqueeze(0)
        aligned_flux = lower_flux * (1.0 - weight) + upper_flux * weight
        lower_mask = flat_mask[:, self.lower_index]
        upper_mask = flat_mask[:, self.upper_index]
        aligned_mask = lower_mask * upper_mask * self.scan_valid.unsqueeze(0)
        shape = (batch, visits, self.lower_index.shape[0], self.lower_index.shape[1])
        return aligned_flux.reshape(shape), aligned_mask.reshape(shape)
