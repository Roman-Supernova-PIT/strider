"""Small wavelength-grid helpers shared by the deployment runtime."""

from __future__ import annotations

import numpy as np


def log_wavelength_grid(minimum: float, maximum: float, bins: int) -> np.ndarray:
    """Return the packaged logarithmic observer or rest wavelength grid."""
    return np.geomspace(float(minimum), float(maximum), int(bins), dtype=np.float64)
