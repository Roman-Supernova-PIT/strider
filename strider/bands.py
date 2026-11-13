"""The four observed bands the brightness term compares."""

import numpy as np

__all__ = ["EDGES", "band_fluxes", "full_weights"]

EDGES = np.array([7500.0, 9340.0, 11620.0, 14460.0, 18000.0])  # Å, equal steps in log wavelength
COVERAGE = 0.9  # a band is kept when its usable pixels carry this share of its full weight


def full_weights(profile):
    """Each band's full weight: the pixel weights summed over the noise profile's own pixels in the band."""
    wavelength, sigma = profile
    inside = (wavelength >= EDGES[:-1, None]) & (wavelength < EDGES[1:, None])
    return (inside * np.gradient(wavelength) / sigma**2).sum(1)


def band_fluxes(wavelength, flux, error, profile, full):
    """One visit's mean flux density in each band and its error, NaN where the band is not covered.

    Each pixel counts with its width over the squared noise profile (``profile``: wavelength and sigma), the same for
    every visit, so a band means the same thing in every visit. ``wavelength`` is sorted, in Å; pixels with a
    non-finite flux or error, or an error that is not positive, are not used.
    """
    good = np.isfinite(flux) & np.isfinite(error) & (error > 0)
    inside = (wavelength >= EDGES[:-1, None]) & (wavelength < EDGES[1:, None])
    weights = inside * good * (_widths(wavelength) / np.interp(wavelength, *profile) ** 2)
    total = weights.sum(1)
    kept = total >= COVERAGE * full
    total = np.where(kept, total, np.nan)
    value = weights @ np.where(good, flux, 0.0) / total
    sigma = np.sqrt(weights**2 @ np.where(good, error, 0.0) ** 2) / total
    return value, sigma


def _widths(wavelength):
    """Pixel widths from the spacing of their neighbours; beside a gap, from the spacing on the other side."""
    step = np.diff(wavelength)
    left, right = np.r_[step[0], step], np.r_[step, step[-1]]
    return np.where(left > 1.5 * right, right, np.where(right > 1.5 * left, left, 0.5 * (left + right)))
