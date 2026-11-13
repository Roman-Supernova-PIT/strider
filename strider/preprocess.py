"""From measured prism spectra to the network's inputs.

Each visit is prepared on its own, in float32, as in training, and as the package's ``preprocessing.json``
describes:

1. Native pixels with a non-finite wavelength are left out and the rest sorted by wavelength. A pixel is bad if it
   is masked, its flux or error is not finite, or its error is not positive.
2. The visit scale is the lower quartile of the errors of the pixels not masked, where finite and positive.
3. The visit is used only if its pixels cover the model's wavelength grid, with pixel edges halfway between centres
   and the end edges mirrored, to within `COVERAGE_SHORTFALL` of the grid's span in ln λ at each end; it has at least
   two good pixels; and some grid point is interpolated from good pixels. Otherwise it is dropped, and reported. At
   an end the visit falls short of, the grid points beyond its last pixel are masked.
4. The flux over the scale is interpolated linearly onto the grid, and its error propagated through the same
   interpolation. Every grid point whose interpolation touches a bad pixel is masked; the visit is kept.

The visits are then put in time order, with times in observer-frame days since the first visit kept; visits on one
date are ordered by their content, so their input order does not matter.

With the model's noise profile, each visit kept also gets its four band fluxes and their errors (`strider.bands`),
from its native pixels in float64, for the brightness term.

The brightness term draws its null copies from a seed of the input that does not depend on the flux unit, on the
order of visits on one date, or on ``error_scale``, so the same input, in any unit, gives the same copies:

- σ is the median of the input errors (float64) of the good native pixels of every visit kept;
- each visit's tie keys are the 8-byte BLAKE2b digests, read little-endian, of rint(1000 f / (σ / error_scale)),
  rint(1000 e / σ) and rint(1000 λ), each as little-endian int64, and of its good-pixel mask as uint8: f and e the
  input flux and error of its good native pixels, λ the wavelengths of all its native pixels, in wavelength order;
  visits are ordered by MJD, then by the four keys in that order;
- in that order, the seed is the 8-byte BLAKE2b digest, read little-endian, of rint(10⁶ (MJD − first MJD)) and
  rint(1000 ln(median of the visit's good errors / σ)), each as little-endian int64, then the grid mask as uint8,
  visits by row; rint rounds halves to even.
"""

import hashlib
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from .bands import band_fluxes, full_weights

__all__ = ["Prepared", "measured_error_scale", "prepare", "signal_to_noise", "wavelength_grid"]


@dataclass
class Prepared:
    """One object's visits, ready for the network, in time order.

    Parameters
    ----------
    flux, log_error : numpy.ndarray
        (visits, pixels) float32: the flux over the visit scale on the model grid, and the log of its error over
        the same scale.
    measured : numpy.ndarray
        (visits, pixels) bool, True where a pixel is measured.
    scale, days : numpy.ndarray
        (visits,) float32 visit scales, and observer-frame days since the first visit kept.
    mjd : numpy.ndarray
        (visits,) float64 date of each visit kept.
    band_flux, band_error : numpy.ndarray
        (visits, 4) float64 band fluxes and errors, NaN where a band is not covered or no profile was given.
    seed : int
        The seed of the brightness term's null copies.
    visits : numpy.ndarray
        The input index of each visit kept.
    dropped : dict of int to str
        The input index of each visit dropped, with the reason.
    signal_to_noise : float
        The median S/N per pixel of the visits kept, combined (`signal_to_noise`).
    """

    flux: np.ndarray
    log_error: np.ndarray
    measured: np.ndarray
    scale: np.ndarray
    days: np.ndarray
    mjd: np.ndarray
    band_flux: np.ndarray
    band_error: np.ndarray
    seed: int
    visits: np.ndarray
    dropped: dict[int, str] = field(default_factory=dict)
    signal_to_noise: float = np.nan

    def band_rows(self):
        """The brightness term's rows: each measured band's date, band, flux and error."""
        visit, band = np.nonzero(np.isfinite(self.band_flux))
        return self.mjd[visit], band, self.band_flux[visit, band], self.band_error[visit, band]


class _Visit(NamedTuple):
    """One visit on the model grid, and what the seed, the S/N and the brightness term need from its native pixels."""

    flux: np.ndarray
    log_error: np.ndarray
    measured: np.ndarray
    scale: float
    native_wavelength: np.ndarray
    good: np.ndarray  # the native pixels that are not bad
    good_flux: np.ndarray  # the input flux and error of the good native pixels, float64, for the seed
    good_error: np.ndarray
    band_flux: np.ndarray
    band_error: np.ndarray


# share of the grid's ln λ span a visit may fall short by at each end; the grid points it misses are unmeasured
COVERAGE_SHORTFALL = 0.02
# measured_error_scale leaves out pixels whose two neighbours' mean flux is this many errors or more
BRIGHT = 10.0


def wavelength_grid(settings):
    """The model's observer-frame wavelength grid in Å, log-uniform, from ``preprocessing.json``."""
    grid = settings["wavelength_grid"]
    return np.geomspace(grid["minimum"], grid["maximum"], grid["bins"])


NETWORK_INPUTS = ("flux", "measured", "log_error", "scale", "days")


def network_inputs(prepared, device):
    """The prepared arrays the network takes, in the order of `NETWORK_INPUTS`, as tensors on ``device``
    (``measured`` as float32)."""
    import torch

    values = [getattr(prepared, name) for name in NETWORK_INPUTS]
    values = [value.astype(np.float32) if name == "measured" else value for name, value in zip(NETWORK_INPUTS, values)]
    return tuple(torch.from_numpy(np.ascontiguousarray(value)).to(device) for value in values)


def prepare(wavelength, flux, error, mjd, mask=None, *, settings, grid=None, error_scale=1.0, profile=None):
    """Prepare one object's visits.

    Parameters
    ----------
    wavelength, flux, error : sequence of array_like
        Lists with one array per visit, each visit on its own grid: observer-frame wavelength in Å, flux and its
        1-sigma error. One wavelength array may instead serve every visit.
    mjd : sequence of float
        Date of each visit.
    mask : sequence of array_like, optional
        True for pixels to ignore: a list with one array (or None) per visit.
    settings : dict
        The package's ``preprocessing.json``.
    grid : numpy.ndarray, optional
        The model's wavelength grid, ``wavelength_grid.npy``; by default, the one ``settings`` describes.
    error_scale : float
        The factor ``error`` has already been multiplied by.
    profile : tuple of numpy.ndarray, optional
        The model's noise profile (wavelength, sigma), for the band fluxes.

    Returns
    -------
    Prepared

    Raises
    ------
    ValueError
        If the errors are missing, the visits are inconsistent, or no visit can be used.
    """
    if error is None:
        raise ValueError("flux errors are required; give the 1σ error of every pixel")
    mjd = np.atleast_1d(np.asarray(mjd, dtype=np.float64))
    if not len(mjd):
        raise ValueError("no visit given; wavelength, flux, error and mjd are empty")
    visits = _visits(wavelength, flux, error, mask, len(mjd))
    undated = np.flatnonzero(~np.isfinite(mjd))
    if len(undated):
        raise ValueError(f"visit {undated[0]} has MJD {mjd[undated[0]]}; give every visit a finite MJD")
    limit = settings.get("maximum_visits")
    if limit is not None and len(mjd) > limit:
        raise ValueError(f"this model takes at most {limit} visits, and {len(mjd)} were given; leave some out")

    grid = wavelength_grid(settings) if grid is None else np.asarray(grid, dtype=np.float64)
    bands = None if profile is None else (profile, full_weights(profile))
    kept, dropped, prepared = [], {}, []
    for index, parts in enumerate(visits):
        visit = _prepare_visit(*parts, grid, bands)
        if isinstance(visit, str):
            dropped[index] = visit
        else:
            kept.append(index)
            prepared.append(visit)
    if not kept:
        raise ValueError("no usable visit: " + "; ".join(f"visit {i} {reason}" for i, reason in dropped.items()))
    sigma = np.median(np.concatenate([visit.good_error for visit in prepared]))
    ties = [_tie_keys(visit, sigma / error_scale, sigma) for visit in prepared]
    order = sorted(range(len(kept)), key=lambda i: (mjd[kept[i]], *ties[i]))
    kept, prepared = [kept[i] for i in order], [prepared[i] for i in order]
    flux, log_error, measured = (
        np.stack([getattr(visit, name) for visit in prepared]) for name in ("flux", "log_error", "measured")
    )
    dates = np.rint(1e6 * (mjd[kept] - mjd[kept[0]])).astype("<i8")
    spread = np.array([np.rint(1000.0 * np.log(np.median(v.good_error) / sigma)) for v in prepared]).astype("<i8")
    return Prepared(
        flux=flux,
        log_error=log_error,
        measured=measured,
        scale=np.array([visit.scale for visit in prepared], dtype=np.float32),
        days=(mjd[kept] - mjd[kept[0]]).astype(np.float32),
        mjd=mjd[kept],
        band_flux=np.stack([visit.band_flux for visit in prepared]),
        band_error=np.stack([visit.band_error for visit in prepared]),
        seed=_digest(dates.tobytes() + spread.tobytes() + measured.astype(np.uint8).tobytes()),
        visits=np.asarray(kept),
        dropped=dropped,
        signal_to_noise=signal_to_noise(prepared, grid),
    )


def signal_to_noise(visits, grid, trim=0.05, error_limit=3.0):
    """The median S/N per pixel of visits combined: their inverse-variance coadd on the native pixels of the visit
    with the most pixels (each other visit's pixels on the nearest of them, if within half a pixel), over the model's
    wavelength range less ``trim`` of it in ln λ at each end, without the pixels whose combined error is over
    ``error_limit`` times the median. NaN if no pixel is left."""
    reference = max(visits, key=lambda visit: len(visit.native_wavelength)).native_wavelength.astype(np.float64)
    half = 0.5 * np.gradient(reference)
    unit = np.median(np.concatenate([visit.good_error for visit in visits]))
    precision, weighted = np.zeros(len(reference)), np.zeros(len(reference))
    for visit in visits:
        good = visit.native_wavelength[visit.good].astype(np.float64)
        index = np.clip(np.searchsorted(reference, good), 1, len(reference) - 1)
        index = np.where(good - reference[index - 1] < reference[index] - good, index - 1, index)
        found = np.abs(good - reference[index]) <= half[index]
        weight = np.square(unit / visit.good_error[found])
        np.add.at(precision, index[found], weight)
        np.add.at(weighted, index[found], weight * visit.good_flux[found] / unit)
    low, high = np.log(grid[0]), np.log(grid[-1])
    edges = np.exp([low + trim * (high - low), high - trim * (high - low)])
    inside = (reference >= edges[0]) & (reference <= edges[1]) & (precision > 0.0)
    if not inside.any():
        return np.nan
    error = 1.0 / np.sqrt(precision[inside])
    kept = error <= error_limit * np.median(error)
    return float(np.median(weighted[inside][kept] / np.sqrt(precision[inside][kept])))


def measured_error_scale(wavelength, flux, error, mask=None, minimum=20):
    """The factor by which the visits' scatter exceeds their stated errors, never below 1.

    It compares each pixel with its neighbours two pixels either side, as DER_SNR does (Stoehr et al. 2008). Pixels
    that are masked, not finite, or brighter than about 10 errors are left out; it returns 1 if fewer than
    ``minimum`` comparisons remain. Noise correlated between neighbouring pixels makes it read low.

    Parameters
    ----------
    wavelength, flux, error, mask
        The visits, as `prepare` takes them.
    minimum : int
        The fewest comparisons to measure from.

    Returns
    -------
    float
    """
    ratios = []
    for wave, value, sigma, use in _visits(wavelength, flux, error, mask, len(flux)):
        finite = np.isfinite(wave)
        order = np.argsort(wave[finite])
        good = (use & np.isfinite(value) & np.isfinite(sigma) & (sigma > 0.0))[finite][order]
        value = np.where(good, value[finite][order], np.nan)
        sigma = np.where(good, sigma[finite][order], np.nan)
        difference = 2.0 * value[2:-2] - value[:-4] - value[4:]
        stated = np.sqrt(4.0 * sigma[2:-2] ** 2 + sigma[:-4] ** 2 + sigma[4:] ** 2)
        kept = np.isfinite(difference) & np.isfinite(stated)
        kept &= np.abs(value[:-4] + value[4:]) < 2 * BRIGHT * sigma[2:-2]
        ratios.append(difference[kept] / stated[kept])
    if sum(map(len, ratios)) < minimum:
        return 1.0
    return max(1.0, 1.4826 * float(np.median(np.abs(np.concatenate(ratios)))))


def _tie_keys(visit, flux_unit, error_unit):
    """The keys that order visits on one date: digests of the flux, the errors, the wavelengths and the mask."""
    parts = (visit.good_flux / flux_unit, visit.good_error / error_unit, visit.native_wavelength.astype(np.float64))
    return [_digest(np.rint(1000.0 * part).astype("<i8")) for part in parts] + [_digest(visit.good.astype(np.uint8))]


def _digest(data):
    """The 8-byte BLAKE2b digest of ``data``, as an integer read little-endian."""
    if not isinstance(data, bytes):
        data = data.tobytes()
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


def _visits(wavelength, flux, error, mask, count):
    """Per-visit float64 wavelength, flux and error, and a boolean mask of pixels to use."""
    flux, error = _per_visit(flux, "flux"), _per_visit(error, "error")
    if len(wavelength) and np.ndim(next(iter(wavelength))) == 0:  # one array for every visit
        wavelength = [np.asarray(wavelength, dtype=np.float64)] * len(flux)
    else:
        wavelength = _per_visit(wavelength, "wavelength")
    masks = [None] * len(flux) if mask is None else list(mask)
    if not len(wavelength) == len(flux) == len(error) == len(masks) == count:
        raise ValueError(
            "wavelength, flux, error, mask and mjd must give the same number of visits; they give "
            f"{len(wavelength)}, {len(flux)}, {len(error)}, {len(masks)} and {count}"
        )
    use = [np.ones(len(row), bool) if ignore is None else ~np.asarray(ignore, bool) for row, ignore in zip(flux, masks)]
    for index, parts in enumerate(zip(wavelength, flux, error, use)):
        if any(part.shape != parts[0].shape for part in parts):
            raise ValueError(
                f"visit {index}: wavelength, flux, error and mask must have the same length; their shapes "
                f"are {', '.join(str(part.shape) for part in parts)}"
            )
    return list(zip(wavelength, flux, error, use))


def _per_visit(values, name):
    rows = [np.asarray(row, dtype=np.float64) for row in values]
    if any(row.ndim != 1 for row in rows):
        raise ValueError(f"{name} must be a list with one 1-d array per visit")
    return rows


def _prepare_visit(wavelength, flux, error, use, grid, bands=None):
    """One visit on the model grid (`_Visit`), or the reason it cannot be used; `bands` is the noise profile and the
    bands' full weights."""
    finite = np.isfinite(wavelength)
    exact = wavelength[finite]
    wavelength = exact.astype(np.float32)
    order = np.argsort(wavelength)
    wavelength, exact = wavelength[order], exact[order]
    input_flux, input_error = flux[finite][order], np.where(use, error, np.nan)[finite][order]
    with np.errstate(over="ignore"):  # a value beyond float32's range becomes infinite, so its pixel is bad
        flux, error = input_flux.astype(np.float32), input_error.astype(np.float32)
    stated = np.isfinite(error) & (error > 0.0)
    # lower quartile of the positive errors, whatever the flux
    scale = max(float(np.quantile(error[stated], 0.25)) if stated.any() else 1.0, np.finfo(np.float32).tiny)
    good = stated & np.isfinite(flux)
    if good.sum() < 2:
        return "has fewer than two usable pixels"
    centres = wavelength.astype(np.float64)
    middle = 0.5 * (centres[:-1] + centres[1:])
    low, high = centres[0] - (middle[0] - centres[0]), centres[-1] + (centres[-1] - middle[-1])
    # how far short of each end of the grid the visit falls, in ln λ
    blue_short = np.log(low / grid[0]) if low > grid[0] else 0.0
    red_short = np.log(grid[-1] / high) if high < grid[-1] else 0.0
    allowed = COVERAGE_SHORTFALL * np.log(grid[-1] / grid[0])
    if max(blue_short, red_short) > allowed:
        unit = "; its wavelengths look like µm, and STRIDER needs Å" if centres[-1] < 100.0 else ""
        return (
            f"covers {max(low, 0.0):.0f}-{high:.0f} Å, and STRIDER needs a visit to reach from "
            f"{grid[0] * np.exp(allowed):.0f} Å or below to {grid[-1] * np.exp(-allowed):.0f} Å or above{unit}"
        )
    flux = np.where(good, flux / scale, 0.0).astype(np.float32)
    error = error / scale
    model_flux = np.interp(grid, wavelength, flux).astype(np.float32)
    # the interpolation's error, extrapolated beyond the end centres with the end pair's weights
    upper = np.clip(np.searchsorted(wavelength, grid, side="left"), 1, len(wavelength) - 1)
    lower = upper - 1
    span = wavelength[upper] - wavelength[lower]
    fraction = np.divide(grid - wavelength[lower], span, out=np.zeros_like(grid), where=span > 0.0)
    propagated = np.sqrt(
        np.square(1.0 - fraction) * np.square(error[lower]) + np.square(fraction) * np.square(error[upper])
    )
    measured = good[lower] & good[upper] & (span > 0.0) & np.isfinite(propagated) & (propagated > 0.0)
    if blue_short:  # at an end the visit falls short of, nothing beyond its last pixel
        measured &= grid >= centres[0]
    if red_short:
        measured &= grid <= centres[-1]
    if not measured.any():
        return "has no two neighbouring usable pixels in the model's wavelength range"
    band_flux, band_error = band_fluxes(exact, input_flux, input_error, *bands) if bands else np.full((2, 4), np.nan)
    return _Visit(
        flux=np.where(measured, model_flux, 0.0).astype(np.float32),
        log_error=np.where(measured, np.log(np.where(measured, propagated, 1.0)), 0.0).astype(np.float32),
        measured=measured,
        scale=scale,
        native_wavelength=wavelength,
        good=good,
        good_flux=input_flux[good],
        good_error=input_error[good],
        band_flux=band_flux,
        band_error=band_error,
    )
