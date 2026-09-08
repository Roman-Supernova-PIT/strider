"""Generate controlled observations on native detector wavelength bins.

Inputs are native-bin wavelength in Angstrom, clean simulation flux and a
source-free scale. Outputs are native-bin flux arrays. This simulation-only
module may receive clean flux; inference model modules must never receive it.
Read ``dataset.py`` next for resampling and view selection.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def repeatable_rng(*parts: object) -> np.random.Generator:
    """Return a generator whose state is fully determined by the supplied key."""
    key = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "little", signed=False))


def draw_reported_error_flux(
    clean_flux: np.ndarray,
    flux_error: np.ndarray,
    include_source: bool,
    rng: np.random.Generator,
    noise_scale: float = 1.0,
) -> np.ndarray:
    """Draw independent native-bin Gaussian noise using a reported error array."""
    signal = clean_flux if include_source else np.zeros_like(clean_flux)
    noise = rng.normal(0.0, flux_error).astype(np.float32)
    return signal + float(noise_scale) * noise


def draw_controlled_background_flux(
    wavelength_angstrom: np.ndarray,
    clean_flux: np.ndarray,
    background_scale: float,
    include_source: bool,
    settings: dict[str, Any],
    rng: np.random.Generator,
    noise_scale: float = 1.0,
) -> np.ndarray:
    """Draw the configured source-free background model on native bins."""
    flux, _ = draw_controlled_background_measurement(
        wavelength_angstrom=wavelength_angstrom,
        clean_flux=clean_flux,
        background_scale=background_scale,
        include_source=include_source,
        settings=settings,
        rng=rng,
        noise_scale=noise_scale,
    )
    return flux


def draw_controlled_background_measurement(
    wavelength_angstrom: np.ndarray,
    clean_flux: np.ndarray,
    background_scale: float,
    include_source: bool,
    settings: dict[str, Any],
    rng: np.random.Generator,
    noise_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw controlled flux and return the standard deviation of that draw."""
    midpoint = 0.5 * (float(wavelength_angstrom.min()) + float(wavelength_angstrom.max()))
    half_range = max(0.5 * float(np.ptp(wavelength_angstrom)), 1.0)
    coordinate = (wavelength_angstrom - midpoint) / half_range
    shape_strength = float(settings["background_shape_strength"])
    slope = rng.uniform(-shape_strength, shape_strength)
    curvature = rng.uniform(0.0, shape_strength)
    background_shape = np.clip(
        1.0 + slope * coordinate + curvature * coordinate**2,
        0.25,
        None,
    )
    scale = background_scale * np.exp(
        rng.normal(0.0, float(settings["background_scale_log_std"]))
    )
    variance = (scale * background_shape) ** 2
    source_fraction = float(settings.get("source_variance_fraction", 0.0))
    if source_fraction:
        variance += source_fraction * scale * np.abs(clean_flux)
    standard_deviation = np.sqrt(variance).astype(np.float32)
    noise = rng.normal(0.0, standard_deviation).astype(np.float32)
    signal = clean_flux if include_source else np.zeros_like(clean_flux)
    return (
        signal + float(noise_scale) * noise,
        float(noise_scale) * standard_deviation,
    )
