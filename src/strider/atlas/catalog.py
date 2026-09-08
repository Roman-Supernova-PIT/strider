"""Load the small, explicit ONIR wavelength catalog."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


SPEED_OF_LIGHT_KMS = 299_792.458


@dataclass(frozen=True)
class OnirCatalog:
    names: tuple[str, ...]
    rest_wavelengths: np.ndarray
    half_width_kms: np.ndarray


def load_catalog(path: str | Path) -> OnirCatalog:
    catalog_path = Path(path).expanduser().resolve()
    with catalog_path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    features = document.get("features") if isinstance(document, dict) else None
    if not isinstance(features, list) or not features:
        raise ValueError(f"ONIR catalog has no features: {catalog_path}")
    names = tuple(str(feature["name"]) for feature in features)
    wavelengths = np.asarray(
        [float(feature["rest_wavelength"]) for feature in features], dtype=np.float32
    )
    widths = np.asarray(
        [float(feature["half_width_kms"]) for feature in features], dtype=np.float32
    )
    if len(set(names)) != len(names):
        raise ValueError("ONIR feature names must be unique")
    if not np.all(np.diff(wavelengths) > 0):
        raise ValueError("ONIR rest wavelengths must be strictly increasing")
    if np.any(widths <= 0):
        raise ValueError("ONIR velocity half-widths must be positive")
    return OnirCatalog(names, wavelengths, widths)


def feature_geometry(
    rest_grid: np.ndarray,
    catalog: OnirCatalog,
    maximum_radius_bins: int,
    allow_radius_clipping: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return center indices, per-feature radii, and a padded window mask."""
    log_step = float(np.median(np.diff(np.log(rest_grid))))
    velocity_per_bin = SPEED_OF_LIGHT_KMS * log_step
    centers = np.abs(
        np.log(rest_grid)[None, :] - np.log(catalog.rest_wavelengths)[:, None]
    ).argmin(axis=1)
    requested_radii = np.ceil(catalog.half_width_kms / velocity_per_bin).astype(np.int64)
    clipped = requested_radii > int(maximum_radius_bins)
    if np.any(clipped) and not allow_radius_clipping:
        details = ", ".join(
            f"{name} needs {radius} bins"
            for name, radius in zip(np.asarray(catalog.names)[clipped], requested_radii[clipped])
        )
        raise ValueError(
            f"maximum_radius_bins={maximum_radius_bins} truncates ONIR regions: {details}. "
            "Increase the limit; radius clipping is allowed only in explicit development tests."
        )
    radii = np.clip(requested_radii, 1, int(maximum_radius_bins))
    offsets = np.arange(-maximum_radius_bins, maximum_radius_bins + 1)
    window_mask = np.abs(offsets[None, :]) <= radii[:, None]
    return centers.astype(np.int64), radii, window_mask


def overlap_weights(centers: np.ndarray, window_mask: np.ndarray, bins: int) -> np.ndarray:
    """Downweight repeated coverage so overlapping windows are not independent votes."""
    radius = window_mask.shape[1] // 2
    coverage = np.zeros(int(bins), dtype=np.float64)
    feature_bins: list[np.ndarray] = []
    for center, active in zip(centers, window_mask):
        indices = center + np.arange(-radius, radius + 1)
        valid = active & (indices >= 0) & (indices < bins)
        selected = indices[valid]
        feature_bins.append(selected)
        coverage[selected] += 1.0
    weights = np.asarray(
        [np.sum(1.0 / coverage[selected]) if len(selected) else 0.0 for selected in feature_bins],
        dtype=np.float32,
    )
    positive = weights > 0
    if np.any(positive):
        weights[positive] /= weights[positive].mean()
    return weights
