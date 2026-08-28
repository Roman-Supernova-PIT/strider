"""Truth-free summaries of multimodal redshift posterior probability."""

from __future__ import annotations

from typing import Any

import numpy as np


def posterior_basin_candidates(
    grid: np.ndarray,
    probability_mass: np.ndarray,
    cell_width: np.ndarray,
    *,
    maximum_candidates: int = 3,
    smoothing_sigma_bins: float = 2.0,
    minimum_peak_height_ratio: float = 0.10,
    minimum_peak_mass: float = 0.05,
) -> list[dict[str, Any]]:
    """Return the strongest distinct redshift basins without merging them.

    A lightly smoothed probability density identifies basin topology. Reported
    masses, quantiles and intervals always come from the original probability
    mass. The first candidate contains the highest posterior-density peak.
    """
    if maximum_candidates < 1:
        raise ValueError("maximum_candidates must be positive")
    grid = np.asarray(grid, dtype=np.float64)
    mass = np.asarray(probability_mass, dtype=np.float64)
    width = np.asarray(cell_width, dtype=np.float64)
    if grid.ndim != 1 or mass.shape != grid.shape or width.shape != grid.shape:
        raise ValueError("Posterior candidate inputs must be one-dimensional and aligned")
    if np.any(width <= 0.0) or not np.isfinite(width).all():
        raise ValueError("Posterior cell widths must be finite and positive")
    total = float(mass.sum())
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("Posterior probability must have positive finite mass")
    mass = mass / total
    density = mass / width
    smoothed = _gaussian_smooth_1d(density, smoothing_sigma_bins)
    peak_indices = _local_peak_indices(smoothed)
    dominant_index = int(np.argmax(smoothed))
    if dominant_index not in peak_indices:
        peak_indices = np.sort(np.append(peak_indices, dominant_index)).astype(int)

    boundaries = [0]
    for left, right in zip(peak_indices[:-1], peak_indices[1:], strict=True):
        valley = int(left + np.argmin(smoothed[left : right + 1]))
        boundaries.append(valley + 1)
    boundaries.append(len(grid))
    basin_mass = np.asarray(
        [
            mass[left:right].sum()
            for left, right in zip(boundaries[:-1], boundaries[1:], strict=True)
        ],
        dtype=np.float64,
    )
    peak_height = smoothed[peak_indices]
    log_prominence = _log_peak_to_saddle_ratios(smoothed, peak_indices)
    dominant_position = int(np.argmax(peak_height))
    dominant_height = max(float(peak_height[dominant_position]), 1.0e-300)
    qualifying = [dominant_position]
    qualifying.extend(
        position
        for position in range(len(peak_indices))
        if position != dominant_position
        and peak_height[position] / dominant_height >= minimum_peak_height_ratio
        and basin_mass[position] >= minimum_peak_mass
    )
    secondary = sorted(
        qualifying[1:],
        key=lambda position: (peak_height[position], basin_mass[position]),
        reverse=True,
    )
    ordered = [dominant_position, *secondary][:maximum_candidates]
    largest_mass_position = int(np.argmax(basin_mass))

    strongest_competitor_position = secondary[0] if secondary else None
    if strongest_competitor_position is None:
        strongest_competitor_redshift = float("nan")
        primary_competitor_saddle_contrast = float("nan")
        primary_to_competitor_height_ratio = float("nan")
    else:
        primary_index = int(peak_indices[dominant_position])
        competitor_index = int(peak_indices[strongest_competitor_position])
        lower_peak = min(primary_index, competitor_index)
        upper_peak = max(primary_index, competitor_index)
        connecting_saddle = float(smoothed[lower_peak : upper_peak + 1].min())
        primary_competitor_saddle_contrast = float(
            np.log(dominant_height)
            - np.log(max(connecting_saddle, 1.0e-300))
        )
        primary_to_competitor_height_ratio = float(
            dominant_height
            / max(float(peak_height[strongest_competitor_position]), 1.0e-300)
        )
        strongest_competitor_redshift = _interpolated_peak_at_index(
            grid, smoothed, competitor_index
        )

    candidates: list[dict[str, Any]] = []
    for position in ordered:
        left = int(boundaries[position])
        right = int(boundaries[position + 1])
        conditional_mass = mass[left:right]
        conditional_mass = conditional_mass / conditional_mass.sum()
        local_grid = grid[left:right]
        lower_68, upper_68 = _central_interval(local_grid, conditional_mass, 0.68)
        candidates.append(
            {
                "peak_redshift": _interpolated_peak_at_index(
                    grid, smoothed, int(peak_indices[position])
                ),
                "median_redshift": _posterior_quantile(
                    local_grid, conditional_mass, 0.5
                ),
                "lower_68": lower_68,
                "upper_68": upper_68,
                "mass": float(basin_mass[position]),
                "is_largest_mass_basin": bool(position == largest_mass_position),
                "height_ratio": float(peak_height[position] / dominant_height),
                "peak_density": float(peak_height[position]),
                "log_peak_to_saddle_ratio": float(log_prominence[position]),
                "basin_lower": float(local_grid[0]),
                "basin_upper": float(local_grid[-1]),
                "left_index": left,
                "right_index": right,
            }
        )
    candidates[0].update(
        {
            "strongest_competitor_peak_redshift": float(
                strongest_competitor_redshift
            ),
            "log_peak_to_strongest_competitor_saddle_ratio": float(
                primary_competitor_saddle_contrast
            ),
            "primary_to_strongest_competitor_height_ratio": float(
                primary_to_competitor_height_ratio
            ),
        }
    )
    return candidates


def _central_interval(
    grid: np.ndarray, probability: np.ndarray, mass: float
) -> tuple[float, float]:
    tail = 0.5 * (1.0 - mass)
    return (
        _posterior_quantile(grid, probability, tail),
        _posterior_quantile(grid, probability, 1.0 - tail),
    )


def _posterior_quantile(
    grid: np.ndarray, probability: np.ndarray, quantile: float
) -> float:
    mass = np.asarray(probability, dtype=np.float64)
    mass = mass / mass.sum()
    midpoint_cdf = np.cumsum(mass) - 0.5 * mass
    return float(
        np.interp(
            quantile,
            midpoint_cdf,
            np.asarray(grid, dtype=np.float64),
            left=float(grid[0]),
            right=float(grid[-1]),
        )
    )


def _interpolated_peak_at_index(
    grid: np.ndarray, probability: np.ndarray, index: int
) -> float:
    if index == 0 or index == len(grid) - 1:
        return float(grid[index])
    x = np.log1p(grid[index - 1 : index + 2])
    y = np.log(np.clip(probability[index - 1 : index + 2], 1.0e-300, None))
    curvature, slope, _ = np.polyfit(x, y, 2)
    if not np.isfinite(curvature) or curvature >= 0.0:
        return float(grid[index])
    peak = float(np.clip(-slope / (2.0 * curvature), x[0], x[-1]))
    return float(np.expm1(peak))


def _log_peak_to_saddle_ratios(
    density: np.ndarray, peak_indices: np.ndarray
) -> np.ndarray:
    values = np.asarray(density, dtype=np.float64)
    peaks = np.asarray(peak_indices, dtype=np.int64)
    heights = values[peaks]
    result = np.full(len(peaks), np.inf, dtype=np.float64)
    for position, peak in enumerate(peaks):
        higher = np.flatnonzero(heights > heights[position])
        if not len(higher):
            continue
        saddle = max(
            float(
                values[
                    min(peak, peaks[other]) : max(peak, peaks[other]) + 1
                ].min()
            )
            for other in higher
        )
        result[position] = float(
            np.log(max(float(heights[position]), 1.0e-300))
            - np.log(max(saddle, 1.0e-300))
        )
    return result


def _gaussian_smooth_1d(values: np.ndarray, sigma_bins: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if sigma_bins <= 0.0:
        return values.copy()
    radius = max(1, int(np.ceil(4.0 * sigma_bins)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(offsets / sigma_bins))
    kernel /= kernel.sum()
    padded = np.pad(values, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _local_peak_indices(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 1:
        return np.asarray([0], dtype=int)
    interior = np.flatnonzero(
        (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])
    ) + 1
    edges = []
    if values[0] > values[1]:
        edges.append(0)
    if values[-1] > values[-2]:
        edges.append(len(values) - 1)
    peaks = np.asarray([*edges[:1], *interior, *edges[1:]], dtype=int)
    if peaks.size == 0:
        peaks = np.asarray([int(np.argmax(values))], dtype=int)
    return np.unique(peaks)
