"""Clear public evidence plot for current STRIDER model packages."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from strider.current_io import ObservedSeriesInput


def save_current_evidence_map(
    result: dict[str, Any],
    data: ObservedSeriesInput,
    path: str | Path,
    *,
    object_id: str,
) -> Path:
    """Plot the best measured spectrum and the final class-redshift result."""
    if "joint_probability" not in result:
        raise ValueError("Evidence plotting requires return_joint=True")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = path.parent / ".matplotlib-cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib.pyplot as plt

    class_names = list(result["model"]["classes"])
    grid = np.asarray(result["redshift"]["grid"], dtype=np.float64)
    joint = np.asarray(result["joint_probability"], dtype=np.float64)
    if joint.shape != (len(class_names), len(grid)):
        raise ValueError("Joint probability does not match class and redshift axes")
    widths = _cell_widths(grid)
    joint_density = 100.0 * joint / widths[None, :]
    redshift_density = joint_density.sum(axis=0)
    probabilities = result["classification"]["probabilities"]
    class_probability = np.asarray(
        [float(probabilities[name]) for name in class_names], dtype=np.float64
    )
    best_index, best_snr = _best_visit(data)
    days = data.observer_time - np.min(data.observer_time)

    figure = plt.figure(figsize=(10.8, 8.6), dpi=180, facecolor="white")
    outer_grid = figure.add_gridspec(
        2,
        1,
        height_ratios=(1.15, 2.55),
        left=0.09,
        right=0.96,
        bottom=0.09,
        top=0.89,
        hspace=0.25,
    )
    lower_grid = outer_grid[1].subgridspec(
        2,
        2,
        width_ratios=(4.5, 1.45),
        height_ratios=(1.55, 1.0),
        hspace=0.08,
        wspace=0.28,
    )
    spectrum_axis = figure.add_subplot(outer_grid[0])
    evidence_axis = figure.add_subplot(lower_grid[0, 0])
    redshift_axis = figure.add_subplot(lower_grid[1, 0], sharex=evidence_axis)
    class_axis = figure.add_subplot(lower_grid[:, 1])

    _plot_spectrum(spectrum_axis, data, best_index)
    spectrum_axis.text(
        0.01,
        0.96,
        (
            f"{len(data.observer_time)} visits • best-S/N spectrum: "
            f"{days[best_index]:+.0f} observer days • S/N {best_snr:.2f}"
        ),
        transform=spectrum_axis.transAxes,
        ha="left",
        va="top",
        fontsize=11.5,
        color="#333333",
    )

    edges = _cell_edges(grid)
    image = evidence_axis.pcolormesh(
        edges,
        np.arange(len(class_names) + 1) - 0.5,
        joint_density,
        shading="flat",
        cmap="magma",
        rasterized=True,
    )
    evidence_axis.set_yticks(np.arange(len(class_names)), class_names)
    evidence_axis.invert_yaxis()
    evidence_axis.set_ylabel("Transient class")
    evidence_axis.tick_params(axis="x", labelbottom=False)
    colorbar = figure.colorbar(image, ax=evidence_axis, pad=0.015, fraction=0.035)
    colorbar.set_label("Joint probability density (% per unit redshift)", fontsize=9.5)
    colorbar.ax.tick_params(labelsize=9)

    redshift_axis.fill_between(grid, redshift_density, color="#4C78A8", alpha=0.30)
    redshift_axis.plot(grid, redshift_density, color="#285A8E", linewidth=2.0)
    primary = result["redshift"]["primary_basin"]
    redshift_axis.axvline(
        float(primary["peak_redshift"]),
        color="#D1495B",
        linewidth=2.0,
        label="STRIDER redshift",
    )
    for index, candidate in enumerate(result["redshift"]["candidate_basins"][1:]):
        redshift_axis.axvline(
            float(candidate["peak_redshift"]),
            color="#D1495B",
            linewidth=1.3,
            linestyle="--",
            alpha=0.75,
            label="Alternative solution" if index == 0 else None,
        )
    redshift_axis.set_xlabel("Redshift")
    redshift_axis.set_ylabel("Probability density\n(% per unit redshift)")
    redshift_axis.legend(loc="upper right", frameon=False, fontsize=9.5)

    order = np.argsort(class_probability)
    y = np.arange(len(class_names))
    bars = class_axis.barh(
        y,
        100.0 * class_probability[order],
        color="#4C78A8",
        alpha=0.92,
        height=0.68,
    )
    class_axis.set_yticks(y, np.asarray(class_names, dtype=object)[order])
    class_axis.set_xlim(0.0, 100.0)
    class_axis.set_xlabel("Class probability (%)")
    class_axis.grid(axis="x", color="#DDDDDD", linewidth=0.7, alpha=0.8)
    class_axis.set_axisbelow(True)
    for bar, probability in zip(bars, class_probability[order], strict=True):
        class_axis.text(
            min(100.0 * probability + 2.0, 97.0),
            bar.get_y() + 0.5 * bar.get_height(),
            f"{100.0 * probability:.1f}",
            va="center",
            ha="left" if probability < 0.93 else "right",
            fontsize=10,
            fontweight="bold",
            color="#222222",
        )

    for axis in (spectrum_axis, evidence_axis, redshift_axis, class_axis):
        axis.tick_params(labelsize=10.5)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("#555555")
            spine.set_linewidth(0.8)

    classification = result["classification"]
    signal = result["signal"]
    signal_text = (
        f" • {signal['grade']} signal"
        if signal.get("grade") is not None
        else " • signal calibration pending"
    )
    figure.suptitle(
        (
            f"{object_id}\n"
            f"STRIDER: {classification['class']} "
            f"({100.0 * classification['confidence']:.1f}%) • "
            f"redshift {result['redshift']['z_STRIDER']:.4f}{signal_text}"
        ),
        x=0.09,
        y=0.975,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
        linespacing=1.35,
    )
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _plot_spectrum(axis: Any, data: ObservedSeriesInput, index: int) -> None:
    wavelength = np.asarray(data.wavelength, dtype=np.float64)
    flux = np.asarray(data.flux, dtype=np.float64)
    error = np.asarray(data.flux_error, dtype=np.float64)
    if flux.ndim == 1:
        flux = flux[None, :]
        error = error[None, :]
    valid = (
        np.isfinite(wavelength)
        & np.isfinite(flux[index])
        & np.isfinite(error[index])
        & (error[index] > 0.0)
    )
    scale = float(np.quantile(error[index, valid], 0.25))
    measured = flux[index] / scale
    uncertainty = error[index] / scale
    smooth = _masked_smooth(measured, valid, width=5)
    axis.fill_between(
        wavelength[valid],
        measured[valid] - uncertainty[valid],
        measured[valid] + uncertainty[valid],
        color="#8FB7D7",
        alpha=0.18,
        linewidth=0.0,
    )
    axis.plot(
        wavelength[valid],
        measured[valid],
        color="#7A9BB8",
        alpha=0.45,
        linewidth=0.7,
        label="Measured FLAM",
    )
    axis.plot(
        wavelength[valid],
        smooth[valid],
        color="#1F4E79",
        linewidth=1.8,
        label="Five-bin mean",
    )
    axis.axhline(0.0, color="#777777", linewidth=0.7, alpha=0.65)
    axis.set_xlim(float(np.nanmin(wavelength[valid])), float(np.nanmax(wavelength[valid])))
    axis.set_ylabel("FLAM / visit\nnoise scale")
    axis.set_xlabel("Observed wavelength (Å)")
    axis.legend(loc="upper right", frameon=False, ncol=2, fontsize=9.5)


def _best_visit(data: ObservedSeriesInput) -> tuple[int, float]:
    wavelength = np.asarray(data.wavelength, dtype=np.float64)
    flux = np.atleast_2d(np.asarray(data.flux, dtype=np.float64))
    error = np.atleast_2d(np.asarray(data.flux_error, dtype=np.float64))
    log_minimum = np.log(np.nanmin(wavelength))
    log_maximum = np.log(np.nanmax(wavelength))
    interior = (
        (wavelength >= np.exp(log_minimum + 0.05 * (log_maximum - log_minimum)))
        & (wavelength <= np.exp(log_minimum + 0.95 * (log_maximum - log_minimum)))
    )
    values = []
    for visit_flux, visit_error in zip(flux, error, strict=True):
        valid = (
            interior
            & np.isfinite(visit_flux)
            & np.isfinite(visit_error)
            & (visit_error > 0.0)
        )
        if not valid.any():
            values.append(float("-inf"))
            continue
        median_error = float(np.median(visit_error[valid]))
        valid &= visit_error <= 3.0 * median_error
        values.append(float(np.median(visit_flux[valid] / visit_error[valid])))
    index = int(np.argmax(values))
    return index, float(values[index])


def _masked_smooth(values: np.ndarray, valid: np.ndarray, width: int) -> np.ndarray:
    kernel = np.ones(int(width), dtype=np.float64)
    numerator = np.convolve(np.where(valid, values, 0.0), kernel, mode="same")
    denominator = np.convolve(valid.astype(np.float64), kernel, mode="same")
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )


def _cell_widths(grid: np.ndarray) -> np.ndarray:
    return np.diff(_cell_edges(grid))


def _cell_edges(grid: np.ndarray) -> np.ndarray:
    edges = np.empty(len(grid) + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (grid[:-1] + grid[1:])
    edges[0] = grid[0]
    edges[-1] = grid[-1]
    return edges
