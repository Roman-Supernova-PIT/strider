#!/usr/bin/env python3
"""Diagnose Sundial redshift plateaus, aliases, and quality dependence."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from strider.evaluation.evaluate import _posterior_peak_summary  # noqa: E402
from strider.model.redshift_scan import (  # noqa: E402
    build_redshift_grid,
    redshift_cell_widths,
)


COMPETING_PEAK_MASS_RATIO = 0.5
DENSITY_SMOOTHING_SIGMA = 2.4
DENSITY_DISPLAY_FLOOR = 0.065


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _nmad(values: np.ndarray) -> float:
    centre = np.nanmedian(values)
    return float(1.4826 * np.nanmedian(np.abs(values - centre)))


def _smooth_density(axis, truth: np.ndarray, estimate: np.ndarray, *, vmax: float):
    counts, _, _ = np.histogram2d(
        truth,
        estimate,
        bins=180,
        range=((0.0, 3.0), (0.0, 3.0)),
    )
    density = gaussian_filter(counts.T, sigma=DENSITY_SMOOTHING_SIGMA)
    density[density < DENSITY_DISPLAY_FLOOR] = np.nan
    image = axis.imshow(
        density,
        origin="lower",
        extent=(0.0, 3.0, 0.0, 3.0),
        cmap="viridis",
        norm=LogNorm(vmin=0.035, vmax=vmax),
        interpolation="bilinear",
        aspect="equal",
    )
    axis.plot([0.0, 3.0], [0.0, 3.0], color="0.2", linewidth=1.15)
    axis.set_xlim(0.0, 3.0)
    axis.set_ylim(0.0, 3.0)
    axis.grid(False)
    return image


def _smooth_residual_density(
    axis,
    truth: np.ndarray,
    normalized_residual: np.ndarray,
    *,
    vmax: float,
):
    counts, _, _ = np.histogram2d(
        truth,
        normalized_residual,
        bins=(180, 160),
        range=((0.0, 3.0), (-0.5, 0.5)),
    )
    density = gaussian_filter(counts.T, sigma=DENSITY_SMOOTHING_SIGMA)
    density[density < DENSITY_DISPLAY_FLOOR] = np.nan
    image = axis.imshow(
        density,
        origin="lower",
        extent=(0.0, 3.0, -0.5, 0.5),
        cmap="viridis",
        norm=LogNorm(vmin=0.035, vmax=vmax),
        interpolation="bilinear",
        aspect="auto",
    )
    axis.axhline(0.0, color="0.2", linewidth=1.15)
    axis.set_xlim(0.0, 3.0)
    axis.set_ylim(-0.5, 0.5)
    axis.grid(False)
    return image


def _redshift_summary(
    truth: np.ndarray,
    estimates: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    edges = np.arange(0.0, 3.0001, 0.25)
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = (truth >= lower) & (truth < upper)
        for name, estimate in estimates.items():
            delta = estimate[selected] - truth[selected]
            normalized = delta / (1.0 + truth[selected])
            rows.append(
                {
                    "summary": name,
                    "redshift_min": lower,
                    "redshift_max": upper,
                    "redshift_mid": 0.5 * (lower + upper),
                    "n_objects": int(selected.sum()),
                    "median_abs_delta_z": float(np.median(np.abs(delta))),
                    "nmad_normalized_delta_z": _nmad(normalized),
                    "outlier_fraction_abs_delta_z_gt_0p1": float(
                        np.mean(np.abs(delta) > 0.1)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _plot_summary_comparison(
    truth: np.ndarray,
    estimates: dict[str, np.ndarray],
    summary: pd.DataFrame,
    output: Path,
) -> None:
    density_maximum = 1.0
    for estimate in (estimates["posterior median"], estimates["density mode"]):
        counts, _, _ = np.histogram2d(
            truth, estimate, bins=180, range=((0.0, 3.0), (0.0, 3.0))
        )
        density_maximum = max(
            density_maximum,
            float(
                np.max(
                    gaussian_filter(counts.T, sigma=DENSITY_SMOOTHING_SIGMA)
                )
            ),
        )

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.1), constrained_layout=True)
    image = _smooth_density(
        axes[0, 0], truth, estimates["posterior median"], vmax=density_maximum
    )
    _smooth_density(
        axes[0, 1], truth, estimates["density mode"], vmax=density_maximum
    )
    axes[0, 0].set_title("posterior median")
    axes[0, 1].set_title("dominant peak")
    axes[0, 0].set_ylabel("predicted redshift")
    for axis in axes[0]:
        axis.set_xlabel("true redshift")
    colourbar = fig.colorbar(image, ax=axes[0], fraction=0.028, pad=0.02)
    colourbar.set_label("objects")

    colours = {
        "posterior median": "#247BA0",
        "density mode": "#D95F45",
        "posterior mean": "#6B6B6B",
    }
    for name, colour in colours.items():
        selected = summary[summary["summary"] == name]
        axes[1, 0].plot(
            selected["redshift_mid"],
            selected["median_abs_delta_z"],
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            color=colour,
            label=name,
        )
        axes[1, 1].plot(
            selected["redshift_mid"],
            selected["outlier_fraction_abs_delta_z_gt_0p1"],
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            color=colour,
        )
    axes[1, 0].set_ylabel(r"median $|\Delta z|$")
    axes[1, 1].set_ylabel(r"fraction with $|\Delta z|>0.1$")
    axes[1, 1].set_ylim(-0.03, 1.03)
    for axis in axes[1]:
        axis.set_xlim(0.0, 3.0)
        axis.set_xlabel("true redshift")
        axis.grid(False)
    axes[1, 0].legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _binned_quality(
    values: np.ndarray,
    delta: np.ndarray,
    edges: np.ndarray,
    labels: list[str],
    quantity: str,
) -> pd.DataFrame:
    rows = []
    for lower, upper, label in zip(edges[:-1], edges[1:], labels, strict=True):
        selected = (values >= lower) & (values < upper)
        if not np.any(selected):
            continue
        rows.append(
            {
                "quantity": quantity,
                "bin": label,
                "n_objects": int(selected.sum()),
                "median_abs_delta_z": float(np.median(np.abs(delta[selected]))),
                "outlier_fraction_abs_delta_z_gt_0p1": float(
                    np.mean(np.abs(delta[selected]) > 0.1)
                ),
            }
        )
    return pd.DataFrame(rows)


def _plot_quality(
    frame: pd.DataFrame,
    truth: np.ndarray,
    mode: np.ndarray,
    output: Path,
) -> pd.DataFrame:
    delta = mode - truth
    snr = frame["coadded_clean_signal_to_noise"].to_numpy(float)
    visits = frame["visit_count"].to_numpy(float)
    width = frame["redshift_68_interval_width"].to_numpy(float)
    quality = pd.concat(
        [
            _binned_quality(
                snr,
                delta,
                np.array([0, 0.25, 0.5, 1, 2, 4, 8, np.inf]),
                ["<0.25", "0.25–0.5", "0.5–1", "1–2", "2–4", "4–8", "8+"],
                "coadded S/N",
            ),
            _binned_quality(
                visits,
                delta,
                np.array([1, 2, 5, 9, 17, 33]),
                ["1", "2–4", "5–8", "9–16", "17–32"],
                "visits",
            ),
            _binned_quality(
                width,
                delta,
                np.array([0, 0.05, 0.1, 0.25, 0.5, 1, np.inf]),
                ["<0.05", "0.05–0.1", "0.1–0.25", "0.25–0.5", "0.5–1", "1+"],
                "68% width",
            ),
        ],
        ignore_index=True,
    )

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), constrained_layout=True)
    for axis, quantity, title in zip(
        axes.flat[:3],
        ("coadded S/N", "visits", "68% width"),
        ("coadded S/N", "spectra available", "posterior width"),
        strict=True,
    ):
        selected = quality[quality["quantity"] == quantity]
        x = np.arange(len(selected))
        axis.plot(
            x,
            selected["outlier_fraction_abs_delta_z_gt_0p1"],
            marker="o",
            color="#247BA0",
            linewidth=1.5,
        )
        axis.set_xticks(x, selected["bin"], rotation=35, ha="right")
        axis.set_ylim(-0.03, 1.03)
        axis.set_title(title)
        axis.set_ylabel(r"fraction with $|\Delta z|>0.1$")
        axis.grid(False)

    normalized = delta / (1.0 + truth)
    outliers = np.abs(normalized) > 0.05
    axes[1, 1].hist(
        normalized[outliers],
        bins=np.linspace(-0.8, 0.8, 81),
        color="#247BA0",
        alpha=0.9,
    )
    axes[1, 1].axvline(0.0, color="0.3", linewidth=0.8)
    axes[1, 1].set_xlim(-0.8, 0.8)
    axes[1, 1].set_xlabel(r"$\Delta z/(1+z)$")
    axes[1, 1].set_ylabel("objects")
    axes[1, 1].set_title("outlier residuals")
    axes[1, 1].grid(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return quality


def _plot_quality_cuts(
    predictions: pd.DataFrame,
    ia: pd.DataFrame,
    output: Path,
) -> pd.DataFrame:
    cuts = (
        ("No S/N cut", 0.0),
        ("S/N $> 1$", 1.0),
        ("S/N $> 1.5$", 1.5),
        ("S/N $> 3$", 3.0),
        ("S/N $> 5$", 5.0),
    )
    n_columns = len(cuts)
    cohorts = (("all objects", predictions), ("Ia", ia))
    density_maximum = 1.0
    for _, cohort in cohorts:
        truth = cohort["true_redshift"].to_numpy(float)
        mode = cohort["posterior_density_mode_redshift"].to_numpy(float)
        snr = cohort["coadded_clean_signal_to_noise"].to_numpy(float)
        for _, threshold in cuts:
            selected = snr > threshold if threshold > 0.0 else np.ones(len(cohort), bool)
            redshift_counts, _, _ = np.histogram2d(
                truth[selected],
                mode[selected],
                bins=180,
                range=((0.0, 3.0), (0.0, 3.0)),
            )
            density_maximum = max(
                density_maximum,
                float(
                    np.max(
                        gaussian_filter(
                            redshift_counts.T,
                            sigma=DENSITY_SMOOTHING_SIGMA,
                        )
                    )
                ),
            )
    ia_truth = ia["true_redshift"].to_numpy(float)
    ia_mode = ia["posterior_density_mode_redshift"].to_numpy(float)
    ia_snr = ia["coadded_clean_signal_to_noise"].to_numpy(float)
    for _, threshold in cuts:
        selected = ia_snr > threshold if threshold > 0.0 else np.ones(len(ia), bool)
        normalized = (ia_mode[selected] - ia_truth[selected]) / (1.0 + ia_truth[selected])
        residual_counts, _, _ = np.histogram2d(
            ia_truth[selected],
            normalized,
            bins=(180, 160),
            range=((0.0, 3.0), (-0.5, 0.5)),
        )
        density_maximum = max(
            density_maximum,
            float(
                np.max(
                    gaussian_filter(
                        residual_counts.T,
                        sigma=DENSITY_SMOOTHING_SIGMA,
                    )
                )
            ),
        )

    fig, axes = plt.subplots(
        3,
        n_columns,
        figsize=(18.0, 10.4),
        sharex=True,
        sharey="row",
        gridspec_kw={"height_ratios": (1.0, 1.0, 0.72)},
        constrained_layout=True,
    )
    rows = []
    image = None
    for row, (cohort_name, cohort) in enumerate(cohorts):
        truth = cohort["true_redshift"].to_numpy(float)
        mode = cohort["posterior_density_mode_redshift"].to_numpy(float)
        snr = cohort["coadded_clean_signal_to_noise"].to_numpy(float)
        for column, (label, threshold) in enumerate(cuts):
            selected = snr > threshold if threshold > 0.0 else np.ones(len(cohort), bool)
            image = _smooth_density(
                axes[row, column], truth[selected], mode[selected], vmax=density_maximum
            )
            delta = mode[selected] - truth[selected]
            normalized = delta / (1.0 + truth[selected])
            axes[row, column].text(
                0.96,
                0.95,
                f"N = {int(selected.sum()):,}",
                transform=axes[row, column].transAxes,
                ha="right",
                va="top",
                fontsize=10.5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.8},
            )
            rows.append(
                {
                    "cohort": cohort_name,
                    "selection": label.replace("$", ""),
                    "n_objects": int(selected.sum()),
                    "median_abs_delta_z": float(np.median(np.abs(delta))),
                    "median_normalized_delta_z": float(np.median(normalized)),
                    "nmad_normalized_delta_z": _nmad(normalized),
                    "outlier_fraction_abs_delta_z_gt_0p1": float(
                        np.mean(np.abs(delta) > 0.1)
                    ),
                }
            )
    for column, (label, threshold) in enumerate(cuts):
        axes[0, column].set_title(label, fontsize=15, fontweight="semibold", pad=10)
        selected = ia_snr > threshold if threshold > 0.0 else np.ones(len(ia), bool)
        normalized = (ia_mode[selected] - ia_truth[selected]) / (1.0 + ia_truth[selected])
        _smooth_residual_density(
            axes[2, column],
            ia_truth[selected],
            normalized,
            vmax=density_maximum,
        )
        axes[2, column].text(
            0.96,
            0.95,
            f"N = {int(selected.sum()):,}",
            transform=axes[2, column].transAxes,
            ha="right",
            va="top",
            fontsize=10.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.8},
        )
    axes[0, 0].set_ylabel(r"$z_{\mathrm{STRIDER}}$", fontsize=14)
    axes[1, 0].set_ylabel(r"$z_{\mathrm{STRIDER}}$", fontsize=14)
    axes[2, 0].set_ylabel(r"$\Delta z/(1+z_{\mathrm{true}})$", fontsize=14)
    for axis, row_label in zip(
        axes[:2, 0], ("All objects", "Type Ia"), strict=True
    ):
        axis.text(
            0.04,
            0.95,
            row_label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=11.5,
            fontweight="semibold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.8},
        )
    for axis in axes[2]:
        axis.set_xlabel(r"$z_{\mathrm{true}}$", fontsize=13)
    for axis in axes.flat:
        axis.tick_params(axis="both", which="major", labelsize=11, width=1.0, length=4.5)
        for spine in axis.spines.values():
            spine.set_linewidth(1.0)
    assert image is not None
    colourbar = fig.colorbar(image, ax=axes, fraction=0.026, pad=0.02)
    colourbar.set_label("Object density", fontsize=13)
    colourbar.ax.tick_params(labelsize=11)
    fig.suptitle("Sundial redshift performance", fontsize=17, fontweight="semibold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def _peak_diagnostics(ia: pd.DataFrame) -> pd.DataFrame:
    grid = build_redshift_grid(0.0, 3.0, 500, "log1p")
    cell_width = redshift_cell_widths(grid)
    summaries = [
        _posterior_peak_summary(grid, probability, cell_width)
        for probability in ia["redshift_probability"]
    ]
    result = ia[
        [
            "snid",
            "true_redshift",
            "posterior_density_mode_redshift",
            "coadded_clean_signal_to_noise",
        ]
    ].reset_index(drop=True)
    result["posterior_distinct_peak_count"] = [
        summary["distinct_peak_count"] for summary in summaries
    ]
    result["posterior_secondary_peak_redshift"] = [
        summary["secondary_redshift"] for summary in summaries
    ]
    result["posterior_secondary_to_dominant_mass_ratio"] = [
        summary["secondary_to_dominant_mass_ratio"] for summary in summaries
    ]
    result["competing_peak"] = (
        result["posterior_secondary_to_dominant_mass_ratio"]
        >= COMPETING_PEAK_MASS_RATIO
    )
    normalized = (
        result["posterior_density_mode_redshift"] - result["true_redshift"]
    ) / (1.0 + result["true_redshift"])
    result["normalized_delta_z"] = normalized
    result["lower_alias_branch"] = (
        (result["coadded_clean_signal_to_noise"] > 1.5)
        & result["true_redshift"].between(0.7, 2.1, inclusive="neither")
        & normalized.between(-0.17, -0.09, inclusive="neither")
    )
    return result


def _peak_summary(peak_diagnostics: pd.DataFrame) -> pd.DataFrame:
    normalized = peak_diagnostics["normalized_delta_z"]
    signal_to_noise = peak_diagnostics["coadded_clean_signal_to_noise"]
    selections = {
        "all Ia": np.ones(len(peak_diagnostics), dtype=bool),
        "Ia, S/N > 1": signal_to_noise > 1.0,
        "Ia, S/N > 1.5": signal_to_noise > 1.5,
        "lower alias branch": peak_diagnostics["lower_alias_branch"].to_numpy(bool),
        "high-S/N redshift core": (signal_to_noise > 1.5) & (normalized.abs() < 0.03),
    }
    rows = []
    for name, selected in selections.items():
        cohort = peak_diagnostics.loc[selected]
        rows.append(
            {
                "cohort": name,
                "n_objects": len(cohort),
                "competing_peak_fraction": float(cohort["competing_peak"].mean()),
                "median_secondary_to_dominant_mass_ratio": float(
                    cohort["posterior_secondary_to_dominant_mass_ratio"].median()
                ),
                "median_normalized_delta_z": float(cohort["normalized_delta_z"].median()),
            }
        )
    return pd.DataFrame(rows)


def _overall_summary(truth: np.ndarray, estimates: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, estimate in estimates.items():
        delta = estimate - truth
        normalized = delta / (1.0 + truth)
        rows.append(
            {
                "summary": name,
                "n_objects": len(truth),
                "median_delta_z": float(np.median(delta)),
                "median_abs_delta_z": float(np.median(np.abs(delta))),
                "nmad_normalized_delta_z": _nmad(normalized),
                "outlier_fraction_abs_delta_z_gt_0p1": float(
                    np.mean(np.abs(delta) > 0.1)
                ),
                "outlier_fraction_abs_normalized_delta_z_gt_0p05": float(
                    np.mean(np.abs(normalized) > 0.05)
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    arguments = _arguments()
    predictions = pd.read_parquet(arguments.predictions)
    ia = predictions[predictions["true_class_name"].astype(str).eq("Ia")].copy()
    truth = ia["true_redshift"].to_numpy(float)
    estimates = {
        "posterior median": ia["posterior_median_redshift"].to_numpy(float),
        "density mode": ia["posterior_density_mode_redshift"].to_numpy(float),
        "posterior mean": ia["posterior_mean_redshift"].to_numpy(float),
    }
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    redshift_summary = _redshift_summary(truth, estimates)
    redshift_summary.to_csv(
        arguments.output_dir / "sundial_v3_redshift_summary_by_z.csv",
        index=False,
        float_format="%.5g",
    )
    _overall_summary(truth, estimates).to_csv(
        arguments.output_dir / "sundial_v3_redshift_summary_overall.csv",
        index=False,
        float_format="%.5g",
    )
    _plot_summary_comparison(
        truth,
        estimates,
        redshift_summary,
        arguments.output_dir / "sundial_v3_redshift_summary_comparison.png",
    )
    quality = _plot_quality(
        ia,
        truth,
        estimates["density mode"],
        arguments.output_dir / "sundial_v3_redshift_quality_diagnostics.png",
    )
    quality.to_csv(
        arguments.output_dir / "sundial_v3_redshift_quality_diagnostics.csv",
        index=False,
        float_format="%.5g",
    )
    peak_diagnostics = _peak_diagnostics(ia)
    peak_diagnostics.to_csv(
        arguments.output_dir / "sundial_v3_redshift_peak_diagnostics.csv",
        index=False,
        float_format="%.5g",
    )
    _peak_summary(peak_diagnostics).to_csv(
        arguments.output_dir / "sundial_v3_redshift_peak_summary.csv",
        index=False,
        float_format="%.5g",
    )
    cuts = _plot_quality_cuts(
        predictions,
        ia,
        arguments.output_dir / "sundial_v3_redshift_quality_cuts_clean.png",
    )
    cuts.to_csv(
        arguments.output_dir / "sundial_v3_redshift_quality_cuts.csv",
        index=False,
        float_format="%.5g",
    )
    print(arguments.output_dir / "sundial_v3_redshift_summary_comparison.png")
    print(arguments.output_dir / "sundial_v3_redshift_quality_diagnostics.png")
    print(arguments.output_dir / "sundial_v3_redshift_quality_cuts_clean.png")
    alias = peak_diagnostics["lower_alias_branch"]
    core = (
        (peak_diagnostics["coadded_clean_signal_to_noise"] > 1.5)
        & (peak_diagnostics["normalized_delta_z"].abs() < 0.03)
    )
    print(
        "competing peak: "
        f"{100.0 * peak_diagnostics['competing_peak'].mean():.1f}% overall; "
        f"{100.0 * peak_diagnostics.loc[alias, 'competing_peak'].mean():.1f}% "
        f"on lower alias branch; "
        f"{100.0 * peak_diagnostics.loc[core, 'competing_peak'].mean():.1f}% "
        "in the high-S/N redshift core"
    )


if __name__ == "__main__":
    main()
