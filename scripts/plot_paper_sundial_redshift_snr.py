#!/usr/bin/env python3
"""Create publication figures for Sundial redshift recovery versus S/N."""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import zipfile

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


DEFAULT_OBSERVED_SNR_CUTS = (0.5, 1.0, 2.0)
DEFAULT_OBSERVED_SNR_COLUMN = "median_coadded_observed_signal_to_noise"
NORMALIZED_OUTLIER_LIMIT = 0.05


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--redshift-column",
        default="z_strider",
        help="Prediction column used as the STRIDER point estimate.",
    )
    parser.add_argument(
        "--snr-column",
        default=DEFAULT_OBSERVED_SNR_COLUMN,
        help="Column used for the S/N selections.",
    )
    parser.add_argument(
        "--snr-catalog",
        type=Path,
        help="per-SNID observed S/N parquet/CSV to merge with older predictions",
    )
    parser.add_argument(
        "--snr-cuts",
        nargs="+",
        type=float,
        default=list(DEFAULT_OBSERVED_SNR_CUTS),
        help="fixed measured-S/N thresholds; the no-cut panel is always included",
    )
    parser.add_argument(
        "--tag",
        default="provisional",
        help="Short tag included in output filenames.",
    )
    return parser.parse_args()


def _set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9.5,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "axes.linewidth": 0.9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _nmad(values: np.ndarray) -> float:
    centre = np.nanmedian(values)
    return float(1.4826 * np.nanmedian(np.abs(values - centre)))


def _selection(values: pd.DataFrame, snr_column: str, threshold: float | None) -> pd.DataFrame:
    if threshold is None:
        return values
    return values[values[snr_column].to_numpy(float) >= threshold]


def _cut_label(threshold: float | None) -> str:
    if threshold is None:
        return "No quality cut"
    return rf"Observed S/N $\geq {threshold:g}$"


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            candidates = [
                name
                for name in archive.namelist()
                if name.endswith("test_predictions_original.parquet")
            ]
            if len(candidates) != 1:
                raise ValueError(
                    "Prediction archives must contain exactly one "
                    "test_predictions_original.parquet"
                )
            return pd.read_parquet(io.BytesIO(archive.read(candidates[0])))
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Prediction and S/N tables must be parquet, CSV, or ZIP")


def _attach_snr_catalog(
    predictions: pd.DataFrame,
    catalog_path: Path | None,
    snr_column: str,
) -> pd.DataFrame:
    if catalog_path is None:
        return predictions
    if "snid" not in predictions:
        raise ValueError("Predictions are missing required column: snid")
    catalog = _read_table(catalog_path)
    required = {"snid", snr_column}
    missing = sorted(required - set(catalog.columns))
    if missing:
        raise ValueError(f"S/N catalog is missing required columns: {missing}")
    if predictions["snid"].duplicated().any() or catalog["snid"].duplicated().any():
        raise ValueError("Prediction and S/N catalog SNIDs must be unique")
    result = predictions.drop(columns=[snr_column], errors="ignore").merge(
        catalog[["snid", snr_column]],
        on="snid",
        how="left",
        validate="one_to_one",
    )
    missing_values = int(result[snr_column].isna().sum())
    if missing_values:
        raise ValueError(
            f"S/N catalog lacks measured values for {missing_values} prediction rows"
        )
    return result


def _smoothed_density(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    bins: tuple[int, int],
    smoothing: tuple[float, float],
) -> np.ndarray:
    """Return smoothed two-dimensional counts on a fixed grid."""
    counts, _, _ = np.histogram2d(
        x,
        y,
        bins=bins,
        range=(x_range, y_range),
    )
    return gaussian_filter(counts.T, sigma=smoothing)


def _density_peak(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    bins: tuple[int, int],
    smoothing: tuple[float, float],
) -> float:
    """Return the peak smoothed count for a reference cohort."""
    density = _smoothed_density(
        x,
        y,
        x_range=x_range,
        y_range=y_range,
        bins=bins,
        smoothing=smoothing,
    )
    return float(np.nanmax(density)) if density.size else 0.0


def _relative_density(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    bins: tuple[int, int],
    smoothing: tuple[float, float],
    normalization_peak: float | None = None,
) -> np.ndarray:
    """Return smoothed counts normalized to a supplied or local peak."""
    density = _smoothed_density(
        x,
        y,
        x_range=x_range,
        y_range=y_range,
        bins=bins,
        smoothing=smoothing,
    )
    maximum = (
        float(np.nanmax(density))
        if normalization_peak is None and density.size
        else float(normalization_peak or 0.0)
    )
    if maximum <= 0.0:
        return np.full_like(density, np.nan, dtype=float)
    density = density / maximum
    density[density < 0.006] = np.nan
    return density


def _draw_redshift_density(
    axis: mpl.axes.Axes,
    values: pd.DataFrame,
    redshift_column: str,
    *,
    normalization_peak: float | None = None,
) -> mpl.image.AxesImage:
    truth = values["true_redshift"].to_numpy(float)
    estimate = values[redshift_column].to_numpy(float)
    density = _relative_density(
        truth,
        estimate,
        x_range=(0.0, 3.0),
        y_range=(0.0, 3.0),
        bins=(170, 170),
        smoothing=(2.2, 2.2),
        normalization_peak=normalization_peak,
    )
    image = axis.imshow(
        density,
        origin="lower",
        extent=(0.0, 3.0, 0.0, 3.0),
        aspect="equal",
        cmap="viridis",
        norm=LogNorm(vmin=0.006, vmax=1.0),
        interpolation="bilinear",
    )
    axis.plot((0.0, 3.0), (0.0, 3.0), color="0.16", linewidth=0.9, zorder=4)
    axis.set_xlim(0.0, 3.0)
    axis.set_ylim(0.0, 3.0)
    axis.set_xticks((0.0, 1.0, 2.0, 3.0))
    axis.set_yticks((0.0, 1.0, 2.0, 3.0))
    return image


def _draw_residual_density(
    axis: mpl.axes.Axes,
    values: pd.DataFrame,
    redshift_column: str,
    *,
    normalization_peak: float | None = None,
) -> mpl.image.AxesImage:
    truth = values["true_redshift"].to_numpy(float)
    estimate = values[redshift_column].to_numpy(float)
    residual = estimate - truth
    density = _relative_density(
        truth,
        residual,
        x_range=(0.0, 3.0),
        y_range=(-1.5, 1.5),
        bins=(170, 150),
        smoothing=(2.2, 1.8),
        normalization_peak=normalization_peak,
    )
    image = axis.imshow(
        density,
        origin="lower",
        extent=(0.0, 3.0, -1.5, 1.5),
        aspect="auto",
        cmap="viridis",
        norm=LogNorm(vmin=0.006, vmax=1.0),
        interpolation="bilinear",
    )
    axis.axhline(0.0, color="0.16", linewidth=0.9, zorder=4)
    axis.set_xlim(0.0, 3.0)
    axis.set_ylim(-1.5, 1.5)
    axis.set_xticks((0.0, 1.0, 2.0, 3.0))
    axis.set_yticks((-1.5, -0.75, 0.0, 0.75, 1.5))
    return image


def _metrics(
    values: pd.DataFrame,
    *,
    cohort: str,
    threshold: float | None,
    redshift_column: str,
    snr_column: str,
) -> dict[str, float | int | str]:
    truth = values["true_redshift"].to_numpy(float)
    estimate = values[redshift_column].to_numpy(float)
    delta = estimate - truth
    normalized = delta / (1.0 + truth)
    return {
        "cohort": cohort,
        "snr_threshold": np.nan if threshold is None else threshold,
        "n_objects": len(values),
        "median_delta_z": float(np.nanmedian(delta)),
        "median_abs_delta_z": float(np.nanmedian(np.abs(delta))),
        "standard_deviation_delta_z": float(np.nanstd(delta, ddof=1)),
        "median_normalized_delta_z": float(np.nanmedian(normalized)),
        "standard_deviation_normalized_delta_z": float(
            np.nanstd(normalized, ddof=1)
        ),
        "nmad_normalized_delta_z": _nmad(normalized),
        "median_observed_signal_to_noise": float(
            np.nanmedian(values[snr_column].to_numpy(float))
        ),
        "outlier_fraction_abs_normalized_delta_z_gt_0p05": float(
            np.mean(np.abs(normalized) > NORMALIZED_OUTLIER_LIMIT)
        ),
        "outlier_fraction_abs_normalized_delta_z_gt_0p10": float(
            np.mean(np.abs(normalized) > 0.10)
        ),
        "outlier_fraction_abs_delta_z_gt_0p1": float(np.mean(np.abs(delta) > 0.1)),
    }


def _annotate_count(axis: mpl.axes.Axes, count: int) -> None:
    axis.text(
        0.04,
        0.94,
        rf"$N={count:,}$",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )


def _annotate_metrics(axis: mpl.axes.Axes, row: dict[str, float | int | str]) -> None:
    axis.text(
        0.96,
        0.94,
        "\n".join(
            (
                rf"median $|\Delta z|={float(row['median_abs_delta_z']):.3f}$",
                rf"std $(\Delta z)={float(row['standard_deviation_delta_z']):.3f}$",
            )
        ),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
        linespacing=1.18,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 1.5},
    )


def _finish_axes(axes: np.ndarray) -> None:
    for axis in axes.flat:
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_linewidth(0.9)


def _tighten_panel_grid(axes: np.ndarray) -> None:
    """Keep panels contiguous without duplicate labels or double borders."""
    columns = axes.shape[1]
    for column in range(columns - 1):
        for axis in axes[:, column]:
            axis.spines["right"].set_visible(False)
    axes[-1, 0].set_xticks((0.0, 1.0, 2.0, 3.0))
    for column, axis in enumerate(axes[-1]):
        labels = axis.get_xticklabels()
        if column > 0:
            labels[0].set_visible(False)
        if column < columns - 1:
            labels[-1].set_visible(False)


def _add_colourbar(fig: mpl.figure.Figure, axes: np.ndarray, image: mpl.image.AxesImage) -> None:
    colourbar = fig.colorbar(image, ax=axes, fraction=0.018, pad=0.008)
    colourbar.set_label("Relative density", labelpad=7)
    colourbar.set_ticks((0.01, 0.1, 1.0))
    colourbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))


def _save(fig: mpl.figure.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_full(
    predictions: pd.DataFrame,
    ia: pd.DataFrame,
    *,
    redshift_column: str,
    snr_column: str,
    cuts: tuple[float | None, ...],
    output: Path,
) -> pd.DataFrame:
    fig, axes = plt.subplots(
        3,
        len(cuts),
        figsize=(3.25 * len(cuts) + 0.7, 8.6),
        sharex=True,
        sharey="row",
        gridspec_kw={"height_ratios": (1.0, 1.0, 0.68), "hspace": 0.025, "wspace": 0.025},
    )
    rows: list[dict[str, float | int | str]] = []
    image = None
    for column, threshold in enumerate(cuts):
        all_selected = _selection(predictions, snr_column, threshold)
        ia_selected = _selection(ia, snr_column, threshold)
        image = _draw_redshift_density(axes[0, column], all_selected, redshift_column)
        _draw_redshift_density(axes[1, column], ia_selected, redshift_column)
        _draw_residual_density(axes[2, column], ia_selected, redshift_column)
        all_row = _metrics(
            all_selected,
            cohort="all objects",
            threshold=threshold,
            redshift_column=redshift_column,
            snr_column=snr_column,
        )
        ia_row = _metrics(
            ia_selected,
            cohort="true Ia",
            threshold=threshold,
            redshift_column=redshift_column,
            snr_column=snr_column,
        )
        rows.extend((all_row, ia_row))
        axes[0, column].set_title(_cut_label(threshold), fontweight="semibold", pad=5)
        _annotate_count(axes[0, column], len(all_selected))
        _annotate_count(axes[1, column], len(ia_selected))
        _annotate_metrics(axes[2, column], ia_row)
        axes[2, column].set_xlabel(r"$z_{\mathrm{true}}$")

    axes[0, 0].set_ylabel("All objects\n" + r"$z_{\mathrm{STRIDER}}$")
    axes[1, 0].set_ylabel("True Ia\n" + r"$z_{\mathrm{STRIDER}}$")
    axes[2, 0].set_ylabel(r"$\Delta z$")
    _finish_axes(axes)
    _tighten_panel_grid(axes)
    fig.subplots_adjust(top=0.92)
    fig.suptitle(
        "Sundial redshift recovery by measured spectral S/N",
        fontsize=13,
        fontweight="semibold",
        y=0.995,
    )
    assert image is not None
    _add_colourbar(fig, axes, image)
    _save(fig, output)
    return pd.DataFrame(rows)


def _plot_ia_main(
    ia: pd.DataFrame,
    *,
    redshift_column: str,
    snr_column: str,
    cuts: tuple[float | None, ...],
    output: Path,
) -> pd.DataFrame:
    fig, axes = plt.subplots(
        2,
        len(cuts),
        figsize=(3.15 * len(cuts), 5.35),
        sharex=True,
        sharey="row",
        gridspec_kw={"height_ratios": (1.0, 0.68), "hspace": 0.025, "wspace": 0.025},
    )
    rows: list[dict[str, float | int | str]] = []
    image = None
    for column, threshold in enumerate(cuts):
        selected = _selection(ia, snr_column, threshold)
        image = _draw_redshift_density(axes[0, column], selected, redshift_column)
        _draw_residual_density(axes[1, column], selected, redshift_column)
        row = _metrics(
            selected,
            cohort="true Ia",
            threshold=threshold,
            redshift_column=redshift_column,
            snr_column=snr_column,
        )
        rows.append(row)
        axes[0, column].set_title(_cut_label(threshold), fontweight="semibold", pad=5)
        _annotate_count(axes[0, column], len(selected))
        _annotate_metrics(axes[1, column], row)
        axes[1, column].set_xlabel(r"$z_{\mathrm{true}}$")

    axes[0, 0].set_ylabel(r"$z_{\mathrm{STRIDER}}$")
    axes[1, 0].set_ylabel(r"$\Delta z$")
    _finish_axes(axes)
    _tighten_panel_grid(axes)
    fig.subplots_adjust(top=0.90)
    fig.suptitle(
        "Type Ia redshift recovery",
        fontsize=13,
        fontweight="semibold",
        y=0.995,
    )
    assert image is not None
    _add_colourbar(fig, axes, image)
    _save(fig, output)
    return pd.DataFrame(rows)


def _write_caption(output_dir: Path, tag: str, redshift_column: str, snr_column: str) -> None:
    if snr_column == DEFAULT_OBSERVED_SNR_COLUMN:
        snr_description = (
            "For each Sundial object, observed S/N is the signed median across "
            "native wavelength bins of the inverse-variance coadded observed FLAM "
            "divided by its propagated FLAMERR, using exactly the visits evaluated "
            "by STRIDER. The outer 5% at each end of the configured log-wavelength "
            "range and bins with propagated error above three times the median "
            "in-band error are excluded. These masks use wavelength and FLAMERR "
            "only; the statistic uses neither SIM_FLAM nor class/redshift truth. "
            "The fixed observer-frame band avoids defining data quality with true "
            "redshift, while trimming low-throughput and edge-interpolation regions."
        )
    else:
        snr_description = f"The S/N selection uses column {snr_column!r}."
    caption = (
        "Sundial redshift recovery for the provisional STRIDER v3 binary model. "
        "The upper panels compare the STRIDER point estimate with the simulated "
        "redshift, while the lower panels show the raw residual "
        "Delta z = z_STRIDER - z_true. Columns retain "
        "successively higher-signal subsets using fixed, predeclared thresholds. "
        "Density is normalized within each panel. The annotations give the median "
        "absolute redshift error and standard deviation of Delta z. Normalized NMAD "
        "and the corresponding 5% and 10% normalized outlier fractions are retained "
        "in the accompanying table. "
        f"{snr_description} The point estimate is {redshift_column!r}.\n"
    )
    (output_dir / f"sundial_redshift_snr_{tag}_caption.txt").write_text(
        caption,
        encoding="utf-8",
    )


def main() -> None:
    arguments = _arguments()
    predictions = _read_table(arguments.predictions)
    predictions = _attach_snr_catalog(
        predictions,
        arguments.snr_catalog,
        arguments.snr_column,
    )
    cuts = (None, *tuple(sorted(set(arguments.snr_cuts))))
    required = {
        "true_class_name",
        "true_redshift",
        arguments.redshift_column,
        arguments.snr_column,
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing required columns: {missing}")
    numeric_required = sorted(required.difference({"true_class_name"}))
    nonfinite = {
        column: int((~np.isfinite(predictions[column].to_numpy(float))).sum())
        for column in numeric_required
    }
    nonfinite = {column: count for column, count in nonfinite.items() if count}
    if nonfinite:
        raise ValueError(f"Prediction inputs contain non-finite values: {nonfinite}")
    ia = predictions[predictions["true_class_name"].astype(str).eq("Ia")].copy()
    if ia.empty:
        raise ValueError("Predictions contain no true Ia objects")

    _set_style()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    full_stem = arguments.output_dir / f"sundial_redshift_snr_full_{arguments.tag}"
    main_stem = arguments.output_dir / f"sundial_redshift_snr_ia_{arguments.tag}"
    full_summary = _plot_full(
        predictions,
        ia,
        redshift_column=arguments.redshift_column,
        snr_column=arguments.snr_column,
        cuts=cuts,
        output=full_stem,
    )
    main_summary = _plot_ia_main(
        ia,
        redshift_column=arguments.redshift_column,
        snr_column=arguments.snr_column,
        cuts=cuts,
        output=main_stem,
    )
    full_summary.to_csv(
        full_stem.with_suffix(".csv"), index=False, float_format="%.5g"
    )
    main_summary.to_csv(
        main_stem.with_suffix(".csv"), index=False, float_format="%.5g"
    )
    _write_caption(
        arguments.output_dir,
        arguments.tag,
        arguments.redshift_column,
        arguments.snr_column,
    )
    print(full_stem.with_suffix(".png"))
    print(main_stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
