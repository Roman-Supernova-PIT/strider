#!/usr/bin/env python3
"""Audit the structured redshift aliases among high-S/N Sundial Type Ia objects."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SNR_COLUMN = "median_coadded_observed_signal_to_noise"
DEFAULT_REDSHIFT_COLUMN = "posterior_primary_peak_redshift"
RATIO_BRANCHES = (
    (0.875, "0.875 lower-z branch"),
    (1.125, "1.125 higher-z branch"),
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--snr-catalog", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--snr-threshold", default=2.0, type=float)
    parser.add_argument("--outlier-limit", default=0.1, type=float)
    parser.add_argument("--truth-peak-tolerance", default=0.05, type=float)
    parser.add_argument("--redshift-column", default=DEFAULT_REDSHIFT_COLUMN)
    parser.add_argument("--snr-column", default=DEFAULT_SNR_COLUMN)
    return parser.parse_args()


def _as_float_array(value: object) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Saved posterior-candidate values must be one-dimensional")
    return values


def _read_table(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            candidates = [
                name
                for name in archive.namelist()
                if name.endswith("test_predictions_original.parquet")
            ]
            if len(candidates) != 1:
                raise ValueError(
                    "Prediction archives must contain one "
                    "test_predictions_original.parquet"
                )
            member = candidates[0]
            payload = archive.read(member)
        return pd.read_parquet(io.BytesIO(payload)), {
            "path": str(path.resolve()),
            "member": member,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    if path.suffix.lower() == ".parquet":
        table = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        table = pd.read_csv(path)
    else:
        raise ValueError("Inputs must be parquet, CSV, or a prediction ZIP archive")
    return table, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _prepare_sample(
    predictions: pd.DataFrame,
    snr: pd.DataFrame,
    *,
    redshift_column: str,
    snr_column: str,
    snr_threshold: float,
    outlier_limit: float,
    truth_peak_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_columns = {
        "snid",
        "true_class_name",
        "true_redshift",
        "p_Ia",
        redshift_column,
        "posterior_candidate_peak_redshifts",
        "posterior_candidate_masses",
        "posterior_candidate_lower_68",
        "posterior_candidate_upper_68",
    }
    missing = sorted(prediction_columns - set(predictions))
    if missing:
        raise ValueError(f"Predictions are missing required columns: {missing}")
    missing = sorted({"snid", snr_column} - set(snr))
    if missing:
        raise ValueError(f"S/N catalog is missing required columns: {missing}")
    if predictions["snid"].duplicated().any() or snr["snid"].duplicated().any():
        raise ValueError("Prediction and S/N SNIDs must be unique")

    merged = predictions.merge(
        snr[["snid", snr_column]],
        on="snid",
        how="left",
        validate="one_to_one",
    )
    if merged[snr_column].isna().any():
        raise ValueError("The S/N catalog does not cover every prediction row")
    sample = merged[
        (merged["true_class_name"] == "Ia")
        & (merged[snr_column].to_numpy(float) > snr_threshold)
    ].copy()
    sample["z_strider"] = sample[redshift_column].to_numpy(float)
    sample["delta_z"] = sample["z_strider"] - sample["true_redshift"]
    sample["absolute_delta_z"] = sample["delta_z"].abs()
    sample["one_plus_z_ratio"] = (1.0 + sample["z_strider"]) / (
        1.0 + sample["true_redshift"]
    )
    sample["is_outlier"] = sample["absolute_delta_z"] > outlier_limit

    outliers = sample[sample["is_outlier"]].copy()
    closest_ranks: list[int] = []
    closest_distances: list[float] = []
    truth_near_peak: list[bool] = []
    truth_in_interval: list[bool] = []
    truth_status: list[str] = []
    for row in outliers.itertuples(index=False):
        peaks = _as_float_array(row.posterior_candidate_peak_redshifts)
        lower = _as_float_array(row.posterior_candidate_lower_68)
        upper = _as_float_array(row.posterior_candidate_upper_68)
        if not (len(peaks) == len(lower) == len(upper)) or not len(peaks):
            raise ValueError("Posterior-candidate arrays must be nonempty and aligned")
        distances = np.abs(peaks - float(row.true_redshift))
        closest = int(np.argmin(distances))
        near_peak = bool(distances[closest] <= truth_peak_tolerance)
        in_interval = bool(
            np.any((lower <= float(row.true_redshift)) & (float(row.true_redshift) <= upper))
        )
        closest_ranks.append(closest + 1)
        closest_distances.append(float(distances[closest]))
        truth_near_peak.append(near_peak)
        truth_in_interval.append(in_interval)
        if near_peak:
            truth_status.append(f"truth near candidate {closest + 1}")
        elif in_interval:
            truth_status.append("truth inside a candidate 68% interval")
        else:
            truth_status.append("truth not retained")
    outliers["closest_truth_candidate_rank"] = closest_ranks
    outliers["closest_truth_candidate_distance"] = closest_distances
    outliers["truth_near_reported_candidate"] = truth_near_peak
    outliers["truth_in_candidate_68_interval"] = truth_in_interval
    outliers["truth_status"] = truth_status
    return sample, outliers


def _summary(
    sample: pd.DataFrame,
    outliers: pd.DataFrame,
    *,
    snr_threshold: float,
    outlier_limit: float,
    truth_peak_tolerance: float,
    prediction_source: dict[str, str],
    snr_source: dict[str, str],
) -> dict[str, object]:
    confident = sample["p_Ia"].to_numpy(float) > 0.9
    confident_outlier = confident & sample["is_outlier"].to_numpy(bool)
    lower = outliers["delta_z"].to_numpy(float) < 0.0
    ratio = outliers["one_plus_z_ratio"].to_numpy(float)
    branch_counts = {
        label: int((np.abs(ratio - centre) <= 0.02).sum())
        for centre, label in RATIO_BRANCHES
    }
    redshift_bins = (
        ("z_lt_1", sample["true_redshift"].to_numpy(float) < 1.0),
        (
            "z_1_to_1p5",
            (sample["true_redshift"].to_numpy(float) >= 1.0)
            & (sample["true_redshift"].to_numpy(float) < 1.5),
        ),
        ("z_ge_1p5", sample["true_redshift"].to_numpy(float) >= 1.5),
    )
    by_true_redshift = {}
    for label, selected in redshift_bins:
        selected_sample = sample.loc[selected]
        selected_outliers = selected_sample["is_outlier"].to_numpy(bool)
        selected_confident = selected_sample["p_Ia"].to_numpy(float) > 0.9
        confident_outliers = selected_outliers & selected_confident
        outlier_rows = outliers[
            outliers["snid"].isin(selected_sample["snid"])
        ]
        by_true_redshift[label] = {
            "objects": int(selected.sum()),
            "outliers": int(selected_outliers.sum()),
            "outlier_fraction": float(
                selected_outliers.sum() / max(selected.sum(), 1)
            ),
            "p_Ia_gt_0p9_objects": int(selected_confident.sum()),
            "p_Ia_gt_0p9_outliers": int(confident_outliers.sum()),
            "p_Ia_gt_0p9_outlier_fraction": float(
                confident_outliers.sum() / max(selected_confident.sum(), 1)
            ),
            "outliers_with_truth_near_a_reported_candidate": int(
                outlier_rows["truth_near_reported_candidate"].sum()
            ),
        }
    return {
        "format_version": "strider-sundial-high-snr-alias-audit-v1",
        "selection": {
            "true_class": "Ia",
            "observed_snr_strictly_greater_than": snr_threshold,
            "absolute_delta_z_strictly_greater_than": outlier_limit,
            "truth_candidate_peak_tolerance": truth_peak_tolerance,
        },
        "inputs": {
            "predictions": prediction_source,
            "observed_snr": snr_source,
        },
        "objects": int(len(sample)),
        "outliers": int(len(outliers)),
        "outlier_fraction": float(len(outliers) / max(len(sample), 1)),
        "p_Ia_gt_0p9_objects": int(confident.sum()),
        "p_Ia_gt_0p9_outliers": int(confident_outlier.sum()),
        "p_Ia_gt_0p9_outlier_fraction": float(
            confident_outlier.sum() / max(confident.sum(), 1)
        ),
        "outliers_below_truth": int(lower.sum()),
        "outliers_above_truth": int((~lower).sum()),
        "median_one_plus_z_ratio_for_outliers": float(np.median(ratio)),
        "truth_near_a_reported_candidate": int(
            outliers["truth_near_reported_candidate"].sum()
        ),
        "truth_near_a_reported_candidate_fraction": float(
            outliers["truth_near_reported_candidate"].mean()
        ),
        "truth_in_a_candidate_68_interval": int(
            outliers["truth_in_candidate_68_interval"].sum()
        ),
        "truth_in_a_candidate_68_interval_fraction": float(
            outliers["truth_in_candidate_68_interval"].mean()
        ),
        "closest_truth_candidate_rank_counts": {
            str(int(rank)): int(count)
            for rank, count in outliers["closest_truth_candidate_rank"]
            .value_counts()
            .sort_index()
            .items()
        },
        "retained_truth_candidate_rank_counts": {
            str(int(rank)): int(count)
            for rank, count in outliers.loc[
                outliers["truth_near_reported_candidate"],
                "closest_truth_candidate_rank",
            ]
            .value_counts()
            .sort_index()
            .items()
        },
        "descriptive_ratio_branch_counts_within_0p02": branch_counts,
        "by_true_redshift": by_true_redshift,
    }


def _set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.linewidth": 0.9,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
        }
    )


def _draw_figure(sample: pd.DataFrame, outliers: pd.DataFrame, path: Path) -> None:
    _set_style()
    retained = outliers["truth_near_reported_candidate"].to_numpy(bool)
    truth = sample["true_redshift"].to_numpy(float)
    estimate = sample["z_strider"].to_numpy(float)
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.45), constrained_layout=True)

    axes[0].scatter(truth, estimate, s=7, color="0.78", alpha=0.32, linewidths=0)
    for selected, color, label, marker in (
        (retained, "#168AAD", "truth retained as a candidate", "o"),
        (~retained, "#D95F3D", "truth not retained", "x"),
    ):
        values = outliers[selected]
        axes[0].scatter(
            values["true_redshift"],
            values["z_strider"],
            s=31,
            color=color,
            marker=marker,
            linewidths=1.1,
            label=label,
            zorder=4,
        )
    axes[0].plot((0.0, 3.0), (0.0, 3.0), color="0.18", linewidth=1.0)
    x = np.linspace(0.0, 3.0, 500)
    for ratio, label in RATIO_BRANCHES:
        y = ratio * (1.0 + x) - 1.0
        valid = (y >= 0.0) & (y <= 3.0)
        axes[0].plot(
            x[valid],
            y[valid],
            color="0.35",
            linewidth=0.85,
            linestyle="--",
            alpha=0.7,
            label=label,
        )
    axes[0].set_xlim(0.0, 3.0)
    axes[0].set_ylim(0.0, 3.0)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("true redshift")
    axes[0].set_ylabel("STRIDER redshift")
    axes[0].legend(frameon=False, fontsize=8.4, loc="upper left")

    bins = np.linspace(0.45, 1.28, 70)
    axes[1].hist(
        outliers.loc[~retained, "one_plus_z_ratio"],
        bins=bins,
        histtype="stepfilled",
        color="#D95F3D",
        alpha=0.30,
        label="truth not retained",
    )
    axes[1].hist(
        outliers.loc[retained, "one_plus_z_ratio"],
        bins=bins,
        histtype="step",
        color="#168AAD",
        linewidth=1.8,
        label="truth retained as a candidate",
    )
    axes[1].axvline(1.0, color="0.18", linewidth=1.0)
    for ratio, label in RATIO_BRANCHES:
        axes[1].axvline(ratio, color="0.35", linestyle="--", linewidth=0.9)
        axes[1].text(
            ratio,
            0.98,
            label.split()[0],
            transform=axes[1].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8.5,
            color="0.30",
        )
    axes[1].set_xlim(0.45, 1.28)
    axes[1].set_xlabel(r"$(1+z_{\mathrm{STRIDER}})/(1+z_{\mathrm{true}})$")
    axes[1].set_ylabel("number of outliers")
    axes[1].legend(frameon=False, fontsize=8.8, loc="upper left")

    for axis in axes:
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(True)
    figure.suptitle("High-S/N Type Ia redshift aliases", fontsize=15, fontweight="bold")
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    arguments = _arguments()
    predictions, prediction_source = _read_table(arguments.predictions)
    snr, snr_source = _read_table(arguments.snr_catalog)
    sample, outliers = _prepare_sample(
        predictions,
        snr,
        redshift_column=arguments.redshift_column,
        snr_column=arguments.snr_column,
        snr_threshold=arguments.snr_threshold,
        outlier_limit=arguments.outlier_limit,
        truth_peak_tolerance=arguments.truth_peak_tolerance,
    )
    summary = _summary(
        sample,
        outliers,
        snr_threshold=arguments.snr_threshold,
        outlier_limit=arguments.outlier_limit,
        truth_peak_tolerance=arguments.truth_peak_tolerance,
        prediction_source=prediction_source,
        snr_source=snr_source,
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = arguments.output_dir / "snr2_aliases.csv"
    summary_path = arguments.output_dir / "snr2_aliases_summary.json"
    figure_path = arguments.output_dir / "snr2_aliases.png"
    output_columns = [
        "snid",
        "true_redshift",
        "z_strider",
        "delta_z",
        "absolute_delta_z",
        "one_plus_z_ratio",
        arguments.snr_column,
        "p_Ia",
        "closest_truth_candidate_rank",
        "closest_truth_candidate_distance",
        "truth_near_reported_candidate",
        "truth_in_candidate_68_interval",
        "truth_status",
        "posterior_candidate_peak_redshifts",
        "posterior_candidate_masses",
    ]
    outliers.sort_values(arguments.snr_column, ascending=False)[output_columns].to_csv(
        table_path,
        index=False,
        float_format="%.7g",
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _draw_figure(sample, outliers, figure_path)
    print(json.dumps(summary, indent=2))
    print(table_path)
    print(summary_path)
    print(figure_path)


if __name__ == "__main__":
    main()
