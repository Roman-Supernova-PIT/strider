#!/usr/bin/env python3
"""Compare v2 and v3 on exactly paired Sundial spectra and noise draws."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REDSHIFT_EDGES = np.asarray([0.0, 0.75, 1.25, 1.75, 2.25, 3.0])
MODEL_STYLE = {
    "v2": {"color": "#6a51a3", "linestyle": "--"},
    "v3": {"color": "#0587a1", "linestyle": "-"},
}


def _v2_predictions(path: Path) -> pd.DataFrame:
    paths = sorted(path.glob("predictions*_of_*.csv")) if path.is_dir() else [path]
    if not paths:
        raise FileNotFoundError(f"no v2 prediction shards found in {path}")
    frame = pd.concat([pd.read_csv(item) for item in paths], ignore_index=True)
    frame = frame[frame["condition"].str.startswith("fresh_")].copy()
    frame = frame.rename(
        columns={
            "draw": "noise_repeat",
            "z_true": "true_redshift",
            "z_pred": "predicted_redshift",
        }
    )
    if "is_ia" in frame:
        frame["is_ia"] = _as_boolean(frame["is_ia"])
    elif "true_class" in frame:
        frame["is_ia"] = frame["true_class"].astype(str).eq("Ia")
    else:
        raise ValueError("v2 predictions lack is_ia or true_class")
    frame["model"] = "v2"
    return frame


def _v3_predictions(path: Path, tag: str | None = None) -> pd.DataFrame:
    pattern = (
        f"noise_predictions_{tag}_shard_*.csv"
        if tag is not None
        else "noise_predictions*_shard_*.csv"
    )
    paths = sorted(path.glob(pattern)) if path.is_dir() else [path]
    if not paths:
        raise FileNotFoundError(f"no v3 prediction shards found in {path}")
    frame = pd.concat([pd.read_csv(item) for item in paths], ignore_index=True)
    frame = frame[frame["input_kind"].eq("source")].copy()
    if "p_Ia" in frame and "p_ia" not in frame:
        frame = frame.rename(columns={"p_Ia": "p_ia"})
    if "true_class_name" in frame:
        frame["is_ia"] = frame["true_class_name"].astype(str).eq("Ia")
    elif "true_class" in frame:
        frame["is_ia"] = frame["true_class"].eq(0)
    else:
        raise ValueError("v3 predictions lack true_class_name or true_class")
    frame["model"] = "v3"
    return frame


def _as_boolean(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false", "1", "0"}).all():
        raise ValueError("is_ia contains values that cannot be interpreted as boolean")
    return normalized.isin({"true", "1"})


def _paired(v2: pd.DataFrame, v3: pd.DataFrame) -> pd.DataFrame:
    keys = ["snid", "noise_scale", "noise_repeat"]
    needed = keys + ["true_redshift", "predicted_redshift", "p_ia", "is_ia"]
    left = v2[needed].rename(
        columns={column: f"{column}_v2" for column in needed if column not in keys}
    )
    right = v3[needed].rename(
        columns={column: f"{column}_v3" for column in needed if column not in keys}
    )
    merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError(
            f"prediction keys do not match exactly: v2={len(left)}, "
            f"v3={len(right)}, paired={len(merged)}"
        )
    if not np.allclose(
        merged["true_redshift_v2"], merged["true_redshift_v3"], atol=1.0e-5
    ):
        raise ValueError("paired v2 and v3 rows disagree on true redshift")
    if not merged["is_ia_v2"].eq(merged["is_ia_v3"]).all():
        raise ValueError("paired v2 and v3 rows disagree on the Ia label")
    merged["true_redshift"] = merged.pop("true_redshift_v2")
    merged["is_ia"] = merged.pop("is_ia_v2")
    return merged.drop(columns=["true_redshift_v3", "is_ia_v3"])


def _wide_predictions(predictions: pd.DataFrame, model: str) -> pd.DataFrame:
    """Put one model's predictions into the wide comparison-table contract."""
    columns = [
        "snid",
        "noise_scale",
        "noise_repeat",
        "true_redshift",
        "predicted_redshift",
        "p_ia",
        "is_ia",
    ]
    return predictions[columns].rename(
        columns={
            "predicted_redshift": f"predicted_redshift_{model}",
            "p_ia": f"p_ia_{model}",
        }
    )


def _available_models(frame: pd.DataFrame) -> tuple[str, ...]:
    models = tuple(
        model
        for model in ("v2", "v3")
        if f"predicted_redshift_{model}" in frame
        and f"p_ia_{model}" in frame
    )
    if not models:
        raise ValueError("prediction table contains no recognized model outputs")
    return models


def _redshift_summary(paired: pd.DataFrame) -> pd.DataFrame:
    true_ia = paired[paired["is_ia"]].copy()
    final_edge = np.nextafter(REDSHIFT_EDGES[-1], np.inf)
    labels = [
        f"{lower:.2f}\N{EN DASH}{upper:.2f}"
        for lower, upper in zip(REDSHIFT_EDGES[:-1], REDSHIFT_EDGES[1:], strict=True)
    ]
    true_ia["redshift_bin"] = pd.cut(
        true_ia["true_redshift"],
        [*REDSHIFT_EDGES[:-1], final_edge],
        labels=labels,
        right=False,
        include_lowest=True,
    )
    rows = []
    for model in _available_models(paired):
        delta = true_ia[f"predicted_redshift_{model}"] - true_ia["true_redshift"]
        work = true_ia.assign(delta_z=delta)
        for (label, scale), group in work.groupby(
            ["redshift_bin", "noise_scale"], observed=True, sort=False
        ):
            values = group["delta_z"].to_numpy(float)
            median_delta = float(np.median(values))
            rows.append(
                {
                    "model": model,
                    "redshift_bin": str(label),
                    "noise_scale": float(scale),
                    "noise_percent": 100.0 * float(scale),
                    "n_ia_draws": len(group),
                    "median_delta_z": median_delta,
                    "sigma_delta_z": float(np.std(values, ddof=0)),
                    "median_abs_delta_z": float(np.median(np.abs(values))),
                    "nmad_delta_z": float(
                        1.4826 * np.median(np.abs(values - median_delta))
                    ),
                    "outlier_fraction_abs_delta_z_gt_0p1": float(
                        np.mean(np.abs(values) > 0.1)
                    ),
                    "median_p_ia": float(np.median(group[f"p_ia_{model}"])),
                    "fraction_p_ia_ge_0p9": float(
                        np.mean(group[f"p_ia_{model}"] >= 0.9)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _normal_noise_summary(
    summary: pd.DataFrame,
    *,
    nominal_scale: float = 1.0,
) -> pd.DataFrame:
    """Return the interpretable redshift metrics at the nominal noise level."""
    nominal = summary[np.isclose(summary["noise_scale"], nominal_scale)].copy()
    columns = [
        "model",
        "redshift_bin",
        "noise_scale",
        "n_ia_draws",
        "median_delta_z",
        "median_abs_delta_z",
        "nmad_delta_z",
        "outlier_fraction_abs_delta_z_gt_0p1",
        "median_p_ia",
        "fraction_p_ia_ge_0p9",
    ]
    return nominal[columns].sort_values(["model", "redshift_bin"])


def _classification_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in _available_models(paired):
        for scope, selected_scope in (
            ("all", np.ones(len(paired), dtype=bool)),
            ("z_lt_2", paired["true_redshift"].to_numpy() < 2.0),
        ):
            work = paired.loc[selected_scope]
            for scale, group in work.groupby("noise_scale", sort=True):
                selected = group[f"p_ia_{model}"].to_numpy(float) >= 0.9
                is_ia = group["is_ia"].to_numpy(bool)
                true_positive = int(np.sum(selected & is_ia))
                selected_count = int(np.sum(selected))
                ia_count = int(np.sum(is_ia))
                has_non_ia = bool(np.any(~is_ia))
                ia_group = group.loc[is_ia]
                delta = (
                    ia_group[f"predicted_redshift_{model}"]
                    - ia_group["true_redshift"]
                ).to_numpy(float)
                selected_ia = ia_group[f"p_ia_{model}"].to_numpy(float) >= 0.9
                rows.append(
                    {
                        "model": model,
                        "scope": scope,
                        "noise_scale": float(scale),
                        "noise_percent": 100.0 * float(scale),
                        "object_draws": len(group),
                        "unique_objects": int(group["snid"].nunique()),
                        "selected": selected_count,
                        "true_ia": ia_count,
                        "true_positive": true_positive,
                        "purity_p_ia_ge_0p9": _safe_ratio(
                            true_positive, selected_count
                        ) if has_non_ia else float("nan"),
                        "completeness_p_ia_ge_0p9": _safe_ratio(
                            true_positive, ia_count
                        ),
                        "ia_median_abs_delta_z": float(np.median(np.abs(delta))),
                        "ia_outlier_fraction": float(np.mean(np.abs(delta) > 0.1)),
                        "selected_ia_median_abs_delta_z": (
                            float(np.median(np.abs(delta[selected_ia])))
                            if selected_ia.any()
                            else float("nan")
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _reliability_summary(
    paired: pd.DataFrame,
    *,
    nominal_scale: float = 1.0,
    bins: int = 10,
) -> pd.DataFrame:
    nominal = paired[np.isclose(paired["noise_scale"], nominal_scale)].copy()
    rows = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for model in _available_models(paired):
        per_object = nominal.groupby("snid", as_index=False).agg(
            p_ia=(f"p_ia_{model}", "mean"),
            is_ia=("is_ia", "first"),
        )
        per_object["probability_bin"] = pd.cut(
            per_object["p_ia"], edges, labels=False, include_lowest=True
        )
        for bin_index, group in per_object.groupby(
            "probability_bin", observed=True, sort=True
        ):
            rows.append(
                {
                    "model": model,
                    "probability_bin": int(bin_index),
                    "n": len(group),
                    "mean_predicted_probability": float(group["p_ia"].mean()),
                    "observed_ia_fraction": float(group["is_ia"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _redshift_colours(labels: list[str]) -> np.ndarray:
    return np.asarray(plt_colours(len(labels)))


def plt_colours(count: int) -> list[tuple[float, float, float, float]]:
    import matplotlib.pyplot as plt

    return list(plt.cm.viridis(np.linspace(0.08, 0.9, count)))


def _plot_redshift(summary: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    models = tuple(model for model in ("v2", "v3") if model in set(summary["model"]))
    fig, axes = plt.subplots(
        len(models),
        4,
        figsize=(15.2, 3.2 * len(models)),
        sharex=True,
        squeeze=False,
    )
    columns = (
        ("median_delta_z", r"median $\Delta z$"),
        ("nmad_delta_z", "NMAD"),
        ("outlier_fraction_abs_delta_z_gt_0p1", r"fraction $|\Delta z|>0.1$"),
        ("median_p_ia", r"median $P(\mathrm{Ia})$"),
    )
    labels = sorted(
        summary["redshift_bin"].unique(),
        key=lambda text: float(text.split("\N{EN DASH}", 1)[0]),
    )
    colours = _redshift_colours(labels)
    for row, model in enumerate(models):
        model_rows = summary[summary["model"].eq(model)]
        for label, colour in zip(labels, colours, strict=True):
            group = model_rows[model_rows["redshift_bin"].eq(label)].sort_values(
                "noise_scale"
            )
            for column, (metric, _) in enumerate(columns):
                axes[row, column].plot(
                    group["noise_percent"],
                    group[metric],
                    "o-",
                    color=colour,
                    linewidth=1.7,
                    markersize=4.0,
                    label=label,
                )
        axes[row, 0].axhline(0.0, color="0.35", linewidth=0.8)
        axes[row, 3].axhline(0.9, color="0.55", linewidth=0.8, linestyle=":")
        axes[row, 0].set_ylabel(f"STRIDER {model}")
        for axis in axes[row]:
            axis.axvline(100.0, color="0.5", linewidth=0.9, linestyle="--")
            axis.grid(False)
    for column, (_, title) in enumerate(columns):
        axes[0, column].set_title(title)
        axes[-1, column].set_xlabel("noise [% FLAMERR]")
    for row in range(len(models)):
        axes[row, 2].set_ylim(-0.02, 1.02)
        axes[row, 3].set_ylim(-0.02, 1.02)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        title="true redshift",
        ncol=len(labels),
        loc="upper center",
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_classification(summary: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.5), sharex=True, sharey=True)
    metrics = (
        ("purity_p_ia_ge_0p9", "purity"),
        ("completeness_p_ia_ge_0p9", "completeness"),
    )
    for row, (scope, label) in enumerate((("all", "all z"), ("z_lt_2", "z < 2"))):
        scoped = summary[summary["scope"].eq(scope)]
        for column, (metric, title) in enumerate(metrics):
            axis = axes[row, column]
            for model in ("v2", "v3"):
                if model not in set(scoped["model"]):
                    continue
                group = scoped[scoped["model"].eq(model)].sort_values("noise_scale")
                axis.plot(
                    group["noise_percent"],
                    group[metric],
                    marker="o",
                    linewidth=2.0,
                    markersize=4.5,
                    label=model,
                    **MODEL_STYLE[model],
                )
            axis.axvline(100.0, color="0.5", linewidth=0.9, linestyle=":")
            axis.set_ylim(-0.02, 1.02)
            axis.grid(False)
            if row == 0:
                axis.set_title(title)
            if row == 1:
                axis.set_xlabel("noise [% FLAMERR]")
        axes[row, 0].set_ylabel(label)
    axes[0, 1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_reliability(summary: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    models = tuple(model for model in ("v2", "v3") if model in set(summary["model"]))
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(4.4 * len(models), 4.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for axis, model in zip(axes[0], models, strict=True):
        group = summary[summary["model"].eq(model)]
        axis.plot([0, 1], [0, 1], color="0.55", linestyle=":", linewidth=1.2)
        axis.plot(
            group["mean_predicted_probability"],
            group["observed_ia_fraction"],
            marker="o",
            linewidth=2.0,
            **MODEL_STYLE[model],
        )
        axis.set_title(model)
        axis.set_xlabel(r"predicted $P(\mathrm{Ia})$")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(False)
    axes[0, 0].set_ylabel("observed Ia fraction")
    fig.tight_layout()
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_nominal_redshift(
    paired: pd.DataFrame,
    output: Path,
    *,
    nominal_scale: float = 1.0,
) -> None:
    import matplotlib.pyplot as plt

    nominal = paired[np.isclose(paired["noise_scale"], nominal_scale)].copy()
    models = _available_models(paired)
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(4.5 * len(models), 4.2),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for axis, model in zip(axes[0], models, strict=True):
        columns = {
            "true_redshift": "first",
            "is_ia": "first",
            f"predicted_redshift_{model}": "median",
            f"p_ia_{model}": "mean",
        }
        per_object = nominal.groupby("snid", as_index=False).agg(columns)
        per_object = per_object[per_object["is_ia"]]
        selected = per_object[f"p_ia_{model}"] >= 0.9
        axis.scatter(
            per_object["true_redshift"],
            per_object[f"predicted_redshift_{model}"],
            s=7,
            color="0.72",
            alpha=0.35,
            linewidths=0,
            label="all Ia",
        )
        axis.scatter(
            per_object.loc[selected, "true_redshift"],
            per_object.loc[selected, f"predicted_redshift_{model}"],
            s=8,
            color=MODEL_STYLE[model]["color"],
            alpha=0.6,
            linewidths=0,
            label=r"$P(\mathrm{Ia})\geq0.9$",
        )
        axis.plot([0, 3], [0, 3], color="0.2", linewidth=1.0)
        axis.set_title(model)
        axis.set_xlabel("true redshift")
        axis.set_xlim(0, 3)
        axis.set_ylim(0, 3)
        axis.grid(False)
    axes[0, 0].set_ylabel("predicted redshift")
    axes[0, -1].legend(frameon=False, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v2",
        type=Path,
        help="v2 prediction file or shard directory; omit for a v3-only figure",
    )
    parser.add_argument("--v3", type=Path, required=True)
    parser.add_argument(
        "--v3-tag",
        help="base output tag used by the v3 sharded evaluator",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    v3 = _v3_predictions(args.v3, tag=args.v3_tag)
    paired = (
        _paired(_v2_predictions(args.v2), v3)
        if args.v2 is not None
        else _wide_predictions(v3, "v3")
    )
    redshift = _redshift_summary(paired)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(
        args.output_dir / "paired_predictions.csv", index=False, float_format="%.7g"
    )
    redshift.to_csv(
        args.output_dir / "true_ia_noise_summary.csv",
        index=False,
        float_format="%.6g",
    )
    _normal_noise_summary(redshift).to_csv(
        args.output_dir / "normal_noise_typical_errors.csv",
        index=False,
        float_format="%.6g",
    )
    suffix = "v2_v3" if args.v2 is not None else "v3"
    _plot_redshift(redshift, args.output_dir / f"true_ia_noise_{suffix}.png")
    _plot_nominal_redshift(paired, args.output_dir / f"redshift_{suffix}.png")
    if (~paired["is_ia"]).any():
        classification = _classification_summary(paired)
        reliability = _reliability_summary(paired)
        classification.to_csv(
            args.output_dir / "mixed_class_noise_summary.csv",
            index=False,
            float_format="%.6g",
        )
        reliability.to_csv(
            args.output_dir / "probability_calibration_summary.csv",
            index=False,
            float_format="%.6g",
        )
        _plot_classification(
            classification, args.output_dir / "mixed_class_noise_v2_v3.png"
        )
        _plot_reliability(
            reliability, args.output_dir / "probability_calibration_v2_v3.png"
        )
    print(
        f"analysed {paired['snid'].nunique():,} objects, "
        f"{paired['noise_repeat'].nunique()} draws and "
        f"{paired['noise_scale'].nunique()} scales"
    )


if __name__ == "__main__":
    main()
