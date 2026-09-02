"""Measure how a frozen model responds as generated noise changes.

Uses the selected evaluation split and reports Ia classification, redshift,
and evidence scores for source and blank inputs. Flux stays on the observer
wavelength grid and no simulation truth enters the model.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

from strider.config import project_path
from strider.data.dataset import SundialDataset

from .checkpoint import load_trained_model
from .evaluate import _predict
from .loader import inference_loader
from .metrics import metrics_by_redshift, source_metrics


@torch.no_grad()
def run_noise_check(
    config: dict[str, Any],
    noise_scales: list[float],
    max_objects: int | None = None,
    *,
    noise_family: str = "controlled-background",
    objects_per_redshift_bin: int | None = None,
    redshift_edges: list[float] | None = None,
    repeats: int = 1,
    split: str | None = None,
    output_tag: str | None = None,
    object_list: Path | None = None,
    paired_noise_seed: int | None = None,
    save_predictions: bool = False,
    ia_only: bool = False,
) -> dict[str, Any]:
    scales = [float(value) for value in noise_scales]
    if not scales or any(value < 0.0 for value in scales):
        raise ValueError("noise scales must be a nonempty list of nonnegative values")
    if noise_family not in {"controlled-background", "reported-error"}:
        raise ValueError(
            "noise_family must be 'controlled-background' or 'reported-error'"
        )
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if objects_per_redshift_bin is not None and objects_per_redshift_bin <= 0:
        raise ValueError("objects_per_redshift_bin must be positive")
    if max_objects is not None and objects_per_redshift_bin is not None:
        raise ValueError(
            "max_objects and objects_per_redshift_bin are mutually exclusive"
        )
    if object_list is not None and objects_per_redshift_bin is not None:
        raise ValueError(
            "object_list and objects_per_redshift_bin are mutually exclusive"
        )
    if object_list is not None and max_objects is not None:
        raise ValueError("object_list and max_objects are mutually exclusive")
    if ia_only and max_objects is not None:
        raise ValueError("ia_only and max_objects are mutually exclusive")
    if paired_noise_seed is not None and noise_family != "reported-error":
        raise ValueError("paired_noise_seed requires noise_family='reported-error'")
    selection_modes = sum(
        value is not None
        for value in (object_list, objects_per_redshift_bin)
    ) + int(ia_only)
    if selection_modes > 1:
        raise ValueError(
            "object_list, objects_per_redshift_bin and ia_only are mutually exclusive"
        )
    if output_tag is not None and re.fullmatch(r"[A-Za-z0-9_.-]+", output_tag) is None:
        raise ValueError(
            "output_tag may contain only letters, numbers, dots, dashes and underscores"
        )
    sweep_edges = _validated_redshift_edges(redshift_edges)

    config = deepcopy(config)
    evaluation_split = split or str(config["evaluation"].get("split", "test"))
    output_dir = project_path(config, config["project"]["output_dir"])
    model, checkpoint, device = load_trained_model(config)
    if max_objects is not None:
        if max_objects <= 0:
            raise ValueError("max_objects must be positive")
        config["data"].setdefault("runtime_object_limits", {})[
            evaluation_split
        ] = max_objects
    threshold = float(config["evaluation"]["outlier_delta_z"])
    batch_size = int(config["training"]["batch_size"])
    workers = int(config["training"]["num_workers"])

    source_view = (
        "generated"
        if noise_family == "controlled-background"
        else "reported_error_with_source"
    )
    blank_view = (
        "no_source"
        if noise_family == "controlled-background"
        else "reported_error_no_source"
    )
    paired_noise_keys: dict[int, str] | None = None
    if object_list is not None:
        cohort_indices, paired_noise_keys = _indices_from_object_list(
            config,
            evaluation_split,
            source_view,
            object_list,
        )
    elif ia_only:
        cohort_indices = _all_ia_indices(
            config, evaluation_split, source_view
        )
    else:
        cohort_indices = _balanced_ia_indices(
            config,
            evaluation_split,
            source_view,
            sweep_edges,
            objects_per_redshift_bin,
        )

    rows: list[dict[str, Any]] = []
    redshift_rows: list[dict[str, Any]] = []
    amplitude_rows: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []
    print(f"\nNoise response — {evaluation_split}")
    print(
        f"  family {noise_family} | repeats {repeats}"
        + (
            f" | {objects_per_redshift_bin} Ia per redshift interval"
            if objects_per_redshift_bin is not None
            else ""
        )
    )
    print(
        "  scale | argmax Ia: P R F1 | P(Ia)>=0.9: purity completeness | "
        "true-Ia P(Ia): median/>=0.9 | Brier/ECE | Ia |dz| | "
        "evidence: transient/noise | blank lock | input RMS: transient/noise"
    )
    for scale in scales:
        source_input_rms = _mean_input_rms(
            config,
            evaluation_split,
            source_view,
            scale,
            indices=cohort_indices,
            repeat=0,
            paired_noise_seed=paired_noise_seed,
            paired_noise_keys=paired_noise_keys,
        )
        blank_input_rms = _mean_input_rms(
            config,
            evaluation_split,
            blank_view,
            scale,
            indices=cohort_indices,
            repeat=0,
            paired_noise_seed=paired_noise_seed,
            paired_noise_keys=paired_noise_keys,
        )
        source_parts = []
        blank_parts = []
        for repeat in range(repeats):
            source_part = _predictions(
                config,
                evaluation_split,
                source_view,
                scale,
                model,
                device,
                batch_size,
                workers,
                indices=cohort_indices,
                repeat=repeat,
                paired_noise_seed=paired_noise_seed,
                paired_noise_keys=paired_noise_keys,
            )
            source_part["noise_repeat"] = repeat
            source_part["noise_scale"] = scale
            source_part["input_kind"] = "source"
            source_parts.append(source_part)
            blank_part = _predictions(
                config,
                evaluation_split,
                blank_view,
                scale,
                model,
                device,
                batch_size,
                workers,
                indices=cohort_indices,
                repeat=repeat,
                paired_noise_seed=paired_noise_seed,
                paired_noise_keys=paired_noise_keys,
            )
            blank_part["noise_repeat"] = repeat
            blank_part["noise_scale"] = scale
            blank_part["input_kind"] = "blank"
            blank_parts.append(blank_part)
        source = pd.concat(source_parts, ignore_index=True)
        blank = pd.concat(blank_parts, ignore_index=True)
        if save_predictions:
            prediction_parts.extend((source, blank))
        metrics = source_metrics(source, threshold)
        selected_ia = _ia_probability_metrics(source, threshold=0.9)
        probability = _class_probability_diagnostics(source)
        blank_lock = float(
            ((blank["predicted_redshift"] - blank["true_redshift"]).abs() <= 0.1).mean()
        )
        row = {
            "noise_scale": scale,
            "objects": int(len(source)),
            "balanced_accuracy": metrics["class_balanced_accuracy_present"],
            "Ia_precision": metrics["Ia_precision"],
            "Ia_recall": metrics["Ia_recall"],
            "Ia_f1": metrics["Ia_f1"],
            "Ia_purity_p_ge_0_9": selected_ia["purity"],
            "Ia_completeness_p_ge_0_9": selected_ia["completeness"],
            "Ia_selected_p_ge_0_9": selected_ia["selected"],
            "Ia_true_positives_p_ge_0_9": selected_ia["true_positives"],
            "Ia_mean_probability": probability["Ia_mean_probability"],
            "Ia_median_probability": probability["Ia_median_probability"],
            "Ia_fraction_p_ge_0_9": probability["Ia_fraction_p_ge_0_9"],
            "non_Ia_mean_probability": probability["non_Ia_mean_probability"],
            "class_brier_score": probability["class_brier_score"],
            "class_expected_calibration_error_15_bin": probability[
                "class_expected_calibration_error_15_bin"
            ],
            "class_mean_score_minus_prevalence": probability[
                "class_mean_score_minus_prevalence"
            ],
            "Ia_median_absolute_delta_z": metrics["Ia_median_absolute_delta_z"],
            "source_mean_evidence_score": metrics["mean_evidence_sufficiency"],
            "blank_mean_evidence_score": float(blank["evidence_score"].mean()),
            "blank_mean_Ia_probability": float(blank["p_Ia"].mean()),
            "blank_redshift_match_within_0_1": blank_lock,
            "source_mean_input_rms": source_input_rms,
            "blank_mean_input_rms": blank_input_rms,
        }
        rows.append(row)
        print(
            f"  {scale:5.2f} |"
            f" {100.0 * row['Ia_precision']:6.1f}"
            f" {100.0 * row['Ia_recall']:6.1f}"
            f" {100.0 * row['Ia_f1']:6.1f} |"
            f" {_format_percent(row['Ia_purity_p_ge_0_9'])}"
            f" {_format_percent(row['Ia_completeness_p_ge_0_9'])} |"
            f" {row['Ia_median_probability']:.3f}/"
            f"{_format_percent(row['Ia_fraction_p_ge_0_9']).strip()} |"
            f" {row['class_brier_score']:.3f}/"
            f"{row['class_expected_calibration_error_15_bin']:.3f} |"
            f" {row['Ia_median_absolute_delta_z']:7.4f} |"
            f" {row['source_mean_evidence_score']:.2f}/"
            f"{row['blank_mean_evidence_score']:.2f} |"
            f" {100.0 * blank_lock:6.1f}% |"
            f" {source_input_rms:.3g}/{blank_input_rms:.3g}"
        )
        metric_edges = (
            sweep_edges
            if redshift_edges is not None
            else config["evaluation"].get(
                "ia_redshift_edges", config["data"]["redshift_edges"]
            )
        )
        for group in metrics_by_redshift(
            source,
            threshold,
            metric_edges,
        ):
            if group.get("Ia_N", 0) == 0:
                continue
            lower, upper = group["redshift_range"]
            in_bin = source["true_redshift"].between(
                lower,
                upper,
                inclusive="left" if upper < float(metric_edges[-1]) else "both",
            )
            selected_in_bin = _ia_probability_metrics(
                source.loc[in_bin], threshold=0.9
            )
            redshift_rows.append(
                {
                    "noise_scale": scale,
                    "redshift_min": group["redshift_range"][0],
                    "redshift_max": group["redshift_range"][1],
                    "Ia_N": group["Ia_N"],
                    "Ia_precision": group["Ia_precision"],
                    "Ia_recall": group["Ia_recall"],
                    "Ia_f1": group["Ia_f1"],
                    "Ia_purity_p_ge_0_9": selected_in_bin["purity"],
                    "Ia_completeness_p_ge_0_9": selected_in_bin["completeness"],
                    "Ia_median_absolute_delta_z": group["Ia_median_absolute_delta_z"],
                    "Ia_mean_evidence_score": group["Ia_mean_evidence_sufficiency"],
                }
            )
        amplitude_rows.extend(_amplitude_summary(source, scale, sweep_edges))

    suffix = "" if output_tag is None else f"_{output_tag}"
    summary_path = output_dir / f"noise_check_summary{suffix}.json"
    table_path = output_dir / f"noise_check{suffix}.csv"
    redshift_path = output_dir / f"noise_check_by_redshift{suffix}.csv"
    amplitude_path = output_dir / f"noise_amplitude_by_redshift{suffix}.csv"
    predictions_path = output_dir / f"noise_predictions{suffix}.csv"
    cohort_path = output_dir / f"noise_cohort{suffix}.csv"
    report = {
        "device": str(device),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "split": evaluation_split,
        "requested_max_objects": max_objects,
        "noise_family": noise_family,
        "repeats": repeats,
        "objects_per_redshift_bin": objects_per_redshift_bin,
        "object_list": None if object_list is None else str(object_list),
        "paired_noise_seed": paired_noise_seed,
        "ia_only": ia_only,
        "cohort_kind": (
            "manifest"
            if object_list is not None
            else "all_ia"
            if ia_only
            else "balanced_ia"
            if objects_per_redshift_bin is not None
            else "full_mixed_split"
        ),
        "redshift_edges": sweep_edges,
        "output_tag": output_tag,
        "rows": rows,
        "files": {
            "table": str(table_path),
            "by_redshift": str(redshift_path),
        },
    }
    pd.DataFrame(rows).to_csv(table_path, index=False)
    pd.DataFrame(redshift_rows).to_csv(redshift_path, index=False)
    if cohort_indices is not None:
        cohort = SundialDataset(config, evaluation_split, source_view, training=False)
        cohort.objects.iloc[cohort_indices].to_csv(cohort_path, index=False)
        report["files"]["cohort"] = str(cohort_path)
    if save_predictions:
        pd.concat(prediction_parts, ignore_index=True).to_csv(
            predictions_path, index=False
        )
        report["files"]["predictions"] = str(predictions_path)
    if amplitude_rows:
        amplitude = pd.DataFrame(amplitude_rows)
        amplitude.to_csv(amplitude_path, index=False)
        figure_path = output_dir / f"noise_amplitude_by_redshift{suffix}.png"
        _plot_amplitude_summary(
            amplitude,
            figure_path,
            checkpoint_epoch=int(checkpoint["epoch"]),
            noise_family=noise_family,
            repeats=repeats,
            objects_per_bin=objects_per_redshift_bin,
        )
        report["files"].update(
            {
                "amplitude_by_redshift": str(amplitude_path),
                "amplitude_figure": str(figure_path),
            }
        )
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"  results {summary_path}")
    return report


def _mean_input_rms(
    config: dict[str, Any],
    split: str,
    view: str,
    noise_scale: float,
    *,
    indices: list[int] | None,
    repeat: int,
    maximum_objects: int = 32,
    paired_noise_seed: int | None = None,
    paired_noise_keys: dict[int, str] | None = None,
) -> float:
    """Measure the realized dataset input before model normalization.

    Noise-response summaries can remain unchanged either because a model is
    invariant or because a requested scale never altered the input. Recording
    this small deterministic probe makes those cases distinguishable.
    """
    dataset = SundialDataset(
        config,
        split,
        view,
        training=False,
        visit_repeat=repeat,
        generated_noise_scale=noise_scale,
        paired_noise_seed=paired_noise_seed,
    )
    _attach_paired_noise_keys(dataset, paired_noise_keys)
    positions = (
        list(range(min(maximum_objects, len(dataset))))
        if indices is None
        else list(indices[:maximum_objects])
    )
    root_mean_square = []
    for position in positions:
        item = dataset[position]
        measured = item["wavelength_mask"].bool()
        if not measured.any():
            continue
        values = item["flux"][measured].to(torch.float64)
        root_mean_square.append(float(values.square().mean().sqrt()))
    return float(np.mean(root_mean_square)) if root_mean_square else float("nan")


def _predictions(
    config: dict[str, Any],
    split: str,
    view: str,
    noise_scale: float,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    workers: int,
    *,
    indices: list[int] | None = None,
    repeat: int = 0,
    paired_noise_seed: int | None = None,
    paired_noise_keys: dict[int, str] | None = None,
) -> pd.DataFrame:
    dataset = SundialDataset(
        config,
        split,
        view,
        training=False,
        visit_repeat=repeat,
        generated_noise_scale=noise_scale,
        paired_noise_seed=paired_noise_seed,
    )
    _attach_paired_noise_keys(dataset, paired_noise_keys)
    inference_dataset = dataset if indices is None else Subset(dataset, indices)
    loader = inference_loader(
        inference_dataset,
        config,
        batch_size=batch_size,
        num_workers=workers,
    )
    return _predict(model, loader, device)


def _validated_redshift_edges(edges: list[float] | None) -> list[float]:
    values = [0.0, 0.75, 1.25, 1.75, 2.25, 3.0] if edges is None else edges
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 2 or not np.isfinite(array).all() or np.any(np.diff(array) <= 0.0):
        raise ValueError("redshift_edges must be finite and strictly increasing")
    return array.tolist()


def _ia_probability_metrics(
    predictions: pd.DataFrame,
    *,
    threshold: float,
) -> dict[str, float | int]:
    """Summarize an Ia probability cut without disguising Ia-only tests as purity.

    Purity requires both positive and negative examples. It is therefore left
    undefined for a true-Ia-only cohort, while completeness remains meaningful.
    """
    is_ia = predictions["true_class_name"].eq("Ia").to_numpy()
    selected = predictions["p_Ia"].to_numpy(dtype=np.float64) >= threshold
    true_positives = int(np.sum(is_ia & selected))
    selected_count = int(np.sum(selected))
    ia_count = int(np.sum(is_ia))
    has_negative_examples = bool(np.any(~is_ia))
    purity = (
        float(true_positives / selected_count)
        if has_negative_examples and selected_count > 0
        else float("nan")
    )
    completeness = (
        float(true_positives / ia_count) if ia_count > 0 else float("nan")
    )
    return {
        "purity": purity,
        "completeness": completeness,
        "selected": selected_count,
        "true_positives": true_positives,
        "true_ia": ia_count,
    }


def _class_probability_diagnostics(
    predictions: pd.DataFrame,
    *,
    bins: int = 15,
) -> dict[str, float]:
    """Measure raw Ia-score confidence and reliability for one noise level.

    These are diagnostics of the uncalibrated model score. A mixed-class
    manifest is required to interpret Brier score, ECE, or the mean-score bias
    as population calibration; on an Ia-only cohort they remain descriptive
    measures of confidence response.
    """
    if bins <= 0:
        raise ValueError("bins must be positive")
    score = predictions["p_Ia"].to_numpy(dtype=np.float64)
    truth = predictions["true_class_name"].eq("Ia").to_numpy(dtype=bool)
    finite = np.isfinite(score)
    score = np.clip(score[finite], 0.0, 1.0)
    truth = truth[finite]
    if not len(score):
        return {
            "Ia_mean_probability": float("nan"),
            "Ia_median_probability": float("nan"),
            "Ia_fraction_p_ge_0_9": float("nan"),
            "non_Ia_mean_probability": float("nan"),
            "class_brier_score": float("nan"),
            "class_expected_calibration_error_15_bin": float("nan"),
            "class_mean_score_minus_prevalence": float("nan"),
        }

    ia_score = score[truth]
    non_ia_score = score[~truth]
    bin_index = np.minimum((score * bins).astype(int), bins - 1)
    calibration_error = 0.0
    for index in range(bins):
        selected = bin_index == index
        if not np.any(selected):
            continue
        calibration_error += float(np.mean(selected)) * abs(
            float(np.mean(score[selected])) - float(np.mean(truth[selected]))
        )
    return {
        "Ia_mean_probability": (
            float(np.mean(ia_score)) if len(ia_score) else float("nan")
        ),
        "Ia_median_probability": (
            float(np.median(ia_score)) if len(ia_score) else float("nan")
        ),
        "Ia_fraction_p_ge_0_9": (
            float(np.mean(ia_score >= 0.9)) if len(ia_score) else float("nan")
        ),
        "non_Ia_mean_probability": (
            float(np.mean(non_ia_score)) if len(non_ia_score) else float("nan")
        ),
        "class_brier_score": float(np.mean((score - truth.astype(float)) ** 2)),
        "class_expected_calibration_error_15_bin": calibration_error,
        "class_mean_score_minus_prevalence": float(
            np.mean(score) - np.mean(truth)
        ),
    }


def _format_percent(value: float) -> str:
    return "   n/a" if not np.isfinite(value) else f"{100.0 * value:6.1f}"


def _balanced_ia_indices(
    config: dict[str, Any],
    split: str,
    view: str,
    redshift_edges: list[float],
    objects_per_bin: int | None,
) -> list[int] | None:
    if objects_per_bin is None:
        return None
    dataset = SundialDataset(config, split, view, training=False)
    objects = dataset.objects.reset_index(drop=True).copy()
    ia = objects[objects["class_name"].eq("Ia")].copy()
    final_edge = np.nextafter(float(redshift_edges[-1]), np.inf)
    cut_edges = [*redshift_edges[:-1], final_edge]
    ia["sweep_bin"] = pd.cut(
        ia["redshift"],
        cut_edges,
        labels=False,
        right=False,
        include_lowest=True,
    )
    selected: list[int] = []
    seed = int(config["project"]["seed"])
    for bin_index in range(len(redshift_edges) - 1):
        available = ia[ia["sweep_bin"].eq(bin_index)]
        if len(available) < objects_per_bin:
            raise ValueError(
                f"Split {split!r} has {len(available)} Ia objects in "
                f"[{redshift_edges[bin_index]}, {redshift_edges[bin_index + 1]}), "
                f"but {objects_per_bin} were requested"
            )
        chosen = available.sample(
            n=objects_per_bin,
            random_state=seed + bin_index,
        )
        selected.extend(chosen.index.astype(int).tolist())
    return sorted(selected)


def _all_ia_indices(
    config: dict[str, Any], split: str, view: str
) -> list[int]:
    dataset = SundialDataset(config, split, view, training=False)
    return dataset.objects.index[
        dataset.objects["class_name"].eq("Ia")
    ].astype(int).tolist()


def _indices_from_object_list(
    config: dict[str, Any],
    split: str,
    view: str,
    object_list: Path,
) -> tuple[list[int], dict[int, str]]:
    """Resolve a frozen-v2 manifest onto a prepared v3 split by unique SNID."""
    requested = pd.read_csv(object_list)
    if "snid" not in requested:
        raise ValueError(f"object list lacks an snid column: {object_list}")
    if requested["snid"].duplicated().any():
        raise ValueError(f"object list contains duplicate snid values: {object_list}")
    dataset = SundialDataset(config, split, view, training=False)
    objects = dataset.objects.reset_index(drop=True)
    if objects["snid"].duplicated().any():
        raise ValueError(
            f"prepared split {split!r} has duplicate SNIDs; an explicit product key is required"
        )
    by_snid = {int(snid): int(index) for index, snid in objects["snid"].items()}
    indices: list[int] = []
    keys: dict[int, str] = {}
    missing: list[int] = []
    for row in requested.itertuples(index=False):
        snid = int(row.snid)
        index = by_snid.get(snid)
        if index is None:
            missing.append(snid)
            continue
        if hasattr(row, "z_true"):
            actual = float(objects.iloc[index].redshift)
            if not np.isclose(actual, float(row.z_true), rtol=0.0, atol=1.0e-5):
                raise ValueError(
                    f"SNID {snid} has z={actual:.7f} in v3 but "
                    f"z={float(row.z_true):.7f} in {object_list}"
                )
        indices.append(index)
        keys[index] = str(getattr(row, "paired_noise_key", snid))
    if missing:
        examples = ", ".join(str(value) for value in missing[:8])
        raise ValueError(
            f"{len(missing)} requested objects are absent from split {split!r}: {examples}"
        )
    return indices, keys


def _attach_paired_noise_keys(
    dataset: SundialDataset,
    paired_noise_keys: dict[int, str] | None,
) -> None:
    if paired_noise_keys is None:
        return
    dataset.objects["paired_noise_key"] = dataset.objects.index.map(
        lambda index: paired_noise_keys.get(int(index), "")
    )


def _amplitude_summary(
    predictions: pd.DataFrame,
    noise_scale: float,
    redshift_edges: list[float],
) -> list[dict[str, Any]]:
    final_edge = np.nextafter(float(redshift_edges[-1]), np.inf)
    cut_edges = [*redshift_edges[:-1], final_edge]
    labels = [
        f"{lower:.2f}\N{EN DASH}{upper:.2f}"
        for lower, upper in zip(redshift_edges[:-1], redshift_edges[1:], strict=True)
    ]
    frame = predictions[predictions["true_class_name"].eq("Ia")].copy()
    frame["redshift_bin"] = pd.cut(
        frame["true_redshift"],
        cut_edges,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    rows = []
    for label, group in frame.groupby("redshift_bin", observed=True, sort=False):
        delta = group["predicted_redshift"].to_numpy(dtype=np.float64) - group[
            "true_redshift"
        ].to_numpy(dtype=np.float64)
        rows.append(
            {
                "redshift_bin": str(label),
                "noise_scale": float(noise_scale),
                "noise_percent": 100.0 * float(noise_scale),
                "n_ia": int(len(group)),
                "median_delta_z": float(np.median(delta)),
                "scatter_delta_z": float(np.std(delta, ddof=0)),
                "median_abs_delta_z": float(np.median(np.abs(delta))),
                "fraction_p_ia_ge_0p9": float(np.mean(group["p_Ia"] >= 0.9)),
            }
        )
    return rows


def _plot_amplitude_summary(
    summary: pd.DataFrame,
    output: Path,
    *,
    checkpoint_epoch: int,
    noise_family: str,
    repeats: int,
    objects_per_bin: int | None,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), sharex=True)
    metrics = (
        ("median_delta_z", r"median $\Delta z$"),
        ("scatter_delta_z", r"scatter of $\Delta z$"),
        ("median_abs_delta_z", r"median $|\Delta z|$"),
        ("fraction_p_ia_ge_0p9", r"fraction with $P(\mathrm{Ia})\geq0.9$"),
    )
    labels = sorted(
        set(summary["redshift_bin"].astype(str)),
        key=lambda label: float(label.split("\N{EN DASH}", maxsplit=1)[0]),
    )
    colours = plt.cm.viridis(np.linspace(0.08, 0.9, len(labels)))
    for label, colour in zip(labels, colours, strict=True):
        group = summary[summary["redshift_bin"].astype(str).eq(label)].sort_values(
            "noise_scale"
        )
        for axis, (column, _) in zip(axes.ravel(), metrics, strict=True):
            axis.plot(
                group["noise_percent"],
                group[column],
                "o-",
                color=colour,
                linewidth=1.8,
                markersize=4.5,
                label=label,
            )
    for axis, (_, ylabel) in zip(axes.ravel(), metrics, strict=True):
        axis.axvline(100.0, color="0.45", linewidth=1.0, linestyle="--")
        axis.set_ylabel(ylabel)
    axes[0, 0].axhline(0.0, color="0.35", linewidth=0.8)
    axes[1, 0].set_xlabel("injected noise scale [% of nominal FLAMERR]")
    axes[1, 1].set_xlabel("injected noise scale [% of nominal FLAMERR]")
    axes[1, 1].set_ylim(-0.02, 1.02)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        title="true redshift",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=len(labels),
        frameon=False,
    )
    fig.suptitle(
        f"STRIDER v3 noise response — checkpoint epoch {checkpoint_epoch}",
        y=0.925,
        fontsize=12,
    )
    first_scale = float(summary["noise_scale"].min())
    ia_draws = int(
        summary.loc[summary["noise_scale"].eq(first_scale), "n_ia"].sum()
    )
    ia_objects = ia_draws // repeats
    cohort_text = f"{ia_objects:,} Ia objects"
    if objects_per_bin is not None:
        cohort_text += f" ({objects_per_bin} per interval)"
    fig.text(
        0.5,
        0.012,
        f"{cohort_text}; {repeats} paired {noise_family} draws.",
        ha="center",
        fontsize=9,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.9))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
