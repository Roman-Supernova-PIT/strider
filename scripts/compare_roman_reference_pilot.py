#!/usr/bin/env python3
"""Compare the coadd-first control and Roman-reference selection predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


VIEWS = ("original", "generated", "clean")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _source_metrics(predictions: pd.DataFrame) -> dict[str, float | int]:
    true_class = predictions["true_class"].to_numpy(dtype=np.int64)
    predicted_class = predictions["predicted_class"].to_numpy(dtype=np.int64)
    present = np.unique(true_class)
    recalls = [
        float(np.mean(predicted_class[true_class == value] == value))
        for value in present
    ]
    ia = true_class == 0
    ia_selected = predicted_class == 0
    true_positive = int(np.sum(ia & ia_selected))
    precision = true_positive / max(int(ia_selected.sum()), 1)
    recall = true_positive / max(int(ia.sum()), 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    delta = (
        predictions.loc[ia, "predicted_redshift"].to_numpy(dtype=np.float64)
        - predictions.loc[ia, "true_redshift"].to_numpy(dtype=np.float64)
    )
    return {
        "objects": int(len(predictions)),
        "Ia_objects": int(ia.sum()),
        "balanced_accuracy": float(np.mean(recalls)),
        "Ia_f1": float(f1),
        "Ia_median_absolute_delta_z": float(np.median(np.abs(delta))),
        "Ia_outlier_fraction_abs_delta_z_gt_0p1": float(
            np.mean(np.abs(delta) > 0.1)
        ),
    }


def _cohort_metrics(predictions: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    ia = predictions["true_class"].eq(0)
    redshift = predictions["true_redshift"]
    signal_to_noise = predictions["median_coadded_observed_signal_to_noise"]
    cohorts = {
        "all": np.ones(len(predictions), dtype=bool),
        "Ia_z_ge_1p5": ia & redshift.ge(1.5),
        "Ia_z_ge_1p5_snr_ge_1": ia & redshift.ge(1.5) & signal_to_noise.ge(1.0),
        "Ia_z_ge_1p5_snr_ge_2": ia & redshift.ge(1.5) & signal_to_noise.ge(2.0),
        "Ia_snr_ge_2": ia & signal_to_noise.ge(2.0),
    }
    result = {}
    for name, selected in cohorts.items():
        subset = predictions.loc[np.asarray(selected)]
        if subset.empty:
            continue
        if name == "all":
            result[name] = _source_metrics(subset)
            continue
        delta = (
            subset["predicted_redshift"].to_numpy(dtype=np.float64)
            - subset["true_redshift"].to_numpy(dtype=np.float64)
        )
        result[name] = {
            "objects": int(len(subset)),
            "median_absolute_delta_z": float(np.median(np.abs(delta))),
            "outlier_fraction_abs_delta_z_gt_0p1": float(
                np.mean(np.abs(delta) > 0.1)
            ),
            "median_class_probability_Ia": float(
                subset["class_probability_Ia"].median()
            ),
        }
    return result


def _load_run(run_dir: Path) -> dict[str, Any]:
    predictions = {}
    for view in (*VIEWS, "no_source"):
        path = run_dir / f"selection_predictions_{view}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing selection predictions: {path}")
        predictions[view] = pd.read_parquet(path)
    summary_path = run_dir / "selection_evaluation_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing selection evaluation: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "run_dir": str(run_dir.resolve()),
        "checkpoint_epoch": int(summary["checkpoint_epoch"]),
        "views": {
            view: _cohort_metrics(predictions[view]) for view in VIEWS
        },
        "no_source": {
            "objects": int(len(predictions["no_source"])),
            "mean_enough_reliable_signal": float(
                predictions["no_source"]["evidence_score"].mean()
            ),
            "fraction_enough_reliable_signal_gt_0p5": float(
                predictions["no_source"]["evidence_score"].gt(0.5).mean()
            ),
        },
    }


def compare(control_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    control = _load_run(control_dir)
    candidate = _load_run(candidate_dir)
    rows = []
    for view in VIEWS:
        for cohort in control["views"][view].keys() & candidate["views"][view].keys():
            for metric in control["views"][view][cohort].keys():
                if metric == "objects" or metric not in candidate["views"][view][cohort]:
                    continue
                control_value = float(control["views"][view][cohort][metric])
                candidate_value = float(candidate["views"][view][cohort][metric])
                rows.append(
                    {
                        "view": view,
                        "cohort": cohort,
                        "metric": metric,
                        "control": control_value,
                        "candidate": candidate_value,
                        "candidate_minus_control": candidate_value - control_value,
                    }
                )
    original_control = control["views"]["original"]["all"]
    original_candidate = candidate["views"]["original"]["all"]
    high_control = control["views"]["original"].get("Ia_z_ge_1p5", {})
    high_candidate = candidate["views"]["original"].get("Ia_z_ge_1p5", {})
    checks = {
        "Ia_F1_not_lower_by_more_than_0p01": (
            float(original_candidate["Ia_f1"])
            >= float(original_control["Ia_f1"]) - 0.01
        ),
        "overall_Ia_outlier_fraction_not_higher_by_more_than_0p01": (
            float(original_candidate["Ia_outlier_fraction_abs_delta_z_gt_0p1"])
            <= float(original_control["Ia_outlier_fraction_abs_delta_z_gt_0p1"])
            + 0.01
        ),
        "high_redshift_Ia_outlier_fraction_improves_by_at_least_0p03": (
            bool(high_control)
            and bool(high_candidate)
            and float(high_candidate["outlier_fraction_abs_delta_z_gt_0p1"])
            <= float(high_control["outlier_fraction_abs_delta_z_gt_0p1"]) - 0.03
        ),
        "source_free_false_acceptance_not_higher_by_more_than_0p005": (
            float(candidate["no_source"]["fraction_enough_reliable_signal_gt_0p5"])
            <= float(control["no_source"]["fraction_enough_reliable_signal_gt_0p5"])
            + 0.005
        ),
    }
    return {
        "control": control,
        "candidate": candidate,
        "comparison_rows": rows,
        "predefined_checks": checks,
        "passes_first_selection_gate": bool(all(checks.values())),
        "decision_scope": (
            "selection only; passing permits a second seed, not calibration "
            "or test access"
        ),
    }


def main() -> None:
    arguments = _arguments()
    report = compare(arguments.control, arguments.candidate)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Roman spectral-reference selection gate")
    for name, passed in report["predefined_checks"].items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"  decision: {'pass' if report['passes_first_selection_gate'] else 'do not promote'}")
    print(f"  results: {arguments.output}")


if __name__ == "__main__":
    main()
