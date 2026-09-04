"""Metrics reported in direct redshift units."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def source_metrics(predictions: pd.DataFrame, outlier_delta_z: float) -> dict[str, Any]:
    true_class = predictions["true_class"].to_numpy()
    predicted_class = predictions["predicted_class"].to_numpy()
    ia_true = true_class == 0
    ia_predicted = predicted_class == 0
    interval_contains = (
        (predictions["true_redshift"] >= predictions["redshift_lower_68"])
        & (predictions["true_redshift"] <= predictions["redshift_upper_68"])
    )
    class_report = _class_metrics(true_class, predicted_class)
    ia_precision = _safe_fraction(np.sum(ia_true & ia_predicted), np.sum(ia_predicted))
    ia_recall = _safe_fraction(np.sum(ia_true & ia_predicted), np.sum(ia_true))
    ia_f1 = (
        2.0 * ia_precision * ia_recall / (ia_precision + ia_recall)
        if np.isfinite(ia_precision)
        and np.isfinite(ia_recall)
        and ia_precision + ia_recall > 0.0
        else 0.0
    )
    report = {
        "N": int(len(predictions)),
        "class_accuracy": float(np.mean(true_class == predicted_class)),
        "class_macro_f1_present": class_report["macro_f1"],
        "class_balanced_accuracy_present": class_report["balanced_accuracy"],
        "class_metrics_by_index": class_report["by_index"],
        "Ia_N": int(np.sum(ia_true)),
        "Ia_predicted_N": int(np.sum(ia_predicted)),
        "Ia_precision": ia_precision,
        "Ia_recall": ia_recall,
        "Ia_f1": float(ia_f1),
        "mean_evidence_sufficiency": float(predictions["evidence_score"].mean()),
    }
    report.update(_redshift_metrics(predictions, outlier_delta_z, interval_contains))
    if {
        "full_posterior_lower_68",
        "full_posterior_upper_68",
    }.issubset(predictions.columns):
        full_contains = (
            (predictions["true_redshift"] >= predictions["full_posterior_lower_68"])
            & (
                predictions["true_redshift"]
                <= predictions["full_posterior_upper_68"]
            )
        )
        report["full_posterior_68_interval_coverage"] = float(full_contains.mean())
    report["redshift_point_estimator_comparison"] = _point_estimator_comparison(
        predictions,
        outlier_delta_z,
    )
    report.update(_candidate_redshift_metrics(predictions, outlier_delta_z))
    report.update(_joint_candidate_metrics(predictions, outlier_delta_z))
    report.update(_phase_metrics(predictions))
    report["metrics_by_class"] = _metrics_by_class(
        predictions,
        class_report["by_index"],
        outlier_delta_z,
    )

    if ia_true.any():
        ia_predictions = predictions[ia_true]
        report["Ia_mean_evidence_sufficiency"] = float(
            ia_predictions["evidence_score"].mean()
        )
        ia_contains = (
            (ia_predictions["true_redshift"] >= ia_predictions["redshift_lower_68"])
            & (ia_predictions["true_redshift"] <= ia_predictions["redshift_upper_68"])
        )
        ia_redshift = _redshift_metrics(
            ia_predictions,
            outlier_delta_z,
            ia_contains,
        )
        report.update({f"Ia_{name}": value for name, value in ia_redshift.items()})
        if {
            "full_posterior_lower_68",
            "full_posterior_upper_68",
        }.issubset(ia_predictions.columns):
            ia_full_contains = (
                ia_predictions["true_redshift"]
                >= ia_predictions["full_posterior_lower_68"]
            ) & (
                ia_predictions["true_redshift"]
                <= ia_predictions["full_posterior_upper_68"]
            )
            report["Ia_full_posterior_68_interval_coverage"] = float(
                ia_full_contains.mean()
            )
        report["Ia_redshift_point_estimator_comparison"] = (
            _point_estimator_comparison(ia_predictions, outlier_delta_z)
        )
        report.update(
            {
                f"Ia_{name}": value
                for name, value in _candidate_redshift_metrics(
                    ia_predictions, outlier_delta_z
                ).items()
            }
        )
        report.update(
            {
                f"Ia_{name}": value
                for name, value in _joint_candidate_metrics(
                    ia_predictions, outlier_delta_z
                ).items()
            }
        )
        report.update(
            {f"Ia_{name}": value for name, value in _phase_metrics(ia_predictions).items()}
        )

    if "predicted_redshift_given_true_class" in predictions:
        conditional_contains = (
            (predictions["true_redshift"] >= predictions["true_class_redshift_lower_68"])
            & (predictions["true_redshift"] <= predictions["true_class_redshift_upper_68"])
        )
        conditional = _redshift_metrics(
            predictions,
            outlier_delta_z,
            conditional_contains,
            predicted_column="predicted_redshift_given_true_class",
        )
        report.update({f"true_class_fixed_{name}": value for name, value in conditional.items()})
    if "predicted_class_given_true_redshift" in predictions:
        conditional_class = _class_metrics(
            true_class,
            predictions["predicted_class_given_true_redshift"].to_numpy(),
        )
        report["true_redshift_fixed_class_accuracy"] = float(
            np.mean(true_class == predictions["predicted_class_given_true_redshift"].to_numpy())
        )
        report["true_redshift_fixed_class_balanced_accuracy_present"] = conditional_class[
            "balanced_accuracy"
        ]
        report["true_redshift_fixed_class_macro_f1_present"] = conditional_class["macro_f1"]
    if "posterior_primary_predicted_class" in predictions:
        primary_class = predictions["posterior_primary_predicted_class"].to_numpy()
        primary_report = _class_metrics(true_class, primary_class)
        report["primary_basin_class_accuracy"] = float(
            np.mean(true_class == primary_class)
        )
        report["primary_basin_class_balanced_accuracy_present"] = primary_report[
            "balanced_accuracy"
        ]
        report["primary_basin_class_macro_f1_present"] = primary_report["macro_f1"]
    if "joint_primary_predicted_class" in predictions:
        joint_primary_class = predictions["joint_primary_predicted_class"].to_numpy()
        joint_primary_report = _class_metrics(true_class, joint_primary_class)
        report["joint_primary_class_accuracy"] = float(
            np.mean(true_class == joint_primary_class)
        )
        report["joint_primary_class_balanced_accuracy_present"] = (
            joint_primary_report["balanced_accuracy"]
        )
        report["joint_primary_class_macro_f1_present"] = joint_primary_report[
            "macro_f1"
        ]
    probability_columns = [
        name for name in predictions.columns if name.startswith("class_probability_")
    ]
    if probability_columns and int(np.max(true_class)) < len(probability_columns):
        probability = predictions[probability_columns].to_numpy(dtype=np.float64)
        probability = probability / probability.sum(axis=1, keepdims=True).clip(min=1e-12)
        true_probability = probability[np.arange(len(probability)), true_class]
        one_hot = np.zeros_like(probability)
        one_hot[np.arange(len(probability)), true_class] = 1.0
        report["mean_class_log_loss"] = float(
            -np.log(true_probability.clip(min=1e-12)).mean()
        )
        report["mean_class_brier_score"] = float(
            np.square(probability - one_hot).sum(axis=1).mean()
        )
    return report


def _point_estimator_comparison(
    predictions: pd.DataFrame,
    outlier_delta_z: float,
) -> dict[str, dict[str, float]]:
    """Compare the released basin estimate with retained posterior summaries."""
    columns = {
        "primary_basin_peak": "posterior_primary_peak_redshift",
        "median": "posterior_median_redshift",
        "density_mode": "posterior_density_mode_redshift",
        "primary_basin_median": "posterior_primary_redshift",
        "joint_primary_basin_median": "joint_primary_redshift",
        "mean": "posterior_mean_redshift",
    }
    truth = predictions["true_redshift"].to_numpy(dtype=np.float64)
    comparison = {}
    for name, column in columns.items():
        if column not in predictions:
            continue
        delta = predictions[column].to_numpy(dtype=np.float64) - truth
        comparison[name] = {
            "median_delta_z": float(np.median(delta)),
            "median_absolute_delta_z": float(np.median(np.abs(delta))),
            "outlier_fraction": float(np.mean(np.abs(delta) > outlier_delta_z)),
        }
    return comparison


def _candidate_redshift_metrics(
    predictions: pd.DataFrame,
    outlier_delta_z: float,
) -> dict[str, float]:
    """Measure whether distinct posterior solutions retain the true redshift."""
    required = {
        "posterior_candidate_redshifts",
        "posterior_candidate_lower_68",
        "posterior_candidate_upper_68",
    }
    if not required.issubset(predictions.columns):
        return {}
    truth = predictions["true_redshift"].to_numpy(dtype=np.float64)
    best_distance = []
    interval_contains = []
    candidate_counts = []
    for row_index, (_, row) in enumerate(predictions.iterrows()):
        candidates = np.asarray(
            row["posterior_candidate_redshifts"], dtype=np.float64
        )
        lower = np.asarray(row["posterior_candidate_lower_68"], dtype=np.float64)
        upper = np.asarray(row["posterior_candidate_upper_68"], dtype=np.float64)
        if (
            not len(candidates)
            or lower.shape != candidates.shape
            or upper.shape != candidates.shape
        ):
            raise ValueError(
                "Posterior candidate columns must contain aligned non-empty lists"
            )
        candidate_counts.append(len(candidates))
        best_distance.append(float(np.min(np.abs(candidates - truth[row_index]))))
        interval_contains.append(
            bool(np.any((truth[row_index] >= lower) & (truth[row_index] <= upper)))
        )
    distance = np.asarray(best_distance, dtype=np.float64)
    counts = np.asarray(candidate_counts, dtype=np.int64)
    result = {
        "mean_posterior_candidate_count": float(np.mean(counts)),
        "fraction_with_multiple_posterior_candidates": float(np.mean(counts > 1)),
        "candidate_oracle_median_absolute_delta_z": float(np.median(distance)),
        "candidate_oracle_outlier_fraction": float(
            np.mean(distance > outlier_delta_z)
        ),
        "candidate_68_interval_oracle_coverage": float(np.mean(interval_contains)),
    }
    if {
        "posterior_candidate_class_names",
        "true_class_name",
    }.issubset(predictions.columns):
        class_retained = []
        class_and_redshift = []
        top1_class_and_redshift = []
        for _, row in predictions.iterrows():
            candidates = np.asarray(
                row["posterior_candidate_redshifts"], dtype=np.float64
            )
            classes = np.asarray(row["posterior_candidate_class_names"], dtype=str)
            if candidates.shape != classes.shape:
                raise ValueError(
                    "Posterior candidate redshifts and classes must align"
                )
            class_match = classes == str(row["true_class_name"])
            redshift_match = (
                np.abs(candidates - float(row["true_redshift"]))
                <= outlier_delta_z
            )
            class_retained.append(bool(np.any(class_match)))
            class_and_redshift.append(bool(np.any(class_match & redshift_match)))
            top1_class_and_redshift.append(bool(class_match[0] & redshift_match[0]))
        result.update(
            {
                "candidate_correct_class_retained_fraction": float(
                    np.mean(class_retained)
                ),
                "candidate_class_and_redshift_success_fraction": float(
                    np.mean(class_and_redshift)
                ),
                "primary_candidate_class_and_redshift_success_fraction": float(
                    np.mean(top1_class_and_redshift)
                ),
            }
        )
    return result


def _joint_candidate_metrics(
    predictions: pd.DataFrame,
    outlier_delta_z: float,
) -> dict[str, float]:
    """Measure retained joint class--redshift hypotheses."""
    required = {
        "joint_candidate_redshifts",
        "joint_candidate_class_indices",
        "true_class",
        "true_redshift",
    }
    if not required.issubset(predictions.columns):
        return {}
    correct_solution = []
    primary_correct_solution = []
    correct_class_retained = []
    candidate_counts = []
    best_correct_class_distance = []
    for _, row in predictions.iterrows():
        redshifts = np.asarray(row["joint_candidate_redshifts"], dtype=np.float64)
        classes = np.asarray(row["joint_candidate_class_indices"], dtype=np.int64)
        if not len(redshifts) or redshifts.shape != classes.shape:
            raise ValueError("Joint posterior candidates must contain aligned lists")
        class_match = classes == int(row["true_class"])
        distance = np.abs(redshifts - float(row["true_redshift"]))
        candidate_counts.append(len(redshifts))
        correct_class_retained.append(bool(np.any(class_match)))
        correct_solution.append(bool(np.any(class_match & (distance <= outlier_delta_z))))
        primary_correct_solution.append(
            bool(class_match[0] & (distance[0] <= outlier_delta_z))
        )
        best_correct_class_distance.append(
            float(np.min(distance[class_match])) if np.any(class_match) else np.inf
        )
    counts = np.asarray(candidate_counts, dtype=np.int64)
    finite_distance = np.asarray(best_correct_class_distance, dtype=np.float64)
    return {
        "mean_joint_candidate_count": float(np.mean(counts)),
        "fraction_with_multiple_joint_candidates": float(np.mean(counts > 1)),
        "joint_candidate_correct_class_retained_fraction": float(
            np.mean(correct_class_retained)
        ),
        "joint_candidate_class_and_redshift_success_fraction": float(
            np.mean(correct_solution)
        ),
        "joint_primary_class_and_redshift_success_fraction": float(
            np.mean(primary_correct_solution)
        ),
        "joint_candidate_correct_class_redshift_oracle_outlier_fraction": float(
            np.mean(finite_distance > outlier_delta_z)
        ),
    }


def _redshift_metrics(
    predictions: pd.DataFrame,
    outlier_delta_z: float,
    interval_contains: pd.Series | np.ndarray,
    predicted_column: str = "predicted_redshift",
) -> dict[str, float]:
    delta = predictions[predicted_column].to_numpy() - predictions["true_redshift"].to_numpy()
    return {
        "mean_delta_z": float(np.mean(delta)),
        "median_delta_z": float(np.median(delta)),
        "delta_z_16th_percentile": float(np.quantile(delta, 0.16)),
        "delta_z_84th_percentile": float(np.quantile(delta, 0.84)),
        "population_scatter_delta_z": float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0,
        "median_absolute_delta_z": float(np.median(np.abs(delta))),
        "outlier_fraction": float(np.mean(np.abs(delta) > outlier_delta_z)),
        "outlier_fraction_abs_delta_z_gt_0_05": float(np.mean(np.abs(delta) > 0.05)),
        "outlier_fraction_abs_delta_z_gt_0_10": float(np.mean(np.abs(delta) > 0.10)),
        "posterior_68_interval_coverage": float(np.asarray(interval_contains).mean()),
    }


def _metrics_by_class(
    predictions: pd.DataFrame,
    class_rows: dict[str, dict[str, float | int]],
    outlier_delta_z: float,
) -> list[dict[str, Any]]:
    rows = []
    conditional_class_rows = None
    if "predicted_class_given_true_redshift" in predictions:
        conditional_class_rows = _class_metrics(
            predictions["true_class"].to_numpy(),
            predictions["predicted_class_given_true_redshift"].to_numpy(),
        )["by_index"]
    for class_index_text, class_metrics in class_rows.items():
        class_index = int(class_index_text)
        selected = predictions[predictions["true_class"] == class_index]
        interval_contains = (
            (selected["true_redshift"] >= selected["redshift_lower_68"])
            & (selected["true_redshift"] <= selected["redshift_upper_68"])
        )
        row: dict[str, Any] = {
            "class_index": class_index,
            "class_name": (
                str(selected["true_class_name"].iloc[0])
                if "true_class_name" in selected
                else str(class_index)
            ),
            **class_metrics,
            **_redshift_metrics(selected, outlier_delta_z, interval_contains),
            **_phase_metrics(selected),
            "mean_evidence_sufficiency": float(selected["evidence_score"].mean()),
        }
        if "predicted_redshift_given_true_class" in selected:
            conditional_contains = (
                (selected["true_redshift"] >= selected["true_class_redshift_lower_68"])
                & (selected["true_redshift"] <= selected["true_class_redshift_upper_68"])
            )
            conditional = _redshift_metrics(
                selected,
                outlier_delta_z,
                conditional_contains,
                predicted_column="predicted_redshift_given_true_class",
            )
            row.update({f"true_class_fixed_{name}": value for name, value in conditional.items()})
        if conditional_class_rows is not None:
            row.update(
                {
                    f"true_redshift_fixed_{name}": value
                    for name, value in conditional_class_rows[class_index_text].items()
                    if name not in {"N", "predicted_N"}
                }
            )
        rows.append(row)
    return rows


def _phase_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    required = {
        "true_class_redshift_phase_median_delta_days",
        "true_class_redshift_phase_median_absolute_error_days",
        "true_class_redshift_phase_68_interval_coverage",
        "true_class_redshift_phase_order_accuracy",
    }
    if not required.issubset(predictions.columns):
        return {}
    return {
        "median_phase_delta_days": float(
            predictions["true_class_redshift_phase_median_delta_days"].median()
        ),
        "median_phase_absolute_error_days": float(
            predictions[
                "true_class_redshift_phase_median_absolute_error_days"
            ].median()
        ),
        "mean_phase_68_interval_coverage": float(
            predictions[
                "true_class_redshift_phase_68_interval_coverage"
            ].mean()
        ),
        "mean_phase_order_accuracy": float(
            predictions["true_class_redshift_phase_order_accuracy"].mean()
        ),
    }


def metrics_by_redshift(
    predictions: pd.DataFrame,
    outlier_delta_z: float,
    redshift_edges: list[float] | tuple[float, ...],
) -> list[dict[str, Any]]:
    edges = tuple(float(value) for value in redshift_edges)
    if len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("redshift_edges must be a strictly increasing sequence")
    reports = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = predictions[
            (predictions["true_redshift"] >= lower) & (predictions["true_redshift"] < upper)
        ]
        if selected.empty:
            continue
        report = source_metrics(selected, outlier_delta_z)
        report["redshift_range"] = [lower, upper]
        reports.append(report)
    return reports


def _safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _class_metrics(
    true_class: np.ndarray, predicted_class: np.ndarray
) -> dict[str, Any]:
    present = np.unique(true_class)
    rows: dict[str, dict[str, float | int]] = {}
    f1_values = []
    recall_values = []
    for class_index in present:
        true_positive = int(np.sum((true_class == class_index) & (predicted_class == class_index)))
        false_positive = int(np.sum((true_class != class_index) & (predicted_class == class_index)))
        false_negative = int(np.sum((true_class == class_index) & (predicted_class != class_index)))
        precision = _safe_fraction(true_positive, true_positive + false_positive)
        recall = _safe_fraction(true_positive, true_positive + false_negative)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if np.isfinite(precision) and precision + recall > 0.0
            else 0.0
        )
        rows[str(int(class_index))] = {
            "N": int(np.sum(true_class == class_index)),
            "predicted_N": int(np.sum(predicted_class == class_index)),
            "precision": precision,
            "recall": recall,
            "f1": float(f1),
        }
        f1_values.append(float(f1))
        recall_values.append(float(recall))
    return {
        "macro_f1": float(np.mean(f1_values)),
        "balanced_accuracy": float(np.mean(recall_values)),
        "by_index": rows,
    }
