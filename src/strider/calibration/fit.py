"""Fit STRIDER calibration from the reserved calibration split only."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from strider.config import project_path, resolved_config_sha256
from strider.model.redshift_scan import (
    build_redshift_grid,
    redshift_cell_widths,
)

from .core import (
    apply_calibration,
    calibrate_class_probabilities,
    posterior_rank_mass,
    visit_band,
)


_EPSILON = 1.0e-7


def fit_calibration(
    config: dict[str, Any],
    *,
    source_predictions: str | Path | None = None,
    blank_predictions: str | Path | None = None,
    output: str | Path | None = None,
    folds: int = 2,
    coverage_levels: tuple[float, ...] = (0.68, 0.90),
    minimum_stratum_size: int = 200,
    gold_purity: float = 0.95,
) -> dict[str, Any]:
    """Fit and serialize the three independent STRIDER calibration layers."""
    run_dir = project_path(config, config["project"]["output_dir"])
    source_path = Path(source_predictions) if source_predictions else (
        run_dir / "calibration_predictions_original.parquet"
    )
    blank_path = Path(blank_predictions) if blank_predictions else (
        run_dir / "calibration_predictions_no_source.parquet"
    )
    output_path = Path(output) if output else run_dir / "calibration.json"
    source = pd.read_parquet(source_path)
    blank = pd.read_parquet(blank_path)
    classes = list(config["model"]["classes"])
    config_digest = resolved_config_sha256(config)
    provenance = _validate_inputs(
        source,
        blank,
        source_path=source_path,
        blank_path=blank_path,
        classes=classes,
        config_digest=config_digest,
    )
    if folds < 2:
        raise ValueError("Calibration diagnostics require at least two folds")
    if minimum_stratum_size < 1:
        raise ValueError("minimum_stratum_size must be positive")
    levels = tuple(float(level) for level in coverage_levels)
    if not levels or any(level <= 0.0 or level >= 1.0 for level in levels):
        raise ValueError("coverage_levels must be probabilities strictly between 0 and 1")

    raw_class = source[
        [f"class_probability_{name}" for name in classes]
    ].to_numpy(dtype=np.float64)
    truth = source["true_class_name"].map(classes.index).to_numpy(dtype=np.int64)
    source_fold = _fold_assignments(source["snid"], folds)
    blank_fold = _fold_assignments(blank["snid"], folds)

    class_calibration = _fit_class_calibration(raw_class, truth, classes)
    oof_class = np.empty_like(raw_class)
    for fold in range(folds):
        train = source_fold != fold
        held_out = ~train
        fold_calibration = _fit_class_calibration(
            raw_class[train], truth[train], classes
        )
        oof_class[held_out] = calibrate_class_probabilities(
            raw_class[held_out], fold_calibration
        )
    class_diagnostics = {
        "folds": folds,
        "raw": _class_diagnostics(raw_class, truth),
        "cross_fitted": _class_diagnostics(oof_class, truth),
    }
    class_calibration["operating_points"] = _operating_points(
        oof_class, truth, classes, gold_purity
    )
    class_calibration["diagnostics"] = class_diagnostics

    grid = build_redshift_grid(
        float(config["model"]["redshift_min"]),
        float(config["model"]["redshift_max"]),
        int(config["model"]["redshift_bins"]),
        str(config["model"].get("redshift_spacing", "linear")),
    ).astype(np.float64)
    redshift_probability = _redshift_probability(source, len(grid))
    conformity = _redshift_conformity_scores(
        redshift_probability,
        source["true_redshift"].to_numpy(dtype=np.float64),
        grid,
    )
    full_calibrated_class = calibrate_class_probabilities(
        raw_class, class_calibration
    )
    predicted_names = np.asarray(classes, dtype=object)[
        full_calibrated_class.argmax(axis=1)
    ]
    redshift_calibration = _fit_redshift_calibration(
        conformity,
        predicted_names,
        source["visit_count"].to_numpy(),
        levels,
        minimum_stratum_size,
    )
    redshift_calibration["posterior_basis"] = (
        "raw full redshift marginal P(z); class calibration is reported separately"
    )
    redshift_calibration["diagnostics"] = _cross_fitted_redshift_diagnostics(
        conformity=conformity,
        predicted_names=np.asarray(classes, dtype=object)[oof_class.argmax(axis=1)],
        visit_count=source["visit_count"].to_numpy(),
        folds=source_fold,
        levels=levels,
        minimum_stratum_size=minimum_stratum_size,
    )

    sufficiency_calibration = _fit_sufficiency_calibration(
        source["evidence_score"].to_numpy(dtype=np.float64),
        blank["evidence_score"].to_numpy(dtype=np.float64),
        source_fold,
        blank_fold,
    )

    artifact: dict[str, Any] = {
        "format_version": "strider-calibration-v1",
        "status": "fitted",
        "config_sha256": config_digest,
        "checkpoint_epoch": provenance["checkpoint_epoch"],
        "calibration_split": "calibration",
        "source_view": provenance["source_view"],
        "blank_view": provenance["blank_view"],
        "source_predictions": str(source_path),
        "blank_predictions": str(blank_path),
        "source_objects": int(len(source)),
        "blank_objects": int(len(blank)),
        "classes": classes,
        "class_calibration": class_calibration,
        "redshift_sets": redshift_calibration,
        "signal_sufficiency": sufficiency_calibration,
        "separation_of_meanings": {
            "class_probability": "calibrated probability of transient class",
            "redshift_set": "coverage-calibrated, possibly disconnected set in redshift",
            "source_probability": (
                "probability of source-bearing versus matched blank input under a "
                "50/50 calibration reference; not redshift confidence"
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, artifact)

    calibrated_path = output_path.parent / f"{source_path.stem}_calibrated.parquet"
    calibrated = apply_calibration(source, artifact, grid)
    calibrated.to_parquet(calibrated_path, index=False)
    summary = {
        "calibration": str(output_path),
        "calibrated_source_predictions": str(calibrated_path),
        "class_method": class_calibration["method"],
        "redshift_coverage": list(levels),
        "source_objects": int(len(source)),
        "blank_objects": int(len(blank)),
        "diagnostics": {
            "class": class_diagnostics,
            "redshift": redshift_calibration["diagnostics"],
            "signal_sufficiency": sufficiency_calibration["diagnostics"],
        },
    }
    summary_path = output_path.parent / "calibration_summary.json"
    _write_json(summary_path, summary)
    summary["summary"] = str(summary_path)
    return summary


def _fit_class_calibration(
    probability: np.ndarray,
    truth: np.ndarray,
    classes: list[str],
) -> dict[str, Any]:
    if len(classes) == 2:
        positive_index = classes.index("Ia") if "Ia" in classes else 0
        positive = np.clip(
            probability[:, positive_index], _EPSILON, 1.0 - _EPSILON
        )
        feature = np.log(positive / (1.0 - positive))[:, None]
        target = (truth == positive_index).astype(np.int64)
        if len(np.unique(target)) != 2:
            raise ValueError("Binary calibration data must contain both classes")
        model = LogisticRegression(C=100.0, solver="lbfgs")
        model.fit(feature, target)
        return {
            "method": "binary_affine_logit",
            "positive_class": classes[positive_index],
            "positive_class_index": positive_index,
            "slope": float(model.coef_[0, 0]),
            "intercept": float(model.intercept_[0]),
            "regularization_C": 100.0,
            "empirical_positive_prevalence": float(target.mean()),
        }
    temperature = _fit_temperature(probability, truth)
    return {
        "method": "multiclass_temperature",
        "temperature": temperature,
        "empirical_class_prevalence": {
            name: float(np.mean(truth == index))
            for index, name in enumerate(classes)
        },
    }


def _fit_temperature(probability: np.ndarray, truth: np.ndarray) -> float:
    log_probability = np.log(np.clip(probability, _EPSILON, 1.0))

    def objective(log_temperature: float) -> float:
        logits = log_probability / math.exp(log_temperature)
        logits -= logits.max(axis=1, keepdims=True)
        normalized = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        return float(-normalized[np.arange(len(truth)), truth].mean())

    lower, upper = -4.0, 4.0
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    left_value, right_value = objective(left), objective(right)
    for _ in range(80):
        if left_value < right_value:
            upper, right, right_value = right, left, left_value
            left = upper - ratio * (upper - lower)
            left_value = objective(left)
        else:
            lower, left, left_value = left, right, right_value
            right = lower + ratio * (upper - lower)
            right_value = objective(right)
    return float(math.exp(0.5 * (lower + upper)))


def _fit_redshift_calibration(
    conformity: np.ndarray,
    predicted_names: np.ndarray,
    visit_count: np.ndarray,
    levels: tuple[float, ...],
    minimum_stratum_size: int,
) -> dict[str, Any]:
    bands = np.asarray([visit_band(value) for value in visit_count], dtype=object)
    keys = np.asarray(
        [f"{name}|{band}" for name, band in zip(predicted_names, bands, strict=True)],
        dtype=object,
    )
    level_records = []
    for coverage in levels:
        strata: dict[str, float] = {}
        counts: dict[str, int] = {}
        for key in sorted(set(keys.tolist())):
            selected = keys == key
            count = int(selected.sum())
            counts[key] = count
            if count >= minimum_stratum_size:
                strata[key] = _conformal_quantile(conformity[selected], coverage)
        level_records.append(
            {
                "coverage": coverage,
                "global_quantile": _conformal_quantile(conformity, coverage),
                "strata": strata,
                "stratum_counts": counts,
            }
        )
    return {
        "method": "class-and-visit-aware-conformal-highest-density-set",
        "conformity_score": "posterior mass at density >= density(true-z cell)",
        "visit_bands": ["1-4", "5-16", "17+"],
        "minimum_stratum_size": minimum_stratum_size,
        "fallback": "global_quantile",
        "levels": level_records,
    }


def _cross_fitted_redshift_diagnostics(
    *,
    conformity: np.ndarray,
    predicted_names: np.ndarray,
    visit_count: np.ndarray,
    folds: np.ndarray,
    levels: tuple[float, ...],
    minimum_stratum_size: int,
) -> dict[str, Any]:
    achieved: dict[str, list[bool]] = {str(level): [] for level in levels}
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        held_out = ~train
        fitted = _fit_redshift_calibration(
            conformity[train],
            predicted_names[train],
            visit_count[train],
            levels,
            minimum_stratum_size,
        )
        held_bands = [visit_band(value) for value in visit_count[held_out]]
        held_names = predicted_names[held_out]
        held_scores = conformity[held_out]
        for record in fitted["levels"]:
            outcomes = achieved[str(record["coverage"])]
            for score, name, band in zip(
                held_scores, held_names, held_bands, strict=True
            ):
                key = f"{name}|{band}"
                quantile = record["strata"].get(key, record["global_quantile"])
                outcomes.append(bool(score <= quantile + 1.0e-12))
    return {
        "folds": int(len(np.unique(folds))),
        "cross_fitted_coverage": {
            key: float(np.mean(values)) for key, values in achieved.items()
        },
    }


def _fit_sufficiency_calibration(
    source_score: np.ndarray,
    blank_score: np.ndarray,
    source_folds: np.ndarray,
    blank_folds: np.ndarray,
) -> dict[str, Any]:
    score = np.concatenate([source_score, blank_score])
    target = np.concatenate(
        [np.ones(len(source_score), dtype=np.int64), np.zeros(len(blank_score), dtype=np.int64)]
    )
    folds = np.concatenate([source_folds, blank_folds])
    oof = np.empty(len(score), dtype=np.float64)
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        held_out = ~train
        model = _fit_logit_model(score[train], target[train], balanced=True)
        oof[held_out] = _apply_logit_model(score[held_out], model)
    fitted = _fit_logit_model(score, target, balanced=True)
    calibrated_blank = _apply_logit_model(blank_score, fitted)
    thresholds = {
        "high": _threshold_for_false_positive_rate(calibrated_blank, 0.001),
        "medium": _threshold_for_false_positive_rate(calibrated_blank, 0.01),
        "low": _threshold_for_false_positive_rate(calibrated_blank, 0.05),
    }
    diagnostics = _binary_diagnostics(target, oof)
    diagnostics["cross_fitted_blank_false_positive_rate"] = {
        grade: float(np.mean(oof[len(source_score) :] >= threshold))
        for grade, threshold in thresholds.items()
    }
    diagnostics["cross_fitted_source_acceptance_rate"] = {
        grade: float(np.mean(oof[: len(source_score)] >= threshold))
        for grade, threshold in thresholds.items()
    }
    return {
        "method": "balanced-affine-logit",
        "reference_prior": {"source": 0.5, "blank": 0.5},
        "slope": fitted["slope"],
        "intercept": fitted["intercept"],
        "grade_thresholds": thresholds,
        "blank_false_positive_caps": {"high": 0.001, "medium": 0.01, "low": 0.05},
        "meaning": "source-bearing versus matched blank; not redshift confidence",
        "diagnostics": diagnostics,
    }


def _fit_logit_model(
    score: np.ndarray, target: np.ndarray, *, balanced: bool
) -> dict[str, float]:
    clipped = np.clip(score, _EPSILON, 1.0 - _EPSILON)
    feature = np.log(clipped / (1.0 - clipped))[:, None]
    model = LogisticRegression(
        C=100.0,
        solver="lbfgs",
        class_weight="balanced" if balanced else None,
    )
    model.fit(feature, target)
    return {"slope": float(model.coef_[0, 0]), "intercept": float(model.intercept_[0])}


def _apply_logit_model(score: np.ndarray, model: dict[str, float]) -> np.ndarray:
    clipped = np.clip(score, _EPSILON, 1.0 - _EPSILON)
    logit = np.log(clipped / (1.0 - clipped))
    value = model["slope"] * logit + model["intercept"]
    return 1.0 / (1.0 + np.exp(-np.clip(value, -700.0, 700.0)))


def _redshift_conformity_scores(
    probability: np.ndarray,
    true_redshift: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    widths = redshift_cell_widths(grid).astype(np.float64)
    scores = np.empty(len(probability), dtype=np.float64)
    for index, mass in enumerate(probability):
        true_index = int(np.abs(grid - true_redshift[index]).argmin())
        scores[index] = posterior_rank_mass(mass, widths)[true_index]
    return scores


def _redshift_probability(frame: pd.DataFrame, bins: int) -> np.ndarray:
    if "redshift_probability" not in frame:
        raise ValueError(
            "Calibration predictions require redshift_probability; set "
            "evaluation.save_redshift_probability: true"
        )
    values = np.stack(frame["redshift_probability"].map(np.asarray)).astype(np.float64)
    if values.shape != (len(frame), bins):
        raise ValueError(
            f"redshift_probability has shape {values.shape}; expected ({len(frame)}, {bins})"
        )
    return values


def _conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    values = np.sort(np.asarray(scores, dtype=np.float64))
    if len(values) == 0:
        raise ValueError("Cannot fit a conformal quantile from zero objects")
    rank = min(int(math.ceil((len(values) + 1) * coverage)), len(values))
    return float(values[rank - 1])


def _operating_points(
    probability: np.ndarray,
    truth: np.ndarray,
    classes: list[str],
    gold_purity: float,
) -> dict[str, Any]:
    if "Ia" not in classes:
        return {"status": "not_applicable_without_Ia"}
    ia_index = classes.index("Ia")
    score = probability[:, ia_index]
    target = truth == ia_index
    candidates = np.unique(np.r_[0.0, score, 1.0])
    best: dict[str, float] | None = None
    gold: dict[str, float] | None = None
    for threshold in candidates:
        selected = score >= threshold
        true_positive = int(np.sum(selected & target))
        false_positive = int(np.sum(selected & ~target))
        false_negative = int(np.sum(~selected & target))
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, _EPSILON)
        row = {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "selected": int(selected.sum()),
        }
        if best is None or (row["f1"], row["recall"]) > (best["f1"], best["recall"]):
            best = row
        lower = _wilson_lower_bound(true_positive, true_positive + false_positive)
        if lower >= gold_purity and (gold is None or row["recall"] > gold["recall"]):
            gold = {**row, "precision_lower_95": lower}
    return {
        "selection_basis": "cross-fitted calibrated probabilities",
        "balanced_maximum_f1": best,
        "high_purity": {
            "target_precision_lower_95": gold_purity,
            "status": "available" if gold is not None else "unavailable",
            "operating_point": gold,
        },
    }


def _wilson_lower_bound(successes: int, total: int, z_value: float = 1.645) -> float:
    if total == 0:
        return 0.0
    rate = successes / total
    denominator = 1.0 + z_value**2 / total
    centre = rate + z_value**2 / (2.0 * total)
    radius = z_value * math.sqrt(
        rate * (1.0 - rate) / total + z_value**2 / (4.0 * total**2)
    )
    return float((centre - radius) / denominator)


def _class_diagnostics(probability: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probability, _EPSILON, 1.0 - _EPSILON)
    one_hot = np.eye(probability.shape[1])[truth]
    return {
        "negative_log_likelihood": float(log_loss(truth, clipped, labels=range(probability.shape[1]))),
        "multiclass_brier_score": float(np.mean(np.sum((clipped - one_hot) ** 2, axis=1))),
        "expected_calibration_error": _expected_calibration_error(clipped, truth),
    }


def _binary_diagnostics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(target, probability)),
        "average_precision": float(average_precision_score(target, probability)),
        "negative_log_likelihood": float(log_loss(target, probability)),
        "brier_score": float(brier_score_loss(target, probability)),
    }


def _expected_calibration_error(
    probability: np.ndarray, truth: np.ndarray, bins: int = 15
) -> float:
    confidence = probability.max(axis=1)
    correct = probability.argmax(axis=1) == truth
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            error += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return error


def _threshold_for_false_positive_rate(values: np.ndarray, cap: float) -> float:
    for threshold in np.unique(values):
        if float(np.mean(values >= threshold)) <= cap:
            return float(threshold)
    return float(np.nextafter(np.max(values), np.inf))


def _fold_assignments(snids: pd.Series, folds: int) -> np.ndarray:
    values = []
    for snid in snids:
        digest = hashlib.blake2b(str(snid).encode("utf-8"), digest_size=8).digest()
        values.append(int.from_bytes(digest, "little") % folds)
    result = np.asarray(values, dtype=np.int64)
    if len(np.unique(result)) != folds:
        raise ValueError("Too few distinct objects for the requested calibration folds")
    return result


def _validate_inputs(
    source: pd.DataFrame,
    blank: pd.DataFrame,
    *,
    source_path: Path,
    blank_path: Path,
    classes: list[str],
    config_digest: str,
) -> dict[str, Any]:
    common = {"snid", "evidence_score", "data_split", "data_view", "checkpoint_epoch", "config_sha256"}
    source_required = common | {
        "true_class_name",
        "true_redshift",
        "visit_count",
        "redshift_probability",
        *(f"class_probability_{name}" for name in classes),
    }
    for name, frame, required in (
        ("source", source, source_required),
        ("blank", blank, common),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(
                f"{name} calibration predictions lack provenance or required fields: {missing}. "
                "Regenerate them with the current evaluator."
            )
        if frame.empty:
            raise ValueError(f"{name} calibration predictions are empty")
        if set(frame["data_split"].astype(str)) != {"calibration"}:
            raise ValueError(f"{name} predictions must come only from the calibration split")
        if set(frame["config_sha256"].astype(str)) != {config_digest}:
            raise ValueError(f"{name} predictions do not match the resolved configuration")
    source_epoch = set(source["checkpoint_epoch"].astype(int))
    blank_epoch = set(blank["checkpoint_epoch"].astype(int))
    if len(source_epoch) != 1 or source_epoch != blank_epoch:
        raise ValueError("Source and blank predictions must use the same checkpoint epoch")
    source_view = set(source["data_view"].astype(str))
    blank_view = set(blank["data_view"].astype(str))
    if len(source_view) != 1 or len(blank_view) != 1:
        raise ValueError("Each calibration parquet must contain exactly one view")
    if source_view == {"no_source"} or blank_view != {"no_source"}:
        raise ValueError("Use a source-bearing view and the matched no_source blank view")
    if set(source["snid"]) != set(blank["snid"]):
        raise ValueError("Source and blank calibration views must contain the same SNIDs")
    return {
        "checkpoint_epoch": int(next(iter(source_epoch))),
        "source_view": next(iter(source_view)),
        "blank_view": next(iter(blank_view)),
        "source_path": str(source_path),
        "blank_path": str(blank_path),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
