"""Fit-free application of a serialized STRIDER calibration artifact.

The trained network is not changed.  Class calibration, redshift-set coverage,
and source-sufficiency calibration remain separate so none of the three can be
mistaken for another.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from strider.engine.model.redshift_scan import redshift_cell_widths


_EPSILON = 1.0e-7


def calibrate_class_probabilities(
    probability: np.ndarray,
    calibration: dict[str, Any],
) -> np.ndarray:
    """Apply binary affine-logit or multiclass temperature calibration."""
    values = _probability_matrix(probability)
    method = str(calibration["method"])
    if method == "binary_affine_logit":
        if values.shape[1] != 2:
            raise ValueError("Binary affine-logit calibration requires two classes")
        positive_index = int(calibration["positive_class_index"])
        positive = np.clip(values[:, positive_index], _EPSILON, 1.0 - _EPSILON)
        logit = np.log(positive / (1.0 - positive))
        calibrated_positive = _sigmoid(
            float(calibration["slope"]) * logit
            + float(calibration["intercept"])
        )
        result = np.empty_like(values)
        result[:, positive_index] = calibrated_positive
        result[:, 1 - positive_index] = 1.0 - calibrated_positive
        return result
    if method == "multiclass_temperature":
        temperature = float(calibration["temperature"])
        if temperature <= 0.0:
            raise ValueError("Calibration temperature must be positive")
        logits = np.log(np.clip(values, _EPSILON, 1.0)) / temperature
        logits -= logits.max(axis=1, keepdims=True)
        result = np.exp(logits)
        return result / result.sum(axis=1, keepdims=True)
    raise ValueError(f"Unsupported class calibration method: {method}")


def calibrate_joint_probability(
    joint_probability: np.ndarray,
    calibration: dict[str, Any],
) -> np.ndarray:
    """Reweight class marginals while preserving each raw ``P(z | class)``.

    Input may have shape ``(classes, redshift)`` or
    ``(objects, classes, redshift)``.  This is the coherent joint-posterior
    application path for deployment consumers that retain the full tensor.
    """
    joint = np.asarray(joint_probability, dtype=np.float64)
    squeeze = joint.ndim == 2
    if squeeze:
        joint = joint[None, ...]
    if joint.ndim != 3:
        raise ValueError("joint_probability must have shape (C, Z) or (N, C, Z)")
    if np.any(joint < 0.0):
        raise ValueError("joint_probability cannot contain negative values")
    total = joint.sum(axis=(1, 2), keepdims=True)
    if np.any(total <= 0.0):
        raise ValueError("Every joint posterior must contain positive mass")
    joint = joint / total
    raw_class = joint.sum(axis=2)
    calibrated_class = calibrate_class_probabilities(raw_class, calibration)
    conditional = np.divide(
        joint,
        raw_class[:, :, None],
        out=np.zeros_like(joint),
        where=raw_class[:, :, None] > 0.0,
    )
    result = conditional * calibrated_class[:, :, None]
    return result[0] if squeeze else result


def posterior_rank_mass(
    probability: np.ndarray,
    cell_width: np.ndarray,
) -> np.ndarray:
    """Return highest-density rank mass for every redshift cell.

    Cells with equal density receive the same rank mass, including all members
    of the tie.  Selecting cells whose rank mass is below a conformal quantile
    therefore gives a possibly disconnected highest-density set.
    """
    mass = _probability_vector(probability)
    width = np.asarray(cell_width, dtype=np.float64)
    if width.shape != mass.shape or np.any(width <= 0.0):
        raise ValueError("cell_width must be positive and match probability")
    density = mass / width
    order = np.argsort(-density, kind="stable")
    sorted_density = density[order]
    cumulative = np.cumsum(mass[order])
    ranked = np.empty_like(mass)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and np.isclose(
            sorted_density[stop], sorted_density[start], rtol=1.0e-12, atol=0.0
        ):
            stop += 1
        ranked[order[start:stop]] = cumulative[stop - 1]
        start = stop
    return ranked


def highest_density_set(
    redshift_grid: np.ndarray,
    probability: np.ndarray,
    rank_mass_quantile: float,
) -> dict[str, Any]:
    """Build a possibly disconnected conformal highest-density redshift set."""
    grid = np.asarray(redshift_grid, dtype=np.float64)
    widths = redshift_cell_widths(grid).astype(np.float64)
    ranks = posterior_rank_mass(probability, widths)
    selected = ranks <= float(rank_mass_quantile) + 1.0e-12
    if not selected.any():
        selected[int(np.argmin(ranks))] = True
    edges = _cell_edges(grid)
    intervals: list[list[float]] = []
    start: int | None = None
    for index, included in enumerate(np.r_[selected, False]):
        if included and start is None:
            start = index
        elif not included and start is not None:
            intervals.append([float(edges[start]), float(edges[index])])
            start = None
    total_width = float(sum(upper - lower for lower, upper in intervals))
    return {
        "intervals": intervals,
        "total_width": total_width,
        "component_count": len(intervals),
        "posterior_mass": float(_probability_vector(probability)[selected].sum()),
    }


def apply_calibration(
    predictions: pd.DataFrame,
    artifact: dict[str, Any] | str | Path,
    redshift_grid: np.ndarray,
) -> pd.DataFrame:
    """Add calibrated class, redshift-set, and source-sufficiency columns."""
    calibration = _load_artifact(artifact)
    if calibration.get("status") != "fitted":
        raise ValueError("Calibration artifact is not fitted")
    result = predictions.copy()
    class_names = list(calibration["classes"])
    raw_class = result[
        [f"class_probability_{name}" for name in class_names]
    ].to_numpy(dtype=np.float64)
    calibrated_class = calibrate_class_probabilities(
        raw_class, calibration["class_calibration"]
    )
    for index, name in enumerate(class_names):
        result[f"calibrated_class_probability_{name}"] = calibrated_class[:, index]
    predicted_index = calibrated_class.argmax(axis=1)
    result["calibrated_predicted_class_name"] = [
        class_names[index] for index in predicted_index
    ]
    if "Ia" in class_names:
        result["calibrated_p_Ia"] = calibrated_class[:, class_names.index("Ia")]

    evidence = np.clip(
        result["evidence_score"].to_numpy(dtype=np.float64),
        _EPSILON,
        1.0 - _EPSILON,
    )
    sufficiency = calibration["signal_sufficiency"]
    logit = np.log(evidence / (1.0 - evidence))
    source_probability = _sigmoid(
        float(sufficiency["slope"]) * logit
        + float(sufficiency["intercept"])
    )
    result["calibrated_source_probability"] = source_probability
    thresholds = sufficiency["grade_thresholds"]
    result["signal_grade"] = [
        _signal_grade(value, thresholds) for value in source_probability
    ]

    redshift = calibration["redshift_sets"]
    if "redshift_probability" not in result:
        raise ValueError(
            "Predictions do not contain redshift_probability; enable "
            "evaluation.save_redshift_probability"
        )
    probability = np.stack(result["redshift_probability"].map(np.asarray))
    visit_bands = result["visit_count"].map(visit_band).to_numpy()
    for level in redshift["levels"]:
        coverage = float(level["coverage"])
        label = _coverage_label(coverage)
        sets = []
        for index in range(len(result)):
            stratum = f"{class_names[predicted_index[index]]}|{visit_bands[index]}"
            quantile = float(
                level["strata"].get(stratum, level["global_quantile"])
            )
            sets.append(
                highest_density_set(redshift_grid, probability[index], quantile)
            )
        result[f"redshift_set_{label}_intervals"] = [
            item["intervals"] for item in sets
        ]
        result[f"redshift_set_{label}_total_width"] = [
            item["total_width"] for item in sets
        ]
        result[f"redshift_set_{label}_component_count"] = [
            item["component_count"] for item in sets
        ]
        result[f"redshift_set_{label}_posterior_mass"] = [
            item["posterior_mass"] for item in sets
        ]
    return result


def visit_band(visit_count: int | float) -> str:
    count = int(visit_count)
    if count <= 4:
        return "1-4"
    if count <= 16:
        return "5-16"
    return "17+"


def _coverage_label(coverage: float) -> str:
    return str(int(round(100.0 * coverage)))


def _signal_grade(value: float, thresholds: dict[str, float]) -> str:
    if value >= float(thresholds["high"]):
        return "high"
    if value >= float(thresholds["medium"]):
        return "medium"
    if value >= float(thresholds["low"]):
        return "low"
    return "limited"


def _cell_edges(grid: np.ndarray) -> np.ndarray:
    edges = np.empty(len(grid) + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (grid[:-1] + grid[1:])
    edges[0] = grid[0]
    edges[-1] = grid[-1]
    return edges


def _probability_matrix(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("probability must have shape (objects, classes)")
    if np.any(values < 0.0) or np.any(values.sum(axis=1) <= 0.0):
        raise ValueError("Every probability row must contain non-negative mass")
    return values / values.sum(axis=1, keepdims=True)


def _probability_vector(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    if values.ndim != 1 or np.any(values < 0.0) or values.sum() <= 0.0:
        raise ValueError("probability must be one-dimensional non-negative mass")
    return values / values.sum()


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    result = np.empty_like(value)
    positive = value >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponent = np.exp(value[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def _load_artifact(value: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    with Path(value).open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Calibration artifact must contain a JSON object")
    return loaded
