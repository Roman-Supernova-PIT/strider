"""Test how visit accumulation changes redshift uncertainty and blank controls.

The trained checkpoint is loaded with its original configuration. Only the
ONIR visit exponent is changed in memory, so this is a sensitivity test rather
than a new fit. Results are measured on generated source and no-source views.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from torch.utils.data import Subset

from strider.config import project_path
from strider.data.dataset import SundialDataset

from .checkpoint import load_trained_model
from .evaluate import _predict
from .loader import inference_loader
from .metrics import source_metrics
from .subset import stratified_positions


DEFAULT_EXPONENTS = (0.0, 0.15, 0.25, 0.35, 0.5)
DEFAULT_VISIT_COUNTS = (1, 4, 16, 32)


def run_evidence_growth_sweep(
    config: dict[str, Any],
    exponents: Iterable[float] = DEFAULT_EXPONENTS,
    visit_counts: Iterable[int] = DEFAULT_VISIT_COUNTS,
    max_objects: int = 500,
    repeats: int = 1,
) -> dict[str, Any]:
    """Evaluate source coverage and blank behaviour for candidate exponents."""
    exponent_values = _validated_exponents(exponents)
    visit_values = _validated_visit_counts(visit_counts)
    if max_objects < 1:
        raise ValueError("max_objects must be positive")
    if repeats < 1:
        raise ValueError("repeats must be positive")

    output_dir = project_path(config, config["project"]["output_dir"])
    model, checkpoint, device = load_trained_model(config)
    onir = getattr(model, "onir", None)
    if onir is None or not hasattr(onir, "visit_evidence_exponent"):
        raise ValueError("Evidence growth requires a visit-combined ONIR model")

    split = str(config["training"].get("selection_split", "selection"))
    threshold = float(config["evaluation"]["outlier_delta_z"])
    reference = SundialDataset(config, split, "generated", training=False)
    positions = stratified_positions(
        reference.objects,
        max_objects,
        int(config["project"]["seed"]),
        config["data"]["redshift_edges"],
    )
    source_ids = [reference._source_keys[position] for position in positions]
    cases: dict[str, dict[str, dict[str, float | int]]] = {
        _exponent_key(value): {} for value in exponent_values
    }

    original_exponent = float(onir.visit_evidence_exponent)
    try:
        for view in ("generated", "no_source"):
            for visits in visit_values:
                reports_by_exponent: dict[float, list[dict[str, float | int]]] = {
                    value: [] for value in exponent_values
                }
                for repeat in range(repeats):
                    dataset = SundialDataset(
                        config,
                        split,
                        view,
                        training=False,
                        visit_selection="random",
                        visit_repeat=repeat,
                    )
                    dataset.max_visits = visits
                    selected_ids = [dataset._source_keys[position] for position in positions]
                    if selected_ids != source_ids:
                        raise RuntimeError("Evidence-growth cases use different objects")
                    loader = inference_loader(
                        Subset(dataset, positions),
                        config,
                        num_workers=0,
                    )
                    for exponent in exponent_values:
                        onir.visit_evidence_exponent = exponent
                        predictions = _predict(model, loader, device)
                        reports_by_exponent[exponent].append(
                            _case_metrics(predictions, view, threshold)
                        )
                for exponent in exponent_values:
                    cases[_exponent_key(exponent)][f"{view}:{visits}"] = (
                        _mean_reports(reports_by_exponent[exponent])
                    )
    finally:
        onir.visit_evidence_exponent = original_exponent

    assessment = _assess_sweep(cases, exponent_values, visit_values)
    report = {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "device": str(device),
        "split": split,
        "objects": len(positions),
        "repeats": int(repeats),
        "exponents": list(exponent_values),
        "visit_counts": list(visit_values),
        "cases": cases,
        "assessment": assessment,
        "notes": [
            "The checkpoint is unchanged; the ONIR visit exponent is varied only at inference.",
            "The exponent is selected on the training selection split; calibration is untouched.",
            "Every case uses the same class-redshift stratified objects.",
            "The exponent changes ONIR scale relative to the shape and temporal routes.",
            "A candidate must be retrained before it can replace the baseline.",
        ],
    }
    path = output_dir / "evidence_growth_summary.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    _print_report(report, path)
    return report


def _case_metrics(
    predictions: pd.DataFrame,
    view: str,
    outlier_delta_z: float,
) -> dict[str, float | int]:
    if view == "generated":
        report = source_metrics(predictions, outlier_delta_z)
        return {
            "N": int(report["N"]),
            "Ia_N": int(report["Ia_N"]),
            "Ia_f1": float(report["Ia_f1"]),
            "Ia_median_absolute_delta_z": float(
                report["Ia_median_absolute_delta_z"]
            ),
            "Ia_posterior_68_interval_coverage": float(
                report["Ia_posterior_68_interval_coverage"]
            ),
            "mean_evidence_sufficiency": float(
                report["mean_evidence_sufficiency"]
            ),
        }
    delta = np.abs(
        predictions["predicted_redshift"] - predictions["true_redshift"]
    )
    return {
        "N": int(len(predictions)),
        "median_absolute_delta_z_to_simulation": float(np.median(delta)),
        "fraction_within_delta_z_0_1_of_simulation": float((delta <= 0.1).mean()),
        "mean_evidence_sufficiency": float(
            predictions["evidence_score"].mean()
        ),
        "mean_redshift_information_gain_nats": float(
            predictions["redshift_information_gain_nats"].mean()
        ),
    }


def _mean_reports(
    reports: list[dict[str, float | int]],
) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for name in reports[0]:
        values = [report[name] for report in reports]
        result[name] = int(values[0]) if name in {"N", "Ia_N"} else float(np.mean(values))
    return result


def _assess_sweep(
    cases: dict[str, dict[str, dict[str, float | int]]],
    exponents: tuple[float, ...],
    visits: tuple[int, ...],
) -> dict[str, Any]:
    first_visit = visits[0]
    last_visit = visits[-1]
    rows = []
    for exponent in exponents:
        key = _exponent_key(exponent)
        blank_first = cases[key][f"no_source:{first_visit}"]
        blank_last = cases[key][f"no_source:{last_visit}"]
        blank_cases = [cases[key][f"no_source:{count}"] for count in visits]
        source_cases = [cases[key][f"generated:{count}"] for count in visits]
        lock_growth = float(
            blank_last["fraction_within_delta_z_0_1_of_simulation"]
            - blank_first["fraction_within_delta_z_0_1_of_simulation"]
        )
        information_growth = float(
            blank_last["mean_redshift_information_gain_nats"]
            - blank_first["mean_redshift_information_gain_nats"]
        )
        maximum_blank_sufficiency = max(
            float(case["mean_evidence_sufficiency"]) for case in blank_cases
        )
        coverage_error = float(
            np.mean(
                [
                    abs(float(case["Ia_posterior_68_interval_coverage"]) - 0.68)
                    for case in source_cases
                ]
            )
        )
        blank_passed = (
            lock_growth <= 0.03
            and information_growth <= 0.02
            and maximum_blank_sufficiency <= 0.05
        )
        rows.append(
            {
                "exponent": exponent,
                "blank_target_lock_growth": lock_growth,
                "blank_information_gain_growth_nats": information_growth,
                "maximum_blank_sufficiency": maximum_blank_sufficiency,
                "Ia_coverage_mean_absolute_error": coverage_error,
                "blank_gate_passed": blank_passed,
            }
        )
    eligible = [row for row in rows if row["blank_gate_passed"]]
    best = min(
        eligible,
        key=lambda row: (row["Ia_coverage_mean_absolute_error"], -row["exponent"]),
        default=None,
    )
    return {
        "rows": rows,
        "coverage_target": 0.68,
        "blank_gate_tolerances": {
            "maximum_target_lock_growth": 0.03,
            "maximum_information_gain_growth_nats": 0.02,
            "maximum_mean_sufficiency": 0.05,
        },
        "best_safe_exponent_by_coverage": (
            None if best is None else float(best["exponent"])
        ),
    }


def _validated_exponents(values: Iterable[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(value < 0.0 or value > 1.0 for value in result):
        raise ValueError("exponents must contain values in [0, 1]")
    if len(set(result)) != len(result):
        raise ValueError("exponents must be unique")
    return result


def _validated_visit_counts(values: Iterable[int]) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or any(value < 1 for value in result):
        raise ValueError("visit_counts must contain positive integers")
    if tuple(sorted(set(result))) != result:
        raise ValueError("visit_counts must be unique and increasing")
    return result


def _exponent_key(exponent: float) -> str:
    return f"alpha={exponent:g}"


def _print_report(report: dict[str, Any], path: Path) -> None:
    visits = report["visit_counts"]
    print("\nVisit evidence growth", flush=True)
    print(
        f"  checkpoint epoch {report['checkpoint_epoch']} | "
        f"{report['objects']:,} selection objects | visits {visits}",
        flush=True,
    )
    print(
        "  alpha | blank lock 1->last | blank info 1->last | "
        "blank evidence | Ia coverage error | gate",
        flush=True,
    )
    for row in report["assessment"]["rows"]:
        status = "pass" if row["blank_gate_passed"] else "hold"
        print(
            f"  {row['exponent']:>5.2f} | "
            f"{row['blank_target_lock_growth']:>+18.3f} | "
            f"{row['blank_information_gain_growth_nats']:>+19.3f} | "
            f"{row['maximum_blank_sufficiency']:>10.3f} | "
            f"{row['Ia_coverage_mean_absolute_error']:>17.3f} | {status}",
            flush=True,
        )
    best = report["assessment"]["best_safe_exponent_by_coverage"]
    label = "none" if best is None else f"{best:g}"
    print(f"  best safe sensitivity value: {label}", flush=True)
    print(f"  results {path}", flush=True)
