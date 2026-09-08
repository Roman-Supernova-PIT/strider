"""Measure how inference changes as more visits become available."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from torch.utils.data import Subset

from strider.config import project_path
from strider.data.dataset import SundialDataset

from .checkpoint import load_trained_model
from .evaluate import _predict
from .loader import inference_loader
from .metrics import source_metrics
from .subset import stratified_positions


def run_visit_controls(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = project_path(config, config["project"]["output_dir"])
    model, _, device = load_trained_model(config)
    repeats = int(config["evaluation"].get("visit_control_repeats", 3))
    visit_counts = tuple(
        None if isinstance(value, str) and value.lower() == "all" else int(value)
        for value in config["evaluation"].get(
            "visit_control_counts", [1, 2, 4, 8, 16, 24, 32]
        )
    )
    finite_visit_counts = [value for value in visit_counts if value is not None]
    if not visit_counts or any(value < 1 for value in finite_visit_counts):
        raise ValueError(
            "evaluation.visit_control_counts must contain positive integers or 'all'"
        )
    split = str(config["evaluation"].get("split", "calibration"))
    max_objects = int(config["evaluation"].get("visit_control_max_objects", 1600))
    visit_selection = str(
        config["evaluation"].get("visit_control_selection", "random")
    )
    if visit_selection not in {"span", "random"}:
        raise ValueError("evaluation.visit_control_selection must be span or random")
    reference = SundialDataset(config, split, "generated", training=False)
    positions = stratified_positions(
        reference.objects,
        max_objects,
        int(config["project"]["seed"]),
        config["data"]["redshift_edges"],
    )
    source_ids = [reference._source_keys[position] for position in positions]
    report: dict[str, Any] = {
        "device": str(device),
        "repeats": repeats,
        "visit_counts": ["all" if value is None else value for value in visit_counts],
        "objects": len(positions),
        "visit_selection": visit_selection,
        "cases": {},
    }
    for view in ("generated", "original", "no_source"):
        for visits in visit_counts:
            visit_label = "all" if visits is None else str(visits)
            key = f"{view}:{visit_label}_visits"
            repeat_reports = []
            for repeat in range(repeats):
                dataset = SundialDataset(
                    config,
                    split,
                    view,
                    training=False,
                    visit_selection=visit_selection,
                    visit_repeat=repeat,
                )
                dataset.max_visits = visits
                selected_ids = [dataset._source_keys[position] for position in positions]
                if selected_ids != source_ids:
                    raise RuntimeError("Visit-control cases use different objects")
                loader = inference_loader(
                    Subset(dataset, positions),
                    config,
                    num_workers=0,
                )
                predictions = _predict(model, loader, device)
                if view == "no_source":
                    delta = np.abs(
                        predictions["predicted_redshift"] - predictions["true_redshift"]
                    )
                    one_report = {
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
                else:
                    one_report = source_metrics(
                        predictions, float(config["evaluation"]["outlier_delta_z"])
                    )
                one_report["repeat"] = repeat
                repeat_reports.append(one_report)
            numeric_names = [
                name
                for name, value in repeat_reports[0].items()
                if name not in {"N", "repeat"} and isinstance(value, (int, float))
            ]
            report["cases"][key] = {
                "N_per_repeat": int(repeat_reports[0]["N"]),
                "runs": repeat_reports,
                "mean_across_repeats": {
                    name: float(np.nanmean([run[name] for run in repeat_reports]))
                    for name in numeric_names
                },
            }
    with (output_dir / "visit_control_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report
