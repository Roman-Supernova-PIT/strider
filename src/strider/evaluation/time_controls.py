"""Measure what observer-time information contributes to source and no-source inputs."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import pandas as pd
import torch

from strider.config import project_path
from strider.data.dataset import SundialDataset

from .checkpoint import load_trained_model
from .controls import blank_redshift_metrics
from .evaluate import _predict
from .loader import inference_loader
from .metrics import source_metrics


def run_time_controls(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = project_path(config, config["project"]["output_dir"])
    model, _, device = load_trained_model(config)
    split = str(config["evaluation"].get("split", "calibration"))
    report: dict[str, Any] = {"device": str(device), "cases": {}}
    for view in ("generated", "clean", "no_source"):
        dataset = SundialDataset(config, split, view, training=False)
        loader = inference_loader(dataset, config, num_workers=0)
        time_reassignment = _dataset_time_reassignment(dataset)
        report.setdefault("reassigned_objects", {})[view] = len(time_reassignment)
        spectral_predictions = _predict(
            model,
            loader,
            device,
            logit_source="spectral",
        )
        report["cases"][f"{view}:spectral_only"] = _summarize(
            spectral_predictions,
            view,
            config,
        )
        for transform in (
            "none",
            "zero",
            "reverse_within_object",
            "reassigned_within_redshift",
        ):
            predictions = _predict(
                model,
                loader,
                device,
                time_transform=transform,
                time_reassignment=time_reassignment,
            )
            report["cases"][f"{view}:{transform}"] = _summarize(
                predictions,
                view,
                config,
            )
    with (output_dir / "time_control_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def _dataset_time_reassignment(
    dataset: SundialDataset,
) -> dict[int, torch.Tensor]:
    """Exchange date patterns among matched objects across the full split."""
    groups: dict[tuple[int, int], list[tuple[int, torch.Tensor]]] = defaultdict(list)
    for index in range(len(dataset)):
        item = dataset[index]
        redshift_group = int(torch.floor(item["redshift"] / 0.25).item())
        observer_days = item["observer_days"].clone()
        key = (redshift_group, len(observer_days))
        groups[key].append((int(item["snid"]), observer_days))

    reassignment: dict[int, torch.Tensor] = {}
    for matched in groups.values():
        if len(matched) < 2:
            continue
        source_times = [entry[1] for entry in matched[-1:] + matched[:-1]]
        for (snid, target_times), replacement in zip(matched, source_times):
            shifted = replacement - replacement[:1] + target_times[:1]
            reassignment[snid] = shifted
    return reassignment


def _summarize(
    predictions: pd.DataFrame,
    view: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    if view in {"generated", "clean"}:
        return source_metrics(
            predictions, float(config["evaluation"]["outlier_delta_z"])
        )
    blank = blank_redshift_metrics(
        predictions,
        int(config["project"]["seed"]),
    )
    return {
        **blank,
        "mean_evidence_sufficiency": float(
            predictions["evidence_score"].mean()
        ),
    }
