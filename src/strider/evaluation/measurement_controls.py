"""Ablations for the optional wavelength-dependent FLAMERR input."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import pandas as pd
import torch

from strider.config import project_path
from strider.data.dataset import SundialDataset

from .checkpoint import load_trained_model
from .evaluate import _predict
from .loader import inference_loader
from .metrics import source_metrics


CONTROLS = ("normal", "flux_only", "error_only", "shuffled_error")


def run_measurement_controls(
    config: dict[str, Any],
    *,
    split: str | None = None,
    view: str = "original",
) -> dict[str, Any]:
    """Measure whether FLAMERR helps or carries a shortcut by itself."""
    if not bool(config["data"].get("include_flux_error_channel", False)):
        raise ValueError("Measurement controls require include_flux_error_channel")
    if not bool(config["model"].get("use_flux_error_channel", False)):
        raise ValueError("Measurement controls require use_flux_error_channel")

    evaluation_split = split or str(config["evaluation"].get("split", "calibration"))
    output_dir = project_path(config, config["project"]["output_dir"])
    model, checkpoint, device = load_trained_model(config)
    threshold = float(config["evaluation"]["outlier_delta_z"])
    predictions: dict[str, pd.DataFrame] = {}
    reports: dict[str, Any] = {}

    for control in CONTROLS:
        dataset = SundialDataset(
            config,
            evaluation_split,
            view,
            training=False,
            pair_no_source=False,
        )
        loader = inference_loader(dataset, config)
        controlled_loader = _controlled_batches(loader, control)
        result = _predict(model, controlled_loader, device)
        result.to_parquet(
            output_dir
            / f"measurement_control_predictions_{evaluation_split}_{view}_{control}.parquet",
            index=False,
        )
        predictions[control] = result
        reports[control] = source_metrics(result, threshold)

    normal = predictions["normal"]
    report = {
        "device": str(device),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "split": evaluation_split,
        "view": view,
        "controls": reports,
        "change_from_normal": {
            name: _change_from_normal(normal, result)
            for name, result in predictions.items()
            if name != "normal"
        },
        "learned_flux_error_scale": float(
            torch.tanh(model.flux_error_scale_raw).detach().cpu()
        ),
    }
    path = output_dir / f"measurement_control_summary_{evaluation_split}_{view}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    report["summary_path"] = str(path)
    return report


def _controlled_batches(
    loader: Iterable[dict[str, torch.Tensor]], control: str
) -> Iterator[dict[str, torch.Tensor]]:
    if control not in CONTROLS:
        raise ValueError(f"Unsupported measurement control: {control}")
    for batch in loader:
        yield apply_measurement_control(batch, control)


def apply_measurement_control(
    batch: dict[str, torch.Tensor], control: str
) -> dict[str, torch.Tensor]:
    """Return a copy with one measurement channel ablated or disrupted."""
    if "flux_error_shape" not in batch:
        raise KeyError("Measurement controls need flux_error_shape")
    result = dict(batch)
    if control == "normal":
        return result
    if control == "flux_only":
        result["flux_error_shape"] = torch.zeros_like(batch["flux_error_shape"])
        return result
    if control == "error_only":
        result["flux"] = torch.zeros_like(batch["flux"])
        return result
    if control != "shuffled_error":
        raise ValueError(f"Unsupported measurement control: {control}")

    shuffled = batch["flux_error_shape"].clone()
    mask = batch["wavelength_mask"] > 0
    for object_index, snid in enumerate(batch["snid"].tolist()):
        for visit_index in range(shuffled.shape[1]):
            valid = torch.where(mask[object_index, visit_index])[0]
            if len(valid) < 2:
                continue
            # Rotate only valid wavelength cells. This preserves each visit's
            # FLAMERR distribution while breaking alignment with the spectrum.
            offset = 1 + (abs(int(snid)) % (len(valid) - 1))
            values = shuffled[object_index, visit_index, valid]
            shuffled[object_index, visit_index, valid] = torch.roll(
                values, shifts=offset
            )
    result["flux_error_shape"] = shuffled
    return result


def _change_from_normal(
    normal: pd.DataFrame, controlled: pd.DataFrame
) -> dict[str, float]:
    joined = normal[["snid", "predicted_class", "predicted_redshift"]].merge(
        controlled[["snid", "predicted_class", "predicted_redshift"]],
        on="snid",
        suffixes=("_normal", "_controlled"),
        validate="one_to_one",
    )
    redshift_change = np.abs(
        joined["predicted_redshift_controlled"]
        - joined["predicted_redshift_normal"]
    )
    return {
        "N": int(len(joined)),
        "same_predicted_class_fraction": float(
            np.mean(
                joined["predicted_class_controlled"]
                == joined["predicted_class_normal"]
            )
        ),
        "median_absolute_redshift_change": float(np.median(redshift_change)),
        "fraction_redshift_change_above_0_1": float(
            np.mean(redshift_change > 0.1)
        ),
    }
