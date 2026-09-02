"""Paired source and no-source evaluation on identical objects and visit dates."""

from __future__ import annotations

import copy
import json
import math
from typing import Any

import numpy as np

from strider.config import project_path
from strider.data.dataset import SundialDataset

from .checkpoint import load_trained_model
from .evaluate import _predict
from .loader import inference_loader


def run_paired_controls(
    config: dict[str, Any],
    max_objects: int | None = None,
) -> dict[str, Any]:
    output_dir = project_path(config, config["project"]["output_dir"])
    model, _, device = load_trained_model(config)
    split = str(config["evaluation"].get("split", "calibration"))
    evaluation_config = _config_with_object_limit(config, split, max_objects)
    predictions = {}
    for view in ("generated", "no_source"):
        dataset = SundialDataset(
            evaluation_config,
            split,
            view,
            training=False,
            pair_no_source=False,
        )
        loader = inference_loader(dataset, evaluation_config)
        predictions[view] = _predict(model, loader, device)
    paired = predictions["generated"].merge(
        predictions["no_source"],
        on=["snid", "true_redshift", "true_class"],
        suffixes=("_source", "_no_source"),
        validate="one_to_one",
    )
    threshold = float(config["evaluation"]["outlier_delta_z"])
    source_delta = np.abs(paired["predicted_redshift_source"] - paired["true_redshift"])
    no_source_delta = np.abs(
        paired["predicted_redshift_no_source"] - paired["true_redshift"]
    )
    source_success = np.asarray(source_delta <= threshold)
    no_source_success = np.asarray(no_source_delta <= threshold)
    source_only = int(np.sum(source_success & ~no_source_success))
    no_source_only = int(np.sum(~source_success & no_source_success))
    differences = source_success.astype(float) - no_source_success.astype(float)
    rng = np.random.default_rng(int(config["project"]["seed"]))
    bootstrap = np.asarray(
        [
            differences[rng.integers(0, len(differences), len(differences))].mean()
            for _ in range(5000)
        ]
    )
    report = {
        "device": str(device),
        "N": int(len(paired)),
        "requested_max_objects": max_objects,
        "delta_z_threshold": threshold,
        "source_fraction_within_threshold": float(source_success.mean()),
        "no_source_fraction_within_threshold": float(no_source_success.mean()),
        "paired_difference": float(differences.mean()),
        "paired_difference_bootstrap_95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "source_only_successes": source_only,
        "no_source_only_successes": no_source_only,
        "paired_exact_p_value": _two_sided_binomial_p(source_only, no_source_only),
        "source_median_absolute_delta_z": float(np.median(source_delta)),
        "no_source_median_absolute_delta_z": float(np.median(no_source_delta)),
        "source_mean_redshift_information_gain_nats": float(
            paired["redshift_information_gain_nats_source"].mean()
        ),
        "no_source_mean_redshift_information_gain_nats": float(
            paired["redshift_information_gain_nats_no_source"].mean()
        ),
        "source_mean_68_interval_width": float(
            paired["redshift_68_interval_width_source"].mean()
        ),
        "no_source_mean_68_interval_width": float(
            paired["redshift_68_interval_width_no_source"].mean()
        ),
        "source_mean_largest_joint_probability": float(
            paired["largest_joint_probability_source"].mean()
        ),
        "no_source_mean_largest_joint_probability": float(
            paired["largest_joint_probability_no_source"].mean()
        ),
        "source_mean_evidence_sufficiency": float(
            paired["evidence_score_source"].mean()
        ),
        "no_source_mean_evidence_sufficiency": float(
            paired["evidence_score_no_source"].mean()
        ),
    }
    filename = (
        "paired_control_summary.json"
        if max_objects is None
        else f"paired_control_summary_{max_objects}.json"
    )
    with (output_dir / filename).open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def _config_with_object_limit(
    config: dict[str, Any],
    split: str,
    max_objects: int | None,
) -> dict[str, Any]:
    if max_objects is None:
        return config
    if max_objects <= 0:
        raise ValueError("max_objects must be positive")
    evaluation_config = copy.deepcopy(config)
    limits = dict(evaluation_config["data"].get("runtime_object_limits", {}))
    limits[split] = int(max_objects)
    evaluation_config["data"]["runtime_object_limits"] = limits
    return evaluation_config


def _two_sided_binomial_p(first: int, second: int) -> float:
    discordant = first + second
    if discordant == 0:
        return 1.0
    tail = min(first, second)
    probability = sum(math.comb(discordant, value) for value in range(tail + 1)) / 2**discordant
    return float(min(1.0, 2.0 * probability))
