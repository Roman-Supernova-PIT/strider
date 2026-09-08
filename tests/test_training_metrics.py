from __future__ import annotations

import numpy as np
import torch

from strider.training.trainer import (
    _empty_validation_progress,
    _finish_z_lt_2_progress,
    _posterior_median_redshift,
)


def test_epoch_metrics_report_every_class_below_redshift_two() -> None:
    progress = _empty_validation_progress(["Ia", "H-rich CC", "other"])
    progress["z_lt_2_confusion"] = np.asarray(
        [
            [2, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
        ],
        dtype=np.int64,
    )
    progress["z_lt_2_delta_z_by_class"] = [
        [0.01, -0.02, 0.03],
        [0.10, -0.20],
        [0.40],
    ]

    metrics = _finish_z_lt_2_progress(progress)

    assert metrics["z_lt_2_N"] == 6
    assert np.isclose(metrics["ia_z_lt_2_precision"], 1.0)
    assert np.isclose(metrics["ia_z_lt_2_recall"], 2.0 / 3.0)
    assert np.isclose(metrics["ia_z_lt_2_f1"], 0.8)
    assert np.isclose(metrics["ia_z_lt_2_median_absolute_delta_z"], 0.02)
    assert len(metrics["z_lt_2_metrics_by_class"]) == 3
    assert metrics["z_lt_2_metrics_by_class"][1]["class_name"] == "H-rich CC"
    assert np.isclose(
        metrics["z_lt_2_metrics_by_class"][1]["population_scatter_delta_z"],
        np.std([0.10, -0.20], ddof=1),
    )


def test_absent_low_redshift_class_is_reported_as_unavailable() -> None:
    progress = _empty_validation_progress(["Ia", "PISN"])
    progress["z_lt_2_confusion"] = np.asarray([[1, 0], [0, 0]])
    progress["z_lt_2_delta_z_by_class"] = [[0.01], []]

    metrics = _finish_z_lt_2_progress(progress)
    pisn = metrics["z_lt_2_metrics_by_class"][1]

    assert pisn["N"] == 0
    assert np.isnan(pisn["f1"])
    assert np.isnan(pisn["median_absolute_delta_z"])


def test_epoch_redshift_metric_uses_interpolated_posterior_median() -> None:
    grid = torch.tensor([0.0, 1.0, 2.0])
    probability = torch.tensor(
        [
            [0.2, 0.6, 0.2],
            [1.0, 0.0, 0.0],
        ]
    )

    median = _posterior_median_redshift(probability, grid)

    assert torch.allclose(median, torch.tensor([1.0, 0.0]))
