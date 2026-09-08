from __future__ import annotations

import numpy as np

from strider.evaluation.metadata_baseline import _metrics


def test_metadata_metrics_report_binary_classification_and_redshift() -> None:
    true_class = np.asarray([0, 0, 1, 1])
    predicted_class = np.asarray([0, 1, 1, 1])
    true_redshift = np.asarray([0.2, 0.8, 1.2, 2.0])
    predicted_redshift = np.asarray([0.2, 0.9, 1.1, 2.0])

    report = _metrics(
        true_class, predicted_class, true_redshift, predicted_redshift
    )

    assert report["Ia_precision"] == 1.0
    assert report["Ia_recall"] == 0.5
    assert 0.0 < report["Ia_low_z_median_absolute_delta_z_over_1_plus_z"] < 0.1
