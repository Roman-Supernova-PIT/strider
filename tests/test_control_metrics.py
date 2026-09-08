from __future__ import annotations

import numpy as np
import pandas as pd

from strider.evaluation.controls import blank_redshift_metrics, sufficiency_auc


def test_blank_metrics_find_a_truth_locked_prediction() -> None:
    truth = np.linspace(0.1, 2.9, 100)
    predictions = pd.DataFrame(
        {"true_redshift": truth, "predicted_redshift": truth + 0.01}
    )

    report = blank_redshift_metrics(predictions, seed=3)

    assert report["blank_redshift_lock"] == 1.0
    assert report["blank_redshift_truth_correlation"] > 0.99
    assert report["blank_lock_z_score"] > 2.0


def test_sufficiency_auc_has_the_expected_direction() -> None:
    source = pd.DataFrame({"evidence_score": [0.8, 0.9, 1.0]})
    blank = pd.DataFrame({"evidence_score": [0.0, 0.1, 0.2]})

    assert sufficiency_auc(source, blank) == 1.0
