import numpy as np
import pandas as pd
import torch

from strider.evaluation.evaluate import _class_probability_at_redshift, _ia_metric_rows
from strider.evaluation.metrics import metrics_by_redshift, source_metrics


def test_redshift_metrics_use_direct_delta_z_and_keep_the_full_scatter() -> None:
    predictions = pd.DataFrame(
        {
            "predicted_redshift": [0.9, 1.5, 2.4, 2.8],
            "true_redshift": [1.0, 1.5, 2.0, 3.0],
            "true_class": [0, 0, 1, 1],
            "true_class_name": ["Ia", "Ia", "IIP", "IIP"],
            "predicted_class": [0, 1, 1, 1],
            "redshift_lower_68": [0.8, 1.4, 2.2, 2.7],
            "redshift_upper_68": [1.1, 1.6, 2.5, 2.9],
            "evidence_score": [0.9, 0.8, 0.7, 0.6],
        }
    )

    report = source_metrics(predictions, outlier_delta_z=0.10)
    delta_z = np.asarray([-0.1, 0.0, 0.4, -0.2])

    assert np.isclose(report["median_delta_z"], np.median(delta_z))
    assert np.isclose(report["delta_z_16th_percentile"], np.quantile(delta_z, 0.16))
    assert np.isclose(report["delta_z_84th_percentile"], np.quantile(delta_z, 0.84))
    assert np.isclose(report["population_scatter_delta_z"], np.std(delta_z, ddof=1))
    assert np.isclose(report["median_absolute_delta_z"], np.median(np.abs(delta_z)))
    assert report["outlier_fraction_abs_delta_z_gt_0_05"] == 0.75
    assert report["outlier_fraction_abs_delta_z_gt_0_10"] == 0.5
    assert np.isclose(report["class_macro_f1_present"], (2.0 / 3.0 + 0.8) / 2.0)
    assert np.isclose(report["class_balanced_accuracy_present"], 0.75)
    assert np.isclose(report["Ia_f1"], 2.0 / 3.0)
    assert np.isclose(report["Ia_median_delta_z"], -0.05)
    assert np.isclose(
        report["Ia_population_scatter_delta_z"],
        np.std(np.asarray([-0.1, 0.0]), ddof=1),
    )
    by_class = {row["class_name"]: row for row in report["metrics_by_class"]}
    assert by_class["Ia"]["N"] == 2
    assert np.isclose(by_class["Ia"]["median_delta_z"], -0.05)
    assert np.isclose(by_class["IIP"]["median_absolute_delta_z"], 0.3)


def test_phase_metrics_are_reported_for_ia_and_each_class() -> None:
    predictions = pd.DataFrame(
        {
            "predicted_redshift": [1.0, 1.2, 0.8],
            "true_redshift": [1.0, 1.1, 0.9],
            "true_class": [0, 0, 1],
            "true_class_name": ["Ia", "Ia", "IIP"],
            "predicted_class": [0, 0, 1],
            "redshift_lower_68": [0.9, 1.0, 0.7],
            "redshift_upper_68": [1.1, 1.3, 1.0],
            "evidence_score": [0.9, 0.8, 0.7],
            "true_class_redshift_phase_median_delta_days": [1.0, -2.0, 3.0],
            "true_class_redshift_phase_median_absolute_error_days": [1.0, 2.0, 3.0],
            "true_class_redshift_phase_68_interval_coverage": [1.0, 0.5, 0.0],
            "true_class_redshift_phase_order_accuracy": [1.0, 0.75, 0.5],
        }
    )

    report = source_metrics(predictions, outlier_delta_z=0.10)

    assert report["Ia_median_phase_absolute_error_days"] == 1.5
    assert report["Ia_mean_phase_order_accuracy"] == 0.875
    by_class = {row["class_name"]: row for row in report["metrics_by_class"]}
    assert by_class["Ia"]["median_phase_absolute_error_days"] == 1.5
    assert by_class["IIP"]["median_phase_absolute_error_days"] == 3.0


def test_ia_csv_rows_keep_the_full_sample_and_redshift_groups() -> None:
    rows = _ia_metric_rows(
        {
            "Ia_N": 20,
            "Ia_precision": 0.8,
            "redshift_groups": [
                {
                    "redshift_range": [1.0, 1.5],
                    "Ia_N": 7,
                    "Ia_precision": 0.75,
                }
            ],
        }
    )

    assert rows[0]["Ia_N"] == 20
    assert np.isnan(rows[0]["redshift_min"])
    assert rows[1]["redshift_min"] == 1.0
    assert rows[1]["redshift_max"] == 1.5
    assert rows[1]["Ia_N"] == 7


def test_ia_redshift_rows_include_non_ia_false_positives() -> None:
    predictions = pd.DataFrame(
        {
            "predicted_redshift": [0.2, 0.3, 0.2, 0.4],
            "true_redshift": [0.2, 0.3, 0.2, 0.4],
            "true_class": [0, 0, 1, 1],
            "true_class_name": ["Ia", "Ia", "IIP", "IIP"],
            "predicted_class": [0, 1, 0, 1],
            "redshift_lower_68": [0.1, 0.2, 0.1, 0.3],
            "redshift_upper_68": [0.3, 0.4, 0.3, 0.5],
            "evidence_score": [0.9, 0.8, 0.7, 0.6],
        }
    )
    report = source_metrics(predictions, outlier_delta_z=0.10)
    report["redshift_groups"] = metrics_by_redshift(
        predictions,
        outlier_delta_z=0.10,
        redshift_edges=[0.0, 0.5],
    )

    row = _ia_metric_rows(report)[1]

    assert row["Ia_N"] == 2
    assert row["Ia_predicted_N"] == 2
    assert row["Ia_precision"] == 0.5
    assert row["Ia_recall"] == 0.5
    assert row["Ia_f1"] == 0.5


def test_class_probability_scores_reward_the_true_class() -> None:
    predictions = pd.DataFrame(
        {
            "predicted_redshift": [1.0, 2.0],
            "true_redshift": [1.0, 2.0],
            "true_class": [0, 1],
            "true_class_name": ["Ia", "IIP"],
            "predicted_class": [0, 1],
            "redshift_lower_68": [0.9, 1.9],
            "redshift_upper_68": [1.1, 2.1],
            "evidence_score": [0.9, 0.8],
            "class_probability_Ia": [0.9, 0.1],
            "class_probability_IIP": [0.1, 0.9],
        }
    )

    report = source_metrics(predictions, outlier_delta_z=0.10)

    assert np.isclose(report["mean_class_log_loss"], -np.log(0.9))
    assert np.isclose(report["mean_class_brier_score"], 0.02)


def test_class_probability_at_true_redshift_interpolates_density() -> None:
    joint = torch.tensor([[[0.10, 0.20, 0.10], [0.05, 0.15, 0.40]]])
    grid = torch.tensor([0.0, 1.0, 3.0])
    width = torch.tensor([1.0, 1.5, 2.0])

    probability = _class_probability_at_redshift(
        joint,
        torch.tensor([2.0]),
        grid,
        width,
    )

    expected = torch.tensor([[0.0916667, 0.15]])
    expected = expected / expected.sum(dim=1, keepdim=True)
    assert torch.allclose(probability, expected, atol=1.0e-5)


def test_candidate_metrics_measure_retained_alternative_solutions() -> None:
    predictions = pd.DataFrame(
        {
            "predicted_redshift": [2.0, 1.0],
            "posterior_median_redshift": [2.0, 1.0],
            "posterior_primary_redshift": [2.0, 1.0],
            "posterior_density_mode_redshift": [2.0, 1.0],
            "posterior_mean_redshift": [2.0, 1.0],
            "true_redshift": [1.0, 1.0],
            "true_class": [0, 0],
            "true_class_name": ["Ia", "Ia"],
            "predicted_class": [0, 0],
            "redshift_lower_68": [1.8, 0.9],
            "redshift_upper_68": [2.2, 1.1],
            "evidence_score": [0.8, 0.9],
            "posterior_candidate_redshifts": [[2.0, 1.02], [1.0]],
            "posterior_candidate_lower_68": [[1.9, 0.98], [0.95]],
            "posterior_candidate_upper_68": [[2.1, 1.06], [1.05]],
            "posterior_candidate_class_names": [["IIP", "Ia"], ["Ia"]],
            "joint_primary_redshift": [2.0, 1.0],
            "joint_primary_predicted_class": [1, 0],
            "joint_candidate_redshifts": [[2.0, 1.02], [1.0]],
            "joint_candidate_class_indices": [[1, 0], [0]],
        }
    )

    report = source_metrics(predictions, outlier_delta_z=0.1)

    assert report["fraction_with_multiple_posterior_candidates"] == 0.5
    assert report["candidate_oracle_outlier_fraction"] == 0.0
    assert report["candidate_68_interval_oracle_coverage"] == 1.0
    assert report["candidate_class_and_redshift_success_fraction"] == 1.0
    assert report["primary_candidate_class_and_redshift_success_fraction"] == 0.5
    assert report["Ia_fraction_with_multiple_posterior_candidates"] == 0.5
    assert report["Ia_candidate_oracle_outlier_fraction"] == 0.0
    assert report["joint_primary_class_accuracy"] == 0.5
    assert report["joint_candidate_correct_class_retained_fraction"] == 1.0
    assert report["joint_candidate_class_and_redshift_success_fraction"] == 1.0
    assert report["joint_primary_class_and_redshift_success_fraction"] == 0.5
    assert report["Ia_joint_candidate_class_and_redshift_success_fraction"] == 1.0
