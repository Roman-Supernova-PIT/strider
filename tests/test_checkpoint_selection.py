import pytest
import torch

from strider.training.trainer import (
    _checkpoint_metric_view,
    _ia_outlier_score,
    _macro_redshift_outlier_score,
    _resume_candidate_metrics,
    _resume_macro_redshift_metrics,
    _save_candidate_checkpoint,
    _science_score,
)


def _view(
    macro_f1: float,
    ia_outliers: float,
    other_outliers: float = 0.5,
    other_count: int = 200,
) -> dict:
    return {
        "z_lt_2_macro_f1_present": macro_f1,
        "z_lt_2_metrics_by_class": [
            {
                "N": 300,
                "outlier_fraction_abs_delta_z_gt_0_1": ia_outliers,
            },
            {
                "N": other_count,
                "outlier_fraction_abs_delta_z_gt_0_1": other_outliers,
            },
        ],
    }


def test_candidate_scores_use_deployment_view() -> None:
    views = {
        "original": _view(0.60, 0.30),
        "generated": _view(0.90, 0.12),
    }
    assert _science_score(views, "original") == pytest.approx(0.60)
    assert _ia_outlier_score(views, "original") == pytest.approx(0.30)
    assert _macro_redshift_outlier_score(views, "original") == pytest.approx(
        0.40
    )


def test_macro_redshift_score_excludes_unstable_rare_classes() -> None:
    views = {
        "original": {
            "z_lt_2_metrics_by_class": [
                {"N": 300, "outlier_fraction_abs_delta_z_gt_0_1": 0.1},
                {"N": 200, "outlier_fraction_abs_delta_z_gt_0_1": 0.3},
                {"N": 17, "outlier_fraction_abs_delta_z_gt_0_1": 1.0},
            ]
        }
    }

    assert _macro_redshift_outlier_score(views, "original") == pytest.approx(
        0.2
    )


def test_checkpoint_metric_view_must_be_validated() -> None:
    weights = {"original": 2.0, "generated": 1.0}

    assert _checkpoint_metric_view({}, weights) == "original"
    assert _checkpoint_metric_view({}, {"generated": 1.0}) == "generated"
    assert (
        _checkpoint_metric_view({"checkpoint_metric_view": "original"}, weights)
        == "original"
    )
    with pytest.raises(ValueError, match="validation_view_weights"):
        _checkpoint_metric_view({"checkpoint_metric_view": "clean"}, weights)


def test_old_resume_history_reconstructs_candidate_leaders() -> None:
    history = [
        {
            "epoch": 1,
            "validation_views": {
                "original": _view(0.70, 0.25),
                "generated": _view(0.76, 0.19),
            },
        },
        {
            "epoch": 2,
            "validation_views": {
                "original": _view(0.74, 0.22),
                "generated": _view(0.77, 0.25),
            },
        },
    ]

    science, science_epoch, redshift, redshift_epoch = _resume_candidate_metrics(
        {}, history, "original"
    )

    assert science == pytest.approx(0.74)
    assert science_epoch == 2
    assert redshift == pytest.approx(0.22)
    assert redshift_epoch == 2


def test_new_resume_fields_override_reconstructed_values() -> None:
    state = {
        "best_science_score": 0.81,
        "best_science_epoch": 4,
        "best_redshift_score": 0.15,
        "best_redshift_epoch": 3,
        "checkpoint_metric_view": "generated",
    }

    result = _resume_candidate_metrics(state, [], "generated")

    assert result == (0.81, 4, 0.15, 3)


def test_macro_redshift_resume_reconstructs_old_history() -> None:
    history = [
        {
            "epoch": 1,
            "validation_views": {
                "original": _view(0.7, 0.2, other_outliers=0.6),
            },
        },
        {
            "epoch": 2,
            "validation_views": {
                "original": _view(0.7, 0.25, other_outliers=0.35),
            },
        },
    ]

    score, epoch = _resume_macro_redshift_metrics({}, history, "original")

    assert score == pytest.approx(0.3)
    assert epoch == 2


def test_old_history_record_can_be_saved_as_candidate(tmp_path) -> None:
    record = {
        "epoch": 3,
        "selection_score": 4.2,
        "validation_views": {
            "original": {"loss": 3.0, **_view(0.74, 0.22)},
            "generated": {"loss": 3.6, **_view(0.77, 0.25)},
        },
    }
    path = tmp_path / "candidate.pt"

    _save_candidate_checkpoint(
        path,
        torch.nn.Linear(2, 2),
        record,
        {"original": 2.0, "generated": 1.0},
        "original",
        "digest",
        ["Ia", "other"],
        torch.tensor([0.0, 0.5]),
        "flat",
        role="science",
    )

    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["science_score"] == pytest.approx(0.74)
    assert saved["redshift_outlier_score"] == pytest.approx(0.22)
    assert saved["macro_redshift_outlier_score"] == pytest.approx(0.36)
