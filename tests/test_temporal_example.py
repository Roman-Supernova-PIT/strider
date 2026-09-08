from pathlib import Path

from strider.evaluation.temporal_example import run_temporal_example


def test_controlled_temporal_example_keeps_identical_visits_neutral(
    tmp_path: Path,
) -> None:
    report = run_temporal_example(
        output=tmp_path / "summary.json",
        epochs=1,
        training_objects=20,
        test_objects=10,
    )

    assert (tmp_path / "summary.json").is_file()
    assert set(report["cases"]) == {
        "correct_dates",
        "shifted_dates",
        "reversed_dates",
        "reassigned_dates",
    }
    assert report["starting_phase_range_days"] == [-25.0, 25.0]
    assert report["training_rest_gap_range_days"] == [4, 18]
    assert report["redshift_bins"] == 15
    assert report["class_names"] == ["class 0", "class 1", "class 2"]
    assert report["identical_visit_max_temporal_logit"] == 0.0
    assert report["absolute_date_shift_max_logit_difference"] < 1.0e-5
    assert set(report["correct_date_branches"]) == {
        "shape_only",
        "temporal_only",
        "combined",
    }
    assert set(report["visit_counts"]) == {"1", "2", "3", "5"}
