from pathlib import Path

from strider.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_uncertainty_eight_visit_test_changes_only_the_sequence_length() -> None:
    baseline = load_config(ROOT / "configs/nersc/uncertainty_test.yaml")
    eight_visit = load_config(ROOT / "configs/nersc/uncertainty_8_test.yaml")

    assert baseline["reference"]["sequence_visits"] == 6
    assert eight_visit["reference"]["sequence_visits"] == 8
    assert eight_visit["project"]["name"] == "uncertainty_8_test"
    assert eight_visit["project"]["output_dir"].endswith("/uncertainty_8_test")
    for section in ("data", "model", "training", "observation"):
        assert eight_visit[section] == baseline[section]
    assert {
        key: value
        for key, value in eight_visit["reference"].items()
        if key != "sequence_visits"
    } == {
        key: value
        for key, value in baseline["reference"].items()
        if key != "sequence_visits"
    }
