import json
from pathlib import Path

from strider.training.plot_history import plot_training_history


def test_training_plot_shows_strider_diagnostics(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    history = []
    for epoch in range(1, 4):
        history.append(
            {
                "epoch": epoch,
                "train": {"loss": 5.0 / epoch},
                "validation": {
                    "joint_loss": 4.0 / epoch,
                    "evidence_sufficiency_loss": 0.5 / epoch,
                    "no_source_redshift_loss": 0.2 / epoch,
                    "no_source_class_loss": 0.1 / epoch,
                    "ia_precision": 0.6 + 0.1 * epoch,
                    "ia_recall": 0.5 + 0.1 * epoch,
                    "ia_f1": 0.55 + 0.1 * epoch,
                    "ia_median_absolute_delta_z": 0.4 / epoch,
                    "source_mean_evidence_sufficiency": 0.5 + 0.1 * epoch,
                    "no_source_mean_evidence_sufficiency": 0.3 - 0.05 * epoch,
                },
                "selection_score": 4.5 / epoch,
                "science_score": [0.60, 0.75, 0.70][epoch - 1],
                "redshift_outlier_score": [0.30, 0.20, 0.25][epoch - 1],
                "learning_rate": 1e-3 / epoch,
                "learned_scales": {"shape": 0.4, "temporal": 0.1 * epoch},
            }
        )
    (output / "training_history.json").write_text(
        json.dumps(history), encoding="utf-8"
    )
    config = {
        "_project_root": str(tmp_path),
        "project": {"name": "test_run", "output_dir": str(output)},
    }

    report = plot_training_history(config)

    assert report["epochs"] == 3
    assert report["best_epoch"] == 3
    assert report["checkpoint_epochs"] == {
        "posterior": 3,
        "science": 2,
        "redshift": 2,
    }
    assert Path(report["figure"]).is_file()
