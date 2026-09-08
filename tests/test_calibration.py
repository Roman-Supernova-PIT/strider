import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strider.calibration import (
    calibrate_class_probabilities,
    calibrate_joint_probability,
    fit_calibration,
    highest_density_set,
)
from strider.config import resolved_config_sha256
from strider.evaluation.evaluate import _attach_prediction_provenance
from strider.model_package import _calibration_for_package
from strider.model.redshift_scan import build_redshift_grid, redshift_cell_widths


def test_binary_calibration_reweights_joint_without_changing_class_conditionals() -> None:
    joint = np.asarray(
        [
            [0.15, 0.35, 0.10],
            [0.08, 0.12, 0.20],
        ]
    )
    calibration = {
        "method": "binary_affine_logit",
        "positive_class_index": 0,
        "slope": 0.8,
        "intercept": -0.3,
    }

    calibrated = calibrate_joint_probability(joint, calibration)
    expected_class = calibrate_class_probabilities(
        joint.sum(axis=1)[None, :], calibration
    )[0]

    assert calibrated.sum() == pytest.approx(1.0)
    assert calibrated.sum(axis=1) == pytest.approx(expected_class)
    assert calibrated[0] / calibrated[0].sum() == pytest.approx(
        joint[0] / joint[0].sum()
    )
    assert calibrated[1] / calibrated[1].sum() == pytest.approx(
        joint[1] / joint[1].sum()
    )


def test_highest_density_set_can_report_disconnected_redshift_solutions() -> None:
    grid = np.arange(5, dtype=np.float64)
    result = highest_density_set(
        grid,
        np.asarray([0.40, 0.05, 0.10, 0.05, 0.40]),
        rank_mass_quantile=0.81,
    )

    assert result["component_count"] == 2
    assert result["intervals"] == [[0.0, 0.5], [3.5, 4.0]]
    assert result["posterior_mass"] == pytest.approx(0.8)


def test_prediction_provenance_is_embedded_and_conflicts_are_rejected() -> None:
    frame = pd.DataFrame({"snid": [1, 2]})
    result = _attach_prediction_provenance(
        frame,
        split="calibration",
        view="original",
        checkpoint_epoch=6,
        config_sha256="abc",
    )

    assert list(result.columns[:4]) == [
        "data_split",
        "data_view",
        "checkpoint_epoch",
        "config_sha256",
    ]
    with pytest.raises(ValueError, match="conflict"):
        _attach_prediction_provenance(
            result,
            split="test",
            view="original",
            checkpoint_epoch=6,
            config_sha256="abc",
        )


def test_external_prediction_provenance_keeps_model_and_data_ids() -> None:
    result = _attach_prediction_provenance(
        pd.DataFrame({"snid": [1]}),
        split="test",
        view="original",
        checkpoint_epoch=12,
        config_sha256="model-digest",
        data_config_sha256="data-digest",
        dataset_tag="sundial_jun24",
    )

    assert result.loc[0, "config_sha256"] == "model-digest"
    assert result.loc[0, "data_config_sha256"] == "data-digest"
    assert result.loc[0, "dataset_tag"] == "sundial_jun24"


def test_fit_calibration_writes_complete_artifact_and_calibrated_predictions(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    source, blank = _prediction_frames(config, objects=240)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    source_path = run_dir / "calibration_predictions_original.parquet"
    blank_path = run_dir / "calibration_predictions_no_source.parquet"
    source.to_parquet(source_path, index=False)
    blank.to_parquet(blank_path, index=False)

    summary = fit_calibration(
        config,
        source_predictions=source_path,
        blank_predictions=blank_path,
        minimum_stratum_size=10,
    )

    artifact = json.loads((run_dir / "calibration.json").read_text())
    calibrated = pd.read_parquet(summary["calibrated_source_predictions"])
    assert artifact["status"] == "fitted"
    assert artifact["class_calibration"]["method"] == "binary_affine_logit"
    assert artifact["redshift_sets"]["method"].startswith("class-and-visit-aware")
    assert artifact["signal_sufficiency"]["reference_prior"] == {
        "source": 0.5,
        "blank": 0.5,
    }
    assert "calibrated_p_Ia" in calibrated
    assert "redshift_set_90_intervals" in calibrated
    assert "calibrated_source_probability" in calibrated
    assert set(calibrated["signal_grade"]) <= {"high", "medium", "low", "limited"}


def test_fit_calibration_refuses_test_predictions(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source, blank = _prediction_frames(config, objects=40)
    source["data_split"] = "test"
    source_path = tmp_path / "source.parquet"
    blank_path = tmp_path / "blank.parquet"
    source.to_parquet(source_path, index=False)
    blank.to_parquet(blank_path, index=False)

    with pytest.raises(ValueError, match="calibration split"):
        fit_calibration(
            config,
            source_predictions=source_path,
            blank_predictions=blank_path,
            output=tmp_path / "calibration.json",
            minimum_stratum_size=5,
        )


def test_model_package_accepts_only_matching_fitted_calibration(tmp_path: Path) -> None:
    fitted = {
        "format_version": "strider-calibration-v1",
        "status": "fitted",
        "config_sha256": "digest",
        "checkpoint_epoch": 6,
    }
    (tmp_path / "calibration.json").write_text(json.dumps(fitted))

    packaged = _calibration_for_package(
        tmp_path, config_sha256="digest", checkpoint_epoch=6
    )
    assert packaged == {
        **fitted,
        "format_version": "strider-calibration-v1",
    }
    with pytest.raises(ValueError, match="checkpoint"):
        _calibration_for_package(
            tmp_path, config_sha256="digest", checkpoint_epoch=7
        )


def _config(root: Path) -> dict:
    return {
        "project": {"name": "calibration-test", "seed": 7, "output_dir": "run"},
        "model": {
            "classes": ["Ia", "other"],
            "redshift_min": 0.05,
            "redshift_max": 3.0,
            "redshift_bins": 31,
            "redshift_spacing": "log1p",
        },
        "_project_root": str(root),
        "_config_path": str(root / "config.yaml"),
    }


def _prediction_frames(config: dict, objects: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    random = np.random.default_rng(12)
    grid = build_redshift_grid(0.05, 3.0, 31, "log1p").astype(np.float64)
    widths = redshift_cell_widths(grid).astype(np.float64)
    true_class = np.arange(objects) % 2
    true_redshift = grid[(np.arange(objects) * 7) % len(grid)]
    raw_ia = np.where(true_class == 0, 0.70, 0.30)
    raw_ia = np.clip(raw_ia + random.normal(0.0, 0.12, objects), 0.02, 0.98)
    redshift_probability = []
    coordinate = np.log1p(grid)
    for redshift in true_redshift:
        density = np.exp(
            -0.5 * ((coordinate - np.log1p(redshift)) / 0.055) ** 2
        )
        mass = density * widths
        redshift_probability.append((mass / mass.sum()).astype(np.float32).tolist())
    digest = resolved_config_sha256(config)
    common = {
        "snid": np.arange(1000, 1000 + objects),
        "data_split": "calibration",
        "checkpoint_epoch": 6,
        "config_sha256": digest,
    }
    source = pd.DataFrame(
        {
            **common,
            "data_view": "original",
            "true_class_name": np.where(true_class == 0, "Ia", "other"),
            "true_redshift": true_redshift,
            "visit_count": 1 + np.arange(objects) % 24,
            "class_probability_Ia": raw_ia,
            "class_probability_other": 1.0 - raw_ia,
            "redshift_probability": redshift_probability,
            "evidence_score": random.uniform(0.55, 0.98, objects),
        }
    )
    blank = pd.DataFrame(
        {
            **common,
            "data_view": "no_source",
            "evidence_score": random.uniform(0.01, 0.40, objects),
        }
    )
    return source, blank
