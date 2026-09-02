import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from strider.atlas.roman_reference import RomanReferenceBank
from strider.config import (
    load_config,
    resolved_config,
    resolved_config_sha256,
)
from strider.data.classes import HOURGLASS_15_CLASSES
from strider.deployment import (
    RESULT_FORMAT,
    load_model_package,
    prepare_observed_series,
)
from strider.model import Strider
from strider.model_package import export_model_package


ROOT = Path(__file__).resolve().parents[1]


def test_observed_series_uses_measurements_and_orders_visits() -> None:
    config = load_config(ROOT / "configs/local_pilot.yaml")
    wavelength = np.geomspace(7400.0, 18500.0, 80)
    flux = np.stack(
        [np.sin(np.linspace(0.0, 4.0, 80)), np.cos(np.linspace(0.0, 4.0, 80))]
    )
    error = np.stack([np.full(80, 4.0), np.full(80, 2.0)])

    prepared = prepare_observed_series(
        config,
        wavelength=wavelength,
        flux=flux,
        flux_error=error,
        observer_time=[60010.0, 60000.0],
    )

    assert prepared.observer_time.tolist() == [60000.0, 60010.0]
    assert prepared.batch["observer_days"].tolist() == [[0.0, 10.0]]
    assert prepared.visit_noise_scale.tolist() == [2.0, 4.0]
    assert prepared.batch["peak_date_valid"].item() == 0.0
    assert prepared.batch["flux"].shape == (1, 2, 256)
    assert torch.isfinite(prepared.batch["flux"]).all()
    assert torch.all(prepared.batch["wavelength_mask"] > 0)


def test_observed_series_requires_reported_uncertainty() -> None:
    config = load_config(ROOT / "configs/local_pilot.yaml")
    with pytest.raises(ValueError, match="fewer than two valid bins"):
        prepare_observed_series(
            config,
            wavelength=[7500.0, 8000.0, 8500.0],
            flux=[1.0, 2.0, 3.0],
            flux_error=[np.nan, np.nan, np.nan],
            observer_time=60000.0,
        )


def test_observed_series_propagates_variance_through_resampling() -> None:
    config = load_config(ROOT / "configs/local_pilot.yaml")
    config["data"]["include_flux_error_channel"] = True
    wavelength = np.geomspace(7000.0, 19500.0, 9)
    flux = np.linspace(-1.0, 2.0, len(wavelength))
    error = np.linspace(0.5, 4.5, len(wavelength))

    prepared = prepare_observed_series(
        config,
        wavelength=wavelength,
        flux=flux,
        flux_error=error,
        observer_time=62000.0,
    )

    output_index = 111
    target = prepared.wavelength_grid_angstrom[output_index]
    upper = int(np.searchsorted(wavelength, target, side="left"))
    lower = upper - 1
    fraction = (target - wavelength[lower]) / (
        wavelength[upper] - wavelength[lower]
    )
    scale = float(np.quantile(error, 0.25))
    expected_error = np.sqrt(
        (1.0 - fraction) ** 2 * error[lower] ** 2
        + fraction**2 * error[upper] ** 2
    ) / scale
    actual_log_error = prepared.batch["flux_error_shape"][0, 0, output_index]

    assert actual_log_error.item() == pytest.approx(np.log(expected_error))
    assert prepared.batch["wavelength_mask"][0, 0, output_index].item() == 1.0


def test_complete_support_uses_native_bin_edges_at_detector_boundary() -> None:
    config = load_config(ROOT / "configs/local_pilot.yaml")
    wavelength = np.linspace(7450.55, 18174.85, 64)
    flux = np.linspace(-2.0, 3.0, len(wavelength))
    error = np.full(len(wavelength), 0.5)

    prepared = prepare_observed_series(
        config,
        wavelength=wavelength,
        flux=flux,
        flux_error=error,
        observer_time=62000.0,
    )

    assert torch.all(prepared.batch["wavelength_mask"] == 1.0)
    assert prepared.batch["flux"][0, 0, -1].item() == pytest.approx(
        flux[-1] / 0.5
    )


def test_exported_package_loads_and_classifies_without_truth(tmp_path: Path) -> None:
    package = _export_test_package(tmp_path)
    deployed = load_model_package(package)
    wavelength = np.geomspace(7400.0, 18500.0, 96)
    coordinate = np.linspace(0.0, 2.0 * np.pi, len(wavelength))
    result = deployed.classify(
        wavelength=wavelength,
        flux=np.stack([np.sin(coordinate), np.sin(coordinate + 0.2)]),
        flux_error=np.full((2, len(wavelength)), 0.8),
        observer_time=[62000.0, 62012.0],
    )
    json.dumps(result, allow_nan=False)

    assert result["format_version"] == RESULT_FORMAT
    assert result["model"]["calibration_status"] == "not_fitted"
    assert result["input"]["observer_days"] == [0.0, 12.0]
    assert result["classification"]["probability_type"] == "raw"
    assert sum(result["classification"]["probabilities"].values()) == pytest.approx(1.0)
    assert 0.05 <= result["redshift"]["z_STRIDER"] <= 3.0
    assert sum(result["redshift"]["probability"]) == pytest.approx(1.0)
    assert result["signal"]["source_probability"] is None


def test_reference_package_is_self_contained_and_uses_its_reference_bank(
    tmp_path: Path,
) -> None:
    package = _export_reference_test_package(tmp_path)

    assert (package / "reference_bank.npz").is_file()
    assert not (package / "onir_bank.npz").exists()
    deployed = load_model_package(package)
    wavelength = np.geomspace(7000.0, 19000.0, 96)
    coordinate = np.linspace(0.0, 4.0 * np.pi, len(wavelength))
    result = deployed.classify(
        wavelength=wavelength,
        flux=np.stack([np.sin(coordinate), np.sin(coordinate + 0.15)]),
        flux_error=np.full((2, len(wavelength)), 0.7),
        observer_time=[62000.0, 62009.0],
    )

    assert result["model"]["architecture"] == "roman_reference"
    assert result["input"]["observer_days"] == [0.0, 9.0]
    assert sum(result["classification"]["probabilities"].values()) == pytest.approx(
        1.0
    )


def test_package_v1_without_descriptive_architecture_fields_still_loads(
    tmp_path: Path,
) -> None:
    package = _export_test_package(tmp_path)
    model_info_path = package / "model_info.json"
    model_info = json.loads(model_info_path.read_text())
    model_info.pop("architecture")
    model_info.pop("model_assets")
    model_info_path.write_text(json.dumps(model_info))

    deployed = load_model_package(package, verify_checksums=False)

    assert deployed.model_info["architecture"] == "full_scan"
    assert deployed.model_info["model_assets"] == []


def test_package_checksum_failure_is_reported(tmp_path: Path) -> None:
    package = _export_test_package(tmp_path)
    info_path = package / "model_info.json"
    info = json.loads(info_path.read_text())
    info["model_name"] = "tampered"
    info_path.write_text(json.dumps(info))

    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_model_package(package)


def test_reexport_is_atomic_and_packages_only_matching_test_metrics(
    tmp_path: Path,
) -> None:
    package = _export_test_package(tmp_path)
    resolved_path = tmp_path / "run" / "config.resolved.yaml"
    config = yaml.safe_load(resolved_path.read_text())
    config["_project_root"] = str(tmp_path)
    config["_config_path"] = str(tmp_path / "config.yaml")
    digest = resolved_config_sha256(config)
    (tmp_path / "run" / "evaluation_summary.json").write_text(
        json.dumps({"split": "calibration", "checkpoint_epoch": 2})
    )
    (tmp_path / "run" / "test_evaluation_summary.json").write_text(
        json.dumps(
            {
                "split": "test",
                "checkpoint_epoch": 2,
                "config_sha256": digest,
                "metric": float("nan"),
            }
        )
    )
    (tmp_path / "run" / "calibration.json").write_text(
        json.dumps(
            {
                "status": "fitted",
                "format_version": "strider-calibration-v1",
                "config_sha256": digest,
                "checkpoint_epoch": 2,
                "classes": ["Ia", "other"],
                "class_calibration": {
                    "method": "binary_affine_logit",
                    "positive_class_index": 0,
                    "slope": 1.0,
                    "intercept": 0.0,
                },
                "signal_sufficiency": {
                    "slope": 1.0,
                    "intercept": 0.0,
                    "grade_thresholds": {"high": 0.9, "medium": 0.5, "low": 0.1},
                },
                "redshift_sets": {"levels": []},
            }
        )
    )

    report = export_model_package(config, replace=True)

    assert Path(report["package_dir"]) == package
    assert Path(report["replaced_package_backup"]).is_dir()
    metrics = json.loads((package / "metrics.json").read_text())
    assert metrics["split"] == "test"
    assert metrics["metric"] is None
    assert report["metrics_status"] == "frozen_test"
    calibration = json.loads((package / "calibration.json").read_text())
    assert calibration["format_version"] == "strider-calibration-v1"
    assert report["calibration_status"] == "fitted"
    load_model_package(package)


def _export_test_package(root: Path) -> Path:
    config = load_config(ROOT / "configs/local_pilot.yaml")
    config["_project_root"] = str(root)
    config["_config_path"] = str(root / "config.yaml")
    config["project"]["output_dir"] = "run"
    run_dir = root / "run"
    run_dir.mkdir()
    model = Strider(config)
    checkpoint = {
        "epoch": 2,
        "config_sha256": resolved_config_sha256(config),
        "model_state": model.state_dict(),
        "redshift_grid": model.redshift_grid.detach().cpu(),
    }
    torch.save(checkpoint, run_dir / "best_model.pt")
    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(resolved_config(config), sort_keys=False)
    )
    (run_dir / "environment.json").write_text(json.dumps({"python": "test"}))

    report = export_model_package(config)
    return Path(report["package_dir"])


def _export_reference_test_package(root: Path) -> Path:
    config = load_config(ROOT / "configs/experiments/encoded_onir_named_clean.yaml")
    config["_project_root"] = str(root)
    config["_config_path"] = str(root / "config.yaml")
    config["project"]["output_dir"] = "reference_run"
    config["data"].update(
        {
            "class_scheme": "normal_ia_binary",
            "include_flux_error_channel": True,
            "max_visits": 3,
        }
    )
    config["model"].update(
        {
            "architecture": "roman_reference",
            "classes": ["Ia", "other"],
            "dense_rest_frame_scan": False,
            "dense_continuum_detail": False,
            "full_spectrum_context": False,
            "temporal_mode": "none",
            "phase_auxiliary_bins": 0,
            "candidate_phase_consistency": False,
            "use_flux_error_channel": False,
            "coadd_weighting": "reported_error",
            "coadd_maximum_relative_error": None,
            "coadd_edge_trim_fraction": 0.0,
            "redshift_bins": 31,
        }
    )
    bank_path = _write_reference_test_bank(root / "reference.npz")
    config["reference"] = {
        "bank_path": str(bank_path),
        "continuum_width_km_s": 12_000.0,
        "minimum_profile_support": 5,
        "minimum_rest_fraction": 0.10,
        "minimum_shared_fraction": 0.50,
        "prototype_temperature": 0.08,
        "fine_class_temperature": 0.10,
        "phase_temperature": 0.10,
        "initial_continuum_removed_fraction": 0.60,
        "initial_coadd_scale": 0.75,
        "initial_sequence_scale": 0.20,
        "evidence_scale": 10.0,
        "redshift_chunk_size": 7,
        "sequence_visits": 2,
        "minimum_sequence_visits": 2,
        "spectral_encoder": "direct",
        "sequence_combination": "mean",
        "maximum_relative_coadd_error": None,
        "edge_trim_fraction": 0.0,
        "edge_taper_fraction": 0.05,
        "spectral_uncertainty_weighting": "inverse_variance",
        "minimum_relative_spectral_precision": float(torch.finfo(torch.float32).eps),
    }
    run_dir = root / "reference_run"
    run_dir.mkdir()
    model = Strider(config)
    checkpoint = {
        "epoch": 2,
        "config_sha256": resolved_config_sha256(config),
        "model_state": model.state_dict(),
        "redshift_grid": model.redshift_grid.detach().cpu(),
    }
    torch.save(checkpoint, run_dir / "best_model.pt")
    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(resolved_config(config), sort_keys=False)
    )
    (run_dir / "environment.json").write_text(json.dumps({"python": "test"}))

    report = export_model_package(config)
    return Path(report["package_dir"])


def _write_reference_test_bank(path: Path) -> Path:
    rest_bins = 64
    class_count = len(HOURGLASS_15_CLASSES)
    rest = np.geomspace(2500.0, 10000.0, rest_bins).astype(np.float32)
    coordinate = np.linspace(0.0, 5.0 * np.pi, rest_bins, dtype=np.float32)
    coadd = np.stack(
        [np.sin(coordinate + 0.13 * index) for index in range(class_count)]
    )[:, None, :].astype(np.float32)
    phase = np.stack(
        [
            np.stack(
                [
                    np.sin(coordinate + 0.13 * class_index + 0.17 * phase_index)
                    for phase_index in range(2)
                ]
            )
            for class_index in range(class_count)
        ]
    )[:, :, None, :].astype(np.float32)
    bank = RomanReferenceBank(
        class_names=HOURGLASS_15_CLASSES,
        rest_wavelength=rest,
        phase_edges_days=np.asarray([-20.0, 20.0, 80.0], dtype=np.float32),
        coadd_full_profiles=coadd,
        coadd_continuum_removed_profiles=coadd - coadd.mean(axis=-1, keepdims=True),
        coadd_profile_masks=np.ones_like(coadd, dtype=bool),
        coadd_support_counts=np.full(coadd.shape[:-1], 20, dtype=np.int64),
        phase_full_profiles=phase,
        phase_continuum_removed_profiles=phase - phase.mean(axis=-1, keepdims=True),
        phase_profile_masks=np.ones_like(phase, dtype=bool),
        phase_support_counts=np.full(phase.shape[:-1], 20, dtype=np.int64),
        metadata={"source_split": "train", "truth_used_at_runtime": False},
    )
    return bank.save(path)
