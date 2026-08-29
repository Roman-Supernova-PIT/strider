import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from strider import load_model
from strider.engine.config import resolved_config_sha256
from strider.engine.deployment import MODEL_PACKAGE_FORMAT, prepare_observed_series
from strider.engine.model import StriderModel
from strider.engine.model.redshift_scan import redshift_cell_widths
from strider.engine.posterior_summary import posterior_basin_candidates
from strider.engine.wavelength import log_wavelength_grid


def test_load_model_dispatches_model_package_directories(
    tmp_path: Path, monkeypatch
) -> None:
    package = tmp_path / "model-package"
    package.mkdir()
    sentinel = object()
    calls = []

    def fake_load(path: Path, *, device: str):
        calls.append((path, device))
        return sentinel

    monkeypatch.setattr("strider.engine.deployment.load_model_package", fake_load)

    assert load_model(package, device="cpu") is sentinel
    assert calls == [(package, "cpu")]


def test_current_redshift_summary_preserves_two_distinct_basins() -> None:
    grid = np.linspace(0.0, 1.0, 101)
    density = np.exp(-0.5 * ((grid - 0.25) / 0.035) ** 2)
    density += 0.65 * np.exp(-0.5 * ((grid - 0.78) / 0.045) ** 2)
    widths = redshift_cell_widths(grid)
    mass = density * widths
    mass /= mass.sum()

    candidates = posterior_basin_candidates(grid, mass, widths)

    assert len(candidates) == 2
    assert candidates[0]["peak_redshift"] == pytest.approx(0.25, abs=0.01)
    assert candidates[1]["peak_redshift"] == pytest.approx(0.78, abs=0.01)


def test_public_model_package_loads_and_runs_end_to_end(tmp_path: Path) -> None:
    package, config = _model_package(tmp_path)
    deployed = load_model(package)
    wavelength = np.linspace(7440.0, 18220.0, 64)
    coordinate = np.linspace(0.0, 2.0 * np.pi, len(wavelength))

    result = deployed.classify(
        wavelength=wavelength,
        flux=np.stack([np.sin(coordinate), np.sin(coordinate + 0.2)]),
        flux_error=np.full((2, len(wavelength)), 0.8),
        observer_time=[62012.0, 62000.0],
        return_joint=True,
    )
    json.dumps(result, allow_nan=False)

    assert result["model"]["package_format"] == MODEL_PACKAGE_FORMAT
    assert result["input"]["observer_days"] == [0.0, 12.0]
    assert result["classification"]["probability_type"] == "raw"
    assert sum(result["classification"]["probabilities"].values()) == pytest.approx(1.0)
    assert sum(result["redshift"]["probability"]) == pytest.approx(1.0)
    assert result["signal"]["source_probability"] is None

    prepared = prepare_observed_series(
        config,
        wavelength=wavelength,
        flux=np.sin(coordinate),
        flux_error=np.full(len(wavelength), 0.8),
        observer_time=62000.0,
    )
    assert torch.all(prepared.batch["wavelength_mask"] == 1.0)


def test_package_checksum_must_cover_every_required_file(tmp_path: Path) -> None:
    package, _ = _model_package(tmp_path)
    checksum_path = package / "SHA256SUMS"
    lines = checksum_path.read_text().splitlines()
    checksum_path.write_text(
        "\n".join(line for line in lines if not line.endswith("  MODEL_CARD.md")) + "\n"
    )

    with pytest.raises(ValueError, match="does not cover: MODEL_CARD.md"):
        load_model(package)


def _model_package(root: Path) -> tuple[Path, dict]:
    package = root / "model-package"
    package.mkdir()
    config = {
        "project": {"name": "STRIDER test", "seed": 7, "output_dir": "run"},
        "data": {"max_visits": 4},
        "observation": {
            "wavelength_min": 7500.0,
            "wavelength_max": 18175.0,
            "wavelength_bins": 48,
            "require_complete_wavelength_coverage": True,
        },
        "model": {
            "rest_wavelength_min": 2500.0,
            "rest_wavelength_max": 10000.0,
            "rest_wavelength_bins": 32,
            "redshift_min": 0.05,
            "redshift_max": 3.0,
            "redshift_bins": 24,
            "redshift_spacing": "linear",
            "redshift_prior": "flat_z",
            "hidden_dim": 16,
            "phase_features": 8,
            "use_phase": True,
            "dropout": 0.0,
            "classes": ["Ia", "other"],
        },
        "training": {"evidence_sufficiency_loss_weight": 0.25},
        "evaluation": {"views": ["original"], "outlier_delta_z": 0.1},
        "onir": {"bank_path": "onir_bank.npz", "catalog_path": "onir_features.yaml"},
    }
    digest = resolved_config_sha256(config)
    model = StriderModel({**config, "_project_root": str(package)})
    checkpoint = {
        "epoch": 2,
        "config_sha256": digest,
        "model_state": model.state_dict(),
    }
    torch.save(checkpoint, package / "weights.pt")
    (package / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    np.save(package / "redshift_grid.npy", model.redshift_grid.detach().numpy())
    np.save(
        package / "wavelength_grid_angstrom.npy",
        log_wavelength_grid(7500.0, 18175.0, 48),
    )
    (package / "preprocessing.yaml").write_text(
        yaml.safe_dump(
            {
                "wavelength_frame": "observer",
                "wavelength_unit": "Angstrom",
                "flux_error_input": False,
                "maximum_visits": 4,
            },
            sort_keys=False,
        )
    )
    np.savez(package / "onir_bank.npz", placeholder=np.asarray([1], dtype=np.int8))
    (package / "onir_features.yaml").write_text("features: []\n")
    (package / "environment.json").write_text(json.dumps({"python": "test"}))
    (package / "MODEL_CARD.md").write_text("# STRIDER test\n")
    model_info = {
        "format_version": MODEL_PACKAGE_FORMAT,
        "model_name": "STRIDER test",
        "config_sha256": digest,
        "checkpoint_epoch": 2,
        "classes": ["Ia", "other"],
        "redshift_min": float(model.redshift_grid[0]),
        "redshift_max": float(model.redshift_grid[-1]),
        "redshift_bins": len(model.redshift_grid),
        "redshift_spacing": "linear",
        "redshift_prior": "flat_z",
        "calibration_status": "not_fitted",
    }
    (package / "model_info.json").write_text(json.dumps(model_info))
    (package / "calibration.json").write_text(json.dumps({"status": "not_fitted"}))
    checksums = []
    for path in sorted(package.iterdir()):
        if path.name == "SHA256SUMS":
            continue
        digest_value = hashlib.sha256(path.read_bytes()).hexdigest()
        checksums.append(f"{digest_value}  {path.name}\n")
    (package / "SHA256SUMS").write_text("".join(checksums))
    return package, config
