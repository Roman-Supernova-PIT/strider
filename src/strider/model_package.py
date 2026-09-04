"""Export a readable, self-contained STRIDER model directory.

The package records weights, resolved configuration, preprocessing, class and
redshift definitions, the architecture's fixed reference assets, environment
and measured results. It contains no object-level training spectra or truth
redshifts.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import yaml

from strider.config import project_path, resolved_config_sha256
from strider.data.dataset import log_wavelength_grid


MODEL_PACKAGE_FORMAT = "strider-model-package-v1"
CALIBRATION_FORMAT = "strider-calibration-v1"


def model_package_asset_specs(
    config: dict[str, Any],
) -> tuple[tuple[str, str, str], ...]:
    """Return ``(section, config key, package name)`` for fixed model assets."""
    architecture = str(config["model"].get("architecture", "full_scan"))
    if architecture == "roman_reference":
        return (("reference", "bank_path", "reference_bank.npz"),)
    if architecture in {"onir", "encoded_onir", "factored_onir"}:
        return (
            ("onir", "bank_path", "onir_bank.npz"),
            ("onir", "catalog_path", "onir_features.yaml"),
        )
    return ()


def export_model_package(
    config: dict[str, Any], *, replace: bool = False
) -> dict[str, Any]:
    run_dir = project_path(config, config["project"]["output_dir"])
    checkpoint_path = run_dir / "best_model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing trained checkpoint: {checkpoint_path}")
    package_dir = run_dir / "model_package"
    if package_dir.exists() and not replace:
        raise FileExistsError(
            f"Model package already exists: {package_dir}. "
            "Use export-model --replace after calibration or frozen evaluation."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config_digest = resolved_config_sha256(config)
    if checkpoint.get("config_sha256") != config_digest:
        raise ValueError("The checkpoint and resolved configuration do not match")
    redshift_grid = checkpoint.get("redshift_grid")
    if redshift_grid is None:
        raise ValueError("Checkpoint does not contain its candidate-redshift grid")
    observed_grid = log_wavelength_grid(
        float(config["observation"]["wavelength_min"]),
        float(config["observation"]["wavelength_max"]),
        int(config["observation"]["wavelength_bins"]),
    )
    learned_uncertainty_channel = bool(
        config["model"].get("use_flux_error_channel", False)
    )
    architecture = str(config["model"].get("architecture", "full_scan"))
    if architecture == "roman_reference":
        uncertainty_use = (
            "resampling, inverse-variance accumulation, continuous spectral "
            "reliability, relative-brightness evolution and visit selection; "
            "not a learned class-redshift input channel"
        )
        input_normalization = "scale-invariant full and continuum-removed spectra"
    elif learned_uncertainty_channel:
        uncertainty_use = (
            "visit scaling, measurement weighting and a learned relative "
            "wavelength-dependent uncertainty channel"
        )
        input_normalization = config.get("onir", {}).get(
            "input_normalization", "none"
        )
    else:
        uncertainty_use = (
            "visit scaling and deterministic measurement weighting; not a "
            "learned wavelength-dependent uncertainty channel"
        )
        input_normalization = config.get("onir", {}).get(
            "input_normalization", "none"
        )
    preprocessing = {
        "wavelength_frame": "observer",
        "wavelength_unit": "Angstrom",
        "resampling": "linear interpolation after native-bin noise generation",
        "visit_scaling": "25th percentile of positive reported uncertainty",
        "input_normalization": input_normalization,
        # Retained for package-v1 compatibility. This means a learned channel,
        # not whether reported uncertainty is used by deterministic operations.
        "flux_error_input": learned_uncertainty_channel,
        "learned_uncertainty_channel": learned_uncertainty_channel,
        "reported_uncertainty_required": True,
        "uncertainty_use": uncertainty_use,
        "truth_redshift_input": False,
        "truth_phase_input": False,
        "simulation_clean_flux_input": False,
        "maximum_visits": config["data"]["max_visits"],
    }
    asset_sources = tuple(
        (
            project_path(config, config[section][key]),
            package_name,
        )
        for section, key, package_name in model_package_asset_specs(config)
    )
    required_sources = (
        run_dir / "config.resolved.yaml",
        run_dir / "environment.json",
        *(source for source, _ in asset_sources),
    )
    missing_sources = [str(path) for path in required_sources if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(
            "Required model-package file is missing: " + ", ".join(missing_sources)
        )
    calibration = _calibration_for_package(
        run_dir,
        config_sha256=config_digest,
        checkpoint_epoch=int(checkpoint["epoch"]),
    )
    metrics = _test_evaluation_metrics(
        run_dir,
        config_sha256=config_digest,
        checkpoint_epoch=int(checkpoint["epoch"]),
    )
    model_info = {
        "format_version": MODEL_PACKAGE_FORMAT,
        "model_name": str(config["project"]["name"]),
        "architecture": architecture,
        "model_assets": [package_name for _, package_name in asset_sources],
        "config_sha256": config_digest,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "classes": list(config["model"]["classes"]),
        "redshift_min": float(redshift_grid[0]),
        "redshift_max": float(redshift_grid[-1]),
        "redshift_bins": int(len(redshift_grid)),
        "redshift_spacing": str(config["model"].get("redshift_spacing", "linear")),
        "redshift_prior": str(config["model"].get("redshift_prior", "flat_z")),
        "conditional_output": "P(class, redshift | usable spectral evidence)",
        "separate_output": (
            "measured-signal reliability; source probability and descriptive "
            "boundaries require separate calibration"
        ),
        "metrics_status": "frozen_test" if metrics is not None else "not_included",
    }
    model_info["calibration_status"] = calibration["status"]

    run_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=".model-package-", dir=run_dir))
    backup_dir: Path | None = None
    try:
        shutil.copy2(checkpoint_path, staging_dir / "weights.pt")
        _copy_required(run_dir / "config.resolved.yaml", staging_dir)
        _copy_required(run_dir / "environment.json", staging_dir)
        np.save(staging_dir / "redshift_grid.npy", redshift_grid.cpu().numpy())
        np.save(staging_dir / "wavelength_grid_angstrom.npy", observed_grid)
        (staging_dir / "preprocessing.yaml").write_text(
            yaml.safe_dump(preprocessing, sort_keys=False), encoding="utf-8"
        )
        for source, package_name in asset_sources:
            _copy_required(source, staging_dir, package_name)
        _write_json(staging_dir / "model_info.json", model_info)
        _write_json(staging_dir / "calibration.json", calibration)
        if metrics is not None:
            _write_json(staging_dir / "metrics.json", metrics)
        (staging_dir / "MODEL_CARD.md").write_text(
            _model_card(model_info), encoding="utf-8"
        )
        checksums = {
            path.name: _sha256(path)
            for path in sorted(staging_dir.iterdir())
            if path.is_file() and path.name != "SHA256SUMS"
        }
        (staging_dir / "SHA256SUMS").write_text(
            "".join(f"{digest}  {name}\n" for name, digest in checksums.items()),
            encoding="utf-8",
        )
        if package_dir.exists():
            backup_dir = run_dir / f"model_package.backup-{uuid4().hex[:8]}"
            package_dir.rename(backup_dir)
        try:
            staging_dir.rename(package_dir)
        except Exception:
            if backup_dir is not None and not package_dir.exists():
                backup_dir.rename(package_dir)
            raise
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

    report = {
        "package_dir": str(package_dir),
        "files": sorted(checksums),
        "calibration_status": calibration["status"],
        "metrics_status": model_info["metrics_status"],
        "replaced_package_backup": None if backup_dir is None else str(backup_dir),
    }
    _write_json(run_dir / "model_package_summary.json", report)
    return report


def _copy_required(source: Path, destination: Path, name: str | None = None) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Required model-package file is missing: {source}")
    shutil.copy2(source, destination / (name or source.name))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _calibration_for_package(
    run_dir: Path,
    *,
    config_sha256: str,
    checkpoint_epoch: int,
) -> dict[str, Any]:
    path = run_dir / "calibration.json"
    if not path.is_file():
        return {"status": "not_fitted"}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or value.get("status") != "fitted":
        raise ValueError(f"Calibration artifact is present but not fitted: {path}")
    format_parts = str(value.get("format_version", "")).rsplit("-", 2)
    if format_parts[-2:] != ["calibration", "v1"]:
        raise ValueError(f"Unsupported calibration artifact format: {path}")
    if value.get("config_sha256") != config_sha256:
        raise ValueError("Calibration artifact and resolved configuration do not match")
    if int(value.get("checkpoint_epoch", -1)) != checkpoint_epoch:
        raise ValueError("Calibration artifact and exported checkpoint do not match")
    public_value = dict(value)
    public_value["format_version"] = CALIBRATION_FORMAT
    return public_value


def _test_evaluation_metrics(
    run_dir: Path,
    *,
    config_sha256: str,
    checkpoint_epoch: int,
) -> dict[str, Any] | None:
    """Return only a frozen, matching test summary; never package calibration metrics."""
    for path in (
        run_dir / "test_evaluation_summary.json",
        run_dir / "evaluation_summary.json",
    ):
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict) or value.get("split") != "test":
            continue
        if int(value.get("checkpoint_epoch", -1)) != checkpoint_epoch:
            raise ValueError(f"Test evaluation and exported checkpoint do not match: {path}")
        recorded_digest = value.get("config_sha256")
        if recorded_digest is not None and recorded_digest != config_sha256:
            raise ValueError(f"Test evaluation and resolved configuration do not match: {path}")
        return value
    return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _model_card(info: dict[str, Any]) -> str:
    classes = ", ".join(info["classes"])
    calibration_sentence = (
        "The packaged class probabilities, redshift sets, and source-sufficiency "
        "mapping have a fitted calibration artifact."
        if info["calibration_status"] == "fitted"
        else "Calibration is not fitted; probability and grade claims are provisional."
    )
    spacing = (
        "uniformly in log(1+z)"
        if info["redshift_spacing"] == "log1p"
        else "linearly in redshift"
    )
    prior = (
        "a flat-redshift prior"
        if info["redshift_prior"] == "flat_z"
        else f"the {info['redshift_prior']} prior"
    )
    metrics_sentence = (
        "`metrics.json` contains the matching frozen test evaluation."
        if info["metrics_status"] == "frozen_test"
        else "No frozen test metrics are included in this package."
    )
    return f"""# {info['model_name']}

This STRIDER model estimates a joint transient class and redshift distribution
from a time series of observer-frame Roman prism spectra. Its class set is:
{classes}.

The class-redshift distribution is conditional on the measured spectra
containing enough information for the model. Always inspect the separate
measured-signal reliability result, posterior width and multimodality before
using a point estimate. Do not treat the reliability score as a calibrated
probability unless `calibration.json` records a fitted calibration.

The model scans {info['redshift_bins']} candidate values from
{info['redshift_min']:.3f} to {info['redshift_max']:.3f}, {spacing}, and reports
probability mass under {prior}.

This package does not establish performance on real Roman observations. Read
the matching simulation evaluation and sample sizes when they are included.
Calibration is recorded separately in `calibration.json`. {metrics_sentence}
{calibration_sentence}
"""
