"""Portable inference from an exported STRIDER model package.

The training and evaluation pipeline works with prepared Sundial stores.  This
module is the smaller deployment boundary: it accepts measured observer-frame
spectra, reproduces the training-time measurement preprocessing, runs one
self-contained model package, and returns a truth-free public result.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from strider.engine.calibration import (
    calibrate_class_probabilities,
    highest_density_set,
    visit_band,
)
from strider.engine.config import resolved_config_sha256
from strider.engine.model import StriderModel, measurement_inputs
from strider.engine.model.posterior import joint_probability
from strider.engine.posterior_summary import posterior_basin_candidates
from strider.engine.wavelength import log_wavelength_grid


MODEL_PACKAGE_FORMAT = "strider-model-package-v1"
CALIBRATION_FORMAT = "strider-calibration-v1"
RESULT_FORMAT = "strider-inference-result-v1"
_EPSILON = 1.0e-7
_REQUIRED_PACKAGE_FILES = (
    "weights.pt",
    "config.resolved.yaml",
    "model_info.json",
    "calibration.json",
    "redshift_grid.npy",
    "wavelength_grid_angstrom.npy",
    "preprocessing.yaml",
    "onir_bank.npz",
    "onir_features.yaml",
    "environment.json",
    "MODEL_CARD.md",
    "SHA256SUMS",
)
_REQUIRED_MODEL_INFO_FIELDS = (
    "format_version",
    "model_name",
    "config_sha256",
    "checkpoint_epoch",
    "classes",
    "redshift_min",
    "redshift_max",
    "redshift_bins",
    "redshift_spacing",
    "redshift_prior",
    "calibration_status",
)


@dataclass(frozen=True)
class PreparedSeries:
    """One measured object after deterministic deployment preprocessing."""

    batch: dict[str, torch.Tensor]
    wavelength_grid_angstrom: np.ndarray
    observer_time: np.ndarray
    visit_noise_scale: np.ndarray
    wavelength_coverage_fraction: np.ndarray


@dataclass
class DeployedStrider:
    """A verified STRIDER model package ready for arbitrary measured spectra."""

    package_dir: Path
    model: StriderModel
    config: dict[str, Any]
    model_info: dict[str, Any]
    calibration: dict[str, Any]
    device: torch.device

    def classify(
        self,
        wavelength: np.ndarray | Sequence[float],
        flux: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
        flux_error: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
        observer_time: float | np.ndarray | Sequence[float],
        *,
        peak_time: float | None = None,
        return_joint: bool = False,
    ) -> dict[str, Any]:
        """Classify one spectrum or one spectral time series.

        ``observer_time`` may contain MJD values or any observer-frame day
        coordinate.  Only offsets from the first chronological visit enter the
        network. ``peak_time`` is optional measured light-curve information;
        it is never inferred from simulation truth.
        """
        prepared = prepare_observed_series(
            self.config,
            wavelength=wavelength,
            flux=flux,
            flux_error=flux_error,
            observer_time=observer_time,
            peak_time=peak_time,
        )
        device_batch = {
            name: value.to(self.device) for name, value in prepared.batch.items()
        }
        with torch.inference_mode():
            output = self.model(measurement_inputs(device_batch))
            joint = joint_probability(
                output["joint_logits"],
                self.model.redshift_cell_width,
                self.model.redshift_prior,
            )[0]
        return _public_result(
            self,
            output=output,
            raw_joint=joint.detach().cpu().numpy().astype(np.float64),
            prepared=prepared,
            return_joint=return_joint,
        )


def load_model_package(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    verify_checksums: bool = True,
) -> DeployedStrider:
    """Verify and load a directory written by ``export-model``."""
    package_dir = Path(path).expanduser().resolve()
    missing = [
        name for name in _REQUIRED_PACKAGE_FILES if not (package_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Model package {package_dir} is missing: {', '.join(missing)}"
        )
    if verify_checksums:
        _verify_checksums(package_dir, _REQUIRED_PACKAGE_FILES)

    config = _read_yaml(package_dir / "config.resolved.yaml")
    if not isinstance(config, dict):
        raise ValueError("config.resolved.yaml must contain a mapping")
    config_digest = resolved_config_sha256(config)
    checkpoint = torch.load(
        package_dir / "weights.pt", map_location="cpu", weights_only=True
    )
    if checkpoint.get("config_sha256") != config_digest:
        raise ValueError("Packaged checkpoint and resolved configuration do not match")
    if "model_state" not in checkpoint:
        raise ValueError("Packaged checkpoint does not contain model_state")
    if "epoch" not in checkpoint:
        raise ValueError("Packaged checkpoint does not identify its selected epoch")

    model_info = _read_json(package_dir / "model_info.json")
    missing_info = [name for name in _REQUIRED_MODEL_INFO_FIELDS if name not in model_info]
    if missing_info:
        raise ValueError("model_info.json is missing: " + ", ".join(missing_info))
    if model_info.get("format_version") != MODEL_PACKAGE_FORMAT:
        raise ValueError(
            f"Unsupported model-package format: {model_info.get('format_version')!r}"
        )
    if model_info["config_sha256"] != config_digest:
        raise ValueError("model_info.json and resolved configuration do not match")
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    if int(model_info["checkpoint_epoch"]) != checkpoint_epoch:
        raise ValueError("model_info.json and checkpoint epoch do not match")

    runtime_config = _runtime_config(config, package_dir)
    model = StriderModel(runtime_config)
    model.load_state_dict(checkpoint["model_state"])
    selected_device = _available_device(device)
    model.to(selected_device).eval()

    packaged_redshift = np.load(package_dir / "redshift_grid.npy")
    model_redshift = model.redshift_grid.detach().cpu().numpy()
    if packaged_redshift.shape != model_redshift.shape or not np.allclose(
        packaged_redshift, model_redshift, rtol=1.0e-6, atol=1.0e-7
    ):
        raise ValueError("Packaged redshift grid does not match the reconstructed model")
    if int(model_info["redshift_bins"]) != len(model_redshift) or not np.isclose(
        float(model_info["redshift_min"]), float(model_redshift[0])
    ) or not np.isclose(
        float(model_info["redshift_max"]), float(model_redshift[-1])
    ):
        raise ValueError("model_info.json redshift grid does not match the model")
    expected_wavelength = log_wavelength_grid(
        runtime_config["observation"]["wavelength_min"],
        runtime_config["observation"]["wavelength_max"],
        runtime_config["observation"]["wavelength_bins"],
    )
    packaged_wavelength = np.load(package_dir / "wavelength_grid_angstrom.npy")
    if packaged_wavelength.shape != expected_wavelength.shape or not np.allclose(
        packaged_wavelength, expected_wavelength, rtol=1.0e-7, atol=1.0e-4
    ):
        raise ValueError(
            "Packaged observer-wavelength grid does not match the resolved configuration"
        )
    preprocessing = _read_yaml(package_dir / "preprocessing.yaml")
    _validate_preprocessing(preprocessing, config=runtime_config)

    calibration = _read_json(package_dir / "calibration.json")
    _validate_calibration(
        calibration,
        config_digest=config_digest,
        checkpoint_epoch=checkpoint_epoch,
    )
    if model_info.get("calibration_status") != calibration.get(
        "status", "not_fitted"
    ):
        raise ValueError("model_info.json and calibration.json disagree on status")
    if list(model_info.get("classes", [])) != list(model.class_names):
        raise ValueError("model_info.json classes do not match the reconstructed model")
    if calibration.get("status") == "fitted" and list(
        calibration.get("classes", [])
    ) != list(model.class_names):
        raise ValueError("Calibration classes do not match the reconstructed model")
    if str(model_info["redshift_spacing"]) != str(
        runtime_config["model"].get("redshift_spacing", "linear")
    ) or str(model_info["redshift_prior"]) != str(
        runtime_config["model"].get("redshift_prior", "flat_z")
    ):
        raise ValueError("model_info.json redshift definition does not match the model")
    return DeployedStrider(
        package_dir=package_dir,
        model=model,
        config=runtime_config,
        model_info=model_info,
        calibration=calibration,
        device=selected_device,
    )


def prepare_observed_series(
    config: dict[str, Any],
    *,
    wavelength: np.ndarray | Sequence[float],
    flux: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
    flux_error: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
    observer_time: float | np.ndarray | Sequence[float],
    peak_time: float | None = None,
) -> PreparedSeries:
    """Reproduce the deployable part of ``SundialDataset`` preprocessing."""
    wave = np.asarray(wavelength, dtype=np.float64)
    values = np.asarray(flux, dtype=np.float64)
    errors = np.asarray(flux_error, dtype=np.float64)
    times = np.atleast_1d(np.asarray(observer_time, dtype=np.float64))
    if wave.ndim != 1:
        raise ValueError("wavelength must be a shared one-dimensional grid")
    if values.ndim == 1:
        values = values[None, :]
    if errors.ndim == 1:
        errors = errors[None, :]
    if values.ndim != 2 or errors.shape != values.shape:
        raise ValueError("flux and flux_error must have shape (visits, wavelength)")
    if values.shape[1] != len(wave):
        raise ValueError("wavelength length does not match the spectral bins")
    if values.shape[0] != len(times):
        raise ValueError("observer_time must contain one value per spectrum")
    if len(times) == 0 or not np.isfinite(times).all():
        raise ValueError("observer_time must contain finite values")
    maximum_visits = config["data"].get("max_visits")
    if maximum_visits is not None and str(maximum_visits).lower() != "all":
        limit = int(maximum_visits)
        if len(times) > limit:
            raise ValueError(
                f"Model package supports at most {limit} visits; received {len(times)}"
            )

    visit_order = np.argsort(times, kind="stable")
    times = times[visit_order]
    values = values[visit_order]
    errors = errors[visit_order]
    output_wave = log_wavelength_grid(
        config["observation"]["wavelength_min"],
        config["observation"]["wavelength_max"],
        config["observation"]["wavelength_bins"],
    )
    resampled_flux: list[np.ndarray] = []
    resampled_mask: list[np.ndarray] = []
    resampled_error_shape: list[np.ndarray] = []
    visit_scales: list[float] = []
    coverage: list[float] = []
    use_error_shape = bool(config["model"].get("use_flux_error_channel", False))
    for visit_index, (visit_flux, visit_error) in enumerate(
        zip(values, errors, strict=True), start=1
    ):
        valid = (
            np.isfinite(wave)
            & np.isfinite(visit_flux)
            & np.isfinite(visit_error)
            & (visit_error > 0.0)
        )
        if int(valid.sum()) < 2:
            raise ValueError(f"Spectrum {visit_index} has fewer than two valid bins")
        native_wave = wave[valid]
        native_flux = visit_flux[valid]
        native_error = visit_error[valid]
        order = np.argsort(native_wave, kind="stable")
        native_wave = native_wave[order]
        native_flux = native_flux[order]
        native_error = native_error[order]
        unique = np.r_[True, np.diff(native_wave) > 0.0]
        native_wave = native_wave[unique]
        native_flux = native_flux[unique]
        native_error = native_error[unique]
        if len(native_wave) < 2:
            raise ValueError(f"Spectrum {visit_index} has fewer than two unique wavelengths")
        scale = float(np.quantile(native_error, 0.25))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Spectrum {visit_index} has no positive uncertainty scale")
        inside = _deployment_wavelength_mask(
            config["observation"], output_wave, native_wave, visit_index=visit_index
        )
        if not inside.any():
            raise ValueError(
                f"Spectrum {visit_index} does not overlap the model wavelength range"
            )
        model_flux = np.zeros_like(output_wave, dtype=np.float32)
        model_flux[inside] = np.interp(
            output_wave[inside], native_wave, native_flux / scale
        ).astype(np.float32)
        resampled_flux.append(model_flux)
        resampled_mask.append(inside.astype(np.float32))
        if use_error_shape:
            model_error = np.zeros_like(output_wave, dtype=np.float32)
            model_error[inside] = np.interp(
                output_wave[inside],
                native_wave,
                np.log(native_error / scale),
            ).astype(np.float32)
            resampled_error_shape.append(model_error)
        visit_scales.append(scale)
        coverage.append(float(inside.mean()))

    first_time = float(times[0])
    peak_valid = peak_time is not None and np.isfinite(float(peak_time))
    batch: dict[str, torch.Tensor] = {
        "flux": torch.from_numpy(np.stack(resampled_flux))[None, ...],
        "wavelength_mask": torch.from_numpy(np.stack(resampled_mask))[None, ...],
        "visit_mask": torch.ones((1, len(times)), dtype=torch.float32),
        "observer_days": torch.from_numpy((times - first_time).astype(np.float32))[None, ...],
        "visit_flux_scale": torch.tensor(visit_scales, dtype=torch.float32)[None, ...],
        "peak_day_offset": torch.tensor(
            [float(peak_time) - first_time if peak_valid else 0.0],
            dtype=torch.float32,
        ),
        "peak_date_valid": torch.tensor([float(peak_valid)], dtype=torch.float32),
    }
    if use_error_shape:
        batch["flux_error_shape"] = torch.from_numpy(
            np.stack(resampled_error_shape)
        )[None, ...]
    return PreparedSeries(
        batch=batch,
        wavelength_grid_angstrom=output_wave,
        observer_time=times,
        visit_noise_scale=np.asarray(visit_scales, dtype=np.float64),
        wavelength_coverage_fraction=np.asarray(coverage, dtype=np.float64),
    )


def _public_result(
    deployed: DeployedStrider,
    *,
    output: dict[str, torch.Tensor],
    raw_joint: np.ndarray,
    prepared: PreparedSeries,
    return_joint: bool,
) -> dict[str, Any]:
    class_names = list(deployed.model.class_names)
    grid = deployed.model.redshift_grid.detach().cpu().numpy().astype(np.float64)
    widths = deployed.model.redshift_cell_width.detach().cpu().numpy().astype(np.float64)
    raw_class = raw_joint.sum(axis=1)
    redshift_probability = raw_joint.sum(axis=0)
    calibrated_class = raw_class.copy()
    calibration_status = str(deployed.calibration.get("status", "not_fitted"))
    if calibration_status == "fitted":
        calibrated_class = calibrate_class_probabilities(
            raw_class[None, :], deployed.calibration["class_calibration"]
        )[0]
    class_index = int(np.argmax(calibrated_class))
    candidates = posterior_basin_candidates(grid, redshift_probability, widths)
    primary = candidates[0]
    evidence_score = float(torch.sigmoid(output["evidence_sufficiency_logit"])[0].cpu())
    source_probability = None
    signal_grade = None
    redshift_sets: dict[str, Any] = {}
    if calibration_status == "fitted":
        source_probability = _calibrated_source_probability(
            evidence_score, deployed.calibration["signal_sufficiency"]
        )
        signal_grade = _signal_grade(
            source_probability,
            deployed.calibration["signal_sufficiency"]["grade_thresholds"],
        )
        redshift_sets = _calibrated_redshift_sets(
            deployed.calibration["redshift_sets"],
            predicted_class=class_names[class_index],
            visit_count=len(prepared.observer_time),
            grid=grid,
            probability=redshift_probability,
        )
    result: dict[str, Any] = {
        "format_version": RESULT_FORMAT,
        "model": {
            "name": deployed.model_info.get("model_name", "STRIDER"),
            "package_format": MODEL_PACKAGE_FORMAT,
            "checkpoint_epoch": int(deployed.model_info["checkpoint_epoch"]),
            "classes": class_names,
            "calibration_status": calibration_status,
        },
        "input": {
            "visit_count": int(len(prepared.observer_time)),
            "observer_time": prepared.observer_time.tolist(),
            "observer_days": (
                prepared.observer_time - prepared.observer_time[0]
            ).tolist(),
            "wavelength_range_angstrom": [
                float(prepared.wavelength_grid_angstrom[0]),
                float(prepared.wavelength_grid_angstrom[-1]),
            ],
            "wavelength_coverage_fraction": (
                prepared.wavelength_coverage_fraction.tolist()
            ),
            "visit_noise_scale": prepared.visit_noise_scale.tolist(),
        },
        "classification": {
            "class": class_names[class_index],
            "confidence": float(calibrated_class[class_index]),
            "probability_type": (
                "calibrated" if calibration_status == "fitted" else "raw"
            ),
            "probabilities": {
                name: float(calibrated_class[index])
                for index, name in enumerate(class_names)
            },
            "raw_probabilities": {
                name: float(raw_class[index])
                for index, name in enumerate(class_names)
            },
        },
        "redshift": {
            "z_STRIDER": float(primary["peak_redshift"]),
            "point_estimator": "primary_basin_peak",
            "primary_basin": _json_safe_candidate(primary),
            "candidate_basins": [_json_safe_candidate(item) for item in candidates],
            "calibrated_sets": redshift_sets,
            "grid": grid.tolist(),
            "probability": redshift_probability.tolist(),
        },
        "signal": {
            "raw_evidence_score": evidence_score,
            "source_probability": source_probability,
            "grade": signal_grade,
        },
    }
    if "Ia" in class_names:
        ia_index = class_names.index("Ia")
        result["classification"]["p_Ia"] = float(calibrated_class[ia_index])
        result["classification"]["raw_p_Ia"] = float(raw_class[ia_index])
    if return_joint:
        result["joint_probability"] = raw_joint.tolist()
    return result


def _runtime_config(config: dict[str, Any], package_dir: Path) -> dict[str, Any]:
    runtime = json.loads(json.dumps(config))
    runtime["_project_root"] = str(package_dir)
    runtime["_config_path"] = str(package_dir / "config.resolved.yaml")
    runtime.setdefault("onir", {})["bank_path"] = "onir_bank.npz"
    runtime["onir"]["catalog_path"] = "onir_features.yaml"
    return runtime


def _validate_calibration(
    calibration: dict[str, Any], *, config_digest: str, checkpoint_epoch: int
) -> None:
    status = calibration.get("status", "not_fitted")
    if status == "not_fitted":
        return
    if status != "fitted":
        raise ValueError(f"Unsupported calibration status: {status!r}")
    if calibration.get("format_version") != CALIBRATION_FORMAT:
        raise ValueError("Unsupported fitted calibration format")
    if calibration.get("config_sha256") != config_digest:
        raise ValueError("Calibration and resolved configuration do not match")
    if int(calibration.get("checkpoint_epoch", -1)) != checkpoint_epoch:
        raise ValueError("Calibration and checkpoint epoch do not match")


def _verify_checksums(package_dir: Path, required_files: Sequence[str]) -> None:
    lines = (package_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("SHA256SUMS is empty")
    listed: set[str] = set()
    for line in lines:
        try:
            expected, name = line.split(maxsplit=1)
        except ValueError as error:
            raise ValueError(f"Invalid SHA256SUMS line: {line!r}") from error
        name = name.strip()
        if Path(name).name != name or name == "SHA256SUMS":
            raise ValueError(f"Unsafe model-package checksum path: {name!r}")
        if name in listed:
            raise ValueError(f"Duplicate model-package checksum path: {name!r}")
        listed.add(name)
        path = package_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Checksummed model-package file is missing: {path}")
        if _sha256(path) != expected:
            raise ValueError(f"Checksum mismatch for model-package file: {name}")
    required_payload = set(required_files) - {"SHA256SUMS"}
    missing = sorted(required_payload - listed)
    if missing:
        raise ValueError("SHA256SUMS does not cover: " + ", ".join(missing))


def _validate_preprocessing(value: Any, *, config: dict[str, Any]) -> None:
    if not isinstance(value, dict):
        raise ValueError("preprocessing.yaml must contain a mapping")
    if value.get("wavelength_frame") != "observer":
        raise ValueError("Model package wavelength_frame must be observer")
    if str(value.get("wavelength_unit", "")).lower() != "angstrom":
        raise ValueError("Model package wavelength_unit must be Angstrom")
    expected_error_channel = bool(config["model"].get("use_flux_error_channel", False))
    if bool(value.get("flux_error_input")) != expected_error_channel:
        raise ValueError("preprocessing.yaml disagrees on the FLAMERR input channel")


def _template_support_policy(observation: dict[str, Any]) -> str:
    configured = observation.get("template_support_policy")
    legacy = observation.get("require_complete_wavelength_coverage")
    if configured is None:
        return "complete" if bool(legacy) else "retain"
    policy = str(configured)
    if policy not in {"complete", "retain"}:
        raise ValueError("template_support_policy must be 'complete' or 'retain'")
    if legacy is not None and bool(legacy) != (policy == "complete"):
        raise ValueError(
            "template_support_policy conflicts with require_complete_wavelength_coverage"
        )
    return policy


def _deployment_wavelength_mask(
    observation: dict[str, Any],
    output_wave: np.ndarray,
    native_wave: np.ndarray,
    *,
    visit_index: int,
) -> np.ndarray:
    """Match the training-time complete/retained template-support semantics."""
    if _template_support_policy(observation) != "complete":
        return (output_wave >= native_wave[0]) & (output_wave <= native_wave[-1])
    edges = _wavelength_cell_edges(native_wave)
    if edges[0] > output_wave[0] or edges[-1] < output_wave[-1]:
        raise ValueError(
            f"Spectrum {visit_index} does not provide complete native-bin support "
            "over the model wavelength range"
        )
    return np.ones(output_wave.shape, dtype=bool)


def _wavelength_cell_edges(centers: np.ndarray) -> np.ndarray:
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    edges = np.empty(len(centers) + 1, dtype=np.float64)
    edges[1:-1] = midpoints
    edges[0] = centers[0] - (midpoints[0] - centers[0])
    edges[-1] = centers[-1] + (centers[-1] - midpoints[-1])
    return edges


def _calibrated_source_probability(
    evidence_score: float, calibration: dict[str, Any]
) -> float:
    clipped = float(np.clip(evidence_score, _EPSILON, 1.0 - _EPSILON))
    logit = np.log(clipped / (1.0 - clipped))
    value = float(calibration["slope"]) * logit + float(calibration["intercept"])
    if value >= 0.0:
        return float(1.0 / (1.0 + np.exp(-value)))
    exponent = float(np.exp(value))
    return exponent / (1.0 + exponent)


def _signal_grade(value: float, thresholds: dict[str, float]) -> str:
    if value >= float(thresholds["high"]):
        return "high"
    if value >= float(thresholds["medium"]):
        return "medium"
    if value >= float(thresholds["low"]):
        return "low"
    return "limited"


def _calibrated_redshift_sets(
    calibration: dict[str, Any],
    *,
    predicted_class: str,
    visit_count: int,
    grid: np.ndarray,
    probability: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    stratum = f"{predicted_class}|{visit_band(visit_count)}"
    for level in calibration["levels"]:
        coverage = float(level["coverage"])
        quantile = float(level["strata"].get(stratum, level["global_quantile"]))
        result[str(int(round(100.0 * coverage)))] = highest_density_set(
            grid, probability, quantile
        )
    return result


def _json_safe_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in candidate.items():
        if isinstance(value, (bool, np.bool_)):
            result[key] = bool(value)
        elif isinstance(value, (int, np.integer)):
            result[key] = int(value)
        else:
            number = float(value)
            result[key] = number if np.isfinite(number) else None
    return result


def _available_device(value: str | torch.device) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return device


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
