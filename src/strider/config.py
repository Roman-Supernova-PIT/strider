"""Configuration loading with explicit paths and lightweight validation."""

from __future__ import annotations

import hashlib
import difflib
import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    parent_setting = config.pop("extends", None)
    if parent_setting is not None:
        parent_path = (config_path.parent / str(parent_setting)).resolve()
        parent = load_config(parent_path)
        parent.pop("_config_path", None)
        parent.pop("_project_root", None)
        config = _deep_merge(parent, config)
    config = _expand_environment_strings(config)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(_find_project_root(config_path))
    _validate_known_keys(config, config_path)
    return config


def project_path(config: dict[str, Any], value: str) -> Path:
    """Resolve a project-relative setting without depending on the shell cwd."""
    path = Path(os.path.expandvars(value)).expanduser()
    if path.is_absolute():
        return path
    return Path(config["_project_root"]) / path


def resolved_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the complete user-facing configuration without loader metadata."""
    return {
        key: value
        for key, value in config.items()
        if not str(key).startswith("_")
    }


def resolved_config_text(config: dict[str, Any]) -> str:
    """Return deterministic YAML suitable for run records and comparisons."""
    return yaml.safe_dump(resolved_config(config), sort_keys=True)


def resolved_config_sha256(config: dict[str, Any]) -> str:
    return hashlib.sha256(resolved_config_text(config).encode("utf-8")).hexdigest()


def _deep_merge(base: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _expand_environment_strings(value: Any) -> Any:
    """Resolve environment variables before a config is recorded or compared."""
    if isinstance(value, dict):
        return {key: _expand_environment_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment_strings(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def _find_project_root(config_path: Path) -> Path:
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"Could not find pyproject.toml above {config_path}")


_KNOWN_KEYS = {
    "project": {"name", "seed", "output_dir"},
    "data": {
        "class_scheme", "source_dir", "source_products", "prepared_dir", "train_blocks",
        "validation_blocks", "test_blocks", "split_blocks", "max_objects",
        "runtime_object_limits", "redshift_edges", "max_visits",
        "training_visit_counts", "require_all_training_classes",
        "minimum_class_counts", "training_sample_by_block",
        "include_flux_error_channel", "include_clean_flux_target",
    },
    "observation": {
        "wavelength_min", "wavelength_max", "wavelength_bins",
        "require_complete_wavelength_coverage",
        "template_support_policy",
        "generated_noise", "no_source_fraction",
    },
    "model": {
        "architecture", "rest_wavelength_min", "rest_wavelength_max",
        "rest_wavelength_bins", "redshift_min", "redshift_max", "redshift_bins",
        "redshift_spacing", "redshift_prior", "hidden_dim", "phase_features",
        "use_phase", "temporal_mode", "temporal_initial_scale",
        "factored_attention_heads", "factored_shape_initial_scale",
        "phase_auxiliary_bins", "phase_auxiliary_min_days",
        "phase_auxiliary_max_days",
        "candidate_phase_consistency", "candidate_phase_initial_scale",
        "candidate_phase_minimum_visits",
        "candidate_phase_use_peak_date",
        "candidate_phase_peak_uncertainty_days",
        "candidate_phase_peak_quadrature_points",
        "candidate_phase_peak_outlier_fraction",
        "candidate_phase_minimum_coverage_fraction",
        "full_spectrum_context", "context_token_dim", "context_patch_size",
        "context_attention_heads", "context_attention_layers",
        "context_initial_scale", "context_visit_attention",
        "context_input_normalization", "context_minimum_support",
        "dense_rest_frame_scan", "dense_scan_initial_scale",
        "dense_scan_evidence_scale", "dense_scan_chunk_size",
        "dense_scan_minimum_overlap", "dense_scan_overlap_exponent",
        "dense_scan_token_dim", "dense_scan_patch_size", "dense_scan_rest_bins",
        "dense_scan_view", "dense_scan_input_mode",
        "coadd_maximum_relative_error", "coadd_edge_trim_fraction",
        "coadd_weighting", "coadd_reconstruction",
        "dense_continuum_detail", "dense_continuum_sigma_bins",
        "dense_continuum_sigma_km_s",
        "dense_initial_detail_weight",
        "dense_minimum_whole_weight",
        "relative_brightness_evolution", "relative_brightness_initial_scale",
        "relative_amplitude_mode",
        "use_flux_error_channel", "flux_error_initial_scale",
        "evidence_visit_count_reference", "evidence_use_visit_count_and_span",
        "dropout", "classes",
    },
    "training": {
        "epochs", "batch_size", "learning_rate", "learning_rate_schedule",
        "warmup_epochs", "minimum_learning_rate_fraction", "weight_decay",
        "num_workers", "persistent_workers", "prefetch_factor",
        "benchmark_worker_counts", "benchmark_batch_sizes", "benchmark_batches",
        "batch_by_visit_count", "maximum_visits_per_batch",
        "maximum_squared_visits_per_batch",
        "full_visit_training_fraction",
        "early_stopping_patience", "selection_split", "paired_no_source",
        "paired_reported_error_fraction", "observed_flux_fraction",
        "observed_flux_blank_family",
        "evidence_sufficiency_loss_weight",
        "noise_scale_augmentation_fraction", "noise_scale_range",
        "no_source_redshift_loss_weight", "no_source_class_loss_weight",
        "class_weight_mode", "class_weight_power", "validation_view_weights",
        "checkpoint_metric_view",
        "timing_baseline_epochs", "mixed_precision", "max_gradient_norm",
        "initial_checkpoint", "temporal_only",
        "phase_loss_weight",
        "coadd_reconstruction_loss_weight",
        "alias_ranking_loss_weight", "alias_ranking_minimum_delta_z",
        "alias_ranking_margin",
    },
    "evaluation": {
        "split", "views", "outlier_delta_z", "visit_control_repeats",
        "visit_control_counts", "visit_control_max_objects",
        "visit_control_selection",
        "save_redshift_probability", "evidence_map_view",
        "evidence_map_redshifts", "evidence_map_objects_per_redshift",
        "evidence_map_object_list", "evidence_map_layout",
        "evidence_gif_max_frames", "evidence_gif_layout",
        "evidence_grade_thresholds",
        "competing_peak_mass_ratio",
        "ia_examples_per_redshift",
        "ia_redshift_edges",
    },
    "onir": {
        "catalog_path", "bank_path", "bank_view", "anchor_mode",
        "bank_input_mode",
        "random_anchor_seed", "profile_initialization", "input_mode",
        "maximum_radius_bins", "minimum_valid_fraction", "minimum_support",
        "allow_radius_clipping", "maximum_windows_per_cell",
        "prototype_count", "profile_rest_phase_min_days",
        "profile_rest_phase_max_days", "evidence_scale", "drift_loss_weight",
        "token_dim", "encoded_samples_per_feature", "encoded_sample_extent_fraction",
        "minimum_encoder_support", "minimum_prototype_support",
        "coverage_log_weight", "prototype_temperature", "input_normalization", "token_activation",
        "visit_evidence_exponent", "train_profiles",
    },
    "reference": {
        "bank_path", "rest_wavelength_bins", "phase_edges_days",
        "coadd_profiles_per_class", "phase_profiles_per_cell",
        "maximum_training_objects_per_class",
        "maximum_coadd_profiles_per_class",
        "maximum_phase_profiles_per_cell", "minimum_bin_fraction",
        "minimum_rest_bins", "continuum_width_km_s",
        "maximum_relative_coadd_error", "edge_trim_fraction",
        "edge_taper_fraction",
        "spectral_uncertainty_weighting",
        "minimum_relative_spectral_precision",
        "minimum_profile_support", "minimum_rest_fraction",
        "minimum_shared_fraction", "prototype_temperature",
        "fine_class_temperature", "phase_temperature",
        "initial_continuum_removed_fraction",
        "initial_coadd_scale", "initial_sequence_scale", "evidence_scale",
        "redshift_chunk_size", "sequence_visits",
        "minimum_sequence_visits", "spectral_encoder", "token_dim",
        "token_pool_size", "minimum_encoder_support", "attention_heads",
        "sequence_combination", "time_attention_hidden_dim",
        "temporal_hidden_dim", "temporal_attention_heads",
        "temporal_layers", "temporal_feedforward_multiplier",
        "temporal_initial_correction_scale", "temporal_use_signal_to_noise",
        "relative_flux_evolution",
    },
}


def _validate_known_keys(config: dict[str, Any], path: Path) -> None:
    allowed_sections = set(_KNOWN_KEYS) | {"_config_path", "_project_root"}
    for section in config:
        if section not in allowed_sections:
            _raise_unknown_key(str(section), sorted(_KNOWN_KEYS), path)
    for section, allowed in _KNOWN_KEYS.items():
        values = config.get(section)
        if values is None:
            continue
        if not isinstance(values, dict):
            raise ValueError(f"Config section {section} must be a mapping: {path}")
        for key in values:
            if key not in allowed:
                _raise_unknown_key(f"{section}.{key}", sorted(allowed), path)


def _raise_unknown_key(key: str, choices: list[str], path: Path) -> None:
    leaf = key.rsplit(".", 1)[-1]
    suggestion = difflib.get_close_matches(leaf, choices, n=1)
    detail = f"; did you mean {suggestion[0]!r}?" if suggestion else ""
    raise ValueError(f"Unknown config key {key!r} in {path}{detail}")
