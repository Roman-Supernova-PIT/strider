"""Build phase-neutral ONIR profiles from one explicit training-split view."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from strider.config import project_path
from strider.data.dataset import SundialDataset, log_wavelength_grid
from strider.model.redshift_scan import build_redshift_grid

from .catalog import feature_geometry, load_catalog, overlap_weights


def build_onir_bank(config: dict[str, Any]) -> dict[str, Any]:
    settings = config["onir"]
    output_path = project_path(config, settings["bank_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog(project_path(config, settings["catalog_path"]))
    rest_grid = log_wavelength_grid(
        config["model"]["rest_wavelength_min"],
        config["model"]["rest_wavelength_max"],
        config["model"]["rest_wavelength_bins"],
    )
    wavelengths = catalog.rest_wavelengths.copy()
    if settings.get("anchor_mode", "named") == "random":
        wavelengths = _random_anchor_wavelengths(
            rest_grid,
            wavelengths,
            int(config["project"]["seed"]) + int(settings.get("random_anchor_seed", 0)),
        )
        catalog = type(catalog)(catalog.names, wavelengths, catalog.half_width_kms)
    centers, radii, window_mask = feature_geometry(
        rest_grid,
        catalog,
        int(settings["maximum_radius_bins"]),
        allow_radius_clipping=bool(settings.get("allow_radius_clipping", False)),
    )
    geometry_weight = overlap_weights(centers, window_mask, len(rest_grid))
    class_names = tuple(str(value) for value in config["model"]["classes"])
    collected: list[list[list[np.ndarray]]] = [
        [[] for _ in catalog.names] for _ in class_names
    ]
    collected_masks: list[list[list[np.ndarray]]] = [
        [[] for _ in catalog.names] for _ in class_names
    ]
    available_windows = np.zeros((len(class_names), len(catalog.names)), dtype=np.int64)
    maximum_windows = int(settings.get("maximum_windows_per_cell", 5_000))
    if maximum_windows < 1:
        raise ValueError("ONIR maximum_windows_per_cell must be positive")
    sampling_rng = np.random.default_rng(int(config["project"]["seed"]) + 41_003)
    bank_view = str(settings.get("bank_view", "clean"))
    if bank_view not in {"clean", "original"}:
        raise ValueError("ONIR bank_view must be 'clean' or 'original'")
    source_flux = "SIM_FLAM" if bank_view == "clean" else "FLAM"
    bank_input_mode = str(settings.get("bank_input_mode", "individual_visits"))
    if bank_input_mode not in {"individual_visits", "coadded_flux"}:
        raise ValueError(
            "ONIR bank_input_mode must be 'individual_visits' or 'coadded_flux'"
        )
    dataset = SundialDataset(
        config, "train", bank_view, training=False, pair_no_source=False
    )
    observed_grid = dataset.output_wavelength
    minimum_valid_fraction = float(settings.get("minimum_valid_fraction", 0.8))
    rest_phase_min_days = float(settings.get("profile_rest_phase_min_days", -20.0))
    rest_phase_max_days = float(settings.get("profile_rest_phase_max_days", 80.0))
    if rest_phase_max_days <= rest_phase_min_days:
        raise ValueError("ONIR profile rest-phase maximum must exceed its minimum")
    for item_index in range(len(dataset)):
        item = dataset[item_index]
        class_index = int(item["class_index"])
        redshift = float(item["redshift"])
        spectra = _profile_input_spectra(
            item,
            bank_input_mode,
            rest_phase_min_days,
            rest_phase_max_days,
        )
        for flux, mask in spectra:
            rest_flux, rest_mask = align_to_rest_grid(
                observed_grid, flux, mask, redshift, rest_grid
            )
            windows, window_support, usable = extract_feature_windows(
                rest_flux,
                rest_mask,
                centers,
                window_mask,
                minimum_valid_fraction,
            )
            complete_window = np.all(window_support | ~window_mask, axis=1)
            usable &= complete_window
            for feature_index in np.flatnonzero(usable):
                available_windows[class_index, feature_index] += 1
                seen = int(available_windows[class_index, feature_index])
                profiles = collected[class_index][feature_index]
                masks = collected_masks[class_index][feature_index]
                if len(profiles) < maximum_windows:
                    profiles.append(windows[feature_index])
                    masks.append(window_support[feature_index])
                else:
                    replacement = int(sampling_rng.integers(seen))
                    if replacement < maximum_windows:
                        profiles[replacement] = windows[feature_index]
                        masks[replacement] = window_support[feature_index]
    classes = len(class_names)
    features = len(catalog.names)
    window = window_mask.shape[1]
    means = np.zeros((classes, features, window), dtype=np.float32)
    mean_masks = np.zeros((classes, features, window), dtype=bool)
    medoids = np.zeros_like(means)
    medoid_masks = np.zeros_like(mean_masks)
    support = np.zeros((classes, features), dtype=np.int64)
    prototype_count = int(settings.get("prototype_count", 3))
    if prototype_count < 1:
        raise ValueError("ONIR prototype_count must be positive")
    prototypes = np.zeros((classes, features, prototype_count, window), dtype=np.float32)
    prototype_masks = np.zeros_like(prototypes, dtype=bool)
    prototype_support = np.zeros((classes, features, prototype_count), dtype=np.int64)
    for class_index in range(classes):
        for feature_index in range(features):
            rows = collected[class_index][feature_index]
            support[class_index, feature_index] = len(rows)
            if not rows:
                continue
            values = np.stack(rows).astype(np.float32)
            masks = np.stack(collected_masks[class_index][feature_index]).astype(bool)
            mean, mean_mask = _masked_mean_profile(
                values, masks, window_mask[feature_index]
            )
            means[class_index, feature_index] = mean
            mean_masks[class_index, feature_index] = mean_mask
            similarity = _masked_cosine(values, masks, mean[None, :], mean_mask[None, :])[:, 0]
            medoid_index = int(np.argmax(similarity))
            medoids[class_index, feature_index] = values[medoid_index]
            medoid_masks[class_index, feature_index] = masks[medoid_index]
            cell_profiles, cell_masks, cell_support = _cluster_masked_profiles(
                values,
                masks,
                window_mask[feature_index],
                prototype_count,
            )
            prototypes[class_index, feature_index] = cell_profiles
            prototype_masks[class_index, feature_index] = cell_masks
            prototype_support[class_index, feature_index] = cell_support
    _assert_geometric_profile_masks(
        mean_masks,
        medoid_masks,
        support,
        prototype_masks,
        prototype_support,
        window_mask,
    )
    np.savez_compressed(
        output_path,
        format_version=np.asarray("strider-onir-phase-neutral-v1"),
        mean_profiles=means,
        mean_profile_mask=mean_masks,
        medoid_profiles=medoids,
        medoid_profile_mask=medoid_masks,
        prototype_profiles=prototypes,
        prototype_profile_mask=prototype_masks,
        prototype_support_counts=prototype_support,
        support_counts=support,
        available_window_counts=available_windows,
        class_names=np.asarray(class_names),
        feature_names=np.asarray(catalog.names),
        rest_wavelengths=catalog.rest_wavelengths,
        feature_radii=radii,
        window_mask=window_mask,
        overlap_weights=geometry_weight,
        rest_grid=rest_grid,
        source_split=np.asarray("train"),
        source_flux=np.asarray(source_flux),
        bank_view=np.asarray(bank_view),
        bank_input_mode=np.asarray(bank_input_mode),
        anchor_mode=np.asarray(settings.get("anchor_mode", "named")),
        profile_rest_phase_min_days=np.asarray(rest_phase_min_days, dtype=np.float32),
        profile_rest_phase_max_days=np.asarray(rest_phase_max_days, dtype=np.float32),
    )
    report = _audit_report(
        config,
        output_path,
        catalog.names,
        catalog.rest_wavelengths,
        centers,
        window_mask,
        geometry_weight,
        support,
        available_windows,
        rest_grid,
        source_flux,
        bank_view,
        bank_input_mode,
    )
    with output_path.with_suffix(".audit.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def _profile_input_spectra(
    item: dict[str, Any],
    bank_input_mode: str,
    rest_phase_min_days: float,
    rest_phase_max_days: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return individual visits or one equal-weight coadd for bank construction."""
    flux = item["flux"].numpy()
    mask = item["wavelength_mask"].numpy()
    rest_phase = item["simulation_rest_phase_days"].numpy()
    selected = (
        (rest_phase >= rest_phase_min_days)
        & (rest_phase <= rest_phase_max_days)
    )
    if not np.any(selected):
        return []
    if bank_input_mode == "individual_visits":
        return [(row_flux, row_mask) for row_flux, row_mask in zip(flux[selected], mask[selected])]
    if bank_input_mode != "coadded_flux":
        raise ValueError(
            "ONIR bank_input_mode must be 'individual_visits' or 'coadded_flux'"
        )
    measured = mask[selected]
    count = measured.sum(axis=0)
    coadded_mask = (count > 0).astype(np.float32)
    # Match the runtime coadd exactly. FLAMERR is intentionally not used here.
    coadded_flux = (flux[selected] * measured).sum(axis=0) / np.maximum(count, 1.0)
    coadded_flux = coadded_flux.astype(np.float32) * coadded_mask
    return [(coadded_flux, coadded_mask)]


def _cluster_profiles(
    values: np.ndarray,
    active: np.ndarray,
    prototype_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic spherical clusters for phase-neutral profile diversity."""
    masks = np.broadcast_to(active[None, :], values.shape).copy()
    profiles, _, support = _cluster_masked_profiles(
        values, masks, active, prototype_count
    )
    return profiles, support


def _cluster_masked_profiles(
    values: np.ndarray,
    masks: np.ndarray,
    active: np.ndarray,
    prototype_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cluster profiles while comparing only wavelength bins shared by both inputs."""
    count, window = values.shape
    output = np.zeros((prototype_count, window), dtype=np.float32)
    output_mask = np.zeros((prototype_count, window), dtype=bool)
    support = np.zeros(prototype_count, dtype=np.int64)
    used = min(prototype_count, count)
    mean, mean_mask = _masked_mean_profile(values, masks, active)
    centers = [mean]
    center_masks = [mean_mask]
    while len(centers) < used:
        similarity = _masked_cosine(
            values, masks, np.stack(centers), np.stack(center_masks)
        )
        next_index = int(np.argmin(similarity.max(axis=1)))
        centers.append(values[next_index])
        center_masks.append(masks[next_index])
    centers_array = np.stack(centers).astype(np.float32)
    center_mask_array = np.stack(center_masks).astype(bool)
    assignments = np.zeros(count, dtype=np.int64)
    for _ in range(8):
        assignments = np.argmax(
            _masked_cosine(values, masks, centers_array, center_mask_array), axis=1
        )
        updated = centers_array.copy()
        updated_masks = center_mask_array.copy()
        for index in range(used):
            selected = assignments == index
            members = values[selected]
            if len(members):
                updated[index], updated_masks[index] = _masked_mean_profile(
                    members, masks[selected], active
                )
        if np.allclose(updated, centers_array, atol=1e-6) and np.array_equal(
            updated_masks, center_mask_array
        ):
            centers_array = updated
            center_mask_array = updated_masks
            break
        centers_array = updated
        center_mask_array = updated_masks
    output[:used] = centers_array
    output_mask[:used] = center_mask_array
    support[:used] = np.bincount(assignments, minlength=used)
    return output, output_mask, support


def _masked_mean_profile(
    values: np.ndarray, masks: np.ndarray, geometry: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    counts = masks.sum(axis=0)
    active = np.asarray(geometry, dtype=bool) & (counts > 0)
    mean = np.zeros(values.shape[1], dtype=np.float32)
    mean[active] = (values * masks).sum(axis=0)[active] / counts[active]
    return _unit_profile(mean, active), active


def _masked_cosine(
    left: np.ndarray,
    left_mask: np.ndarray,
    right: np.ndarray,
    right_mask: np.ndarray,
) -> np.ndarray:
    """Cosine matrix using only bins available in each pair of profiles."""
    numerator = left @ right.T
    left_energy = (left * left) @ right_mask.astype(np.float32).T
    right_energy = left_mask.astype(np.float32) @ (right * right).T
    denominator = np.sqrt(np.maximum(left_energy * right_energy, 1e-16))
    return numerator / denominator


def align_to_rest_grid(
    observed_grid: np.ndarray,
    flux: np.ndarray,
    mask: np.ndarray,
    redshift: float,
    rest_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    observed_target = rest_grid * (1.0 + float(redshift))
    inside = (observed_target >= observed_grid[0]) & (observed_target <= observed_grid[-1])
    output = np.zeros_like(rest_grid, dtype=np.float32)
    valid = np.zeros_like(rest_grid, dtype=np.float32)
    output[inside] = np.interp(observed_target[inside], observed_grid, flux).astype(np.float32)
    valid[inside] = np.interp(observed_target[inside], observed_grid, mask).astype(np.float32)
    valid = (valid > 0.999).astype(np.float32)
    output[valid == 0] = 0.0
    return output, valid


def extract_feature_windows(
    rest_flux: np.ndarray,
    rest_mask: np.ndarray,
    centers: np.ndarray,
    window_mask: np.ndarray,
    minimum_valid_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius = window_mask.shape[1] // 2
    offsets = np.arange(-radius, radius + 1)
    windows = np.zeros(window_mask.shape, dtype=np.float32)
    measured_windows = np.zeros(window_mask.shape, dtype=bool)
    usable = np.zeros(len(centers), dtype=bool)
    for feature_index, center in enumerate(centers):
        indices = center + offsets
        geometry_valid = (indices >= 0) & (indices < len(rest_flux)) & window_mask[feature_index]
        selected = indices[geometry_valid]
        if not len(selected):
            continue
        measured = rest_mask[selected] > 0
        if measured.mean() < minimum_valid_fraction:
            continue
        values = np.zeros(window_mask.shape[1], dtype=np.float32)
        positions = np.flatnonzero(geometry_valid)[measured]
        values[positions] = rest_flux[selected[measured]]
        measured_windows[feature_index, positions] = True
        windows[feature_index] = _unit_profile(values, measured_windows[feature_index])
        usable[feature_index] = np.linalg.norm(windows[feature_index]) > 0
    return windows, measured_windows, usable


def _unit_profile(values: np.ndarray, active: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float32)
    selected = np.asarray(values[active], dtype=np.float32)
    if not len(selected):
        return result
    selected = selected - selected.mean()
    norm = float(np.linalg.norm(selected))
    if norm <= 1e-8:
        return result
    result[active] = selected / norm
    return result


def _random_anchor_wavelengths(
    rest_grid: np.ndarray,
    reference_wavelengths: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Draw one control anchor from each named anchor's local wavelength interval.

    Sampling the full grid uniformly changes detector coverage as well as feature
    identity.  Local intervals preserve the broad wavelength distribution, so
    this control asks whether the exact named positions matter.
    """
    rng = np.random.default_rng(seed)
    reference = np.asarray(reference_wavelengths, dtype=np.float64)
    log_reference = np.log(reference)
    midpoints = np.exp(0.5 * (log_reference[:-1] + log_reference[1:]))
    first_edge = np.exp(log_reference[0] - 0.5 * (log_reference[1] - log_reference[0]))
    last_edge = np.exp(log_reference[-1] + 0.5 * (log_reference[-1] - log_reference[-2]))
    lower_limits = np.concatenate(([max(first_edge, rest_grid[8])], midpoints))
    upper_limits = np.concatenate((midpoints, [min(last_edge, rest_grid[-9])]))
    selected = []
    for lower, upper in zip(lower_limits, upper_limits):
        candidates = rest_grid[(rest_grid >= lower) & (rest_grid < upper)]
        if not len(candidates):
            raise ValueError("Rest grid is too coarse for stratified random ONIR anchors")
        selected.append(float(rng.choice(candidates)))
    return np.asarray(selected, dtype=np.float32)


def _audit_report(
    config: dict[str, Any],
    output_path: Path,
    names: tuple[str, ...],
    wavelengths: np.ndarray,
    centers: np.ndarray,
    window_mask: np.ndarray,
    geometry_weight: np.ndarray,
    support: np.ndarray,
    available_windows: np.ndarray,
    rest_grid: np.ndarray,
    source_flux: str,
    bank_view: str,
    bank_input_mode: str,
) -> dict[str, Any]:
    radius = window_mask.shape[1] // 2
    feature_sets = []
    for center, active in zip(centers, window_mask):
        indices = center + np.arange(-radius, radius + 1)
        feature_sets.append(set(indices[active & (indices >= 0) & (indices < len(rest_grid))]))
    adjacent_overlap = []
    for left, right in zip(feature_sets[:-1], feature_sets[1:]):
        denominator = max(1, min(len(left), len(right)))
        adjacent_overlap.append(len(left & right) / denominator)
    redshift_grid = build_redshift_grid(
        float(config["model"]["redshift_min"]),
        float(config["model"]["redshift_max"]),
        int(config["model"]["redshift_bins"]),
        str(config["model"].get("redshift_spacing", "linear")),
    )
    observed_min = float(config["observation"]["wavelength_min"])
    observed_max = float(config["observation"]["wavelength_max"])
    coverage = []
    for redshift in redshift_grid:
        observed = wavelengths * (1.0 + redshift)
        center_visible = (observed >= observed_min) & (observed <= observed_max)
        coverage.append(int(center_visible.sum()))
    return {
        "format_version": "strider-onir-audit-v1",
        "bank_path": str(output_path),
        "source_split": "train",
        "source_flux": source_flux,
        "bank_view": bank_view,
        "bank_input_mode": bank_input_mode,
        "classes": list(config["model"]["classes"]),
        "features": list(names),
        "rest_wavelengths": wavelengths.tolist(),
        "support_counts": support.tolist(),
        "available_window_counts_before_sampling": available_windows.tolist(),
        "retained_window_limit_per_cell": int(
            config["onir"].get("maximum_windows_per_cell", 5_000)
        ),
        "unsupported_cells": int((support == 0).sum()),
        "unsupported_cells_by_class": {
            str(name): int((support[index] == 0).sum())
            for index, name in enumerate(config["model"]["classes"])
        },
        "single_window_cells": int((support == 1).sum()),
        "profile_masks_match_geometry": True,
        "mean_adjacent_overlap": float(np.mean(adjacent_overlap)),
        "maximum_adjacent_overlap": float(np.max(adjacent_overlap)),
        "overlap_weights": geometry_weight.tolist(),
        "visible_feature_centers_by_redshift": coverage,
        "redshift_grid": redshift_grid.tolist(),
        "silent_phase_filling": False,
        "uses_target_class_masks": False,
    }


def _assert_geometric_profile_masks(
    mean_masks: np.ndarray,
    medoid_masks: np.ndarray,
    support: np.ndarray,
    prototype_masks: np.ndarray,
    prototype_support: np.ndarray,
    geometry: np.ndarray,
) -> None:
    expected_mean = np.broadcast_to(geometry[None, :, :], mean_masks.shape)
    mean_active = support > 0
    if not np.array_equal(mean_masks[mean_active], expected_mean[mean_active]):
        raise RuntimeError("ONIR mean-profile masks differ from feature geometry")
    if not np.array_equal(medoid_masks[mean_active], expected_mean[mean_active]):
        raise RuntimeError("ONIR medoid-profile masks differ from feature geometry")

    expected_prototype = np.broadcast_to(
        geometry[None, :, None, :], prototype_masks.shape
    )
    prototype_active = prototype_support > 0
    if not np.array_equal(
        prototype_masks[prototype_active],
        expected_prototype[prototype_active],
    ):
        raise RuntimeError("ONIR prototype masks differ from feature geometry")
