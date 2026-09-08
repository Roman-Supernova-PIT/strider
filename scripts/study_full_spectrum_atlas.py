#!/usr/bin/env python3
"""Run the bounded STRIDER full-spectrum atlas feasibility study.

Only the prepared ``train`` role builds reference profiles and only the
``selection`` role compares study designs.  The calibration and test roles are
not accepted by this script. Scores measure relative reference-profile
agreement, not calibrated probabilities.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from strider.atlas.build import align_to_rest_grid
from strider.atlas.full_spectrum import (
    CandidateRestFrameScan,
    FullSpectrumAtlas,
    PhaseIndexedAtlas,
    best_measured_visit,
    build_full_spectrum_atlas,
    build_phase_indexed_atlas,
    combine_view_scores,
    measurement_faithful_coadd,
    phase_sequence_match,
    score_atlas_view,
    score_phase_atlas_view,
)
from strider.config import (
    load_config,
    project_path,
    resolved_config_sha256,
)
from strider.data.dataset import SundialDataset, log_wavelength_grid
from strider.model.redshift_scan import build_redshift_grid
from strider.model.spectral_tokens import (
    MaskAwareContinuumRemoval,
    velocity_sigma_to_log_bins,
)


FORMAT_VERSION = "strider-full-spectrum-atlas-study-v1"
DEFAULT_REDSHIFT_EDGES = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.01)
DEFAULT_PHASE_EDGES = (-20.0, 0.0, 20.0, 40.0, 80.0)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/local_ia_binary_20k.yaml",
        help="Prepared-data configuration. Train and selection roles are used.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/atlas_study/local_binary",
        help="Study output directory.",
    )
    parser.add_argument(
        "--train-objects",
        type=int,
        default=800,
        help="Deterministic class/redshift-stratified training count; 0 uses all.",
    )
    parser.add_argument(
        "--selection-objects",
        type=int,
        default=240,
        help="Deterministic class/redshift-stratified selection count; 0 uses all.",
    )
    parser.add_argument("--prototypes-per-class", type=int, default=8)
    parser.add_argument("--phase-profiles-per-cell", type=int, default=4)
    parser.add_argument(
        "--phase-visits",
        type=int,
        default=6,
        help="Best measured visits retained for the phase-sequence comparison.",
    )
    parser.add_argument("--phase-starting-points", type=int, default=17)
    parser.add_argument(
        "--phase-sequence-fraction",
        type=float,
        default=0.25,
        help="Fixed phase-sequence share in the coadd-plus-sequence comparison.",
    )
    parser.add_argument("--redshift-bins", type=int, default=161)
    parser.add_argument(
        "--max-visits",
        default="32",
        help="Positive visit count or 'all'. The same rule is used in both roles.",
    )
    parser.add_argument("--continuum-width-km-s", type=float, default=12_000.0)
    parser.add_argument(
        "--continuum-removed-fraction",
        type=float,
        default=0.5,
        help=(
            "Fixed share assigned to the continuum-removed spectrum when the "
            "two spectral views are combined."
        ),
    )
    parser.add_argument("--minimum-rest-fraction", type=float, default=0.15)
    parser.add_argument("--minimum-shared-fraction", type=float, default=0.8)
    parser.add_argument("--prototype-temperature", type=float, default=0.05)
    parser.add_argument("--maximum-relative-coadd-error", type=float, default=3.0)
    parser.add_argument("--edge-trim-fraction", type=float, default=0.05)
    parser.add_argument(
        "--phase-edges",
        default=",".join(str(value) for value in DEFAULT_PHASE_EDGES),
        help="Rest-frame phase edges for the training-support audit.",
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    if tuple(str(value) for value in config["model"]["classes"]) != (
        "Ia",
        "other",
    ):
        raise ValueError(
            "The first full-spectrum atlas study is intentionally restricted "
            "to the normal-Ia binary task"
        )
    max_visits: int | str
    if str(arguments.max_visits).lower() == "all":
        max_visits = "all"
    else:
        max_visits = int(arguments.max_visits)
        if max_visits < 1:
            raise ValueError("--max-visits must be positive or 'all'")
    config["data"]["max_visits"] = max_visits
    config["data"]["include_flux_error_channel"] = True
    config["data"]["include_clean_flux_target"] = True
    # Sampling is performed explicitly below so this study never inherits an
    # unrelated development cap from a configuration.
    config["data"]["runtime_object_limits"] = {}

    seed = int(config["project"]["seed"] if arguments.seed is None else arguments.seed)
    output_dir = Path(arguments.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = project_path(config, str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    phase_edges = _comma_separated_floats(arguments.phase_edges)
    if len(phase_edges) < 3 or np.any(np.diff(phase_edges) <= 0.0):
        raise ValueError("--phase-edges must contain at least three increasing values")

    train_dataset = SundialDataset(
        config,
        "train",
        "original",
        training=False,
        pair_no_source=False,
    )
    selection_dataset = SundialDataset(
        config,
        "selection",
        "original",
        training=False,
        pair_no_source=False,
    )
    train_indices = _stratified_indices(
        train_dataset.objects,
        arguments.train_objects,
        DEFAULT_REDSHIFT_EDGES,
        seed,
    )
    selection_indices = _stratified_indices(
        selection_dataset.objects,
        arguments.selection_objects,
        DEFAULT_REDSHIFT_EDGES,
        seed + 1,
    )

    observed_grid = train_dataset.output_wavelength.astype(np.float32)
    rest_grid = log_wavelength_grid(
        config["model"]["rest_wavelength_min"],
        config["model"]["rest_wavelength_max"],
        config["model"]["rest_wavelength_bins"],
    ).astype(np.float32)
    redshift_grid = build_redshift_grid(
        float(config["model"]["redshift_min"]),
        float(config["model"]["redshift_max"]),
        int(arguments.redshift_bins),
        str(config["model"]["redshift_spacing"]),
    )
    continuum_sigma_bins = velocity_sigma_to_log_bins(
        float(observed_grid[0]),
        float(observed_grid[-1]),
        len(observed_grid),
        float(arguments.continuum_width_km_s),
    )
    continuum = MaskAwareContinuumRemoval(continuum_sigma_bins).eval()

    print(
        "Building clean training references from "
        f"{len(train_indices):,} train objects"
    )
    training = _collect_training_profiles(
        train_dataset,
        train_indices,
        observed_grid,
        rest_grid,
        continuum,
        phase_edges,
        maximum_relative_error=float(arguments.maximum_relative_coadd_error),
        edge_trim_fraction=float(arguments.edge_trim_fraction),
    )
    class_names = tuple(str(value) for value in config["model"]["classes"])
    coadd_atlas = build_full_spectrum_atlas(
        training["coadd_whole"],
        training["coadd_mask"],
        training["coadd_detail"],
        training["coadd_mask"],
        training["class_index"],
        class_names,
        rest_grid,
        int(arguments.prototypes_per_class),
    )
    best_atlas = build_full_spectrum_atlas(
        training["best_whole"],
        training["best_mask"],
        training["best_detail"],
        training["best_mask"],
        training["class_index"],
        class_names,
        rest_grid,
        int(arguments.prototypes_per_class),
    )
    phase_atlas = build_phase_indexed_atlas(
        training["phase_whole"],
        training["phase_mask"],
        training["phase_detail"],
        training["phase_mask"],
        training["phase_class_index"],
        training["phase_index"],
        class_names,
        rest_grid,
        phase_edges,
        int(arguments.phase_profiles_per_cell),
    )
    coadd_atlas.save(output_dir / "coadd_atlas.npz")
    best_atlas.save(output_dir / "best_spectrum_atlas.npz")
    phase_atlas.save(output_dir / "phase_indexed_atlas.npz")
    training["objects"].to_csv(output_dir / "training_objects.csv", index=False)
    training["phase_support"].to_csv(
        output_dir / "training_phase_support.csv", index=False
    )

    print(
        "Scanning candidate redshifts for "
        f"{len(selection_indices):,} selection objects"
    )
    scan = CandidateRestFrameScan(observed_grid, rest_grid, redshift_grid)
    prediction_rows, selection_phase = _evaluate_selection(
        selection_dataset,
        selection_indices,
        scan,
        coadd_atlas,
        best_atlas,
        phase_atlas,
        continuum,
        phase_edges,
        detail_fraction=float(arguments.continuum_removed_fraction),
        minimum_rest_fraction=float(arguments.minimum_rest_fraction),
        minimum_shared_fraction=float(arguments.minimum_shared_fraction),
        prototype_temperature=float(arguments.prototype_temperature),
        maximum_relative_error=float(arguments.maximum_relative_coadd_error),
        edge_trim_fraction=float(arguments.edge_trim_fraction),
        phase_visits=int(arguments.phase_visits),
        phase_starting_points=int(arguments.phase_starting_points),
        phase_sequence_fraction=float(arguments.phase_sequence_fraction),
    )
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(output_dir / "selection_predictions.csv", index=False)
    selection_phase.to_csv(output_dir / "selection_phase_support.csv", index=False)

    summary = _summarize_predictions(predictions, class_names)
    summary.to_csv(output_dir / "summary.csv", index=False)
    by_redshift = _summarize_slices(
        predictions,
        class_names,
        column="true_redshift",
        edges=DEFAULT_REDSHIFT_EDGES,
        label="redshift_range",
    )
    by_redshift.to_csv(output_dir / "summary_by_redshift.csv", index=False)
    by_signal_to_noise = _summarize_signal_to_noise(predictions, class_names)
    by_signal_to_noise.to_csv(
        output_dir / "summary_by_signal_to_noise.csv", index=False
    )

    report = {
        "format_version": FORMAT_VERSION,
        "purpose": (
            "architecture selection on train/selection roles; reference-match "
            "scores are not calibrated probabilities"
        ),
        "config": str(Path(arguments.config).expanduser().resolve()),
        "config_sha256_after_study_overrides": resolved_config_sha256(config),
        "prepared_data": str(project_path(config, config["data"]["prepared_dir"])),
        "roles_used": ["train", "selection"],
        "roles_not_used": ["calibration", "test"],
        "train_objects": int(len(train_indices)),
        "selection_objects": int(len(selection_indices)),
        "max_visits": max_visits,
        "class_names": list(class_names),
        "redshift_grid": {
            "minimum": float(redshift_grid[0]),
            "maximum": float(redshift_grid[-1]),
            "bins": int(len(redshift_grid)),
            "spacing": str(config["model"]["redshift_spacing"]),
        },
        "rest_wavelength_angstrom": [float(rest_grid[0]), float(rest_grid[-1])],
        "observed_wavelength_angstrom": [
            float(observed_grid[0]),
            float(observed_grid[-1]),
        ],
        "prototypes_per_class": int(arguments.prototypes_per_class),
        "phase_profiles_per_class_and_range": int(
            arguments.phase_profiles_per_cell
        ),
        "phase_sequence": {
            "best_measured_visits": int(arguments.phase_visits),
            "possible_starting_phases": int(arguments.phase_starting_points),
            "fixed_share_when_combined_with_coadd": float(
                arguments.phase_sequence_fraction
            ),
            "truth_informed_upper_bound_is_deployable": False,
        },
        "continuum_width_km_s": float(arguments.continuum_width_km_s),
        "spectral_view_combination": {
            "full_spectrum_fraction": float(
                1.0 - arguments.continuum_removed_fraction
            ),
            "continuum_removed_fraction": float(
                arguments.continuum_removed_fraction
            ),
            "learned": False,
        },
        "coadd": {
            "method": "inverse variance using the reported error for each measurement",
            "maximum_relative_error": float(arguments.maximum_relative_coadd_error),
            "edge_trim_fraction_per_side": float(arguments.edge_trim_fraction),
        },
        "phase_support_edges_days": phase_edges.tolist(),
        "phase_use": (
            "training truth is used only to audit whether a later phase-indexed "
            "atlas has enough examples; this run does not supply phase to selection scoring"
        ),
        "outputs": {
            "selection_predictions": "selection_predictions.csv",
            "summary": "summary.csv",
            "summary_by_redshift": "summary_by_redshift.csv",
            "summary_by_signal_to_noise": "summary_by_signal_to_noise.csv",
            "training_phase_support": "training_phase_support.csv",
            "selection_phase_support": "selection_phase_support.csv",
            "coadd_atlas": "coadd_atlas.npz",
            "best_spectrum_atlas": "best_spectrum_atlas.npz",
            "phase_indexed_atlas": "phase_indexed_atlas.npz",
        },
    }
    with (output_dir / "study_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\nFull-spectrum atlas feasibility summary")
    _print_summary(summary)
    print(f"\nResults: {output_dir}")


def _comma_separated_floats(value: str) -> np.ndarray:
    return np.asarray([float(part.strip()) for part in value.split(",")], dtype=np.float32)


def _stratified_indices(
    objects: pd.DataFrame,
    limit: int,
    redshift_edges: tuple[float, ...],
    seed: int,
) -> np.ndarray:
    if limit < 0:
        raise ValueError("Object limits cannot be negative")
    if limit == 0 or limit >= len(objects):
        return np.arange(len(objects), dtype=np.int64)
    frame = objects[["class_index", "redshift"]].copy()
    frame["position"] = np.arange(len(frame), dtype=np.int64)
    frame["redshift_group"] = pd.cut(
        frame["redshift"],
        bins=np.asarray(redshift_edges),
        include_lowest=True,
        labels=False,
    ).fillna(-1).astype(int)
    rng = np.random.default_rng(seed)
    groups: list[list[int]] = []
    for _, values in frame.groupby(["class_index", "redshift_group"], sort=True):
        positions = values["position"].to_numpy(dtype=np.int64)
        rng.shuffle(positions)
        groups.append(positions.tolist())
    selected: list[int] = []
    offset = 0
    while len(selected) < limit:
        added = False
        for group in groups:
            if offset < len(group):
                selected.append(group[offset])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        offset += 1
    return np.asarray(sorted(selected), dtype=np.int64)


def _continuum_detail(
    flux: np.ndarray,
    mask: np.ndarray,
    continuum: MaskAwareContinuumRemoval,
) -> np.ndarray:
    with torch.no_grad():
        detail = continuum(
            torch.from_numpy(flux)[None, None, :],
            torch.from_numpy(mask.astype(np.float32))[None, None, :],
        )[0, 0]
    return detail.numpy().astype(np.float32)


def _coadd_signal_to_noise(
    flux: np.ndarray,
    error: np.ndarray,
    mask: np.ndarray,
) -> float:
    valid = mask & np.isfinite(error) & (error > 0.0)
    return float(np.median(flux[valid] / error[valid])) if np.any(valid) else float("nan")


def _visit_signal_to_noise(
    item: dict[str, torch.Tensor],
    edge_trim_fraction: float,
) -> np.ndarray:
    flux = item["flux"].numpy().astype(np.float32)
    mask = item["wavelength_mask"].numpy().astype(bool)
    error = np.exp(item["flux_error_shape"].numpy()).astype(np.float32)
    coordinate = np.linspace(0.0, 1.0, flux.shape[-1], dtype=np.float32)
    edge = (coordinate >= edge_trim_fraction) & (
        coordinate <= 1.0 - edge_trim_fraction
    )
    result = np.full(flux.shape[0], -np.inf, dtype=np.float32)
    for visit in range(flux.shape[0]):
        valid = mask[visit] & edge & np.isfinite(error[visit]) & (error[visit] > 0.0)
        if np.any(valid):
            result[visit] = float(np.median(flux[visit, valid] / error[visit, valid]))
    return result


def _phase_rows(
    item: dict[str, torch.Tensor],
    role: str,
    phase_edges: np.ndarray,
) -> list[dict[str, Any]]:
    phase = item["simulation_rest_phase_days"].numpy()
    labels = pd.cut(
        phase,
        bins=phase_edges,
        include_lowest=True,
        right=False,
    )
    counts = pd.Series(labels).value_counts(sort=False)
    rows: list[dict[str, Any]] = []
    for interval, count in counts.items():
        rows.append(
            {
                "role": role,
                "snid": int(item["snid"]),
                "class_index": int(item["class_index"]),
                "phase_range_days": str(interval),
                "visit_count": int(count),
            }
        )
    return rows


def _collect_training_profiles(
    dataset: SundialDataset,
    indices: np.ndarray,
    observed_grid: np.ndarray,
    rest_grid: np.ndarray,
    continuum: MaskAwareContinuumRemoval,
    phase_edges: np.ndarray,
    *,
    maximum_relative_error: float,
    edge_trim_fraction: float,
) -> dict[str, Any]:
    values: dict[str, list[np.ndarray]] = defaultdict(list)
    labels: list[int] = []
    object_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    phase_class_index: list[int] = []
    phase_index: list[int] = []
    for number, index in enumerate(indices, start=1):
        item = dataset[int(index)]
        clean = item["clean_flux_target"]
        coadd, _, coadd_mask = measurement_faithful_coadd(
            clean,
            item["wavelength_mask"],
            item["flux_error_shape"],
            item["visit_flux_scale"],
            maximum_relative_error=maximum_relative_error,
            edge_trim_fraction=edge_trim_fraction,
        )
        observed_best, _, _, _, best_snr = best_measured_visit(
            item["flux"],
            item["wavelength_mask"],
            item["flux_error_shape"],
            edge_trim_fraction=edge_trim_fraction,
        )
        best_clean = clean[observed_best].numpy().astype(np.float32)
        best_mask = item["wavelength_mask"][observed_best].numpy().astype(bool)
        coordinate = np.linspace(0.0, 1.0, len(best_mask), dtype=np.float32)
        best_mask &= (coordinate >= edge_trim_fraction) & (
            coordinate <= 1.0 - edge_trim_fraction
        )
        best_clean = np.where(best_mask, best_clean, 0.0).astype(np.float32)

        redshift = float(item["redshift"])
        coadd_detail = _continuum_detail(coadd, coadd_mask, continuum)
        best_detail = _continuum_detail(best_clean, best_mask, continuum)
        for name, spectrum, mask in (
            ("coadd_whole", coadd, coadd_mask),
            ("coadd_detail", coadd_detail, coadd_mask),
            ("best_whole", best_clean, best_mask),
            ("best_detail", best_detail, best_mask),
        ):
            aligned, aligned_mask = align_to_rest_grid(
                observed_grid,
                spectrum,
                mask.astype(np.float32),
                redshift,
                rest_grid,
            )
            values[name].append(aligned)
            values[name.replace("whole", "mask").replace("detail", "mask")].append(
                aligned_mask.astype(bool)
            )
        labels.append(int(item["class_index"]))
        visit_snr = _visit_signal_to_noise(item, edge_trim_fraction)
        visit_phase = item["simulation_rest_phase_days"].numpy()
        visit_phase_index = np.searchsorted(
            phase_edges, visit_phase, side="right"
        ) - 1
        for phase_value in range(len(phase_edges) - 1):
            candidates = np.flatnonzero(visit_phase_index == phase_value)
            if not len(candidates):
                continue
            selected_visit = int(candidates[np.argmax(visit_snr[candidates])])
            phase_whole = clean[selected_visit].numpy().astype(np.float32)
            phase_mask = item["wavelength_mask"][selected_visit].numpy().astype(bool)
            phase_mask &= (coordinate >= edge_trim_fraction) & (
                coordinate <= 1.0 - edge_trim_fraction
            )
            phase_whole = np.where(phase_mask, phase_whole, 0.0).astype(np.float32)
            phase_detail = _continuum_detail(phase_whole, phase_mask, continuum)
            aligned_whole, aligned_mask = align_to_rest_grid(
                observed_grid,
                phase_whole,
                phase_mask.astype(np.float32),
                redshift,
                rest_grid,
            )
            aligned_detail, _ = align_to_rest_grid(
                observed_grid,
                phase_detail,
                phase_mask.astype(np.float32),
                redshift,
                rest_grid,
            )
            values["phase_whole"].append(aligned_whole)
            values["phase_detail"].append(aligned_detail)
            values["phase_mask"].append(aligned_mask.astype(bool))
            phase_class_index.append(int(item["class_index"]))
            phase_index.append(phase_value)
        object_rows.append(
            {
                "snid": int(item["snid"]),
                "class_index": int(item["class_index"]),
                "redshift": redshift,
                "visits": int(item["flux"].shape[0]),
                "best_measured_visit_index": int(observed_best),
                "best_measured_snr": best_snr,
            }
        )
        phase_rows.extend(_phase_rows(item, "train", phase_edges))
        if number % 100 == 0 or number == len(indices):
            print(f"  prepared {number:,}/{len(indices):,} training objects")

    # Mask keys were appended twice (whole and detail); both views share the
    # same measurement geometry, so retain one copy per object.
    return {
        "coadd_whole": np.stack(values["coadd_whole"]),
        "coadd_detail": np.stack(values["coadd_detail"]),
        "coadd_mask": np.stack(values["coadd_mask"])[::2],
        "best_whole": np.stack(values["best_whole"]),
        "best_detail": np.stack(values["best_detail"]),
        "best_mask": np.stack(values["best_mask"])[::2],
        "class_index": np.asarray(labels, dtype=np.int64),
        "phase_whole": np.stack(values["phase_whole"]),
        "phase_detail": np.stack(values["phase_detail"]),
        "phase_mask": np.stack(values["phase_mask"]),
        "phase_class_index": np.asarray(phase_class_index, dtype=np.int64),
        "phase_index": np.asarray(phase_index, dtype=np.int64),
        "objects": pd.DataFrame(object_rows),
        "phase_support": _summarize_phase_rows(phase_rows),
    }


def _evaluate_selection(
    dataset: SundialDataset,
    indices: np.ndarray,
    scan: CandidateRestFrameScan,
    coadd_atlas: FullSpectrumAtlas,
    best_atlas: FullSpectrumAtlas,
    phase_atlas: PhaseIndexedAtlas,
    continuum: MaskAwareContinuumRemoval,
    phase_edges: np.ndarray,
    *,
    detail_fraction: float,
    minimum_rest_fraction: float,
    minimum_shared_fraction: float,
    prototype_temperature: float,
    maximum_relative_error: float,
    edge_trim_fraction: float,
    phase_visits: int,
    phase_starting_points: int,
    phase_sequence_fraction: float,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    if phase_visits < 2:
        raise ValueError("The phase-sequence study requires at least two visits")
    if phase_starting_points < 1:
        raise ValueError("The phase-sequence study requires a starting-phase grid")
    if not 0.0 <= phase_sequence_fraction <= 1.0:
        raise ValueError("Phase-sequence fraction must lie in [0, 1]")
    rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    for number, index in enumerate(indices, start=1):
        item = dataset[int(index)]
        coadd, coadd_error, coadd_mask = measurement_faithful_coadd(
            item["flux"],
            item["wavelength_mask"],
            item["flux_error_shape"],
            item["visit_flux_scale"],
            maximum_relative_error=maximum_relative_error,
            edge_trim_fraction=edge_trim_fraction,
        )
        best_index, best, _, best_mask, best_snr = best_measured_visit(
            item["flux"],
            item["wavelength_mask"],
            item["flux_error_shape"],
            edge_trim_fraction=edge_trim_fraction,
        )
        coadd_snr = _coadd_signal_to_noise(coadd, coadd_error, coadd_mask)
        object_information = {
            "snid": int(item["snid"]),
            "true_class_index": int(item["class_index"]),
            "true_class": coadd_atlas.class_names[int(item["class_index"])],
            "true_redshift": float(item["redshift"]),
            "visits": int(item["flux"].shape[0]),
            "coadded_measured_snr": coadd_snr,
            "best_spectrum_measured_snr": best_snr,
            "best_spectrum_visit_index": best_index,
            "phase_sequence_visits": min(phase_visits, int(item["flux"].shape[0])),
        }
        coadd_combined_score: np.ndarray | None = None
        coadd_combined_support: np.ndarray | None = None
        for prefix, spectrum, mask, atlas in (
            ("coadd", coadd, coadd_mask, coadd_atlas),
            ("best_spectrum", best, best_mask, best_atlas),
        ):
            detail = _continuum_detail(spectrum, mask, continuum)
            whole_candidates, candidate_mask = scan.align(spectrum, mask)
            detail_candidates, detail_candidate_mask = scan.align(detail, mask)
            whole_score, whole_support = score_atlas_view(
                whole_candidates,
                candidate_mask,
                atlas.whole_profiles,
                atlas.whole_masks,
                atlas.support_counts,
                minimum_rest_fraction=minimum_rest_fraction,
                minimum_shared_fraction=minimum_shared_fraction,
                prototype_temperature=prototype_temperature,
            )
            detail_score, detail_support = score_atlas_view(
                detail_candidates,
                detail_candidate_mask,
                atlas.detail_profiles,
                atlas.detail_masks,
                atlas.support_counts,
                minimum_rest_fraction=minimum_rest_fraction,
                minimum_shared_fraction=minimum_shared_fraction,
                prototype_temperature=prototype_temperature,
            )
            combined_score, combined_support = combine_view_scores(
                whole_score,
                whole_support,
                detail_score,
                detail_support,
                detail_fraction=detail_fraction,
            )
            if prefix == "coadd":
                coadd_combined_score = combined_score
                coadd_combined_support = combined_support
            for view, score, support in (
                ("full_spectrum", whole_score, whole_support),
                ("continuum_removed", detail_score, detail_support),
                ("combined_spectral_view", combined_score, combined_support),
            ):
                rows.append(
                    _prediction_record(
                        object_information,
                        f"{prefix}_{view}",
                        score,
                        support,
                        scan.redshift_grid,
                        atlas.class_names,
                    )
                )
        if coadd_combined_score is None or coadd_combined_support is None:
            raise RuntimeError("Coadd reference match was not constructed")
        phase_scores = _phase_scores_for_object(
            item,
            scan,
            phase_atlas,
            continuum,
            detail_fraction=detail_fraction,
            minimum_rest_fraction=minimum_rest_fraction,
            minimum_shared_fraction=minimum_shared_fraction,
            prototype_temperature=prototype_temperature,
            edge_trim_fraction=edge_trim_fraction,
            maximum_visits=phase_visits,
            starting_phase_points=phase_starting_points,
        )
        for phase_name, (phase_score, phase_support) in phase_scores.items():
            rows.append(
                _prediction_record(
                    object_information,
                    f"visit_sequence_{phase_name}",
                    phase_score,
                    phase_support,
                    scan.redshift_grid,
                    phase_atlas.class_names,
                )
            )
            combined_score, combined_support = combine_view_scores(
                coadd_combined_score,
                coadd_combined_support,
                phase_score,
                phase_support,
                detail_fraction=phase_sequence_fraction,
            )
            rows.append(
                _prediction_record(
                    object_information,
                    f"coadd_plus_{phase_name}",
                    combined_score,
                    combined_support,
                    scan.redshift_grid,
                    phase_atlas.class_names,
                )
            )
        phase_rows.extend(_phase_rows(item, "selection", phase_edges))
        if number % 50 == 0 or number == len(indices):
            print(f"  scanned {number:,}/{len(indices):,} selection objects")
    return rows, _summarize_phase_rows(phase_rows)


def _phase_scores_for_object(
    item: dict[str, torch.Tensor],
    scan: CandidateRestFrameScan,
    atlas: PhaseIndexedAtlas,
    continuum: MaskAwareContinuumRemoval,
    *,
    detail_fraction: float,
    minimum_rest_fraction: float,
    minimum_shared_fraction: float,
    prototype_temperature: float,
    edge_trim_fraction: float,
    maximum_visits: int,
    starting_phase_points: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    visit_snr = _visit_signal_to_noise(item, edge_trim_fraction)
    finite = np.flatnonzero(np.isfinite(visit_snr))
    if len(finite) < 2:
        shape = (len(scan.redshift_grid), len(atlas.class_names))
        unsupported = np.full(shape, -np.inf, dtype=np.float32)
        mask = np.zeros(shape, dtype=bool)
        return {
            "phase_independent": (unsupported, mask),
            "phase_averaged": (unsupported.copy(), mask.copy()),
            "truth_informed_phase_upper_bound": (
                unsupported.copy(),
                mask.copy(),
            ),
        }
    order = finite[np.argsort(visit_snr[finite])[::-1]]
    selected = np.sort(order[:maximum_visits])
    coordinate = np.linspace(0.0, 1.0, item["flux"].shape[-1], dtype=np.float32)
    edge = (coordinate >= edge_trim_fraction) & (
        coordinate <= 1.0 - edge_trim_fraction
    )
    whole_candidates = []
    whole_candidate_masks = []
    detail_candidates = []
    detail_candidate_masks = []
    for visit in selected:
        spectrum = item["flux"][int(visit)].numpy().astype(np.float32)
        mask = item["wavelength_mask"][int(visit)].numpy().astype(bool) & edge
        spectrum = np.where(mask, spectrum, 0.0).astype(np.float32)
        detail = _continuum_detail(spectrum, mask, continuum)
        aligned_whole, aligned_mask = scan.align(spectrum, mask)
        aligned_detail, aligned_detail_mask = scan.align(detail, mask)
        whole_candidates.append(aligned_whole)
        whole_candidate_masks.append(aligned_mask)
        detail_candidates.append(aligned_detail)
        detail_candidate_masks.append(aligned_detail_mask)

    settings = {
        "minimum_rest_fraction": minimum_rest_fraction,
        "minimum_shared_fraction": minimum_shared_fraction,
        "prototype_temperature": prototype_temperature,
    }
    whole_score, whole_support = score_phase_atlas_view(
        np.stack(whole_candidates),
        np.stack(whole_candidate_masks),
        atlas.whole_profiles,
        atlas.whole_masks,
        atlas.support_counts,
        **settings,
    )
    detail_score, detail_support = score_phase_atlas_view(
        np.stack(detail_candidates),
        np.stack(detail_candidate_masks),
        atlas.detail_profiles,
        atlas.detail_masks,
        atlas.support_counts,
        **settings,
    )
    visit_score, visit_support = combine_view_scores(
        whole_score,
        whole_support,
        detail_score,
        detail_support,
        detail_fraction=detail_fraction,
    )
    observer_days = item["observer_days"][selected].numpy()
    possible_start = np.linspace(
        float(atlas.phase_edges_days[0]),
        float(atlas.phase_edges_days[-2]),
        starting_phase_points,
        dtype=np.float32,
    )
    phase_independent = phase_sequence_match(
        visit_score,
        visit_support,
        observer_days,
        scan.redshift_grid,
        atlas.phase_edges_days,
        mode="phase_independent",
    )
    phase_averaged = phase_sequence_match(
        visit_score,
        visit_support,
        observer_days,
        scan.redshift_grid,
        atlas.phase_edges_days,
        mode="phase_averaged",
        starting_phase_grid=possible_start,
    )
    truth_phase = item["simulation_rest_phase_days"][selected].numpy()
    earliest = int(np.argmin(observer_days))
    truth_upper_bound = phase_sequence_match(
        visit_score,
        visit_support,
        observer_days,
        scan.redshift_grid,
        atlas.phase_edges_days,
        mode="truth_phase_upper_bound",
        truth_starting_phase=float(truth_phase[earliest]),
    )
    return {
        "phase_independent": phase_independent,
        "phase_averaged": phase_averaged,
        "truth_informed_phase_upper_bound": truth_upper_bound,
    }


def _prediction_record(
    object_information: dict[str, Any],
    arm: str,
    score: np.ndarray,
    support: np.ndarray,
    redshift_grid: np.ndarray,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    finite = support & np.isfinite(score)
    record = {**object_information, "study_arm": arm}
    if not np.any(finite):
        return {
            **record,
            "prediction_supported": False,
            "predicted_class_index": -1,
            "predicted_class": "not enough wavelength information",
            "predicted_redshift": float("nan"),
            "class_correct": False,
            "delta_redshift": float("nan"),
            "absolute_delta_redshift": float("nan"),
            "redshift_outlier_gt_0p1": False,
            "selected_reference_match": float("nan"),
            "truth_reference_match": float("nan"),
            "truth_match_advantage": float("nan"),
            "true_class_redshift": float("nan"),
        }
    safe_score = np.where(finite, score, -np.inf)
    redshift_index, predicted_class = np.unravel_index(
        int(np.argmax(safe_score)), safe_score.shape
    )
    predicted_redshift = float(redshift_grid[redshift_index])
    true_class = int(record["true_class_index"])
    true_redshift = float(record["true_redshift"])
    truth_index = int(np.argmin(np.abs(redshift_grid - true_redshift)))
    truth_score = float(safe_score[truth_index, true_class])
    competing = finite.copy()
    near_truth_same_class = (
        np.abs(redshift_grid - true_redshift) <= 0.10
    )[:, None] & (np.arange(len(class_names))[None, :] == true_class)
    competing &= ~near_truth_same_class
    competing_score = (
        float(np.max(safe_score[competing])) if np.any(competing) else float("nan")
    )
    true_class_column = safe_score[:, true_class]
    true_class_redshift = float(
        redshift_grid[int(np.argmax(true_class_column))]
    )
    delta = predicted_redshift - true_redshift
    return {
        **record,
        "prediction_supported": True,
        "predicted_class_index": int(predicted_class),
        "predicted_class": class_names[predicted_class],
        "predicted_redshift": predicted_redshift,
        "class_correct": bool(predicted_class == true_class),
        "delta_redshift": delta,
        "absolute_delta_redshift": abs(delta),
        "redshift_outlier_gt_0p1": bool(abs(delta) > 0.1),
        "selected_reference_match": float(safe_score[redshift_index, predicted_class]),
        "truth_reference_match": truth_score,
        "truth_match_advantage": truth_score - competing_score,
        "true_class_redshift": true_class_redshift,
    }


def _summarize_phase_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    by_object = frame.assign(present=frame["visit_count"] > 0)
    return (
        by_object.groupby(
            ["role", "class_index", "phase_range_days"],
            observed=True,
            sort=True,
        )
        .agg(
            objects_with_visits=("present", "sum"),
            visits=("visit_count", "sum"),
        )
        .reset_index()
    )


def _binary_metrics(
    frame: pd.DataFrame,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    supported = frame[frame["prediction_supported"].astype(bool)].copy()
    result: dict[str, Any] = {
        "objects": int(len(frame)),
        "supported_objects": int(len(supported)),
        "supported_fraction": float(len(supported) / len(frame)) if len(frame) else float("nan"),
    }
    if supported.empty:
        return result
    truth = supported["true_class_index"].to_numpy(dtype=int)
    prediction = supported["predicted_class_index"].to_numpy(dtype=int)
    recalls: list[float] = []
    f1_values: list[float] = []
    for class_index in range(len(class_names)):
        true_positive = int(np.sum((truth == class_index) & (prediction == class_index)))
        false_positive = int(np.sum((truth != class_index) & (prediction == class_index)))
        false_negative = int(np.sum((truth == class_index) & (prediction != class_index)))
        recall = true_positive / max(true_positive + false_negative, 1)
        precision = true_positive / max(true_positive + false_positive, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
        recalls.append(recall)
        f1_values.append(f1)
        if class_names[class_index] == "Ia":
            result.update(
                {
                    "ia_precision": precision,
                    "ia_recall": recall,
                    "ia_f1": f1,
                }
            )
    result["balanced_accuracy"] = float(np.mean(recalls))
    result["macro_f1"] = float(np.mean(f1_values))
    ia = supported[supported["true_class"] == "Ia"]
    result["ia_objects"] = int(len(ia))
    result["ia_median_absolute_delta_redshift"] = (
        float(ia["absolute_delta_redshift"].median()) if len(ia) else float("nan")
    )
    result["ia_outlier_fraction_gt_0p1"] = (
        float(ia["redshift_outlier_gt_0p1"].mean()) if len(ia) else float("nan")
    )
    result["ia_median_truth_match_advantage"] = (
        float(ia["truth_match_advantage"].median()) if len(ia) else float("nan")
    )
    return result


def _summarize_predictions(
    predictions: pd.DataFrame,
    class_names: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    for arm, frame in predictions.groupby("study_arm", sort=True):
        rows.append({"study_arm": arm, **_binary_metrics(frame, class_names)})
    return pd.DataFrame(rows)


def _summarize_slices(
    predictions: pd.DataFrame,
    class_names: tuple[str, ...],
    *,
    column: str,
    edges: tuple[float, ...],
    label: str,
) -> pd.DataFrame:
    frame = predictions.copy()
    frame[label] = pd.cut(
        frame[column], bins=np.asarray(edges), include_lowest=True, right=False
    ).astype(str)
    rows = []
    for (arm, value), group in frame.groupby(["study_arm", label], sort=True):
        rows.append(
            {"study_arm": arm, label: value, **_binary_metrics(group, class_names)}
        )
    return pd.DataFrame(rows)


def _summarize_signal_to_noise(
    predictions: pd.DataFrame,
    class_names: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    for column, measurement in (
        ("coadded_measured_snr", "coadd"),
        ("best_spectrum_measured_snr", "best_spectrum"),
    ):
        for threshold in (None, 0.5, 1.0, 2.0):
            selected = predictions if threshold is None else predictions[predictions[column] >= threshold]
            for arm, frame in selected.groupby("study_arm", sort=True):
                rows.append(
                    {
                        "signal_to_noise_measurement": measurement,
                        "minimum_signal_to_noise": (
                            "all" if threshold is None else threshold
                        ),
                        "study_arm": arm,
                        **_binary_metrics(frame, class_names),
                    }
                )
    return pd.DataFrame(rows)


def _print_summary(summary: pd.DataFrame) -> None:
    columns = [
        "study_arm",
        "balanced_accuracy",
        "ia_f1",
        "ia_median_absolute_delta_redshift",
        "ia_outlier_fraction_gt_0p1",
    ]
    visible = summary[columns].copy()
    for column in columns[1:]:
        visible[column] = visible[column].map(
            lambda value: "n/a" if not np.isfinite(value) else f"{value:.4f}"
        )
    print(visible.to_string(index=False))


if __name__ == "__main__":
    main()
