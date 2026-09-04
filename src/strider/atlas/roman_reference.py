"""Training-built Roman spectral references for the deployable STRIDER model.

Truth class, redshift, and phase are used only while this bank is constructed
from the training split.  A trained model later receives the saved references
and compares them with observer-frame measurements over candidate redshifts;
none of those simulation labels are runtime inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from strider.config import project_path, resolved_config_sha256
from strider.data.classes import (
    HOURGLASS_15_CLASSES,
    fine_class_name_for_source,
)
from strider.data.dataset import SundialDataset, log_wavelength_grid
from strider.model.coadd import (
    cosine_edge_taper,
    final_inverse_variance_coadd,
    minimum_relative_precision_mask,
    relative_inverse_variance,
)
from strider.model.spectral_tokens import (
    MaskAwareContinuumRemoval,
    velocity_sigma_to_log_bins,
)

from .build import align_to_rest_grid


REFERENCE_BANK_FORMAT = "strider-roman-spectral-reference-v3"


@dataclass(frozen=True)
class RomanReferenceBank:
    """Multiple clean full-spectrum references for class and phase diversity."""

    class_names: tuple[str, ...]
    rest_wavelength: np.ndarray
    phase_edges_days: np.ndarray
    coadd_full_profiles: np.ndarray
    coadd_continuum_removed_profiles: np.ndarray
    coadd_profile_masks: np.ndarray
    coadd_support_counts: np.ndarray
    phase_full_profiles: np.ndarray
    phase_continuum_removed_profiles: np.ndarray
    phase_profile_masks: np.ndarray
    phase_support_counts: np.ndarray
    metadata: dict[str, Any]

    def validate(self) -> None:
        classes = len(self.class_names)
        if self.class_names != HOURGLASS_15_CLASSES:
            raise ValueError(
                "Roman reference classes must use the canonical 15-class order"
            )
        rest = np.asarray(self.rest_wavelength)
        if rest.ndim != 1 or len(rest) < 2 or np.any(np.diff(rest) <= 0.0):
            raise ValueError("Reference rest wavelengths must increase strictly")
        phases = np.asarray(self.phase_edges_days)
        if phases.ndim != 1 or len(phases) < 3 or np.any(np.diff(phases) <= 0.0):
            raise ValueError("Reference phase edges must contain increasing ranges")

        if self.coadd_full_profiles.ndim != 3:
            raise ValueError(
                "Coadd references need class, profile, and wavelength axes"
            )
        coadd_shape = self.coadd_full_profiles.shape
        if coadd_shape[0] != classes or coadd_shape[-1] != len(rest):
            raise ValueError("Coadd reference dimensions are inconsistent")
        for name, values in (
            ("coadd continuum-removed profiles", self.coadd_continuum_removed_profiles),
            ("coadd masks", self.coadd_profile_masks),
        ):
            if values.shape != coadd_shape:
                raise ValueError(f"{name} must have shape {coadd_shape}")
        if self.coadd_support_counts.shape != coadd_shape[:-1]:
            raise ValueError("Coadd reference support counts are inconsistent")

        if self.phase_full_profiles.ndim != 4:
            raise ValueError(
                "Phase references need class, phase, profile, and wavelength axes"
            )
        phase_shape = self.phase_full_profiles.shape
        expected_prefix = (classes, len(phases) - 1)
        if phase_shape[:2] != expected_prefix or phase_shape[-1] != len(rest):
            raise ValueError("Phase reference dimensions are inconsistent")
        for name, values in (
            (
                "phase continuum-removed profiles",
                self.phase_continuum_removed_profiles,
            ),
            ("phase masks", self.phase_profile_masks),
        ):
            if values.shape != phase_shape:
                raise ValueError(f"{name} must have shape {phase_shape}")
        if self.phase_support_counts.shape != phase_shape[:-1]:
            raise ValueError("Phase reference support counts are inconsistent")

        for name, values in (
            ("coadd full profiles", self.coadd_full_profiles),
            (
                "coadd continuum-removed profiles",
                self.coadd_continuum_removed_profiles,
            ),
            ("phase full profiles", self.phase_full_profiles),
            (
                "phase continuum-removed profiles",
                self.phase_continuum_removed_profiles,
            ),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} contain non-finite values")
        if np.any(self.coadd_support_counts < 0) or np.any(
            self.phase_support_counts < 0
        ):
            raise ValueError("Reference support counts cannot be negative")

    def save(self, path: str | Path) -> Path:
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            format_version=np.asarray(REFERENCE_BANK_FORMAT),
            class_names=np.asarray(self.class_names),
            rest_wavelength=self.rest_wavelength.astype(np.float32),
            phase_edges_days=self.phase_edges_days.astype(np.float32),
            coadd_full_profiles=self.coadd_full_profiles.astype(np.float32),
            coadd_continuum_removed_profiles=(
                self.coadd_continuum_removed_profiles.astype(np.float32)
            ),
            coadd_profile_masks=self.coadd_profile_masks.astype(bool),
            coadd_support_counts=self.coadd_support_counts.astype(np.int64),
            phase_full_profiles=self.phase_full_profiles.astype(np.float32),
            phase_continuum_removed_profiles=(
                self.phase_continuum_removed_profiles.astype(np.float32)
            ),
            phase_profile_masks=self.phase_profile_masks.astype(bool),
            phase_support_counts=self.phase_support_counts.astype(np.int64),
            metadata_json=np.asarray(json.dumps(self.metadata, sort_keys=True)),
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> RomanReferenceBank:
        bank_path = Path(path).expanduser().resolve()
        with np.load(bank_path, allow_pickle=False) as values:
            version = str(values["format_version"].item())
            if version != REFERENCE_BANK_FORMAT:
                raise ValueError(f"Unsupported Roman reference format: {version}")
            bank = cls(
                class_names=tuple(str(value) for value in values["class_names"]),
                rest_wavelength=np.asarray(values["rest_wavelength"], dtype=np.float32),
                phase_edges_days=np.asarray(
                    values["phase_edges_days"], dtype=np.float32
                ),
                coadd_full_profiles=np.asarray(
                    values["coadd_full_profiles"], dtype=np.float32
                ),
                coadd_continuum_removed_profiles=np.asarray(
                    values["coadd_continuum_removed_profiles"], dtype=np.float32
                ),
                coadd_profile_masks=np.asarray(
                    values["coadd_profile_masks"], dtype=bool
                ),
                coadd_support_counts=np.asarray(
                    values["coadd_support_counts"], dtype=np.int64
                ),
                phase_full_profiles=np.asarray(
                    values["phase_full_profiles"], dtype=np.float32
                ),
                phase_continuum_removed_profiles=np.asarray(
                    values["phase_continuum_removed_profiles"], dtype=np.float32
                ),
                phase_profile_masks=np.asarray(
                    values["phase_profile_masks"], dtype=bool
                ),
                phase_support_counts=np.asarray(
                    values["phase_support_counts"], dtype=np.int64
                ),
                metadata=json.loads(str(values["metadata_json"].item())),
            )
        bank.validate()
        return bank


def build_roman_reference_bank(config: dict[str, Any]) -> dict[str, Any]:
    """Build clean fine-class references from the configured training role."""
    settings = config["reference"]
    output_path = project_path(config, settings["bank_path"])
    phase_edges = np.asarray(settings["phase_edges_days"], dtype=np.float32)
    if len(phase_edges) < 3 or np.any(np.diff(phase_edges) <= 0.0):
        raise ValueError("reference.phase_edges_days must increase strictly")
    redshift_edges = np.asarray(config["data"]["redshift_edges"], dtype=np.float32)
    if len(redshift_edges) < 2 or np.any(np.diff(redshift_edges) <= 0.0):
        raise ValueError("data.redshift_edges must increase strictly")

    # The model configuration need not carry a clean target at runtime.  The
    # builder enables it on a private copy solely for training-bank creation.
    build_config = {
        **config,
        "data": {
            **config["data"],
            "include_flux_error_channel": True,
            "include_clean_flux_target": True,
        },
    }
    dataset = SundialDataset(
        build_config,
        "train",
        "original",
        training=False,
        pair_no_source=False,
    )
    rest_grid = log_wavelength_grid(
        float(config["model"]["rest_wavelength_min"]),
        float(config["model"]["rest_wavelength_max"]),
        int(settings["rest_wavelength_bins"]),
    ).astype(np.float32)
    continuum_sigma = velocity_sigma_to_log_bins(
        float(rest_grid[0]),
        float(rest_grid[-1]),
        len(rest_grid),
        float(settings["continuum_width_km_s"]),
    )
    continuum = MaskAwareContinuumRemoval(continuum_sigma).eval()
    class_to_index = {name: index for index, name in enumerate(HOURGLASS_15_CLASSES)}
    coadd_limit = int(settings.get("maximum_coadd_profiles_per_class", 2_000))
    phase_limit = int(settings.get("maximum_phase_profiles_per_cell", 1_000))
    object_limit = int(settings.get("maximum_training_objects_per_class", 0))
    if coadd_limit < 1 or phase_limit < 1:
        raise ValueError("Reference profile limits must be positive")
    if object_limit < 0:
        raise ValueError("Reference training-object limit cannot be negative")
    rng = np.random.default_rng(int(config["project"]["seed"]) + 91_403)

    selected_indices = _balanced_training_indices(
        dataset.objects,
        object_limit,
        rng,
    )
    selected_by_class_and_redshift = _class_redshift_counts(
        dataset.objects,
        selected_indices,
        redshift_edges,
    )

    coadd_rows = [_ProfileReservoir(coadd_limit, rng) for _ in HOURGLASS_15_CLASSES]
    phase_rows = [
        [_ProfileReservoir(phase_limit, rng) for _ in range(len(phase_edges) - 1)]
        for _ in HOURGLASS_15_CLASSES
    ]
    used_objects = np.zeros(len(HOURGLASS_15_CLASSES), dtype=np.int64)
    used_objects_by_redshift = np.zeros(
        (len(HOURGLASS_15_CLASSES), len(redshift_edges) - 1),
        dtype=np.int64,
    )
    used_visits = np.zeros(
        (len(HOURGLASS_15_CLASSES), len(phase_edges) - 1), dtype=np.int64
    )
    edge_mask = _edge_mask(
        len(dataset.output_wavelength),
        float(settings.get("edge_trim_fraction", 0.05)),
    )
    edge_taper_fraction = float(settings.get("edge_taper_fraction", 0.0))
    edge_taper = cosine_edge_taper(
        len(dataset.output_wavelength), edge_taper_fraction
    )
    edge_taper_mask = edge_taper.numpy() > 0.0
    configured_error_limit = settings.get("maximum_relative_coadd_error", 3.0)
    maximum_relative_error = (
        None if configured_error_limit is None else float(configured_error_limit)
    )
    minimum_relative_precision = float(
        settings.get("minimum_relative_spectral_precision", 0.0)
    )
    if not 0.0 <= minimum_relative_precision < 1.0:
        raise ValueError(
            "Reference minimum relative spectral precision must lie in [0, 1)"
        )

    for selected_number, item_index in enumerate(selected_indices, start=1):
        item = dataset[item_index]
        row = dataset.objects.iloc[item_index]
        fine_name = fine_class_name_for_source(
            int(row.gentype), int(row.template_index)
        )
        if fine_name is None:
            continue
        class_index = class_to_index[fine_name]
        redshift = float(item["redshift"])
        clean = item["clean_flux_target"]
        mask = item["wavelength_mask"]
        visit_count = clean.shape[0]
        visit_mask = torch.ones((1, visit_count), dtype=clean.dtype)
        coadd, coadd_error, coadd_mask = final_inverse_variance_coadd(
            clean[None, ...],
            mask[None, ...],
            visit_mask,
            item["flux_error_shape"][None, ...],
            item["visit_flux_scale"][None, ...],
            maximum_relative_error=maximum_relative_error,
        )
        coadd_reliability = relative_inverse_variance(coadd_error, coadd_mask)
        coadd_precision_mask = minimum_relative_precision_mask(
            coadd_reliability,
            coadd_mask,
            minimum_relative_precision,
        )
        final_mask = (
            coadd_precision_mask[0].numpy().astype(bool)
            & edge_mask
            & edge_taper_mask
        )
        rest_full, rest_mask = align_to_rest_grid(
            dataset.output_wavelength,
            coadd[0].numpy(),
            final_mask.astype(np.float32),
            redshift,
            rest_grid,
        )
        rest_mask = rest_mask.astype(bool)
        if rest_mask.sum() >= int(settings.get("minimum_rest_bins", 32)):
            rest_edge_weight, _ = align_to_rest_grid(
                dataset.output_wavelength,
                (coadd_reliability[0] * edge_taper).numpy(),
                final_mask.astype(np.float32),
                redshift,
                rest_grid,
            )
            rest_removed = _continuum_removed(
                rest_full,
                rest_mask,
                continuum,
                rest_edge_weight,
            )
            coadd_rows[class_index].add(rest_full, rest_removed, rest_mask)
            used_objects[class_index] += 1
            redshift_index = _redshift_bin_index(redshift, redshift_edges)
            if redshift_index is not None:
                used_objects_by_redshift[class_index, redshift_index] += 1

        phases = item["simulation_rest_phase_days"].numpy()
        for visit_index, phase in enumerate(phases):
            phase_index = int(np.searchsorted(phase_edges, phase, side="right") - 1)
            if phase_index < 0 or phase_index >= len(phase_edges) - 1:
                continue
            visit_error = torch.exp(item["flux_error_shape"][visit_index])
            visit_reliability = relative_inverse_variance(
                visit_error[None, :],
                mask[visit_index][None, :],
            )[0]
            visit_precision_mask = minimum_relative_precision_mask(
                visit_reliability,
                mask[visit_index],
                minimum_relative_precision,
            )
            visit_rest, visit_rest_mask = align_to_rest_grid(
                dataset.output_wavelength,
                clean[visit_index].numpy(),
                (
                    visit_precision_mask.numpy().astype(bool)
                    & edge_mask
                    & edge_taper_mask
                ).astype(np.float32),
                redshift,
                rest_grid,
            )
            visit_rest_mask = visit_rest_mask.astype(bool)
            if visit_rest_mask.sum() < int(settings.get("minimum_rest_bins", 32)):
                continue
            visit_edge_weight, _ = align_to_rest_grid(
                dataset.output_wavelength,
                (visit_reliability * edge_taper).numpy(),
                (
                    visit_precision_mask.numpy().astype(bool)
                    & edge_mask
                    & edge_taper_mask
                ).astype(np.float32),
                redshift,
                rest_grid,
            )
            visit_removed = _continuum_removed(
                visit_rest,
                visit_rest_mask,
                continuum,
                visit_edge_weight,
            )
            phase_rows[class_index][phase_index].add(
                visit_rest, visit_removed, visit_rest_mask
            )
            used_visits[class_index, phase_index] += 1

        if selected_number % 1_000 == 0:
            print(
                "  collected references from "
                f"{selected_number:,} / {len(selected_indices):,} selected "
                "training objects"
            )

    coadd_profiles = int(settings["coadd_profiles_per_class"])
    phase_profiles = int(settings["phase_profiles_per_cell"])
    minimum_profile_support = int(settings.get("minimum_profile_support", 5))
    if minimum_profile_support < 1:
        raise ValueError("Reference minimum profile support must be positive")
    coadd_full, coadd_removed, coadd_masks, coadd_support = _cluster_cells(
        coadd_rows,
        coadd_profiles,
        len(rest_grid),
        minimum_bin_fraction=float(settings.get("minimum_bin_fraction", 0.5)),
        minimum_profile_support=minimum_profile_support,
    )
    phase_full, phase_removed, phase_masks, phase_support = _cluster_phase_cells(
        phase_rows,
        phase_profiles,
        len(rest_grid),
        minimum_bin_fraction=float(settings.get("minimum_bin_fraction", 0.5)),
        minimum_profile_support=minimum_profile_support,
    )
    missing_coadd = [
        name
        for class_index, name in enumerate(HOURGLASS_15_CLASSES)
        if not np.any(coadd_support[class_index] >= minimum_profile_support)
    ]
    if missing_coadd:
        raise ValueError(
            "Reference construction produced no sufficiently supported coadd "
            "profile for: " + ", ".join(missing_coadd)
        )
    metadata = {
        "format_version": REFERENCE_BANK_FORMAT,
        "source_split": "train",
        "reference_flux": "clean simulated FLAM",
        "coadd_weights": "reported FLAMERR for the retained observations",
        "truth_used_at_runtime": False,
        "config_sha256": resolved_config_sha256(config),
        "training_objects_available": int(len(dataset)),
        "maximum_training_objects_per_class": object_limit,
        "training_objects_selected": int(len(selected_indices)),
        "redshift_edges": redshift_edges.astype(float).tolist(),
        "training_objects_selected_by_fine_class_and_redshift": {
            name: selected_by_class_and_redshift[index].astype(int).tolist()
            for index, name in enumerate(HOURGLASS_15_CLASSES)
        },
        "training_objects_used_by_fine_class": {
            name: int(used_objects[index])
            for index, name in enumerate(HOURGLASS_15_CLASSES)
        },
        "training_objects_used_by_fine_class_and_redshift": {
            name: used_objects_by_redshift[index].astype(int).tolist()
            for index, name in enumerate(HOURGLASS_15_CLASSES)
        },
        "training_visits_used_by_fine_class_and_phase": {
            name: used_visits[index].astype(int).tolist()
            for index, name in enumerate(HOURGLASS_15_CLASSES)
        },
        "continuum_width_km_s": float(settings["continuum_width_km_s"]),
        "maximum_relative_coadd_error": maximum_relative_error,
        "minimum_relative_spectral_precision": minimum_relative_precision,
        "edge_trim_fraction_per_side": float(
            settings.get("edge_trim_fraction", 0.05)
        ),
        "cosine_edge_taper_fraction_per_side": edge_taper_fraction,
        "cosine_edge_taper_application": "reliability_weight_only",
        "minimum_profile_support": minimum_profile_support,
        "supported_coadd_profiles_by_fine_class": {
            name: int(np.sum(coadd_support[index] >= minimum_profile_support))
            for index, name in enumerate(HOURGLASS_15_CLASSES)
        },
        "supported_phase_profiles_by_fine_class_and_phase": {
            name: np.sum(
                phase_support[index] >= minimum_profile_support,
                axis=-1,
            )
            .astype(int)
            .tolist()
            for index, name in enumerate(HOURGLASS_15_CLASSES)
        },
    }
    bank = RomanReferenceBank(
        class_names=HOURGLASS_15_CLASSES,
        rest_wavelength=rest_grid,
        phase_edges_days=phase_edges,
        coadd_full_profiles=coadd_full,
        coadd_continuum_removed_profiles=coadd_removed,
        coadd_profile_masks=coadd_masks,
        coadd_support_counts=coadd_support,
        phase_full_profiles=phase_full,
        phase_continuum_removed_profiles=phase_removed,
        phase_profile_masks=phase_masks,
        phase_support_counts=phase_support,
        metadata=metadata,
    )
    bank.save(output_path)
    audit_path = output_path.with_suffix(".audit.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"bank": str(output_path), "audit": str(audit_path), **metadata}


class _ProfileReservoir:
    def __init__(self, limit: int, rng: np.random.Generator) -> None:
        self.limit = int(limit)
        self.rng = rng
        self.seen = 0
        self.rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def add(
        self, full: np.ndarray, continuum_removed: np.ndarray, mask: np.ndarray
    ) -> None:
        self.seen += 1
        row = (
            np.asarray(full, dtype=np.float32).copy(),
            np.asarray(continuum_removed, dtype=np.float32).copy(),
            np.asarray(mask, dtype=bool).copy(),
        )
        if len(self.rows) < self.limit:
            self.rows.append(row)
            return
        replacement = int(self.rng.integers(self.seen))
        if replacement < self.limit:
            self.rows[replacement] = row


def _balanced_training_indices(
    objects: Any,
    maximum_per_class: int,
    rng: np.random.Generator,
) -> list[int]:
    """Choose a deterministic class-balanced subset of the training role."""
    by_class: dict[str, list[int]] = {name: [] for name in HOURGLASS_15_CLASSES}
    for index, row in enumerate(objects.itertuples(index=False)):
        name = fine_class_name_for_source(int(row.gentype), int(row.template_index))
        if name is not None:
            by_class[name].append(index)

    selected: list[int] = []
    for name in HOURGLASS_15_CLASSES:
        indices = np.asarray(by_class[name], dtype=np.int64)
        if maximum_per_class and len(indices) > maximum_per_class:
            indices = np.sort(
                rng.choice(indices, size=maximum_per_class, replace=False)
            )
        selected.extend(indices.astype(int).tolist())
    return sorted(selected)


def _class_redshift_counts(
    objects: Any,
    indices: list[int],
    redshift_edges: np.ndarray,
) -> np.ndarray:
    counts = np.zeros(
        (len(HOURGLASS_15_CLASSES), len(redshift_edges) - 1),
        dtype=np.int64,
    )
    class_to_index = {name: index for index, name in enumerate(HOURGLASS_15_CLASSES)}
    for index in indices:
        row = objects.iloc[index]
        name = fine_class_name_for_source(int(row.gentype), int(row.template_index))
        if name is None:
            continue
        redshift_index = _redshift_bin_index(float(row.redshift), redshift_edges)
        if redshift_index is not None:
            counts[class_to_index[name], redshift_index] += 1
    return counts


def _redshift_bin_index(
    redshift: float,
    redshift_edges: np.ndarray,
) -> int | None:
    if not np.isfinite(redshift):
        return None
    index = int(np.searchsorted(redshift_edges, redshift, side="right") - 1)
    bins = len(redshift_edges) - 1
    if index == bins and redshift <= float(redshift_edges[-1]):
        index = bins - 1
    return index if 0 <= index < bins else None


def _edge_mask(bins: int, fraction: float) -> np.ndarray:
    if not 0.0 <= fraction < 0.5:
        raise ValueError("Reference edge trim must lie in [0, 0.5)")
    coordinate = np.linspace(0.0, 1.0, bins)
    return (coordinate >= fraction) & (coordinate <= 1.0 - fraction)


def _continuum_removed(
    full: np.ndarray,
    mask: np.ndarray,
    continuum: MaskAwareContinuumRemoval,
    weight: np.ndarray | None = None,
) -> np.ndarray:
    with torch.no_grad():
        result = continuum(
            torch.from_numpy(np.asarray(full, dtype=np.float32))[None, :],
            torch.from_numpy(np.asarray(mask, dtype=np.float32))[None, :],
            None
            if weight is None
            else torch.from_numpy(np.asarray(weight, dtype=np.float32))[None, :],
        )[0]
    return result.numpy().astype(np.float32)


def _cluster_cells(
    reservoirs: list[_ProfileReservoir],
    prototypes: int,
    wavelength_bins: int,
    *,
    minimum_bin_fraction: float,
    minimum_profile_support: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = (len(reservoirs), prototypes, wavelength_bins)
    full = np.zeros(shape, dtype=np.float32)
    removed = np.zeros(shape, dtype=np.float32)
    masks = np.zeros(shape, dtype=bool)
    support = np.zeros(shape[:-1], dtype=np.int64)
    for class_index, reservoir in enumerate(reservoirs):
        if not reservoir.rows:
            continue
        values = np.stack([row[0] for row in reservoir.rows])
        details = np.stack([row[1] for row in reservoir.rows])
        measured = np.stack([row[2] for row in reservoir.rows])
        cell = _paired_spherical_clusters(
            values,
            details,
            measured,
            prototypes,
            minimum_bin_fraction,
            minimum_profile_support,
        )
        (
            full[class_index],
            removed[class_index],
            masks[class_index],
            support[class_index],
        ) = cell
    return full, removed, masks, support


def _cluster_phase_cells(
    reservoirs: list[list[_ProfileReservoir]],
    prototypes: int,
    wavelength_bins: int,
    *,
    minimum_bin_fraction: float,
    minimum_profile_support: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = (
        len(reservoirs),
        len(reservoirs[0]),
        prototypes,
        wavelength_bins,
    )
    full = np.zeros(shape, dtype=np.float32)
    removed = np.zeros(shape, dtype=np.float32)
    masks = np.zeros(shape, dtype=bool)
    support = np.zeros(shape[:-1], dtype=np.int64)
    for class_index, phase_reservoirs in enumerate(reservoirs):
        for phase_index, reservoir in enumerate(phase_reservoirs):
            if not reservoir.rows:
                continue
            values = np.stack([row[0] for row in reservoir.rows])
            details = np.stack([row[1] for row in reservoir.rows])
            measured = np.stack([row[2] for row in reservoir.rows])
            cell = _paired_spherical_clusters(
                values,
                details,
                measured,
                prototypes,
                minimum_bin_fraction,
                minimum_profile_support,
            )
            (
                full[class_index, phase_index],
                removed[class_index, phase_index],
                masks[class_index, phase_index],
                support[class_index, phase_index],
            ) = cell
    return full, removed, masks, support


def _paired_spherical_clusters(
    full: np.ndarray,
    removed: np.ndarray,
    masks: np.ndarray,
    prototype_count: int,
    minimum_bin_fraction: float,
    minimum_profile_support: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cluster both spectral views together and retain paired references."""
    if prototype_count < 1:
        raise ValueError("Reference prototype count must be positive")
    if not 0.0 < minimum_bin_fraction <= 1.0:
        raise ValueError("Reference minimum bin fraction must lie in (0, 1]")
    if minimum_profile_support < 1:
        raise ValueError("Reference minimum profile support must be positive")
    full_n = _normalise_rows(full, masks)
    removed_n = _normalise_rows(removed, masks)
    used = min(
        prototype_count,
        max(1, len(full) // minimum_profile_support),
    )
    # The first centre is the object closest to the two-view population mean.
    population_full = _masked_mean(full_n, masks, minimum_bin_fraction=0.25)[0]
    population_removed = _masked_mean(removed_n, masks, minimum_bin_fraction=0.25)[0]
    similarity = 0.5 * (full_n @ population_full + removed_n @ population_removed)
    centres = [int(np.argmax(similarity))]
    while len(centres) < used:
        pair_similarity = 0.5 * (
            full_n @ full_n[centres].T + removed_n @ removed_n[centres].T
        )
        centres.append(int(np.argmin(pair_similarity.max(axis=1))))

    while True:
        assignment = np.zeros(len(full), dtype=np.int64)
        for _ in range(10):
            centre_full = full_n[centres]
            centre_removed = removed_n[centres]
            score = 0.5 * (full_n @ centre_full.T + removed_n @ centre_removed.T)
            assignment = np.argmax(score, axis=1)
            new_centres: list[int] = []
            for cluster in range(len(centres)):
                selected = assignment == cluster
                if not np.any(selected):
                    new_centres.append(centres[cluster])
                    continue
                mean_full = _normalise_vector(
                    _masked_mean(full[selected], masks[selected], 0.25)[0]
                )
                mean_removed = _normalise_vector(
                    _masked_mean(removed[selected], masks[selected], 0.25)[0]
                )
                members = np.flatnonzero(selected)
                local_score = 0.5 * (
                    full_n[members] @ mean_full + removed_n[members] @ mean_removed
                )
                new_centres.append(int(members[np.argmax(local_score)]))
            if new_centres == centres:
                break
            centres = new_centres

        cluster_counts = np.bincount(assignment, minlength=len(centres))
        if len(centres) == 1 or np.all(cluster_counts >= minimum_profile_support):
            break
        weakest = int(np.argmin(cluster_counts))
        centres.pop(weakest)

    used = len(centres)

    output_full = np.zeros((prototype_count, full.shape[1]), dtype=np.float32)
    output_removed = np.zeros_like(output_full)
    output_mask = np.zeros_like(output_full, dtype=bool)
    counts = np.zeros(prototype_count, dtype=np.int64)
    for cluster in range(used):
        selected = assignment == cluster
        counts[cluster] = int(selected.sum())
        if not counts[cluster]:
            continue
        mean_full, mean_mask = _masked_mean(
            full[selected],
            masks[selected],
            minimum_bin_fraction,
        )
        mean_removed, removed_mask = _masked_mean(
            removed[selected],
            masks[selected],
            minimum_bin_fraction,
        )
        final_mask = mean_mask & removed_mask
        output_full[cluster] = mean_full * final_mask
        output_removed[cluster] = mean_removed * final_mask
        output_mask[cluster] = final_mask
    return output_full, output_removed, output_mask, counts


def _normalise_rows(values: np.ndarray, masks: np.ndarray) -> np.ndarray:
    measured = masks.astype(np.float32)
    scale = np.max(np.abs(values) * measured, axis=1, keepdims=True)
    scale = np.maximum(scale, np.finfo(np.float32).tiny)
    values = values / scale
    count = measured.sum(axis=1, keepdims=True).clip(min=1.0)
    mean = (values * measured).sum(axis=1, keepdims=True) / count
    centred = (values - mean) * measured
    norm = np.linalg.norm(centred, axis=1, keepdims=True).clip(
        min=np.finfo(np.float32).eps
    )
    return (centred / norm).astype(np.float32)


def _normalise_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    scale = max(float(np.max(np.abs(values))), float(np.finfo(np.float32).tiny))
    scaled = values / scale
    norm = max(float(np.linalg.norm(scaled)), float(np.finfo(np.float32).eps))
    return np.asarray(scaled / norm, dtype=np.float32)


def _masked_mean(
    values: np.ndarray,
    masks: np.ndarray,
    minimum_bin_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    count = masks.sum(axis=0)
    required = max(1, int(np.ceil(minimum_bin_fraction * len(values))))
    output_mask = count >= required
    mean = np.divide(
        (values * masks).sum(axis=0),
        np.maximum(count, 1),
        out=np.zeros(values.shape[1], dtype=np.float32),
    )
    return mean.astype(np.float32) * output_mask, output_mask
