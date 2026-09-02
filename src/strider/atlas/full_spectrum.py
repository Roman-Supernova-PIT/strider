"""Full-spectrum reference profiles for the STRIDER atlas feasibility study.

This module is deliberately separate from the production model.  It provides
small, deterministic building blocks for testing whether clean training
spectra can define a useful full-spectrum class--redshift reference atlas.  A
new observation is always scanned over candidate redshifts; truth redshift is
used only while placing training spectra on the atlas rest-frame grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from strider.model.coadd import final_inverse_variance_coadd

from .build import _cluster_masked_profiles


ATLAS_FORMAT = "strider-full-spectrum-atlas-study-v1"
PHASE_ATLAS_FORMAT = "strider-phase-indexed-atlas-study-v1"


@dataclass(frozen=True)
class FullSpectrumAtlas:
    """Clean class reference profiles on one logarithmic rest-frame grid."""

    class_names: tuple[str, ...]
    rest_wavelength: np.ndarray
    whole_profiles: np.ndarray
    whole_masks: np.ndarray
    detail_profiles: np.ndarray
    detail_masks: np.ndarray
    support_counts: np.ndarray

    def validate(self) -> None:
        classes = len(self.class_names)
        if self.rest_wavelength.ndim != 1 or len(self.rest_wavelength) < 2:
            raise ValueError("Atlas rest wavelength must be a one-dimensional grid")
        if np.any(np.diff(self.rest_wavelength) <= 0.0):
            raise ValueError("Atlas rest wavelength must increase strictly")
        expected_prefix = (classes, self.whole_profiles.shape[1])
        expected = (*expected_prefix, len(self.rest_wavelength))
        for name, values in (
            ("whole profiles", self.whole_profiles),
            ("whole masks", self.whole_masks),
            ("detail profiles", self.detail_profiles),
            ("detail masks", self.detail_masks),
        ):
            if values.shape != expected:
                raise ValueError(f"Atlas {name} must have shape {expected}")
        if self.support_counts.shape != expected_prefix:
            raise ValueError(
                f"Atlas support counts must have shape {expected_prefix}"
            )
        if not np.all(np.isfinite(self.whole_profiles)):
            raise ValueError("Atlas whole profiles contain non-finite values")
        if not np.all(np.isfinite(self.detail_profiles)):
            raise ValueError("Atlas detail profiles contain non-finite values")
        if np.any(self.support_counts < 0):
            raise ValueError("Atlas support counts cannot be negative")

    @property
    def prototype_count(self) -> int:
        return int(self.whole_profiles.shape[1])

    def save(self, path: str | Path) -> Path:
        """Write the study atlas as a self-describing compressed NumPy file."""
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            format_version=np.asarray(ATLAS_FORMAT),
            class_names=np.asarray(self.class_names),
            rest_wavelength=self.rest_wavelength.astype(np.float32),
            full_spectrum_profiles=self.whole_profiles.astype(np.float32),
            full_spectrum_masks=self.whole_masks.astype(bool),
            continuum_removed_profiles=self.detail_profiles.astype(np.float32),
            continuum_removed_masks=self.detail_masks.astype(bool),
            support_counts=self.support_counts.astype(np.int64),
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> FullSpectrumAtlas:
        """Load and validate an atlas written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as values:
            format_version = str(values["format_version"].item())
            if format_version != ATLAS_FORMAT:
                raise ValueError(
                    f"Unsupported full-spectrum atlas format: {format_version}"
                )
            atlas = cls(
                class_names=tuple(str(value) for value in values["class_names"]),
                rest_wavelength=np.asarray(
                    values["rest_wavelength"], dtype=np.float32
                ),
                whole_profiles=np.asarray(
                    values["full_spectrum_profiles"], dtype=np.float32
                ),
                whole_masks=np.asarray(values["full_spectrum_masks"], dtype=bool),
                detail_profiles=np.asarray(
                    values["continuum_removed_profiles"], dtype=np.float32
                ),
                detail_masks=np.asarray(values["continuum_removed_masks"], dtype=bool),
                support_counts=np.asarray(values["support_counts"], dtype=np.int64),
            )
        atlas.validate()
        return atlas


@dataclass(frozen=True)
class PhaseIndexedAtlas:
    """Clean single-visit profiles indexed by class and broad phase range."""

    class_names: tuple[str, ...]
    rest_wavelength: np.ndarray
    phase_edges_days: np.ndarray
    whole_profiles: np.ndarray
    whole_masks: np.ndarray
    detail_profiles: np.ndarray
    detail_masks: np.ndarray
    support_counts: np.ndarray

    def validate(self) -> None:
        classes = len(self.class_names)
        phase_edges = np.asarray(self.phase_edges_days)
        if phase_edges.ndim != 1 or len(phase_edges) < 3:
            raise ValueError("Phase-indexed atlas needs at least two phase ranges")
        if np.any(np.diff(phase_edges) <= 0.0):
            raise ValueError("Phase edges must increase strictly")
        phases = len(phase_edges) - 1
        if self.whole_profiles.ndim != 4:
            raise ValueError(
                "Phase-indexed profiles must have class, phase, profile, and wavelength axes"
            )
        expected = (
            classes,
            phases,
            self.whole_profiles.shape[2],
            len(self.rest_wavelength),
        )
        for name, values in (
            ("whole profiles", self.whole_profiles),
            ("whole masks", self.whole_masks),
            ("detail profiles", self.detail_profiles),
            ("detail masks", self.detail_masks),
        ):
            if values.shape != expected:
                raise ValueError(f"Phase-indexed {name} must have shape {expected}")
        if self.support_counts.shape != expected[:-1]:
            raise ValueError(
                f"Phase-indexed support counts must have shape {expected[:-1]}"
            )
        if not np.all(np.isfinite(self.whole_profiles)) or not np.all(
            np.isfinite(self.detail_profiles)
        ):
            raise ValueError("Phase-indexed profiles contain non-finite values")
        if not np.all(np.isfinite(self.rest_wavelength)) or np.any(
            np.diff(self.rest_wavelength) <= 0.0
        ):
            raise ValueError("Phase-indexed rest wavelength must increase strictly")

    @property
    def prototype_count(self) -> int:
        return int(self.whole_profiles.shape[2])

    def save(self, path: str | Path) -> Path:
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            format_version=np.asarray(PHASE_ATLAS_FORMAT),
            class_names=np.asarray(self.class_names),
            rest_wavelength=self.rest_wavelength.astype(np.float32),
            phase_edges_days=self.phase_edges_days.astype(np.float32),
            full_spectrum_profiles=self.whole_profiles.astype(np.float32),
            full_spectrum_masks=self.whole_masks.astype(bool),
            continuum_removed_profiles=self.detail_profiles.astype(np.float32),
            continuum_removed_masks=self.detail_masks.astype(bool),
            support_counts=self.support_counts.astype(np.int64),
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> PhaseIndexedAtlas:
        with np.load(Path(path), allow_pickle=False) as values:
            format_version = str(values["format_version"].item())
            if format_version != PHASE_ATLAS_FORMAT:
                raise ValueError(
                    f"Unsupported phase-indexed atlas format: {format_version}"
                )
            atlas = cls(
                class_names=tuple(str(value) for value in values["class_names"]),
                rest_wavelength=np.asarray(
                    values["rest_wavelength"], dtype=np.float32
                ),
                phase_edges_days=np.asarray(
                    values["phase_edges_days"], dtype=np.float32
                ),
                whole_profiles=np.asarray(
                    values["full_spectrum_profiles"], dtype=np.float32
                ),
                whole_masks=np.asarray(values["full_spectrum_masks"], dtype=bool),
                detail_profiles=np.asarray(
                    values["continuum_removed_profiles"], dtype=np.float32
                ),
                detail_masks=np.asarray(values["continuum_removed_masks"], dtype=bool),
                support_counts=np.asarray(values["support_counts"], dtype=np.int64),
            )
        atlas.validate()
        return atlas


class CandidateRestFrameScan:
    """Align one observer-frame spectrum at every candidate redshift."""

    def __init__(
        self,
        observed_wavelength: np.ndarray,
        rest_wavelength: np.ndarray,
        redshift_grid: np.ndarray,
    ) -> None:
        observed = np.asarray(observed_wavelength, dtype=np.float64)
        rest = np.asarray(rest_wavelength, dtype=np.float64)
        redshift = np.asarray(redshift_grid, dtype=np.float64)
        if observed.ndim != 1 or rest.ndim != 1 or redshift.ndim != 1:
            raise ValueError("Wavelength and redshift grids must be one-dimensional")
        if len(observed) < 2 or len(rest) < 2 or len(redshift) < 2:
            raise ValueError("Candidate scan grids require at least two values")
        if (
            np.any(np.diff(observed) <= 0.0)
            or np.any(np.diff(rest) <= 0.0)
            or np.any(np.diff(redshift) <= 0.0)
        ):
            raise ValueError("Candidate scan grids must increase strictly")

        target = rest[None, :] * (1.0 + redshift[:, None])
        upper = np.searchsorted(observed, target, side="left")
        valid = (upper > 0) & (upper < len(observed))
        upper = np.clip(upper, 1, len(observed) - 1)
        lower = upper - 1
        denominator = observed[upper] - observed[lower]
        weight = np.divide(
            target - observed[lower],
            denominator,
            out=np.zeros_like(target),
            where=denominator > 0.0,
        )
        self.lower = lower
        self.upper = upper
        self.weight = weight.astype(np.float32)
        self.valid = valid
        self.redshift_grid = redshift.astype(np.float32)

    def align(
        self,
        flux: np.ndarray,
        wavelength_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return candidate-aligned flux and measured-bin masks."""
        values = np.asarray(flux, dtype=np.float32)
        measured = np.asarray(wavelength_mask, dtype=bool)
        if values.ndim != 1 or measured.shape != values.shape:
            raise ValueError("Candidate scan input must be one spectrum and mask")
        if self.lower.max(initial=0) >= len(values):
            raise ValueError("Candidate scan indices exceed the observed spectrum")
        lower_values = values[self.lower]
        upper_values = values[self.upper]
        aligned = lower_values * (1.0 - self.weight) + upper_values * self.weight
        support = measured[self.lower] & measured[self.upper] & self.valid
        aligned = np.where(support, aligned, 0.0).astype(np.float32)
        return aligned, support


def measurement_faithful_coadd(
    flux: torch.Tensor,
    wavelength_mask: torch.Tensor,
    log_scaled_error: torch.Tensor,
    visit_flux_scale: torch.Tensor,
    *,
    maximum_relative_error: float = 3.0,
    edge_trim_fraction: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the production-definition IV coadd for one object.

    Inputs have shapes ``(visits, wavelength)`` and ``(visits,)``.  The returned
    arrays are coadded flux, propagated error, and the final measurement mask.
    """
    if flux.ndim != 2:
        raise ValueError("Study coadd flux must have shape (visits, wavelength)")
    if not 0.0 <= edge_trim_fraction < 0.5:
        raise ValueError("Study coadd edge trim must lie in [0, 0.5)")
    visits = flux.shape[0]
    visit_mask = torch.ones((1, visits), dtype=flux.dtype, device=flux.device)
    coadd, error, mask = final_inverse_variance_coadd(
        flux[None, ...],
        wavelength_mask[None, ...],
        visit_mask,
        log_scaled_error[None, ...],
        visit_flux_scale[None, ...],
        maximum_relative_error=maximum_relative_error,
    )
    bins = flux.shape[-1]
    coordinate = np.linspace(0.0, 1.0, bins, dtype=np.float32)
    edge_mask = (coordinate >= edge_trim_fraction) & (
        coordinate <= 1.0 - edge_trim_fraction
    )
    output_mask = mask[0].detach().cpu().numpy().astype(bool) & edge_mask
    output_flux = coadd[0].detach().cpu().numpy().astype(np.float32)
    output_error = error[0].detach().cpu().numpy().astype(np.float32)
    output_flux[~output_mask] = 0.0
    output_error[~output_mask] = 0.0
    return output_flux, output_error, output_mask


def best_measured_visit(
    flux: torch.Tensor,
    wavelength_mask: torch.Tensor,
    log_scaled_error: torch.Tensor,
    *,
    edge_trim_fraction: float = 0.05,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, float]:
    """Select the visit with the largest signed median measured-bin S/N."""
    if flux.ndim != 2 or wavelength_mask.shape != flux.shape:
        raise ValueError("Best-visit inputs must have shape (visits, wavelength)")
    if log_scaled_error.shape != flux.shape:
        raise ValueError("Best-visit error must match the flux")
    if not 0.0 <= edge_trim_fraction < 0.5:
        raise ValueError("Best-visit edge trim must lie in [0, 0.5)")

    values = flux.detach().cpu().numpy().astype(np.float32)
    measured = wavelength_mask.detach().cpu().numpy().astype(bool)
    error = np.exp(log_scaled_error.detach().cpu().numpy()).astype(np.float32)
    coordinate = np.linspace(0.0, 1.0, values.shape[-1], dtype=np.float32)
    edge_mask = (coordinate >= edge_trim_fraction) & (
        coordinate <= 1.0 - edge_trim_fraction
    )
    measured &= edge_mask[None, :]
    scores = np.full(values.shape[0], -np.inf, dtype=np.float64)
    for visit in range(values.shape[0]):
        valid = measured[visit] & np.isfinite(error[visit]) & (error[visit] > 0.0)
        if np.any(valid):
            scores[visit] = float(np.median(values[visit, valid] / error[visit, valid]))
    best = int(np.argmax(scores))
    if not np.isfinite(scores[best]):
        raise ValueError("Object has no measured visit with a valid reported error")
    selected_mask = measured[best]
    selected_flux = np.where(selected_mask, values[best], 0.0).astype(np.float32)
    selected_error = np.where(selected_mask, error[best], 0.0).astype(np.float32)
    return best, selected_flux, selected_error, selected_mask, float(scores[best])


def normalize_masked_profiles(
    values: np.ndarray,
    masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean-centre and unit-normalize profiles over measured bins only."""
    rows = np.asarray(values, dtype=np.float32)
    measured = np.asarray(masks, dtype=bool)
    was_one_dimensional = rows.ndim == 1
    if was_one_dimensional:
        rows = rows[None, :]
        measured = measured[None, :]
    if rows.ndim != 2 or measured.shape != rows.shape:
        raise ValueError("Profiles and masks must have matching one- or two-dimensional shapes")
    count = measured.sum(axis=1, keepdims=True)
    usable = count[:, 0] >= 2
    mean = np.divide(
        (rows * measured).sum(axis=1, keepdims=True),
        np.maximum(count, 1),
    )
    centred = (rows - mean) * measured
    norm = np.linalg.norm(centred, axis=1, keepdims=True)
    usable &= norm[:, 0] > 1.0e-8
    normalized = np.divide(
        centred,
        np.maximum(norm, 1.0e-8),
        out=np.zeros_like(centred),
    )
    final_mask = measured & usable[:, None]
    normalized *= final_mask
    if was_one_dimensional:
        return normalized[0], final_mask[0]
    return normalized, final_mask


def build_full_spectrum_atlas(
    whole_values: np.ndarray,
    whole_masks: np.ndarray,
    detail_values: np.ndarray,
    detail_masks: np.ndarray,
    class_index: np.ndarray,
    class_names: tuple[str, ...],
    rest_wavelength: np.ndarray,
    prototype_count: int,
) -> FullSpectrumAtlas:
    """Cluster clean object-level profiles independently within each class."""
    whole, whole_support = normalize_masked_profiles(whole_values, whole_masks)
    detail, detail_support = normalize_masked_profiles(detail_values, detail_masks)
    labels = np.asarray(class_index, dtype=np.int64)
    if whole.shape != detail.shape or whole_support.shape != detail_support.shape:
        raise ValueError(
            "Full-spectrum and continuum-removed training profiles must have "
            "matching shapes"
        )
    if len(labels) != len(whole):
        raise ValueError("Training labels must match the number of profiles")
    if prototype_count < 1:
        raise ValueError("Full-spectrum prototype count must be positive")

    classes = len(class_names)
    wavelength_bins = whole.shape[1]
    shape = (classes, prototype_count, wavelength_bins)
    whole_profiles = np.zeros(shape, dtype=np.float32)
    whole_masks_out = np.zeros(shape, dtype=bool)
    detail_profiles = np.zeros(shape, dtype=np.float32)
    detail_masks_out = np.zeros(shape, dtype=bool)
    support_counts = np.zeros((classes, prototype_count), dtype=np.int64)
    geometry = np.ones(wavelength_bins, dtype=bool)

    for class_value in range(classes):
        selected = labels == class_value
        selected &= whole_support.any(axis=1) & detail_support.any(axis=1)
        if not np.any(selected):
            raise ValueError(
                f"No usable full-spectrum training profiles for {class_names[class_value]}"
            )
        whole_cluster, whole_cluster_mask, support = _cluster_masked_profiles(
            whole[selected],
            whole_support[selected],
            geometry,
            prototype_count,
        )
        # Assign the same objects to detail clusters independently.  This keeps
        # each view free to represent its own diversity while preserving an
        # equal prototype budget per class.
        detail_cluster, detail_cluster_mask, detail_counts = _cluster_masked_profiles(
            detail[selected],
            detail_support[selected],
            geometry,
            prototype_count,
        )
        whole_profiles[class_value] = whole_cluster
        whole_masks_out[class_value] = whole_cluster_mask
        detail_profiles[class_value] = detail_cluster
        detail_masks_out[class_value] = detail_cluster_mask
        support_counts[class_value] = np.maximum(support, detail_counts)

    atlas = FullSpectrumAtlas(
        class_names=class_names,
        rest_wavelength=np.asarray(rest_wavelength, dtype=np.float32),
        whole_profiles=whole_profiles,
        whole_masks=whole_masks_out,
        detail_profiles=detail_profiles,
        detail_masks=detail_masks_out,
        support_counts=support_counts,
    )
    atlas.validate()
    return atlas


def build_phase_indexed_atlas(
    whole_values: np.ndarray,
    whole_masks: np.ndarray,
    detail_values: np.ndarray,
    detail_masks: np.ndarray,
    class_index: np.ndarray,
    phase_index: np.ndarray,
    class_names: tuple[str, ...],
    rest_wavelength: np.ndarray,
    phase_edges_days: np.ndarray,
    prototype_count: int,
) -> PhaseIndexedAtlas:
    """Build clean single-visit profiles for each supported class/phase cell."""
    whole, whole_support = normalize_masked_profiles(whole_values, whole_masks)
    detail, detail_support = normalize_masked_profiles(detail_values, detail_masks)
    labels = np.asarray(class_index, dtype=np.int64)
    phases = np.asarray(phase_index, dtype=np.int64)
    edges = np.asarray(phase_edges_days, dtype=np.float32)
    if whole.shape != detail.shape or whole_support.shape != detail_support.shape:
        raise ValueError(
            "Full-spectrum and continuum-removed phase profiles must have "
            "matching shapes"
        )
    if len(labels) != len(whole) or len(phases) != len(whole):
        raise ValueError("Phase profile labels must match the number of spectra")
    if len(edges) < 3 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("Phase profile edges must contain increasing phase ranges")
    if prototype_count < 1:
        raise ValueError("Phase profile count must be positive")

    classes = len(class_names)
    phase_count = len(edges) - 1
    wavelength_bins = whole.shape[1]
    shape = (classes, phase_count, prototype_count, wavelength_bins)
    whole_profiles = np.zeros(shape, dtype=np.float32)
    whole_masks_out = np.zeros(shape, dtype=bool)
    detail_profiles = np.zeros(shape, dtype=np.float32)
    detail_masks_out = np.zeros(shape, dtype=bool)
    support_counts = np.zeros(shape[:-1], dtype=np.int64)
    geometry = np.ones(wavelength_bins, dtype=bool)

    for class_value in range(classes):
        for phase_value in range(phase_count):
            selected = (labels == class_value) & (phases == phase_value)
            selected &= whole_support.any(axis=1) & detail_support.any(axis=1)
            if not np.any(selected):
                continue
            whole_cluster, whole_cluster_mask, whole_count = _cluster_masked_profiles(
                whole[selected],
                whole_support[selected],
                geometry,
                prototype_count,
            )
            detail_cluster, detail_cluster_mask, detail_count = _cluster_masked_profiles(
                detail[selected],
                detail_support[selected],
                geometry,
                prototype_count,
            )
            whole_profiles[class_value, phase_value] = whole_cluster
            whole_masks_out[class_value, phase_value] = whole_cluster_mask
            detail_profiles[class_value, phase_value] = detail_cluster
            detail_masks_out[class_value, phase_value] = detail_cluster_mask
            support_counts[class_value, phase_value] = np.maximum(
                whole_count, detail_count
            )

    atlas = PhaseIndexedAtlas(
        class_names=class_names,
        rest_wavelength=np.asarray(rest_wavelength, dtype=np.float32),
        phase_edges_days=edges,
        whole_profiles=whole_profiles,
        whole_masks=whole_masks_out,
        detail_profiles=detail_profiles,
        detail_masks=detail_masks_out,
        support_counts=support_counts,
    )
    atlas.validate()
    return atlas


def score_atlas_view(
    candidate_values: np.ndarray,
    candidate_masks: np.ndarray,
    profiles: np.ndarray,
    profile_masks: np.ndarray,
    support_counts: np.ndarray,
    *,
    minimum_rest_fraction: float = 0.15,
    minimum_shared_fraction: float = 0.8,
    prototype_temperature: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Score candidate redshifts against every class and prototype.

    Scores measure relative reference-profile agreement for architecture
    selection; they are not calibrated probabilities. ``support`` reports which
    class--redshift cells have enough common wavelength information to interpret.
    """
    if not 0.0 < minimum_rest_fraction <= 1.0:
        raise ValueError("Minimum rest-grid fraction must lie in (0, 1]")
    if not 0.0 < minimum_shared_fraction <= 1.0:
        raise ValueError("Minimum shared fraction must lie in (0, 1]")
    if prototype_temperature <= 0.0:
        raise ValueError("Prototype temperature must be positive")

    candidates, candidate_support = normalize_masked_profiles(
        candidate_values, candidate_masks
    )
    classes, prototypes_per_class, bins = profiles.shape
    if candidates.shape[1] != bins or profile_masks.shape != profiles.shape:
        raise ValueError("Candidate and atlas wavelength dimensions must match")
    if support_counts.shape != (classes, prototypes_per_class):
        raise ValueError("Prototype support counts do not match the atlas")

    flat_profiles = profiles.reshape(classes * prototypes_per_class, bins)
    flat_masks = profile_masks.reshape(classes * prototypes_per_class, bins)
    common = candidate_support.astype(np.float32) @ flat_masks.astype(np.float32).T
    numerator = candidates @ flat_profiles.T
    left_energy = (candidates * candidates) @ flat_masks.astype(np.float32).T
    right_energy = candidate_support.astype(np.float32) @ (
        flat_profiles * flat_profiles
    ).T
    denominator = np.sqrt(np.maximum(left_energy * right_energy, 1.0e-16))
    similarity = numerator / denominator

    candidate_count = candidate_support.sum(axis=1, keepdims=True)
    profile_count = flat_masks.sum(axis=1, keepdims=True).T
    shared_denominator = np.maximum(np.minimum(candidate_count, profile_count), 1)
    shared_fraction = common / shared_denominator
    rest_fraction = common / float(bins)
    available = np.broadcast_to(
        support_counts.reshape(1, classes * prototypes_per_class) > 0,
        similarity.shape,
    )
    valid = (
        available
        & (rest_fraction >= minimum_rest_fraction)
        & (shared_fraction >= minimum_shared_fraction)
    )
    similarity = similarity.reshape(-1, classes, prototypes_per_class)
    valid = valid.reshape(-1, classes, prototypes_per_class)

    scaled = np.where(valid, similarity / prototype_temperature, -np.inf)
    maximum = np.max(scaled, axis=2, keepdims=True)
    finite = np.isfinite(maximum)
    safe_maximum = np.where(finite, maximum, 0.0)
    shifted = np.zeros_like(scaled, dtype=np.float32)
    selected = valid & finite
    shifted[selected] = np.exp(
        (scaled - safe_maximum)[selected]
    ).astype(np.float32)
    count = valid.sum(axis=2)
    score = np.full(count.shape, -np.inf, dtype=np.float32)
    usable = count > 0
    log_mean = np.log(np.maximum(shifted.sum(axis=2), 1.0e-30)) - np.log(
        np.maximum(count, 1)
    )
    score[usable] = (
        prototype_temperature
        * (maximum[..., 0][usable] + log_mean[usable])
    ).astype(np.float32)
    return score, usable


def combine_view_scores(
    whole_score: np.ndarray,
    whole_support: np.ndarray,
    detail_score: np.ndarray,
    detail_support: np.ndarray,
    *,
    detail_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine full-spectrum and continuum-removed reference-match scores."""
    if whole_score.shape != detail_score.shape:
        raise ValueError(
            "Full-spectrum and continuum-removed scores must have matching shapes"
        )
    if whole_support.shape != whole_score.shape or detail_support.shape != detail_score.shape:
        raise ValueError(
            "Full-spectrum and continuum-removed support must match their scores"
        )
    if not 0.0 <= detail_fraction <= 1.0:
        raise ValueError("Continuum-removed fraction must lie in [0, 1]")
    support = whole_support & detail_support
    combined = np.full(whole_score.shape, -np.inf, dtype=np.float32)
    combined[support] = (
        (1.0 - detail_fraction) * whole_score[support]
        + detail_fraction * detail_score[support]
    )
    return combined, support


def score_phase_atlas_view(
    candidate_values: np.ndarray,
    candidate_masks: np.ndarray,
    profiles: np.ndarray,
    profile_masks: np.ndarray,
    support_counts: np.ndarray,
    **score_settings: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Score visit/redshift spectra against class/phase reference profiles."""
    values = np.asarray(candidate_values, dtype=np.float32)
    masks = np.asarray(candidate_masks, dtype=bool)
    if values.ndim < 2 or masks.shape != values.shape:
        raise ValueError("Phase candidate spectra and masks must have matching shapes")
    classes, phases, prototypes, bins = profiles.shape
    if values.shape[-1] != bins:
        raise ValueError("Phase candidate and atlas wavelength dimensions must match")
    if profile_masks.shape != profiles.shape:
        raise ValueError("Phase profile masks must match phase profiles")
    if support_counts.shape != (classes, phases, prototypes):
        raise ValueError("Phase profile support counts do not match profiles")
    leading = values.shape[:-1]
    score, support = score_atlas_view(
        values.reshape(-1, bins),
        masks.reshape(-1, bins),
        profiles.reshape(classes * phases, prototypes, bins),
        profile_masks.reshape(classes * phases, prototypes, bins),
        support_counts.reshape(classes * phases, prototypes),
        **score_settings,
    )
    output_shape = (*leading, classes, phases)
    return score.reshape(output_shape), support.reshape(output_shape)


def phase_sequence_match(
    visit_score: np.ndarray,
    visit_support: np.ndarray,
    observer_days: np.ndarray,
    redshift_grid: np.ndarray,
    phase_edges_days: np.ndarray,
    *,
    mode: str,
    starting_phase_grid: np.ndarray | None = None,
    truth_starting_phase: float | None = None,
    minimum_visits: int = 2,
    phase_averaging_scale: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine visit matches using no phase, averaged phase, or a truth upper bound.

    ``visit_score`` has shape ``(visits, redshifts, classes, phase_ranges)``.
    The truth-informed mode is diagnostic only and cannot be used for real data.
    """
    score = np.asarray(visit_score, dtype=np.float32)
    support = np.asarray(visit_support, dtype=bool)
    days = np.asarray(observer_days, dtype=np.float32)
    redshift = np.asarray(redshift_grid, dtype=np.float32)
    edges = np.asarray(phase_edges_days, dtype=np.float32)
    if score.ndim != 4 or support.shape != score.shape:
        raise ValueError(
            "Phase visit scores must have visit, redshift, class, and phase axes"
        )
    if len(days) != score.shape[0] or len(redshift) != score.shape[1]:
        raise ValueError("Phase visit times or redshift grid do not match scores")
    if len(edges) != score.shape[3] + 1 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("Phase edges do not match the phase score axis")
    if minimum_visits < 1:
        raise ValueError("Minimum phase visit count must be positive")
    if phase_averaging_scale <= 0.0:
        raise ValueError("Phase averaging scale must be positive")

    if mode == "phase_independent":
        by_visit, by_visit_support = _log_mean_supported(
            score,
            support,
            axis=3,
            scale=phase_averaging_scale,
        )
        count = by_visit_support.sum(axis=0)
        combined = np.divide(
            np.where(by_visit_support, by_visit, 0.0).sum(axis=0),
            np.maximum(count, 1),
        ).astype(np.float32)
        usable = count >= minimum_visits
        return np.where(usable, combined, -np.inf), usable

    relative_days = days - float(days.min())
    if mode == "phase_averaged":
        if starting_phase_grid is None:
            starting_phase_grid = np.linspace(
                float(edges[0]),
                float(edges[-2]),
                17,
                dtype=np.float32,
            )
        starts = np.asarray(starting_phase_grid, dtype=np.float32)
        if starts.ndim != 1 or not len(starts):
            raise ValueError("Starting phase grid must contain at least one value")
    elif mode == "truth_phase_upper_bound":
        if truth_starting_phase is None or not np.isfinite(truth_starting_phase):
            raise ValueError(
                "Truth-informed phase upper bound requires the simulated starting phase"
            )
        starts = np.asarray([truth_starting_phase], dtype=np.float32)
    else:
        raise ValueError(
            "Phase sequence mode must be 'phase_independent', 'phase_averaged', "
            "or 'truth_phase_upper_bound'"
        )

    trajectory_scores = []
    trajectory_support = []
    for starting_phase in starts:
        target_phase = starting_phase + relative_days[:, None] / (
            1.0 + redshift[None, :]
        )
        phase_index = np.searchsorted(edges, target_phase, side="right") - 1
        inside = (phase_index >= 0) & (phase_index < len(edges) - 1)
        safe_index = np.clip(phase_index, 0, len(edges) - 2)
        selected_score = np.zeros(score.shape[:3], dtype=np.float32)
        selected_support = np.zeros(score.shape[:3], dtype=bool)
        for visit in range(score.shape[0]):
            for redshift_index in range(score.shape[1]):
                phase_value = int(safe_index[visit, redshift_index])
                selected_score[visit, redshift_index] = score[
                    visit, redshift_index, :, phase_value
                ]
                selected_support[visit, redshift_index] = (
                    support[visit, redshift_index, :, phase_value]
                    & inside[visit, redshift_index]
                )
        count = selected_support.sum(axis=0)
        combined = np.divide(
            np.where(selected_support, selected_score, 0.0).sum(axis=0),
            np.maximum(count, 1),
        ).astype(np.float32)
        usable = count >= minimum_visits
        trajectory_scores.append(np.where(usable, combined, -np.inf))
        trajectory_support.append(usable)

    stacked_score = np.stack(trajectory_scores)
    stacked_support = np.stack(trajectory_support)
    if mode == "truth_phase_upper_bound":
        return stacked_score[0], stacked_support[0]
    return _log_mean_supported(
        stacked_score,
        stacked_support,
        axis=0,
        scale=phase_averaging_scale,
    )


def _log_mean_supported(
    values: np.ndarray,
    support: np.ndarray,
    *,
    axis: int,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    scaled = np.where(support, values / scale, -np.inf)
    maximum = np.max(scaled, axis=axis, keepdims=True)
    finite = np.isfinite(maximum)
    safe_maximum = np.where(finite, maximum, 0.0)
    shifted = np.zeros_like(scaled, dtype=np.float32)
    selected = support & finite
    shifted[selected] = np.exp(
        (scaled - safe_maximum)[selected]
    ).astype(np.float32)
    count = support.sum(axis=axis)
    usable = count > 0
    log_mean = np.log(np.maximum(shifted.sum(axis=axis), 1.0e-30)) - np.log(
        np.maximum(count, 1)
    )
    reduced_maximum = np.squeeze(maximum, axis=axis)
    output = np.full(count.shape, -np.inf, dtype=np.float32)
    output[usable] = (
        scale * (reduced_maximum[usable] + log_mean[usable])
    ).astype(np.float32)
    return output, usable
