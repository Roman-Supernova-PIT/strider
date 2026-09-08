"""Learned observer-frame tokens with an explicit ONIR redshift scan."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from strider.atlas import load_onir_bank
from strider.config import project_path

from .spectral_tokens import MaskAwareTokenEncoder, normalize_valid_bins
from .visit_evidence import combine_visit_scores


class EncodedOnirBranch(nn.Module):
    """Encode each visit once, then gather named regions for every trial redshift."""

    def __init__(
        self,
        config: dict[str, Any],
        observed_wavelength: np.ndarray,
        redshift_grid: np.ndarray,
    ) -> None:
        super().__init__()
        settings = config["onir"]
        bank = load_onir_bank(project_path(config, settings["bank_path"]))
        expected_classes = tuple(str(value) for value in config["model"]["classes"])
        if bank.class_names != expected_classes:
            raise ValueError(
                f"ONIR bank classes {bank.class_names} do not match model {expected_classes}"
            )
        token_dim = int(settings.get("token_dim", 16))
        self.encoder = MaskAwareTokenEncoder(
            token_dim,
            float(config["model"]["dropout"]),
            float(settings.get("minimum_encoder_support", 0.5)),
            str(settings.get("token_activation", "gelu")),
        )
        self.input_normalization = str(settings.get("input_normalization", "none"))
        if self.input_normalization not in {"none", "per_visit"}:
            raise ValueError("Encoded ONIR input_normalization must be 'none' or 'per_visit'")
        initialization = str(settings.get("profile_initialization", "clean_prototypes"))
        if initialization == "clean_prototypes":
            profiles = bank.prototype_profiles.copy()
            profile_masks = bank.prototype_profile_mask.copy()
            profile_support = bank.prototype_support_counts.copy()
        elif initialization == "clean_mean":
            profiles = bank.mean_profiles[:, :, None, :].copy()
            profile_masks = bank.mean_profile_mask[:, :, None, :].copy()
            profile_support = bank.support_counts[:, :, None].copy()
        else:
            raise ValueError(f"Unsupported encoded ONIR profile initialization: {initialization}")
        self.profiles = nn.Parameter(
            torch.from_numpy(profiles),
            requires_grad=bool(settings.get("train_profiles", True)),
        )
        self.register_buffer("initial_profiles", torch.from_numpy(profiles.copy()))
        self.register_buffer("raw_profile_mask", torch.from_numpy(profile_masks))
        minimum_profile_support = int(settings.get("minimum_prototype_support", 5))
        supported = (profile_support >= minimum_profile_support) & profile_masks.any(axis=-1)
        missing_classes = np.flatnonzero(~supported.any(axis=(1, 2)))
        if len(missing_classes):
            names = ", ".join(expected_classes[index] for index in missing_classes)
            raise ValueError(
                "The ONIR bank has no usable prototype for declared class(es): "
                f"{names}. Add training support, lower the explicit support threshold, "
                "or remove those classes from this model."
            )
        self.register_buffer("profile_supported", torch.from_numpy(supported))
        self.register_buffer("feature_weights", torch.from_numpy(bank.overlap_weights))
        self.register_buffer("feature_rest_wavelengths", torch.from_numpy(bank.rest_wavelengths))
        self.register_buffer("feature_radii", torch.from_numpy(bank.feature_radii))
        self.minimum_valid_fraction = float(settings.get("minimum_valid_fraction", 0.8))
        self.evidence_scale = float(settings.get("evidence_scale", 10.0))
        self.coverage_log_weight = float(settings.get("coverage_log_weight", 0.2))
        self.prototype_temperature = float(settings.get("prototype_temperature", 0.1))
        if self.prototype_temperature <= 0:
            raise ValueError("ONIR prototype_temperature must be positive")
        self.class_count = len(bank.class_names)
        self.visit_evidence_exponent = float(
            settings.get("visit_evidence_exponent", 0.0)
        )
        self.feature_count = len(bank.feature_names)
        self.prototype_count = profiles.shape[2]
        self.sample_count = int(settings.get("encoded_samples_per_feature", 9))
        self.temporal_representation_dim = token_dim
        if self.sample_count < 3 or self.sample_count % 2 == 0:
            raise ValueError("encoded_samples_per_feature must be an odd integer of at least three")

        observed_log = np.log(np.asarray(observed_wavelength, dtype=np.float64))
        observed_step = _uniform_log_step(observed_log, "observed wavelength")
        rest_log = np.log(np.asarray(bank.rest_grid, dtype=np.float64))
        rest_step = _uniform_log_step(rest_log, "ONIR rest wavelength")
        extent_fraction = float(settings.get("encoded_sample_extent_fraction", 0.8))
        radii_log = bank.feature_radii.astype(np.float64) * rest_step * extent_fraction
        unit_offsets = np.linspace(-1.0, 1.0, self.sample_count)
        sample_log_offsets = radii_log[:, None] * unit_offsets[None, :]

        target_log = (
            np.log1p(redshift_grid.astype(np.float64))[:, None, None]
            + np.log(bank.rest_wavelengths.astype(np.float64))[None, :, None]
            + sample_log_offsets[None, :, :]
        )
        observed_position = (target_log - observed_log[0]) / observed_step
        observed_indices = _fractional_indices(observed_position, len(observed_log))
        self._register_indices("observed", observed_indices)

        maximum_log_radius = float(bank.feature_radii.max()) * rest_step
        context_bins = 8
        canonical_radius = int(np.ceil(maximum_log_radius / observed_step)) + context_bins
        canonical_offsets = np.arange(-canonical_radius, canonical_radius + 1) * observed_step
        raw_position = canonical_offsets[None, :] / rest_step + bank.window_mask.shape[1] // 2
        raw_position = np.broadcast_to(
            raw_position,
            (self.feature_count, raw_position.shape[1]),
        ).copy()
        raw_valid = (
            np.abs(raw_position - bank.window_mask.shape[1] // 2)
            <= bank.feature_radii[:, None] + 1e-6
        )
        raw_indices = _fractional_indices(raw_position, bank.window_mask.shape[1])
        raw_indices = (*raw_indices[:3], raw_indices[3] & raw_valid)
        self._register_indices("profile_input", raw_indices)
        profile_position = canonical_radius + sample_log_offsets / observed_step
        profile_sample_indices = _fractional_indices(profile_position, len(canonical_offsets))
        self._register_indices("profile_sample", profile_sample_indices)

    def _register_indices(
        self,
        prefix: str,
        values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        lower, upper, weight, valid = values
        self.register_buffer(f"{prefix}_lower", torch.from_numpy(lower.astype(np.int64)))
        self.register_buffer(f"{prefix}_upper", torch.from_numpy(upper.astype(np.int64)))
        self.register_buffer(f"{prefix}_weight", torch.from_numpy(weight.astype(np.float32)))
        self.register_buffer(f"{prefix}_valid", torch.from_numpy(valid.astype(np.float32)))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        flux = batch["flux"]
        wavelength_mask = batch["wavelength_mask"]
        visit_mask = batch["visit_mask"]
        batch_size, visits, wavelength_bins = flux.shape
        tokens, token_mask = self.encode_observer_flux(flux, wavelength_mask)
        flat_tokens = tokens.reshape(batch_size * visits, wavelength_bins, tokens.shape[-1])
        flat_mask = token_mask.reshape(batch_size * visits, wavelength_bins)
        encoded_profiles, encoded_profile_mask = self._encoded_profiles()

        feature_scores = []
        feature_support = []
        feature_representations = []
        feature_representation_support = []
        for feature_index in range(self.feature_count):
            measured, measured_mask = interpolate_sequence(
                flat_tokens,
                flat_mask,
                self.observed_lower[:, feature_index],
                self.observed_upper[:, feature_index],
                self.observed_weight[:, feature_index],
                self.observed_valid[:, feature_index],
            )
            valid_fraction = measured_mask.mean(dim=-1)
            measured_valid = valid_fraction >= self.minimum_valid_fraction
            measured_summary = (
                measured * measured_mask[..., None]
            ).sum(dim=-2) / measured_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            profiles = encoded_profiles[:, feature_index]
            profile_mask = encoded_profile_mask[:, feature_index]
            similarity = common_support_cosine(
                measured, measured_mask, profiles, profile_mask
            )
            supported = self.profile_supported[:, feature_index]
            common_count = torch.einsum(
                "nzs,cks->nzck",
                measured_mask,
                profile_mask.to(measured_mask.dtype),
            )
            profile_count = profile_mask.sum(dim=-1).clamp_min(1.0)
            enough_common_support = (
                common_count / profile_count[None, None, :, :]
                >= self.minimum_valid_fraction
            )
            prototype_available = (
                supported[None, None, :, :] & enough_common_support
            )
            similarity = similarity.masked_fill(~prototype_available, -1.0e4)
            supported_count = prototype_available.sum(dim=-1)
            best = self.prototype_temperature * torch.logsumexp(
                similarity / self.prototype_temperature, dim=-1
            ) - self.prototype_temperature * torch.log(
                supported_count.clamp_min(1).to(similarity.dtype)
            )
            class_supported = supported_count > 0
            valid = measured_valid[..., None] & class_supported
            best = torch.where(valid, best, torch.zeros_like(best))
            feature_scores.append(best)
            feature_support.append(valid)
            feature_representations.append(measured_summary)
            feature_representation_support.append(measured_valid)

        scores = torch.stack(feature_scores, dim=2)
        valid = torch.stack(feature_support, dim=2)
        weights = self.feature_weights[None, None, :, None].to(scores.dtype)
        weighted = valid.to(scores.dtype) * weights
        denominator = weighted.sum(dim=2)
        mean_score = (scores * weighted).sum(dim=2) / denominator.clamp_min(1e-6)
        possible = (
            self.profile_supported.any(dim=-1).transpose(0, 1).to(scores.dtype)
            * self.feature_weights[:, None]
        ).sum(dim=0)
        coverage = denominator / possible[None, None, :].clamp_min(1e-6)
        per_visit = mean_score + self.coverage_log_weight * torch.log(coverage.clamp_min(1e-3))

        # Keep the region axis for the factored model. The original temporal
        # branch still receives the weighted mean below.
        representation = torch.stack(feature_representations, dim=2)  # (B*V,Z,F,D)
        representation_valid = torch.stack(
            feature_representation_support, dim=2
        )  # (B*V,Z,F)
        feature_visit_representation = representation.reshape(
            batch_size,
            visits,
            representation.shape[1],
            representation.shape[2],
            representation.shape[3],
        )  # (B,V,Z,F,D)
        feature_visit_support = representation_valid.reshape(
            batch_size,
            visits,
            representation_valid.shape[1],
            representation_valid.shape[2],
        )  # (B,V,Z,F)
        representation_weight = (
            representation_valid.to(representation.dtype)
            * self.feature_weights[None, None, :].to(representation.dtype)
        )
        visit_representation = (
            representation * representation_weight[..., None]
        ).sum(dim=2) / representation_weight.sum(dim=2, keepdim=True).clamp_min(1e-6)
        visit_representation = visit_representation.reshape(
            batch_size,
            visits,
            visit_representation.shape[1],
            visit_representation.shape[2],
        )  # (B,V,Z,D)

        per_visit = per_visit.reshape(batch_size, visits, per_visit.shape[1], self.class_count)
        visit_valid = (denominator > 0).reshape(
            batch_size, visits, denominator.shape[1], self.class_count
        ) & (visit_mask[:, :, None, None] > 0)
        joint = combine_visit_scores(
            per_visit, visit_valid, self.visit_evidence_exponent
        )  # (B,Z,C)
        joint_logits = self.evidence_scale * joint.permute(0, 2, 1).contiguous()
        joint_supported = visit_valid.any(dim=1).permute(0, 2, 1)
        joint_logits = joint_logits.masked_fill(~joint_supported, -1.0e4)

        by_feature = scores.reshape(
            batch_size,
            visits,
            scores.shape[1],
            self.feature_count,
            self.class_count,
        )
        by_feature_valid = valid.reshape_as(by_feature) & (visit_mask[:, :, None, None, None] > 0)
        feature_evidence = (by_feature * by_feature_valid.to(by_feature.dtype)).sum(dim=1)
        feature_evidence = feature_evidence / by_feature_valid.sum(dim=1).clamp_min(1)
        feature_evidence = feature_evidence.permute(0, 2, 3, 1).contiguous()
        return {
            "joint_logits": joint_logits,
            "joint_support": joint_supported,
            "spectral_joint_logits": joint_logits,
            "temporal_joint_logits": torch.zeros_like(joint_logits),
            "feature_evidence": feature_evidence,
            "feature_evidence_support": by_feature_valid.any(dim=1).permute(
                0, 2, 3, 1
            ).contiguous(),
            "feature_coverage": (
                coverage.reshape(
                    batch_size, visits, coverage.shape[1], self.class_count
                )
                * visit_mask[:, :, None, None]
            ).sum(dim=1).div(
                visit_mask.sum(dim=1, keepdim=True)[:, :, None].clamp_min(1.0)
            ).permute(0, 2, 1).contiguous(),
            "profile_drift": self.profile_drift(),
            "mean_valid_feature_count": valid.to(scores.dtype).sum(dim=2).mean(),
            "visit_representation": visit_representation,
            "feature_visit_representation": feature_visit_representation,
            "feature_visit_support": feature_visit_support,
            "observer_tokens": tokens,
            "observer_token_mask": token_mask,
        }

    def encode_observer_flux(
        self,
        flux: torch.Tensor,
        wavelength_mask: torch.Tensor,
        normalize: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if normalize is None:
            normalize = self.input_normalization == "per_visit"
        encoder_flux = (
            normalize_valid_bins(flux, wavelength_mask) if normalize else flux
        )
        return self.encoder(encoder_flux, wavelength_mask)

    def _encoded_profiles(self) -> tuple[torch.Tensor, torch.Tensor]:
        canonical_profiles = []
        canonical_masks = []
        for feature_index in range(self.feature_count):
            raw = self.profiles[:, feature_index]
            raw_support = self.raw_profile_mask[:, feature_index].to(raw.dtype)
            lower = self.profile_input_lower[feature_index]
            upper = self.profile_input_upper[feature_index]
            weight = self.profile_input_weight[feature_index]
            valid = self.profile_input_valid[feature_index]
            measured = raw_support[..., lower] * raw_support[..., upper] * valid
            values = raw[..., lower] * (1.0 - weight) + raw[..., upper] * weight
            values = values * measured
            canonical_profiles.append(values)
            canonical_masks.append(measured)
        canonical = torch.stack(canonical_profiles, dim=1)
        canonical_mask = torch.stack(canonical_masks, dim=1).to(canonical.dtype)
        if self.input_normalization == "per_visit":
            canonical = normalize_valid_bins(canonical, canonical_mask)
        encoded, encoded_mask = self.encoder(canonical, canonical_mask)
        sampled_profiles = []
        sampled_masks = []
        for feature_index in range(self.feature_count):
            feature_tokens = encoded[:, feature_index].reshape(
                self.class_count * self.prototype_count,
                encoded.shape[-2],
                encoded.shape[-1],
            )
            feature_mask = encoded_mask[:, feature_index].reshape(
                self.class_count * self.prototype_count,
                encoded_mask.shape[-1],
            )
            sampled, sampled_mask = interpolate_sequence(
                feature_tokens,
                feature_mask,
                self.profile_sample_lower[feature_index],
                self.profile_sample_upper[feature_index],
                self.profile_sample_weight[feature_index],
                self.profile_sample_valid[feature_index],
            )
            sampled_profiles.append(
                sampled.reshape(
                    self.class_count,
                    self.prototype_count,
                    self.sample_count,
                    encoded.shape[-1],
                )
            )
            sampled_masks.append(
                sampled_mask.reshape(
                    self.class_count,
                    self.prototype_count,
                    self.sample_count,
                )
            )
        return torch.stack(sampled_profiles, dim=1), torch.stack(sampled_masks, dim=1)

    def profile_drift(self) -> torch.Tensor:
        supported = self.profile_supported
        current = self.profiles[supported]
        initial = self.initial_profiles[supported]
        if not len(current):
            return self.profiles.sum() * 0.0
        cosine = functional.cosine_similarity(current, initial, dim=-1)
        return (1.0 - cosine).mean().clamp_min(0.0)


def interpolate_sequence(
    sequence: torch.Tensor,
    support: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    upper_weight: torch.Tensor,
    geometry_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear interpolation over a token sequence for arbitrary target shapes."""
    low = sequence[:, lower]
    high = sequence[:, upper]
    weight = upper_weight[None, ..., None].to(sequence.dtype)
    values = low * (1.0 - weight) + high * weight
    measured = support[:, lower] * support[:, upper] * geometry_valid[None, ...]
    return values * measured[..., None], measured


def common_support_cosine(
    measured: torch.Tensor,
    measured_mask: torch.Tensor,
    profiles: torch.Tensor,
    profile_mask: torch.Tensor,
) -> torch.Tensor:
    """Cosine similarity over samples measured in both inputs.

    measured is ``(N,Z,S,D)`` and profiles is ``(C,K,S,D)``. The result is
    ``(N,Z,C,K)`` for object-visits, redshifts, classes and prototypes.
    """
    numerator = torch.einsum("nzsd,cksd->nzck", measured, profiles)
    measured_energy = measured.square().sum(dim=-1)  # (N,Z,S)
    profile_energy = profiles.square().sum(dim=-1)  # (C,K,S)
    measured_common = torch.einsum(
        "nzs,cks->nzck", measured_energy, profile_mask.to(measured.dtype)
    )
    profile_common = torch.einsum(
        "nzs,cks->nzck", measured_mask.to(profiles.dtype), profile_energy
    )
    denominator = torch.sqrt((measured_common * profile_common).clamp_min(1e-16))
    return numerator / denominator


def _uniform_log_step(values: np.ndarray, name: str) -> float:
    steps = np.diff(values)
    step = float(np.median(steps))
    if not np.allclose(steps, step, rtol=1e-4, atol=1e-8):
        raise ValueError(f"{name} grid must be uniformly spaced in log wavelength")
    return step


def _fractional_indices(
    position: np.ndarray,
    length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lower = np.floor(position).astype(np.int64)
    upper = lower + 1
    valid = (lower >= 0) & (upper < int(length))
    weight = position - lower
    return (
        np.clip(lower, 0, length - 1),
        np.clip(upper, 0, length - 1),
        weight,
        valid,
    )
