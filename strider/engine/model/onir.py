"""Phase-neutral ONIR evidence from named rest-frame spectral regions."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from strider.engine.atlas import load_onir_bank
from strider.engine.config import project_path

from .redshift_scan import RestFrameScan
from .visit_evidence import combine_visit_scores


class PhaseNeutralOnirBranch(nn.Module):
    """Match trial-redshift windows without receiving dates or phase."""

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
        self.scan = RestFrameScan(observed_wavelength, bank.rest_grid, redshift_grid)
        initialization = str(settings.get("profile_initialization", "clean_mean"))
        if initialization == "clean_mean":
            profiles = bank.mean_profiles.copy()
            profile_masks = bank.mean_profile_mask.copy()
        elif initialization == "medoid":
            profiles = bank.medoid_profiles.copy()
            profile_masks = bank.medoid_profile_mask.copy()
        elif initialization == "random":
            profiles = np.random.default_rng(int(config["project"]["seed"])).normal(
                0.0, 1.0, size=bank.mean_profiles.shape
            ).astype(np.float32)
            profile_masks = np.broadcast_to(
                bank.window_mask[None, :, :], profiles.shape
            ).copy()
        else:
            raise ValueError(f"Unsupported ONIR profile initialization: {initialization}")
        self.profiles = nn.Parameter(torch.from_numpy(profiles))
        self.register_buffer("initial_profiles", torch.from_numpy(profiles.copy()))
        support = bank.support_counts >= int(settings.get("minimum_support", 1))
        missing_classes = np.flatnonzero(~support.any(axis=1))
        if len(missing_classes):
            names = ", ".join(expected_classes[index] for index in missing_classes)
            raise ValueError(
                f"The ONIR bank has no usable profile for declared class(es): {names}"
            )
        self.register_buffer("profile_supported", torch.from_numpy(support))
        self.register_buffer("profile_window_mask", torch.from_numpy(profile_masks))
        self.register_buffer("feature_weights", torch.from_numpy(bank.overlap_weights))
        self.register_buffer("feature_rest_wavelengths", torch.from_numpy(bank.rest_wavelengths))
        centers = np.abs(
            np.log(bank.rest_grid)[None, :] - np.log(bank.rest_wavelengths)[:, None]
        ).argmin(axis=1)
        radius = bank.window_mask.shape[1] // 2
        raw_indices = centers[:, None] + np.arange(-radius, radius + 1)[None, :]
        geometry_valid = (raw_indices >= 0) & (raw_indices < len(bank.rest_grid))
        geometry_valid &= bank.window_mask
        self.register_buffer(
            "window_indices",
            torch.from_numpy(np.clip(raw_indices, 0, len(bank.rest_grid) - 1).astype(np.int64)),
        )
        self.register_buffer("geometry_valid", torch.from_numpy(geometry_valid))
        self.minimum_valid_fraction = float(settings.get("minimum_valid_fraction", 0.8))
        self.input_mode = str(settings.get("input_mode", "individual_visits"))
        if self.input_mode not in {"individual_visits", "coadded_flux"}:
            raise ValueError(
                "ONIR input_mode must be 'individual_visits' or 'coadded_flux'"
            )
        self.evidence_scale = float(settings.get("evidence_scale", 10.0))
        if self.evidence_scale <= 0:
            raise ValueError("ONIR evidence_scale must be positive")
        self.class_count = len(bank.class_names)
        self.visit_evidence_exponent = float(
            settings.get("visit_evidence_exponent", 0.0)
        )
        self.feature_count = len(bank.feature_names)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        flux, wavelength_mask, visit_mask = self._spectral_input(batch)
        aligned_flux, aligned_mask = self.scan(flux, wavelength_mask)
        windows = aligned_flux[..., self.window_indices]
        measured = aligned_mask[..., self.window_indices]
        geometry = self.geometry_valid.to(measured.dtype)
        measured = measured * geometry
        active_count = geometry.sum(dim=-1).clamp_min(1.0)
        measured_fraction = measured.sum(dim=-1) / active_count
        feature_valid = measured_fraction >= self.minimum_valid_fraction

        mean = (windows * measured).sum(dim=-1) / measured.sum(dim=-1).clamp_min(1.0)
        centered = (windows - mean[..., None]) * measured
        profile_mask = self.profile_window_mask.to(self.profiles.dtype)
        centered_profiles = self.profiles * profile_mask
        centered_profiles = centered_profiles - (
            centered_profiles.sum(dim=-1, keepdim=True)
            / profile_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        ) * profile_mask
        numerator = torch.einsum("btzfw,cfw->btzfc", centered, centered_profiles)
        measured_energy = centered.square()
        profile_energy = centered_profiles.square()
        measured_common = torch.einsum(
            "btzfw,cfw->btzfc", measured_energy, profile_mask
        )
        profile_common = torch.einsum(
            "btzfw,cfw->btzfc", measured, profile_energy
        )
        scores = numerator / torch.sqrt(
            (measured_common * profile_common).clamp_min(1e-16)
        )

        support = self.profile_supported.transpose(0, 1)[None, None, None, :, :]
        common_count = torch.einsum(
            "btzfw,cfw->btzfc", measured, profile_mask
        )
        profile_count = profile_mask.sum(dim=-1).transpose(0, 1)
        enough_common_support = (
            common_count / profile_count[None, None, None, :, :].clamp_min(1.0)
            >= self.minimum_valid_fraction
        )
        weights = self.feature_weights[None, None, None, :, None]
        valid = feature_valid[..., None] & support & enough_common_support
        weighted = valid.to(scores.dtype) * weights
        per_visit = (scores * weighted).sum(dim=3) / weighted.sum(dim=3).clamp_min(1e-6)
        visit_valid = (weighted.sum(dim=3) > 0) & (visit_mask[:, :, None, None] > 0)
        joint_z_class = combine_visit_scores(
            per_visit, visit_valid, self.visit_evidence_exponent
        )  # (B,Z,C)
        joint_logits = (
            self.evidence_scale * joint_z_class.permute(0, 2, 1).contiguous()
        )
        joint_supported = visit_valid.any(dim=1).permute(0, 2, 1)
        joint_logits = joint_logits.masked_fill(~joint_supported, -1.0e4)

        per_feature_z_class = scores * valid.to(scores.dtype)
        visit_feature_valid = valid & (visit_mask[:, :, None, None, None] > 0)
        feature_evidence = (
            per_feature_z_class * visit_feature_valid.to(scores.dtype)
        ).sum(dim=1) / visit_feature_valid.sum(dim=1).clamp_min(1).to(scores.dtype)
        feature_evidence = feature_evidence.permute(0, 2, 3, 1).contiguous()
        return {
            "joint_logits": joint_logits,
            "joint_support": joint_supported,
            "feature_evidence": feature_evidence,
            "feature_evidence_support": visit_feature_valid.any(dim=1).permute(
                0, 2, 3, 1
            ).contiguous(),
            "profile_drift": self.profile_drift(),
            "mean_valid_feature_count": feature_valid.to(scores.dtype).sum(dim=-1).mean(),
        }

    def _spectral_input(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.input_mode == "individual_visits":
            return batch["flux"], batch["wavelength_mask"], batch["visit_mask"]
        measured = batch["wavelength_mask"] * batch["visit_mask"][:, :, None]
        count = measured.sum(dim=1)
        coadded = (batch["flux"] * measured).sum(dim=1) / count.clamp_min(1.0)
        coadded_mask = (count > 0).to(coadded.dtype)
        coadded = coadded * coadded_mask
        coadded_visit = (batch["visit_mask"].sum(dim=1, keepdim=True) > 0).to(coadded.dtype)
        return coadded[:, None, :], coadded_mask[:, None, :], coadded_visit

    def profile_drift(self) -> torch.Tensor:
        current = self.profiles.reshape(-1, self.profiles.shape[-1])
        initial = self.initial_profiles.reshape_as(current)
        supported = self.profile_supported.reshape(-1)
        if not supported.any():
            return self.profiles.sum() * 0.0
        cosine = functional.cosine_similarity(current[supported], initial[supported], dim=-1)
        return (1.0 - cosine).mean()
