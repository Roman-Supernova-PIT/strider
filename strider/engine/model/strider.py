"""Joint class-redshift model with explicit spectral and time-series axes."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from strider.engine.wavelength import log_wavelength_grid

from .phase import CandidatePhaseEmbedding
from .phase_consistency import CandidatePhaseConsistency
from .phase_prediction import PhasePredictionHead
from .onir import PhaseNeutralOnirBranch
from .encoded_onir import EncodedOnirBranch
from .dense_scan import DenseRestFrameEvidence, WholeDetailRestFrameEvidence
from .evidence_sufficiency import EvidenceSufficiencyHead
from .factored_attention import FactoredOnirEvidence
from .redshift_scan import RestFrameScan, build_redshift_grid, redshift_cell_widths
from .spectral_context import FullSpectrumContext
from .spectral_encoder import SpectralEncoder
from .spectral_tokens import (
    MaskAwareContinuumRemoval,
    object_normalized_flux_amplitude,
    relative_visit_amplitude,
    velocity_sigma_to_log_bins,
)
from .temporal import SpectralEvolutionEvidence


MODEL_INPUT_NAMES = (
    "flux",
    "wavelength_mask",
    "visit_mask",
    "observer_days",
)
OPTIONAL_MODEL_INPUT_NAMES = (
    "visit_flux_scale",
    "peak_day_offset",
    "peak_date_valid",
    "flux_error_shape",
)


def measurement_inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Keep simulation labels outside the model call."""
    missing = [name for name in MODEL_INPUT_NAMES if name not in batch]
    if missing:
        raise KeyError("Model input is missing: " + ", ".join(missing))
    inputs = {name: batch[name] for name in MODEL_INPUT_NAMES}
    inputs.update(
        {name: batch[name] for name in OPTIONAL_MODEL_INPUT_NAMES if name in batch}
    )
    return inputs


class Strider3(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        observation = config["observation"]
        model = config["model"]
        observed_wavelength = log_wavelength_grid(
            observation["wavelength_min"],
            observation["wavelength_max"],
            observation["wavelength_bins"],
        )
        self.register_buffer(
            "amplitude_wavelength_weight",
            torch.from_numpy(np.gradient(observed_wavelength).astype(np.float32)),
            persistent=False,
        )
        rest_wavelength = log_wavelength_grid(
            model["rest_wavelength_min"],
            model["rest_wavelength_max"],
            model["rest_wavelength_bins"],
        )
        redshift_grid = build_redshift_grid(
            float(model["redshift_min"]),
            float(model["redshift_max"]),
            int(model["redshift_bins"]),
            str(model.get("redshift_spacing", "linear")),
        )
        self.register_buffer("redshift_grid", torch.from_numpy(redshift_grid))
        self.register_buffer(
            "redshift_cell_width",
            torch.from_numpy(redshift_cell_widths(redshift_grid)),
        )
        self.redshift_prior = str(model.get("redshift_prior", "flat_z"))
        self.class_names = tuple(str(name) for name in model["classes"])
        self.architecture = str(model.get("architecture", "full_scan"))
        self.use_flux_error_channel = bool(
            model.get("use_flux_error_channel", False)
        )
        if self.use_flux_error_channel:
            initial_error_scale = float(
                model.get("flux_error_initial_scale", 0.1)
            )
            if not -0.99 < initial_error_scale < 0.99:
                raise ValueError("flux_error_initial_scale must lie in (-0.99, 0.99)")
            self.flux_error_scale_raw = nn.Parameter(
                torch.tensor(np.arctanh(initial_error_scale), dtype=torch.float32)
            )
        else:
            self.register_parameter("flux_error_scale_raw", None)
        self.relative_amplitude_mode = str(
            model.get("relative_amplitude_mode", "signal_to_background")
        )
        if self.relative_amplitude_mode not in {
            "signal_to_background",
            "object_normalized_flux",
        }:
            raise ValueError(
                "relative_amplitude_mode must be 'signal_to_background' or "
                "'object_normalized_flux'"
            )
        dropout = float(model["dropout"])
        self.temporal_evidence: SpectralEvolutionEvidence | None = None
        self.factored_evidence: FactoredOnirEvidence | None = None
        self.full_spectrum_context: FullSpectrumContext | None = None
        self.dense_scan: (
            DenseRestFrameEvidence | WholeDetailRestFrameEvidence | None
        ) = None
        self.continuum_removal: MaskAwareContinuumRemoval | None = None
        configured_dense_view = model.get("dense_scan_view")
        self.dense_scan_view = str(
            configured_dense_view
            if configured_dense_view is not None
            else (
                "blend"
                if bool(model.get("dense_continuum_detail", False))
                else "whole"
            )
        )
        if self.dense_scan_view not in {"whole", "detail", "blend"}:
            raise ValueError(
                "dense_scan_view must be 'whole', 'detail', or 'blend'"
            )
        self.phase_consistency: CandidatePhaseConsistency | None = None
        if bool(model.get("dense_continuum_detail", False)) and not bool(
            model.get("dense_rest_frame_scan", False)
        ):
            raise ValueError(
                "Continuum-detail scanning requires dense_rest_frame_scan: true"
            )
        if self.dense_scan_view in {"detail", "blend"} and not bool(
            model.get("dense_continuum_detail", False)
        ):
            raise ValueError(
                "detail and blend dense_scan_view values require "
                "dense_continuum_detail: true"
            )
        if self.dense_scan_view == "whole" and bool(
            model.get("dense_continuum_detail", False)
        ):
            raise ValueError(
                "dense_scan_view: whole requires dense_continuum_detail: false"
            )
        if (
            bool(model.get("dense_rest_frame_scan", False))
            and self.architecture != "factored_onir"
        ):
            raise ValueError(
                "Dense rest-frame scanning requires architecture: factored_onir"
            )
        if self.architecture == "onir":
            self.onir = PhaseNeutralOnirBranch(config, observed_wavelength, redshift_grid)
            self.scan = None
            if str(model.get("temporal_mode", "none")) != "none":
                raise ValueError("The raw-profile ONIR branch supports temporal_mode: none")
        elif self.architecture in {"encoded_onir", "factored_onir"}:
            self.onir = EncodedOnirBranch(config, observed_wavelength, redshift_grid)
            self.scan = None
            self.temporal_mode = str(model.get("temporal_mode", "none"))
            if self.temporal_mode not in {"none", "spectral_evolution"}:
                raise ValueError(
                    "Encoded ONIR temporal_mode must be 'none' or 'spectral_evolution'"
                )
            if self.architecture == "factored_onir":
                if self.temporal_mode != "spectral_evolution":
                    raise ValueError(
                        "Factored ONIR requires temporal_mode: spectral_evolution"
                    )
                self.factored_evidence = FactoredOnirEvidence(
                    hidden_dim=self.onir.temporal_representation_dim,
                    class_count=len(self.class_names),
                    attention_heads=int(model.get("factored_attention_heads", 4)),
                    dropout=dropout,
                    shape_initial_scale=float(
                        model.get("factored_shape_initial_scale", 0.5)
                    ),
                    temporal_initial_scale=float(
                        model.get("temporal_initial_scale", 0.0)
                    ),
                    relative_brightness=bool(
                        model.get("relative_brightness_evolution", False)
                    ),
                    brightness_initial_scale=float(
                        model.get("relative_brightness_initial_scale", 0.0)
                    ),
                )
                if bool(model.get("full_spectrum_context", False)):
                    self.full_spectrum_context = FullSpectrumContext(
                        wavelength_bins=int(observation["wavelength_bins"]),
                        token_dim=int(
                            model.get(
                                "context_token_dim",
                                self.onir.temporal_representation_dim,
                            )
                        ),
                        class_count=len(self.class_names),
                        patch_size=int(model.get("context_patch_size", 8)),
                        attention_heads=int(
                            model.get("context_attention_heads", 4)
                        ),
                        attention_layers=int(
                            model.get("context_attention_layers", 2)
                        ),
                        dropout=dropout,
                        initial_scale=float(model.get("context_initial_scale", 0.25)),
                        input_normalization=str(
                            model.get(
                                "context_input_normalization",
                                config["onir"].get("input_normalization", "none"),
                            )
                        ),
                        use_visit_attention=bool(
                            model.get("context_visit_attention", False)
                        ),
                        minimum_support=float(
                            model.get(
                                "context_minimum_support",
                                config["onir"].get("minimum_encoder_support", 0.5),
                            )
                        ),
                    )
                if bool(model.get("dense_rest_frame_scan", False)):
                    dense_arguments = dict(
                        observed_wavelength=observed_wavelength,
                        rest_wavelength=rest_wavelength,
                        redshift_grid=redshift_grid,
                        hidden_dim=self.onir.temporal_representation_dim,
                        token_dim=int(model.get("dense_scan_token_dim", 16)),
                        class_count=len(self.class_names),
                        patch_size=int(model.get("dense_scan_patch_size", 8)),
                        rest_bins=int(model.get("dense_scan_rest_bins", 256)),
                        initial_scale=float(
                            model.get("dense_scan_initial_scale", 0.0)
                        ),
                        evidence_scale=float(
                            model.get("dense_scan_evidence_scale", 10.0)
                        ),
                        redshift_chunk_size=int(
                            model.get("dense_scan_chunk_size", 16)
                        ),
                        minimum_overlap=float(
                            model.get("dense_scan_minimum_overlap", 0.25)
                        ),
                        overlap_exponent=float(
                            model.get("dense_scan_overlap_exponent", 1.0)
                        ),
                    )
                    if bool(model.get("dense_continuum_detail", False)):
                        has_bin_width = "dense_continuum_sigma_bins" in model
                        has_velocity_width = "dense_continuum_sigma_km_s" in model
                        if has_bin_width and has_velocity_width:
                            raise ValueError(
                                "Set only one dense continuum smoothing width"
                            )
                        if has_velocity_width:
                            continuum_sigma_bins = velocity_sigma_to_log_bins(
                                float(observation["wavelength_min"]),
                                float(observation["wavelength_max"]),
                                int(observation["wavelength_bins"]),
                                float(model["dense_continuum_sigma_km_s"]),
                            )
                        else:
                            continuum_sigma_bins = float(
                                model.get("dense_continuum_sigma_bins", 24.0)
                            )
                        self.continuum_removal = MaskAwareContinuumRemoval(
                            continuum_sigma_bins
                        )
                        self.dense_scan = WholeDetailRestFrameEvidence(
                            **dense_arguments,
                            initial_detail_weight=float(
                                model.get("dense_initial_detail_weight", 0.5)
                            ),
                            minimum_whole_weight=float(
                                model.get("dense_minimum_whole_weight", 0.0)
                            ),
                            scan_view=self.dense_scan_view,
                        )
                    else:
                        self.dense_scan = DenseRestFrameEvidence(**dense_arguments)
            elif self.temporal_mode == "spectral_evolution":
                self.temporal_evidence = SpectralEvolutionEvidence(
                    hidden_dim=self.onir.temporal_representation_dim,
                    class_count=len(self.class_names),
                    dropout=dropout,
                    initial_scale=float(model.get("temporal_initial_scale", 0.0)),
                )
        elif self.architecture == "full_scan":
            self.scan = RestFrameScan(observed_wavelength, rest_wavelength, redshift_grid)
            self.onir = None
        else:
            raise ValueError(f"Unsupported model architecture: {self.architecture}")
        if self.architecture == "full_scan":
            hidden_dim = int(model["hidden_dim"])
            self.spectral_encoder = SpectralEncoder(hidden_dim, dropout)
            self.temporal_mode = str(
                model.get(
                    "temporal_mode",
                    "candidate_phase_addition" if model.get("use_phase", True) else "none",
                )
            )
            if self.temporal_mode not in {
                "none",
                "candidate_phase_addition",
                "spectral_evolution",
            }:
                raise ValueError(f"Unsupported temporal mode: {self.temporal_mode}")
            self.phase_embedding = (
                CandidatePhaseEmbedding(int(model["phase_features"]), hidden_dim)
                if self.temporal_mode == "candidate_phase_addition"
                else None
            )
            self.visit_score = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, len(self.class_names)),
            )
            self.temporal_evidence = (
                SpectralEvolutionEvidence(
                    hidden_dim=hidden_dim,
                    class_count=len(self.class_names),
                    dropout=dropout,
                    initial_scale=float(model.get("temporal_initial_scale", 0.0)),
                )
                if self.temporal_mode == "spectral_evolution"
                else None
            )
        self.evidence_sufficiency = EvidenceSufficiencyHead(
            dropout,
            visit_count_reference=int(model.get("evidence_visit_count_reference", 12)),
            use_visit_count_and_span=bool(
                model.get("evidence_use_visit_count_and_span", True)
            ),
        )
        phase_bins = int(model.get("phase_auxiliary_bins", 0))
        self.phase_head: PhasePredictionHead | None = None
        if phase_bins:
            if phase_bins < 2:
                raise ValueError("phase_auxiliary_bins must be zero or at least two")
            if self.architecture in {"encoded_onir", "factored_onir"}:
                phase_hidden_dim = self.onir.temporal_representation_dim
            elif self.architecture == "full_scan":
                phase_hidden_dim = int(model["hidden_dim"])
            else:
                raise ValueError(
                    "The auxiliary phase head requires encoded spectra"
                )
            phase_min = float(model.get("phase_auxiliary_min_days", -20.0))
            phase_max = float(model.get("phase_auxiliary_max_days", 50.0))
            if phase_max <= phase_min:
                raise ValueError(
                    "phase_auxiliary_max_days must exceed phase_auxiliary_min_days"
                )
            self.register_buffer(
                "phase_grid",
                torch.linspace(phase_min, phase_max, phase_bins),
            )
            self.phase_head = PhasePredictionHead(
                phase_hidden_dim,
                len(self.class_names),
                phase_bins,
                dropout,
            )
        if bool(model.get("candidate_phase_consistency", False)):
            if self.phase_head is None:
                raise ValueError(
                    "Candidate phase consistency requires the auxiliary phase head"
                )
            self.phase_consistency = CandidatePhaseConsistency(
                initial_scale=float(model.get("candidate_phase_initial_scale", 0.0)),
                minimum_visits=int(model.get("candidate_phase_minimum_visits", 2)),
                use_peak_date=bool(
                    model.get("candidate_phase_use_peak_date", False)
                ),
                peak_uncertainty_days=float(
                    model.get("candidate_phase_peak_uncertainty_days", 0.0)
                ),
                peak_quadrature_points=int(
                    model.get("candidate_phase_peak_quadrature_points", 1)
                ),
                peak_outlier_fraction=float(
                    model.get("candidate_phase_peak_outlier_fraction", 0.0)
                ),
                minimum_coverage_fraction=float(
                    model.get("candidate_phase_minimum_coverage_fraction", 0.5)
                ),
            )

    def _relative_amplitude(
        self, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if self.relative_amplitude_mode == "signal_to_background":
            return relative_visit_amplitude(
                batch["flux"],
                batch["wavelength_mask"],
                batch["visit_mask"],
            )
        if "visit_flux_scale" not in batch:
            raise KeyError(
                "Object-normalized FLAM evolution requires visit_flux_scale"
            )
        return object_normalized_flux_amplitude(
            batch["flux"],
            batch["wavelength_mask"],
            batch["visit_mask"],
            batch["visit_flux_scale"],
            self.amplitude_wavelength_weight,
        )

    def _measurement_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Optionally expose relative FLAMERR shape to the existing model routes."""
        if not self.use_flux_error_channel:
            return batch
        if "flux_error_shape" not in batch:
            raise KeyError(
                "use_flux_error_channel requires data.include_flux_error_channel"
            )
        if batch["flux_error_shape"].shape != batch["flux"].shape:
            raise ValueError("flux and flux_error_shape must have the same shape")
        result = dict(batch)
        scale = torch.tanh(self.flux_error_scale_raw)
        result["flux"] = (
            batch["flux"] + scale * batch["flux_error_shape"]
        ) * batch["wavelength_mask"]
        return result

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = self._measurement_batch(batch)
        if self.architecture in {"onir", "encoded_onir", "factored_onir"}:
            output = self.onir(batch)
            feature_representation = output.pop("feature_visit_representation", None)
            feature_support = output.pop("feature_visit_support", None)
            visit_representation = output.pop("visit_representation", None)
            observer_tokens = output.pop("observer_tokens", None)
            observer_token_mask = output.pop("observer_token_mask", None)
            if self.factored_evidence is not None:
                if feature_representation is None or feature_support is None:
                    raise RuntimeError("Factored ONIR representation is missing")
                factored = self.factored_evidence(
                    feature_representation,
                    feature_support,
                    batch["observer_days"],
                    batch["visit_mask"],
                    self.redshift_grid,
                    self._relative_amplitude(batch)
                    if self.factored_evidence.brightness_projection is not None
                    else None,
                )
                onir_logits = output["joint_logits"]
                joint_support = output["joint_support"]
                dense = None
                if self.dense_scan is not None:
                    if observer_tokens is None or observer_token_mask is None:
                        raise RuntimeError("Dense scan observer tokens are missing")
                    if self.continuum_removal is not None:
                        detail_flux = self.continuum_removal(
                            batch["flux"], batch["wavelength_mask"]
                        )
                        detail_tokens, detail_mask = self.onir.encode_observer_flux(
                            detail_flux,
                            batch["wavelength_mask"],
                            normalize=True,
                        )
                        dense = self.dense_scan(
                            observer_tokens,
                            observer_token_mask,
                            detail_tokens,
                            detail_mask,
                            batch["visit_mask"],
                        )
                    else:
                        dense = self.dense_scan(
                            observer_tokens,
                            observer_token_mask,
                            batch["visit_mask"],
                        )
                spectral_logits = onir_logits + factored["shape_joint_logits"]
                if self.full_spectrum_context is not None:
                    context = self.full_spectrum_context(
                        batch["flux"],
                        batch["wavelength_mask"],
                        batch["visit_mask"],
                    )
                    context_joint = context["scaled_context_class_logits"][..., None]
                    context_joint = context_joint.expand(
                        -1, -1, len(self.redshift_grid)
                    )
                    spectral_logits = spectral_logits + context_joint
                    context["context_joint_logits"] = context_joint
                    output.update(context)
                if dense is not None:
                    spectral_logits = spectral_logits + dense["dense_scan_joint_logits"]
                    output.update(dense)
                spectral_logits = spectral_logits.masked_fill(~joint_support, -1.0e4)
                output["onir_joint_logits"] = onir_logits
                output["spectral_joint_logits"] = spectral_logits
                output["joint_support"] = joint_support
                output["joint_logits"] = (
                    spectral_logits + factored["temporal_joint_logits"]
                ).masked_fill(~joint_support, -1.0e4)
                output.update(factored)
            elif self.temporal_evidence is not None:
                temporal_logits, raw_temporal_logits = self.temporal_evidence(
                    visit_representation,
                    batch["observer_days"],
                    batch["visit_mask"],
                    self.redshift_grid,
                )
                output["spectral_joint_logits"] = output["joint_logits"]
                output["joint_logits"] = output["joint_logits"] + temporal_logits
                output["temporal_joint_logits"] = temporal_logits
                output["raw_temporal_joint_logits"] = raw_temporal_logits
            if self.phase_head is not None:
                if visit_representation is None:
                    raise RuntimeError("Encoded visit representation is missing")
                output["phase_logits"] = self.phase_head(visit_representation)
                output["phase_grid"] = self.phase_grid
                self._add_phase_consistency(output, batch)
            output["evidence_sufficiency_logit"] = self.evidence_sufficiency(batch)
            return output
        aligned_flux, aligned_mask = self.scan(batch["flux"], batch["wavelength_mask"])
        spectral = self.spectral_encoder(aligned_flux, aligned_mask)
        if self.phase_embedding is not None:
            scoring_features = spectral + self.phase_embedding(
                batch["observer_days"], self.redshift_grid
            )
        else:
            scoring_features = spectral
        visit_logits = self.visit_score(scoring_features)
        visit_mask = batch["visit_mask"][:, :, None, None]
        spectral_logits = (visit_logits * visit_mask).sum(dim=1) / visit_mask.sum(dim=1).clamp_min(1.0)
        spectral_logits = spectral_logits.permute(0, 2, 1).contiguous()
        if self.temporal_evidence is None:
            temporal_logits = torch.zeros_like(spectral_logits)
            raw_temporal_logits = torch.zeros_like(spectral_logits)
        else:
            temporal_logits, raw_temporal_logits = self.temporal_evidence(
                spectral,
                batch["observer_days"],
                batch["visit_mask"],
                self.redshift_grid,
            )
        joint_logits = spectral_logits + temporal_logits
        output = {
            "joint_logits": joint_logits,
            "evidence_sufficiency_logit": self.evidence_sufficiency(batch),
            "visit_logits": visit_logits,
            "spectral_joint_logits": spectral_logits,
            "temporal_joint_logits": temporal_logits,
            "raw_temporal_joint_logits": raw_temporal_logits,
        }
        if self.phase_head is not None:
            output["phase_logits"] = self.phase_head(spectral)
            output["phase_grid"] = self.phase_grid
            self._add_phase_consistency(output, batch)
        return output

    def _add_phase_consistency(
        self,
        output: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> None:
        if self.phase_consistency is None:
            return
        phase_logits, raw_phase_logits = self.phase_consistency(
            output["phase_logits"],
            batch["observer_days"],
            batch["visit_mask"],
            self.redshift_grid,
            self.phase_grid,
            batch.get("peak_day_offset"),
            batch.get("peak_date_valid"),
        )
        output["phase_consistency_joint_logits"] = phase_logits
        output["raw_phase_consistency_joint_logits"] = raw_phase_logits
        output["joint_logits"] = output["joint_logits"] + phase_logits
        output["temporal_joint_logits"] = (
            output.get("temporal_joint_logits", torch.zeros_like(phase_logits))
            + phase_logits
        )
