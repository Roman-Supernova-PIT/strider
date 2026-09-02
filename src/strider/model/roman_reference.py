"""Coadd-first Roman class--redshift matching with phase-aware references."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from torch import nn

from strider.atlas.roman_reference import RomanReferenceBank
from strider.config import project_path
from strider.data.classes import (
    class_names_for_scheme,
    fine_to_output_class_indices,
)

from .coadd import (
    cosine_edge_taper,
    minimum_relative_precision_mask,
    relative_inverse_variance,
)
from .spectral_tokens import (
    MaskAwareContinuumRemoval,
    MaskAwareMultiscaleAttentionEncoder,
    MaskAwareTokenEncoder,
    normalize_valid_bins,
    object_normalized_flux_amplitude,
    velocity_sigma_to_log_bins,
)


class CandidateTemporalTransformer(nn.Module):
    """Relate class--phase match patterns across measured observer times.

    Each visit is represented by its complete fine-class by possible-starting-
    phase match grid for one candidate redshift.  The transformer therefore
    sees how reference evidence changes across the measured sequence without
    receiving the object's simulated class, redshift, or rest-frame phase.
    """

    def __init__(
        self,
        fine_class_count: int,
        starting_phase_count: int,
        hidden_dim: int,
        attention_heads: int,
        layers: int,
        feedforward_multiplier: int,
        dropout: float,
        initial_correction_scale: float,
        use_signal_to_noise: bool = True,
        use_relative_flux: bool = False,
    ) -> None:
        super().__init__()
        if fine_class_count < 1 or starting_phase_count < 1:
            raise ValueError("Temporal match dimensions must be positive")
        if hidden_dim < 1 or attention_heads < 1 or hidden_dim % attention_heads:
            raise ValueError(
                "Temporal hidden dimension must be divisible by the attention heads"
            )
        if layers < 1 or feedforward_multiplier < 1:
            raise ValueError("Temporal layers and feedforward multiplier must be positive")
        if not 0.0 < initial_correction_scale < 1.0:
            raise ValueError("Initial temporal correction scale must lie in (0, 1)")
        self.fine_class_count = int(fine_class_count)
        self.starting_phase_count = int(starting_phase_count)
        self.use_signal_to_noise = bool(use_signal_to_noise)
        self.use_relative_flux = bool(use_relative_flux)
        match_count = self.fine_class_count * self.starting_phase_count
        # Only change around the masked mean enters the learned path. Static
        # class--redshift evidence remains in the explicit reference baseline.
        # The next three features are rest-frame offset under the candidate
        # redshift and interval magnitude.  The legacy model can additionally
        # receive measured visit S/N; the scientific control disables that
        # learned route while retaining S/N for deterministic visit selection.
        # An optional final feature carries only within-object relative flux
        # evolution; one overall flux scale is removed first.
        self.input_projection = nn.Sequential(
            nn.Linear(
                match_count
                + 2
                + int(self.use_signal_to_noise)
                + int(self.use_relative_flux),
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=feedforward_multiplier * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, match_count)
        # Begin as an almost exact masked-mean reference matcher.  This gives
        # the network a safe baseline while allowing gradients into the full
        # temporal path from the first optimization step.
        nn.init.normal_(self.output_projection.weight, std=1.0e-3)
        nn.init.zeros_(self.output_projection.bias)
        self.correction_scale = nn.Parameter(
            torch.tensor(float(np.arctanh(initial_correction_scale)))
        )

    def forward(
        self,
        selected_score: torch.Tensor,
        selected_support: torch.Tensor,
        rest_offset: torch.Tensor,
        signal_to_noise: torch.Tensor,
        relative_flux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one history score per candidate, class, and start phase.

        ``selected_score`` has shape ``B,V,Z,F,S``.  The transformer is
        factorized over candidate redshift, so its sequence batch is ``B*Z``
        rather than the much larger ``B*Z*F*S`` construction.
        """
        if selected_score.shape != selected_support.shape:
            raise ValueError("Temporal scores and support must have the same shape")
        if selected_score.ndim != 5:
            raise ValueError("Temporal scores must have shape B,V,Z,F,S")
        batch, visits, redshifts, fine_classes, starting_phases = (
            selected_score.shape
        )
        if (
            fine_classes != self.fine_class_count
            or starting_phases != self.starting_phase_count
        ):
            raise ValueError("Temporal score dimensions do not match the transformer")
        if rest_offset.shape != (batch, visits, redshifts):
            raise ValueError("Rest offsets must have shape B,V,Z")
        if signal_to_noise.shape != (batch, visits):
            raise ValueError("Visit S/N must have shape B,V")
        if self.use_relative_flux and (
            relative_flux is None or relative_flux.shape != (batch, visits)
        ):
            raise ValueError(
                "Relative flux must have shape B,V when relative flux evolution is enabled"
            )

        support = selected_support.bool()
        count = support.sum(dim=1)
        baseline = (
            selected_score * support.to(selected_score.dtype)
        ).sum(dim=1) / count.clamp_min(1).to(selected_score.dtype)

        # B,V,Z,F,S -> B,Z,V,(F*S).  Candidate-dependent rest time is retained
        # while class and unknown-start-phase matches form each visit token.
        centred_score = (
            selected_score - baseline[:, None, :, :, :]
        ) * support.to(selected_score.dtype)
        score_features = centred_score.permute(0, 2, 1, 3, 4).flatten(
            start_dim=-2
        )
        support_features = support.permute(0, 2, 1, 3, 4).flatten(
            start_dim=-2
        )
        offset = rest_offset.permute(0, 2, 1)
        interval = torch.log1p(offset.abs()) / np.log(101.0)
        visit_supported = support_features.any(dim=-1)
        feature_parts = [
            score_features,
            torch.tanh(offset / 30.0)[..., None],
            interval[..., None],
        ]
        if self.use_signal_to_noise:
            snr = torch.tanh(signal_to_noise / 3.0)
            snr = snr[:, None, :].expand(batch, redshifts, visits)
            feature_parts.append(snr[..., None])
        centred_relative_flux = None
        if self.use_relative_flux:
            assert relative_flux is not None
            candidate_flux = relative_flux[:, None, :].expand(
                batch, redshifts, visits
            )
            flux_weight = visit_supported.to(candidate_flux.dtype)
            flux_mean = (candidate_flux * flux_weight).sum(dim=-1, keepdim=True)
            flux_mean /= flux_weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
            centred_relative_flux = (
                candidate_flux - flux_mean
            ) * flux_weight
            feature_parts.append(torch.tanh(centred_relative_flux)[..., None])
        features = torch.cat(feature_parts, dim=-1)
        tokens = self.input_projection(features)
        tokens = tokens * visit_supported[..., None].to(tokens.dtype)

        flat_tokens = tokens.reshape(batch * redshifts, visits, -1)
        flat_support = visit_supported.reshape(batch * redshifts, visits)
        active = flat_support.any(dim=1)
        pooled = flat_tokens.new_zeros((batch * redshifts, flat_tokens.shape[-1]))
        if active.any():
            encoded = self.encoder(
                flat_tokens[active],
                src_key_padding_mask=~flat_support[active],
            )
            weight = flat_support[active][..., None].to(encoded.dtype)
            pooled[active] = (encoded * weight).sum(dim=1)
            pooled[active] /= weight.sum(dim=1).clamp_min(1.0)
        correction = self.output_projection(self.output_norm(pooled))
        correction = correction.reshape(
            batch,
            redshifts,
            fine_classes,
            starting_phases,
        )
        correction = torch.tanh(self.correction_scale) * torch.tanh(correction)
        change_energy = centred_score.square().sum(dim=1)
        change_energy /= count.clamp_min(1).to(selected_score.dtype)
        # Use squared change directly. A square root has an infinite derivative
        # at an exactly constant history and can destabilize the shared spectral
        # encoder even though the intended correction is zero.
        change_gate = torch.tanh(8.0 * change_energy.clamp_min(0.0))
        if centred_relative_flux is not None:
            flux_weight = visit_supported.to(centred_relative_flux.dtype)
            relative_flux_energy = (
                centred_relative_flux.square() * flux_weight
            ).sum(dim=-1)
            relative_flux_energy /= flux_weight.sum(dim=-1).clamp_min(1.0)
            relative_flux_gate = torch.tanh(
                8.0 * relative_flux_energy.clamp_min(0.0)
            )
            change_gate = torch.maximum(
                change_gate,
                relative_flux_gate[..., None, None],
            )
        return baseline + change_gate * correction


class RomanReferenceEvidence(nn.Module):
    """Match a final coadd and measured spectral evolution to clean references.

    The final coadd supplies the primary class--redshift evidence.  Individual
    visits supply an independent sequence-consistency term after integrating
    over possible starting phases.  Reference spectra may move slightly during
    training, with their departure from the clean initialization exposed as a
    regularization term.
    """

    def __init__(
        self,
        config: dict[str, Any],
        observed_wavelength: np.ndarray,
        redshift_grid: np.ndarray,
    ) -> None:
        super().__init__()
        settings = config["reference"]
        bank = RomanReferenceBank.load(
            project_path(config, settings["bank_path"])
        )
        output_scheme = str(config["data"]["class_scheme"])
        expected_output = class_names_for_scheme(output_scheme)
        configured_output = tuple(str(name) for name in config["model"]["classes"])
        if expected_output != configured_output:
            raise ValueError(
                f"Configured classes {configured_output} do not match "
                f"class scheme {output_scheme}: {expected_output}"
            )
        self.output_class_names = configured_output
        self.fine_class_names = bank.class_names
        mapping = fine_to_output_class_indices(bank.class_names, output_scheme)
        self.register_buffer(
            "fine_to_output", torch.tensor(mapping, dtype=torch.long)
        )

        rest = np.asarray(bank.rest_wavelength, dtype=np.float64)
        observed = np.asarray(observed_wavelength, dtype=np.float64)
        redshift = np.asarray(redshift_grid, dtype=np.float64)
        target = rest[None, :] * (1.0 + redshift[:, None])
        upper = np.searchsorted(observed, target, side="left")
        valid = (upper > 0) & (upper < len(observed))
        upper = np.clip(upper, 1, len(observed) - 1)
        lower = upper - 1
        denominator = observed[upper] - observed[lower]
        upper_weight = np.divide(
            target - observed[lower],
            denominator,
            out=np.zeros_like(target),
            where=denominator > 0.0,
        )
        self.register_buffer(
            "alignment_lower", torch.from_numpy(lower.astype(np.int64))
        )
        self.register_buffer(
            "alignment_upper", torch.from_numpy(upper.astype(np.int64))
        )
        self.register_buffer(
            "alignment_upper_weight",
            torch.from_numpy(upper_weight.astype(np.float32)),
        )
        self.register_buffer(
            "alignment_valid", torch.from_numpy(valid.astype(np.float32))
        )
        self.register_buffer(
            "phase_edges_days",
            torch.from_numpy(bank.phase_edges_days.astype(np.float32)),
        )
        phase_centres = 0.5 * (
            bank.phase_edges_days[:-1] + bank.phase_edges_days[1:]
        )
        self.register_buffer(
            "phase_starting_points",
            torch.from_numpy(phase_centres.astype(np.float32)),
        )

        for name, values in (
            ("coadd_full_profiles", bank.coadd_full_profiles),
            (
                "coadd_continuum_removed_profiles",
                bank.coadd_continuum_removed_profiles,
            ),
            ("phase_full_profiles", bank.phase_full_profiles),
            (
                "phase_continuum_removed_profiles",
                bank.phase_continuum_removed_profiles,
            ),
        ):
            self.register_buffer(name, torch.from_numpy(values.copy()))
        self.register_buffer(
            "coadd_profile_mask",
            torch.from_numpy(bank.coadd_profile_masks.astype(bool)),
        )
        self.register_buffer(
            "phase_profile_mask",
            torch.from_numpy(bank.phase_profile_masks.astype(bool)),
        )
        minimum_support = int(settings.get("minimum_profile_support", 5))
        self.register_buffer(
            "coadd_profile_supported",
            torch.from_numpy(bank.coadd_support_counts >= minimum_support),
        )
        self.register_buffer(
            "phase_profile_supported",
            torch.from_numpy(bank.phase_support_counts >= minimum_support),
        )
        missing_output = []
        for output_index, output_name in enumerate(configured_output):
            fine = np.asarray(mapping) == output_index
            if not np.any(
                (bank.coadd_support_counts >= minimum_support)[fine]
            ):
                missing_output.append(output_name)
        if missing_output:
            raise ValueError(
                "Reference bank has no supported coadd profile for output "
                "class(es): " + ", ".join(missing_output)
            )

        continuum_sigma = velocity_sigma_to_log_bins(
            float(observed[0]),
            float(observed[-1]),
            len(observed),
            float(settings["continuum_width_km_s"]),
        )
        self.continuum_removal = MaskAwareContinuumRemoval(continuum_sigma)
        self.minimum_rest_fraction = float(
            settings.get("minimum_rest_fraction", 0.15)
        )
        self.minimum_shared_fraction = float(
            settings.get("minimum_shared_fraction", 0.75)
        )
        if not 0.0 < self.minimum_rest_fraction <= 1.0:
            raise ValueError("Reference minimum rest fraction must lie in (0, 1]")
        if not 0.0 < self.minimum_shared_fraction <= 1.0:
            raise ValueError("Reference minimum shared fraction must lie in (0, 1]")
        self.spectral_uncertainty_weighting = str(
            settings.get("spectral_uncertainty_weighting", "none")
        )
        if self.spectral_uncertainty_weighting not in {
            "none",
            "inverse_variance",
        }:
            raise ValueError(
                "Reference spectral_uncertainty_weighting must be 'none' or "
                "'inverse_variance'"
            )
        self.minimum_relative_spectral_precision = float(
            settings.get("minimum_relative_spectral_precision", 0.0)
        )
        if not 0.0 <= self.minimum_relative_spectral_precision < 1.0:
            raise ValueError(
                "Reference minimum relative spectral precision must lie in "
                "[0, 1)"
            )
        if (
            self.minimum_relative_spectral_precision > 0.0
            and self.spectral_uncertainty_weighting != "inverse_variance"
        ):
            raise ValueError(
                "A positive minimum_relative_spectral_precision requires "
                "inverse-variance spectral uncertainty weighting"
            )
        self.prototype_temperature = float(
            settings.get("prototype_temperature", 0.08)
        )
        self.fine_class_temperature = float(
            settings.get("fine_class_temperature", 0.1)
        )
        self.phase_temperature = float(settings.get("phase_temperature", 0.1))
        for name, value in (
            ("prototype_temperature", self.prototype_temperature),
            ("fine_class_temperature", self.fine_class_temperature),
            ("phase_temperature", self.phase_temperature),
        ):
            if value <= 0.0:
                raise ValueError(f"Reference {name} must be positive")

        initial_removed = float(
            settings.get("initial_continuum_removed_fraction", 0.6)
        )
        if not 0.0 < initial_removed < 1.0:
            raise ValueError(
                "Initial continuum-removed fraction must lie in (0, 1)"
            )
        self.continuum_removed_intercept = nn.Parameter(
            torch.tensor(float(np.log(initial_removed / (1.0 - initial_removed))))
        )
        self.continuum_removed_redshift_slope = nn.Parameter(torch.tensor(0.0))
        redshift_coordinate = (redshift - float(np.median(redshift))) / max(
            float(np.ptp(redshift)), 1.0
        )
        self.register_buffer(
            "redshift_coordinate",
            torch.from_numpy(redshift_coordinate.astype(np.float32)),
        )
        self.register_buffer(
            "candidate_redshift",
            torch.from_numpy(redshift.astype(np.float32)),
        )
        self.coadd_scale = nn.Parameter(
            torch.tensor(
                float(np.arctanh(float(settings.get("initial_coadd_scale", 0.75))))
            )
        )
        self.sequence_scale = nn.Parameter(
            torch.tensor(
                float(
                    np.arctanh(float(settings.get("initial_sequence_scale", 0.2)))
                )
            )
        )
        self.evidence_scale = float(settings.get("evidence_scale", 10.0))
        self.redshift_chunk_size = int(settings.get("redshift_chunk_size", 12))
        self.sequence_visits = int(settings.get("sequence_visits", 6))
        self.minimum_sequence_visits = int(
            settings.get("minimum_sequence_visits", 2)
        )
        if self.redshift_chunk_size < 1 or self.sequence_visits < 1:
            raise ValueError("Reference chunk size and sequence visits must be positive")
        if not 1 <= self.minimum_sequence_visits <= self.sequence_visits:
            raise ValueError(
                "Reference minimum sequence visits must lie within the visit limit"
            )
        self.spectral_encoder_mode = str(
            settings.get("spectral_encoder", "direct")
        )
        if self.spectral_encoder_mode not in {
            "direct",
            "shared_cnn",
            "multiscale_attention",
        }:
            raise ValueError(
                "Reference spectral_encoder must be 'direct', 'shared_cnn', "
                "or 'multiscale_attention'"
            )
        self.token_pool_size = int(settings.get("token_pool_size", 2))
        if self.token_pool_size < 1 or len(rest) % self.token_pool_size:
            raise ValueError(
                "Reference token_pool_size must divide the rest wavelength bins"
            )
        if self.spectral_encoder_mode == "shared_cnn":
            self.spectral_encoder: nn.Module | None = MaskAwareTokenEncoder(
                int(settings.get("token_dim", 8)),
                float(config["model"]["dropout"]),
                float(settings.get("minimum_encoder_support", 0.5)),
                "gelu",
            )
        elif self.spectral_encoder_mode == "multiscale_attention":
            self.spectral_encoder = MaskAwareMultiscaleAttentionEncoder(
                int(settings.get("token_dim", 32)),
                int(settings.get("attention_heads", 4)),
                float(config["model"]["dropout"]),
                self.token_pool_size,
                float(settings.get("minimum_encoder_support", 0.5)),
            )
        else:
            self.spectral_encoder = None
        self.sequence_combination = str(
            settings.get("sequence_combination", "mean")
        )
        self.relative_flux_evolution = bool(
            settings.get("relative_flux_evolution", False)
        )
        if self.sequence_combination not in {
            "mean",
            "continuous_time_attention",
            "temporal_transformer",
        }:
            raise ValueError(
                "Reference sequence_combination must be 'mean' or "
                "'continuous_time_attention' or 'temporal_transformer'"
            )
        attention_hidden = int(settings.get("time_attention_hidden_dim", 16))
        self.time_attention = (
            nn.Sequential(
                nn.Linear(4, attention_hidden),
                nn.GELU(),
                nn.Linear(attention_hidden, 1),
            )
            if self.sequence_combination == "continuous_time_attention"
            else None
        )
        if self.sequence_combination == "temporal_transformer":
            self.temporal_transformer: CandidateTemporalTransformer | None = (
                CandidateTemporalTransformer(
                    fine_class_count=len(self.fine_class_names),
                    starting_phase_count=self.phase_starting_points.numel(),
                    hidden_dim=int(settings.get("temporal_hidden_dim", 32)),
                    attention_heads=int(
                        settings.get("temporal_attention_heads", 4)
                    ),
                    layers=int(settings.get("temporal_layers", 1)),
                    feedforward_multiplier=int(
                        settings.get("temporal_feedforward_multiplier", 2)
                    ),
                    dropout=float(config["model"]["dropout"]),
                    initial_correction_scale=float(
                        settings.get("temporal_initial_correction_scale", 0.10)
                    ),
                    use_signal_to_noise=bool(
                        settings.get("temporal_use_signal_to_noise", True)
                    ),
                    use_relative_flux=self.relative_flux_evolution,
                )
            )
        else:
            self.temporal_transformer = None
        if self.relative_flux_evolution and self.temporal_transformer is None:
            raise ValueError(
                "Relative flux evolution requires sequence_combination: "
                "temporal_transformer"
            )
        edge_trim_fraction = float(settings.get("edge_trim_fraction", 0.05))
        if not 0.0 <= edge_trim_fraction < 0.5:
            raise ValueError("Reference edge_trim_fraction must lie in [0, 0.5)")
        edge_taper_fraction = float(settings.get("edge_taper_fraction", 0.0))
        edge_taper = cosine_edge_taper(
            len(observed), edge_taper_fraction
        ).numpy()
        log_coordinate = (
            np.log(observed) - np.log(observed[0])
        ) / np.log(observed[-1] / observed[0])
        hard_edge_mask = (
            (log_coordinate >= edge_trim_fraction)
            & (log_coordinate <= 1.0 - edge_trim_fraction)
        )
        matching_weight = edge_taper * hard_edge_mask.astype(np.float32)
        self.register_buffer(
            "observed_matching_weight",
            torch.from_numpy(matching_weight.astype(np.float32)),
            persistent=False,
        )
        self.register_buffer(
            "observed_matching_mask",
            torch.from_numpy((matching_weight > 0.0).astype(np.float32)),
            persistent=False,
        )
        relative_flux_weight = np.gradient(observed).astype(np.float32)
        relative_flux_weight *= matching_weight
        self.register_buffer(
            "relative_flux_wavelength_weight",
            torch.from_numpy(relative_flux_weight),
            persistent=False,
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        coadded_flux: torch.Tensor,
        coadded_mask: torch.Tensor,
        coadded_error: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        matching_weight = self.observed_matching_weight.to(coadded_flux.dtype)
        matching_mask = self.observed_matching_mask.to(coadded_mask.dtype)
        # Keep the measured spectral shape intact.  The cosine taper controls
        # influence through the reliability path below; multiplying the flux
        # itself would manufacture a shared roll-off before normalization.
        matching_coadd = coadded_flux
        matching_coadd_mask = coadded_mask * matching_mask[None, :]
        coadd_reliability = matching_weight[None, :].expand_as(coadded_flux)
        if self.spectral_uncertainty_weighting == "inverse_variance":
            if coadded_error is None:
                measurement_reliability = matching_coadd_mask.to(
                    coadded_flux.dtype
                )
            else:
                measurement_reliability = relative_inverse_variance(
                    coadded_error,
                    matching_coadd_mask,
                )
            if self.minimum_relative_spectral_precision > 0.0:
                matching_coadd_mask = matching_coadd_mask * (
                    minimum_relative_precision_mask(
                        measurement_reliability,
                        matching_coadd_mask,
                        self.minimum_relative_spectral_precision,
                    ).to(matching_coadd_mask.dtype)
                )
            coadd_reliability = (
                measurement_reliability * matching_weight[None, :]
            )
        coadded_removed = self.continuum_removal(
            matching_coadd,
            matching_coadd_mask,
            coadd_reliability,
        )
        coadd_scores: list[torch.Tensor] = []
        coadd_support: list[torch.Tensor] = []
        sequence_scores: list[torch.Tensor] = []
        sequence_support: list[torch.Tensor] = []
        sequence = self._select_sequence_visits(batch)
        matching_sequence = sequence["flux"]
        matching_sequence_mask = (
            sequence["wavelength_mask"] * matching_mask[None, None, :]
        )
        sequence_reliability = matching_weight[None, None, :].expand_as(
            matching_sequence
        )
        if self.spectral_uncertainty_weighting == "inverse_variance":
            sequence_error = sequence.get("error")
            if sequence_error is None:
                sequence_measurement_reliability = matching_sequence_mask.to(
                    matching_sequence.dtype
                )
            else:
                sequence_measurement_reliability = relative_inverse_variance(
                    sequence_error,
                    matching_sequence_mask,
                )
            if self.minimum_relative_spectral_precision > 0.0:
                matching_sequence_mask = matching_sequence_mask * (
                    minimum_relative_precision_mask(
                        sequence_measurement_reliability,
                        matching_sequence_mask,
                        self.minimum_relative_spectral_precision,
                    ).to(matching_sequence_mask.dtype)
                )
            sequence_reliability = (
                sequence_measurement_reliability
                * matching_weight[None, None, :]
            )
        sequence_removed = self.continuum_removal(
            matching_sequence,
            matching_sequence_mask,
            sequence_reliability,
        )
        redshift_count = self.alignment_lower.shape[0]
        encoded_profiles = self._encoded_profiles()
        removed_weight = torch.sigmoid(
            self.continuum_removed_intercept
            + self.continuum_removed_redshift_slope * self.redshift_coordinate
        )

        for start in range(0, redshift_count, self.redshift_chunk_size):
            stop = min(start + self.redshift_chunk_size, redshift_count)
            aligned_edge_weight, _ = self._align(
                matching_weight[None, None, :],
                matching_mask[None, None, :],
                start,
                stop,
            )
            aligned_full, aligned_mask = self._align(
                matching_coadd[:, None, :],
                matching_coadd_mask[:, None, :],
                start,
                stop,
            )
            aligned_removed, removed_mask = self._align(
                coadded_removed[:, None, :],
                matching_coadd_mask[:, None, :],
                start,
                stop,
            )
            aligned_reliability = aligned_edge_weight[:, 0].expand_as(
                aligned_full[:, 0]
            )
            if (
                self.spectral_uncertainty_weighting == "inverse_variance"
                and coadded_error is not None
            ):
                aligned_error, _ = self._align_error(
                    coadded_error[:, None, :],
                    matching_coadd_mask[:, None, :],
                    start,
                    stop,
                )
                aligned_reliability = relative_inverse_variance(
                    aligned_error[:, 0],
                    aligned_mask[:, 0],
                )
                aligned_reliability = (
                    aligned_reliability * aligned_edge_weight[:, 0]
                )
            full_score, full_support = self._profile_scores(
                aligned_full[:, 0],
                aligned_mask[:, 0],
                self.coadd_full_profiles,
                self.coadd_profile_mask,
                self.coadd_profile_supported,
                encoded_profiles.get("coadd_full"),
                aligned_reliability,
            )
            removed_score, removed_support = self._profile_scores(
                aligned_removed[:, 0],
                removed_mask[:, 0],
                self.coadd_continuum_removed_profiles,
                self.coadd_profile_mask,
                self.coadd_profile_supported,
                encoded_profiles.get("coadd_continuum_removed"),
                aligned_reliability,
            )
            weight = removed_weight[start:stop][None, :, None]
            fine_score = (1.0 - weight) * full_score + weight * removed_score
            fine_support = full_support & removed_support
            output_score, output_support = self._aggregate_fine_classes(
                fine_score, fine_support
            )
            coadd_scores.append(output_score.permute(0, 2, 1))
            coadd_support.append(output_support.permute(0, 2, 1))

            visit_full, visit_mask = self._align(
                matching_sequence, matching_sequence_mask, start, stop
            )
            visit_removed, visit_removed_mask = self._align(
                sequence_removed,
                matching_sequence_mask,
                start,
                stop,
            )
            visit_reliability = aligned_edge_weight.expand_as(visit_full)
            if (
                self.spectral_uncertainty_weighting == "inverse_variance"
                and sequence.get("error") is not None
            ):
                visit_error, _ = self._align_error(
                    sequence["error"],
                    matching_sequence_mask,
                    start,
                    stop,
                )
                visit_reliability = relative_inverse_variance(
                    visit_error,
                    visit_mask,
                )
                visit_reliability = visit_reliability * aligned_edge_weight
            phase_full, phase_full_support = self._phase_profile_scores(
                visit_full,
                visit_mask,
                self.phase_full_profiles,
                encoded_profiles.get("phase_full"),
                visit_reliability,
            )
            phase_removed, phase_removed_support = self._phase_profile_scores(
                visit_removed,
                visit_removed_mask,
                self.phase_continuum_removed_profiles,
                encoded_profiles.get("phase_continuum_removed"),
                visit_reliability,
            )
            phase_weight = removed_weight[start:stop][None, None, :, None, None]
            phase_score = (
                (1.0 - phase_weight) * phase_full
                + phase_weight * phase_removed
            )
            phase_support = phase_full_support & phase_removed_support
            fine_sequence, fine_sequence_support = self._sequence_consistency(
                phase_score,
                phase_support,
                sequence["observer_days"],
                sequence["visit_mask"],
                sequence["signal_to_noise"],
                sequence.get("relative_flux"),
                start,
                stop,
            )
            output_sequence, output_sequence_support = self._aggregate_fine_classes(
                fine_sequence, fine_sequence_support
            )
            sequence_scores.append(output_sequence.permute(0, 2, 1))
            sequence_support.append(output_sequence_support.permute(0, 2, 1))

        raw_coadd = self.evidence_scale * torch.cat(coadd_scores, dim=-1)
        raw_sequence = self.evidence_scale * torch.cat(sequence_scores, dim=-1)
        coadd_joint = torch.tanh(self.coadd_scale) * raw_coadd
        sequence_joint = torch.tanh(self.sequence_scale) * raw_sequence
        joint_support = torch.cat(coadd_support, dim=-1)
        temporal_support = torch.cat(sequence_support, dim=-1)
        sequence_joint = torch.where(
            temporal_support, sequence_joint, torch.zeros_like(sequence_joint)
        )
        joint = (coadd_joint + sequence_joint).masked_fill(~joint_support, -1.0e4)
        return {
            "joint_logits": joint,
            "joint_support": joint_support,
            "spectral_joint_logits": coadd_joint.masked_fill(
                ~joint_support, -1.0e4
            ),
            "temporal_joint_logits": sequence_joint,
            "reference_coadd_joint_logits": coadd_joint,
            "reference_sequence_joint_logits": sequence_joint,
            "reference_sequence_support": temporal_support,
            "reference_continuum_removed_weight": removed_weight.detach(),
        }

    def _align(
        self,
        flux: torch.Tensor,
        mask: torch.Tensor,
        start: int,
        stop: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lower = self.alignment_lower[start:stop]
        upper = self.alignment_upper[start:stop]
        weight = self.alignment_upper_weight[start:stop].to(flux.dtype)
        valid = self.alignment_valid[start:stop].to(flux.dtype)
        values = flux[..., lower] * (1.0 - weight)
        values = values + flux[..., upper] * weight
        measured = mask[..., lower] * mask[..., upper] * valid
        return values * measured, measured

    def _align_error(
        self,
        error: torch.Tensor,
        mask: torch.Tensor,
        start: int,
        stop: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Propagate independent errors through the rest-grid interpolation."""
        lower = self.alignment_lower[start:stop]
        upper = self.alignment_upper[start:stop]
        upper_weight = self.alignment_upper_weight[start:stop].to(error.dtype)
        lower_weight = 1.0 - upper_weight
        valid = self.alignment_valid[start:stop].to(error.dtype)
        variance = (error[..., lower] * lower_weight).square()
        variance = variance + (error[..., upper] * upper_weight).square()
        measured = mask[..., lower] * mask[..., upper] * valid
        propagated = torch.sqrt(variance.clamp_min(0.0))
        return propagated * measured, measured

    def _profile_scores(
        self,
        candidates: torch.Tensor,
        candidate_mask: torch.Tensor,
        profiles: torch.Tensor,
        profile_mask: torch.Tensor,
        profile_supported: torch.Tensor,
        encoded_profiles: tuple[torch.Tensor, torch.Tensor] | None = None,
        candidate_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        leading = candidates.shape[:-1]
        flat_values = candidates.reshape(-1, candidates.shape[-1])
        flat_mask = candidate_mask.reshape(-1, candidate_mask.shape[-1])
        flat_weight = (
            None
            if candidate_weight is None
            else candidate_weight.reshape(-1, candidate_weight.shape[-1])
        )
        profile_shape = profiles.shape
        if self.spectral_encoder is None:
            flat_profiles = profiles.reshape(-1, profiles.shape[-1])
            flat_profile_mask = profile_mask.reshape(-1, profile_mask.shape[-1])
            similarity, valid = _common_support_correlation(
                flat_values,
                flat_mask,
                flat_profiles,
                flat_profile_mask,
                profile_supported.reshape(-1),
                self.minimum_rest_fraction,
                self.minimum_shared_fraction,
                flat_weight,
            )
        else:
            if encoded_profiles is None:
                raise RuntimeError("Encoded reference profiles are missing")
            candidate_tokens, candidate_support = self._encode_spectra(
                flat_values,
                flat_mask,
                flat_weight,
            )
            candidate_token_weight = _pool_wavelength_weight(
                flat_weight,
                flat_mask,
                candidate_tokens.shape[-2],
            )
            profile_tokens, encoded_profile_support = encoded_profiles
            similarity, valid = _common_support_token_cosine(
                candidate_tokens,
                candidate_support,
                profile_tokens,
                encoded_profile_support,
                profile_supported.reshape(-1),
                self.minimum_rest_fraction,
                self.minimum_shared_fraction,
                candidate_token_weight,
            )
        similarity = similarity.reshape(*leading, *profile_shape[:-1])
        valid = valid.reshape(*leading, *profile_shape[:-1])
        score, supported = _supported_log_mean_exp(
            similarity,
            valid,
            dim=-1,
            temperature=self.prototype_temperature,
        )
        return score, supported

    def _phase_profile_scores(
        self,
        candidates: torch.Tensor,
        candidate_mask: torch.Tensor,
        profiles: torch.Tensor,
        encoded_profiles: tuple[torch.Tensor, torch.Tensor] | None,
        candidate_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        score, support = self._profile_scores(
            candidates,
            candidate_mask,
            profiles,
            self.phase_profile_mask,
            self.phase_profile_supported,
            encoded_profiles,
            candidate_weight,
        )
        return score, support

    def _sequence_consistency(
        self,
        phase_score: torch.Tensor,
        phase_support: torch.Tensor,
        observer_days: torch.Tensor,
        visit_mask: torch.Tensor,
        signal_to_noise: torch.Tensor,
        relative_flux: torch.Tensor | None,
        redshift_start: int,
        redshift_stop: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # score: B,V,Z,F,P.  Try every broad training phase as the unknown
        # starting phase, then average those possible histories rather than
        # supplying the simulated phase of the measured object.
        grid = self._redshift_values(redshift_start, redshift_stop).to(
            observer_days.dtype
        )
        first_day = torch.where(
            visit_mask.bool(),
            observer_days,
            torch.full_like(observer_days, torch.inf),
        ).min(dim=1).values
        first_day = torch.where(
            torch.isfinite(first_day), first_day, torch.zeros_like(first_day)
        )
        rest_offset = (observer_days - first_day[:, None])[:, :, None]
        rest_offset = rest_offset / (1.0 + grid[None, None, :])
        possible_phase = (
            rest_offset[..., None]
            + self.phase_starting_points[None, None, None, :].to(
                rest_offset.dtype
            )
        )
        phase_index = torch.bucketize(
            possible_phase,
            self.phase_edges_days.to(possible_phase.dtype),
            right=True,
        ) - 1
        phase_count = phase_score.shape[-1]
        within = (phase_index >= 0) & (phase_index < phase_count)
        safe_index = phase_index.clamp(0, phase_count - 1)

        score = phase_score.unsqueeze(-2).expand(
            *phase_score.shape[:-1], self.phase_starting_points.numel(), phase_count
        )
        support = phase_support.unsqueeze(-2).expand_as(score)
        gather_index = safe_index[:, :, :, None, :, None].expand(
            phase_score.shape[0],
            phase_score.shape[1],
            phase_score.shape[2],
            phase_score.shape[3],
            self.phase_starting_points.numel(),
            1,
        )
        selected_score = torch.gather(score, dim=-1, index=gather_index).squeeze(-1)
        selected_support = torch.gather(
            support, dim=-1, index=gather_index
        ).squeeze(-1)
        selected_support &= within[:, :, :, None, :]
        selected_support &= visit_mask[:, :, None, None, None].bool()
        count = selected_support.sum(dim=1)
        if self.temporal_transformer is not None:
            history_score = self.temporal_transformer(
                selected_score,
                selected_support,
                rest_offset,
                signal_to_noise,
                relative_flux,
            )
        elif self.time_attention is None:
            history_score = (
                selected_score * selected_support.to(selected_score.dtype)
            ).sum(dim=1) / count.clamp_min(1).to(selected_score.dtype)
        else:
            offset_feature = torch.tanh(rest_offset / 30.0)
            interval_feature = torch.log1p(rest_offset.abs()) / np.log(101.0)
            snr_feature = torch.tanh(signal_to_noise / 3.0)
            shape = selected_score.shape
            time_shape = (shape[0], shape[1], shape[2], 1, 1)
            features = torch.stack(
                (
                    selected_score,
                    offset_feature.reshape(*time_shape).expand_as(selected_score),
                    interval_feature.reshape(*time_shape).expand_as(selected_score),
                    snr_feature[:, :, None, None, None].expand_as(selected_score),
                ),
                dim=-1,
            )
            attention_logit = self.time_attention(features).squeeze(-1)
            attention_logit = attention_logit.masked_fill(
                ~selected_support, -torch.inf
            )
            has_supported_visit = selected_support.any(dim=1, keepdim=True)
            attention_logit = torch.where(
                has_supported_visit,
                attention_logit,
                torch.zeros_like(attention_logit),
            )
            attention = torch.softmax(attention_logit, dim=1)
            attention = attention * selected_support.to(attention.dtype)
            history_score = (attention * selected_score).sum(dim=1)
        history_support = count >= self.minimum_sequence_visits
        return _supported_log_mean_exp(
            history_score,
            history_support,
            dim=-1,
            temperature=self.phase_temperature,
        )

    def _encoded_profiles(
        self,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        if self.spectral_encoder is None:
            return {}
        return {
            "coadd_full": self._encode_spectra(
                self.coadd_full_profiles.reshape(-1, self.coadd_full_profiles.shape[-1]),
                self.coadd_profile_mask.reshape(-1, self.coadd_profile_mask.shape[-1]),
            ),
            "coadd_continuum_removed": self._encode_spectra(
                self.coadd_continuum_removed_profiles.reshape(
                    -1, self.coadd_continuum_removed_profiles.shape[-1]
                ),
                self.coadd_profile_mask.reshape(-1, self.coadd_profile_mask.shape[-1]),
            ),
            "phase_full": self._encode_spectra(
                self.phase_full_profiles.reshape(-1, self.phase_full_profiles.shape[-1]),
                self.phase_profile_mask.reshape(-1, self.phase_profile_mask.shape[-1]),
            ),
            "phase_continuum_removed": self._encode_spectra(
                self.phase_continuum_removed_profiles.reshape(
                    -1, self.phase_continuum_removed_profiles.shape[-1]
                ),
                self.phase_profile_mask.reshape(-1, self.phase_profile_mask.shape[-1]),
            ),
        }

    def _encode_spectra(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.spectral_encoder is None:
            raise RuntimeError("Shared spectral encoder is not configured")
        normalized = _scale_invariant_normalize(values, mask, weight)
        # Reliability belongs in the local accumulation, not in the physical
        # flux amplitude. Reliability-aware partial convolutions prevent an
        # uncertain extreme from contaminating neighbouring tokens while
        # preserving the recognizable spectral shape through normalization.
        tokens, support = self.spectral_encoder(normalized, mask, weight)
        if self.spectral_encoder_mode == "multiscale_attention":
            return tokens, support
        if self.token_pool_size == 1:
            return tokens, support
        leading = tokens.shape[:-2]
        pooled_bins = tokens.shape[-2] // self.token_pool_size
        token_dim = tokens.shape[-1]
        token_values = tokens.reshape(
            *leading, pooled_bins, self.token_pool_size, token_dim
        )
        token_support = support.reshape(
            *leading, pooled_bins, self.token_pool_size
        ).to(tokens.dtype)
        count = token_support.sum(dim=-1)
        pooled = (
            token_values * token_support[..., None]
        ).sum(dim=-2) / count[..., None].clamp_min(1.0)
        pooled_support = count >= 0.5 * self.token_pool_size
        return pooled * pooled_support[..., None], pooled_support.to(tokens.dtype)

    def _redshift_values(self, start: int, stop: int) -> torch.Tensor:
        return self.candidate_redshift[start:stop]

    def _aggregate_fine_classes(
        self,
        fine_score: torch.Tensor,
        fine_support: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = []
        supports = []
        for output_index in range(len(self.output_class_names)):
            selected = self.fine_to_output == output_index
            score, support = _supported_log_mean_exp(
                fine_score[..., selected],
                fine_support[..., selected],
                dim=-1,
                temperature=self.fine_class_temperature,
            )
            scores.append(score)
            supports.append(support)
        return torch.stack(scores, dim=-1), torch.stack(supports, dim=-1)

    def _select_sequence_visits(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        batch_size, _, wavelength_bins = batch["flux"].shape
        output_flux = batch["flux"].new_zeros(
            (batch_size, self.sequence_visits, wavelength_bins)
        )
        output_mask = batch["wavelength_mask"].new_zeros(
            output_flux.shape
        )
        output_days = batch["observer_days"].new_zeros(
            (batch_size, self.sequence_visits)
        )
        output_visit_mask = batch["visit_mask"].new_zeros(
            (batch_size, self.sequence_visits)
        )
        output_signal_to_noise = batch["flux"].new_zeros(
            (batch_size, self.sequence_visits)
        )
        output_relative_flux = batch["flux"].new_zeros(
            (batch_size, self.sequence_visits)
        )
        output_error = batch["flux"].new_zeros(output_flux.shape)
        has_reported_error = "flux_error_shape" in batch
        error = (
            torch.exp(batch["flux_error_shape"])
            if has_reported_error
            else None
        )
        spectral_reliability = None
        if (
            self.spectral_uncertainty_weighting == "inverse_variance"
            and error is not None
        ):
            spectral_reliability = relative_inverse_variance(
                error,
                batch["wavelength_mask"],
            )
        relative_flux = None
        if self.relative_flux_evolution:
            if "visit_flux_scale" not in batch:
                raise KeyError(
                    "Relative flux evolution requires visit_flux_scale"
                )
            relative_flux = object_normalized_flux_amplitude(
                batch["flux"],
                batch["wavelength_mask"],
                batch["visit_mask"],
                batch["visit_flux_scale"],
                self.relative_flux_wavelength_weight,
                spectral_reliability,
            )
        if has_reported_error:
            assert error is not None
            ratio = batch["flux"] / error.clamp_min(torch.finfo(error.dtype).tiny)
            ratio = ratio.masked_fill(~batch["wavelength_mask"].bool(), torch.nan)
            # The signed median is already robust to isolated extreme FLAM
            # values because their equally extreme FLAMERR enters the ratio.
            # Keep this reporting/selection statistic stable while using the
            # continuous reliability only in spectral and brightness paths.
            signal_to_noise = torch.nanmedian(ratio, dim=-1).values
            signal_to_noise = torch.nan_to_num(signal_to_noise, nan=-torch.inf)
        else:
            signal_to_noise = batch["wavelength_mask"].sum(dim=-1)
            signal_to_noise = signal_to_noise.masked_fill(
                ~batch["visit_mask"].bool(), -torch.inf
            )

        for object_index in range(batch_size):
            available = torch.nonzero(
                batch["visit_mask"][object_index].bool(), as_tuple=False
            ).flatten()
            if not len(available):
                continue
            if len(available) <= self.sequence_visits:
                selected = available
            else:
                boundaries = torch.linspace(
                    0,
                    len(available),
                    self.sequence_visits + 1,
                    device=available.device,
                ).round().to(torch.long)
                chosen = []
                for segment in range(self.sequence_visits):
                    candidates = available[
                        boundaries[segment] : boundaries[segment + 1]
                    ]
                    local = torch.argmax(signal_to_noise[object_index, candidates])
                    chosen.append(candidates[local])
                selected = torch.stack(chosen).sort().values
            count = len(selected)
            output_flux[object_index, :count] = batch["flux"][
                object_index, selected
            ]
            output_mask[object_index, :count] = batch["wavelength_mask"][
                object_index, selected
            ]
            output_days[object_index, :count] = batch["observer_days"][
                object_index, selected
            ]
            output_visit_mask[object_index, :count] = 1.0
            if has_reported_error:
                output_signal_to_noise[object_index, :count] = signal_to_noise[
                    object_index, selected
                ]
                output_error[object_index, :count] = torch.exp(
                    batch["flux_error_shape"][object_index, selected]
                )
            if relative_flux is not None:
                output_relative_flux[object_index, :count] = relative_flux[
                    object_index, selected
                ]
        result = {
            "flux": output_flux,
            "wavelength_mask": output_mask,
            "observer_days": output_days,
            "visit_mask": output_visit_mask,
            "signal_to_noise": output_signal_to_noise,
        }
        if relative_flux is not None:
            result["relative_flux"] = output_relative_flux
        if has_reported_error:
            result["error"] = output_error
        return result


def _common_support_correlation(
    candidates: torch.Tensor,
    candidate_mask: torch.Tensor,
    profiles: torch.Tensor,
    profile_mask: torch.Tensor,
    profile_supported: torch.Tensor,
    minimum_rest_fraction: float,
    minimum_shared_fraction: float,
    candidate_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    precision = (
        torch.autocast(device_type="cuda", enabled=False)
        if candidates.device.type == "cuda"
        else nullcontext()
    )
    with precision:
        candidates = _scale_invariant_normalize(
            candidates.float(),
            candidate_mask.float(),
            None if candidate_weight is None else candidate_weight.float(),
        )
        profiles = _scale_invariant_normalize(profiles.float(), profile_mask.float())
        measured = candidate_mask.float()
        reference_mask = profile_mask.float()
        reliability = measured
        if candidate_weight is not None:
            reliability = measured * candidate_weight.float().clamp_min(0.0)
        count = torch.einsum("nl,pl->np", measured, reference_mask)
        weighted_count = reliability @ reference_mask.transpose(0, 1)
        weighted_candidates = reliability * candidates
        measured_profiles = reference_mask * profiles
        sum_x = weighted_candidates @ reference_mask.transpose(0, 1)
        sum_y = reliability @ measured_profiles.transpose(0, 1)
        sum_x2 = (reliability * candidates.square()) @ reference_mask.transpose(0, 1)
        sum_y2 = reliability @ (
            reference_mask * profiles.square()
        ).transpose(0, 1)
        sum_xy = weighted_candidates @ measured_profiles.transpose(0, 1)
        safe_weight = weighted_count.clamp_min(torch.finfo(candidates.dtype).tiny)
        covariance = sum_xy - sum_x * sum_y / safe_weight
        variance_x = (sum_x2 - sum_x.square() / safe_weight).clamp_min(0.0)
        variance_y = (sum_y2 - sum_y.square() / safe_weight).clamp_min(0.0)
        denominator = torch.sqrt((variance_x * variance_y).clamp_min(1.0e-12))
        correlation = (covariance / denominator).clamp(-1.0, 1.0)
        candidate_count = measured.sum(dim=-1, keepdim=True)
        profile_count = reference_mask.sum(dim=-1)[None, :]
        shared_fraction = count / torch.minimum(
            candidate_count, profile_count
        ).clamp_min(1.0)
        valid = profile_supported[None, :].bool().expand_as(count).clone()
        valid &= count >= minimum_rest_fraction * candidates.shape[-1]
        valid &= shared_fraction >= minimum_shared_fraction
        valid &= denominator > 1.0e-8
        return torch.where(
            valid, correlation, torch.zeros_like(correlation)
        ), valid


def _scale_invariant_normalize(
    values: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Standardize physical flux without an absolute FLAM-dependent floor."""
    measured = mask.to(values.dtype)
    if weight is None:
        scale = (values.abs() * measured).amax(dim=-1, keepdim=True)
    else:
        reliability = measured * torch.where(
            torch.isfinite(weight) & (weight > 0.0),
            weight.to(values.dtype),
            torch.zeros_like(weight, dtype=values.dtype),
        )
        reliability_sum = reliability.sum(dim=-1, keepdim=True)
        weighted_scale = (values.abs() * reliability).sum(
            dim=-1, keepdim=True
        ) / reliability_sum.clamp_min(torch.finfo(values.dtype).tiny)
        fallback = (values.abs() * measured).amax(dim=-1, keepdim=True)
        scale = torch.where(reliability_sum > 0.0, weighted_scale, fallback)
    safe_scale = scale.clamp_min(torch.finfo(values.dtype).tiny)
    scaled = values / safe_scale
    return normalize_valid_bins(scaled, mask, weight)


def _common_support_token_cosine(
    candidates: torch.Tensor,
    candidate_mask: torch.Tensor,
    profiles: torch.Tensor,
    profile_mask: torch.Tensor,
    profile_supported: torch.Tensor,
    minimum_rest_fraction: float,
    minimum_shared_fraction: float,
    candidate_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compare learned spectral tokens only where both spectra are measured."""
    precision = (
        torch.autocast(device_type="cuda", enabled=False)
        if candidates.device.type == "cuda"
        else nullcontext()
    )
    with precision:
        candidates = candidates.float()
        profiles = profiles.float()
        measured = candidate_mask.float()
        reference_mask = profile_mask.float()
        reliability = measured
        if candidate_weight is not None:
            reliability = measured * candidate_weight.float().clamp_min(0.0)
        weighted_candidates = candidates * reliability[..., None]
        measured_profiles = profiles * reference_mask[..., None]
        # Encoded values are already zero outside their support. Flattening the
        # token and channel axes therefore gives the common-support numerator
        # without constructing an N x P x S tensor.
        numerator = (
            weighted_candidates.flatten(1)
            @ measured_profiles.flatten(1).transpose(0, 1)
        )
        candidate_energy_by_token = candidates.square().sum(dim=-1) * reliability
        profile_energy_by_token = measured_profiles.square().sum(dim=-1)
        candidate_energy = candidate_energy_by_token @ reference_mask.transpose(0, 1)
        profile_energy = reliability @ profile_energy_by_token.transpose(0, 1)
        denominator = torch.sqrt(
            (candidate_energy * profile_energy).clamp_min(1.0e-12)
        )
        similarity = (numerator / denominator).clamp(-1.0, 1.0)

        count = measured @ reference_mask.transpose(0, 1)
        candidate_count = measured.sum(dim=-1, keepdim=True)
        profile_count = reference_mask.sum(dim=-1)[None, :]
        shared_fraction = count / torch.minimum(
            candidate_count, profile_count
        ).clamp_min(1.0)
        valid = profile_supported[None, :].bool().expand_as(count).clone()
        valid &= count >= minimum_rest_fraction * candidates.shape[-2]
        valid &= shared_fraction >= minimum_shared_fraction
        valid &= denominator > 1.0e-8
        return torch.where(valid, similarity, torch.zeros_like(similarity)), valid


def _pool_wavelength_weight(
    weight: torch.Tensor | None,
    mask: torch.Tensor,
    token_count: int,
) -> torch.Tensor | None:
    """Average wavelength reliability onto the encoder token grid."""
    if weight is None:
        return None
    if weight.shape != mask.shape:
        raise ValueError("Spectral reliability weight must match its mask")
    if weight.shape[-1] % token_count:
        raise ValueError("Token count must divide the wavelength weight length")
    pool_size = weight.shape[-1] // token_count
    measured = mask.reshape(*mask.shape[:-1], token_count, pool_size).to(
        weight.dtype
    )
    grouped = weight.reshape(*weight.shape[:-1], token_count, pool_size)
    count = measured.sum(dim=-1)
    return (grouped * measured).sum(dim=-1) / count.clamp_min(1.0)


def _supported_log_mean_exp(
    values: torch.Tensor,
    support: torch.Tensor,
    *,
    dim: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = support.sum(dim=dim)
    scaled = (values / temperature).masked_fill(~support, -torch.inf)
    score = temperature * (
        torch.logsumexp(scaled, dim=dim)
        - torch.log(count.clamp_min(1).to(values.dtype))
    )
    valid = count > 0
    return torch.where(valid, score, torch.zeros_like(score)), valid
