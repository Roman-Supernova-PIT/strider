"""Joint class-redshift evidence from trainable spectral signatures."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from strider.constants import LOG_WAVE_MIN, LOG_WAVE_RANGE, N_WAVE, PHASE_NORM
from strider.network.combiner import CombinerConfig, MultiEpochCombiner
from strider.network.matcher import gather_windows, phase_interpolate_weights
from strider.network.signatures import SignatureConfig, SpectralSignatures
from strider.network.template_bank import TemplateBank


@dataclass
class DetectorConfig:
    """Configuration embedded in the STRIDER model."""

    n_z_bins: int = 500
    z_min: float = 0.0
    z_max: float = 3.0
    patch_size: int = 8
    window_radius_patches: int = 7
    scan_dim: int = 8
    temperature: float = 10.0
    top_k: int = 3
    phase_neighbor_fallback: bool = True
    combiner: CombinerConfig = field(default_factory=CombinerConfig)
    ncc_chunk_size: int = 256


class EvidenceDetector(nn.Module):
    """Signature detector: backbone tokens → class × redshift match surface.

    Templates are not stored as buffers; the signatures module holds them as
    Parameters, so a checkpoint carries the trained signatures rather than
    just the bank initialization.
    """

    def __init__(
        self,
        config: DetectorConfig,
        bank: TemplateBank,
        d_model: int,
        n_classes_override: int | None = None,
    ):
        super().__init__()
        if config.n_z_bins < 2:
            raise ValueError(f'n_z_bins must be >= 2, got {config.n_z_bins}')
        if config.scan_dim <= 0:
            raise ValueError(f'scan_dim must be positive, got {config.scan_dim}')
        if config.temperature <= 0:
            raise ValueError(f'temperature must be positive, got {config.temperature}')

        self.config = config
        self.bank_source_path = bank.source_path
        self.n_patches = N_WAVE // config.patch_size
        self.window_size = 2 * config.window_radius_patches + 1
        self.n_classes = bank.n_classes if n_classes_override is None else n_classes_override
        self.n_features = bank.n_features
        self.n_phase_bins = bank.n_phase_bins
        self.k_max = bank.k_max
        self.d_model = d_model

        self.register_buffer(
            'temperature',
            torch.tensor(float(config.temperature), dtype=torch.float32),
            persistent=True,
        )

        self.signatures = SpectralSignatures(
            SignatureConfig(
                window_size=self.window_size,
                scan_dim=config.scan_dim,
                phase_neighbor_fallback=config.phase_neighbor_fallback,
            ),
            bank,
        )

        # This learned map aligns backbone tokens with the signature subspace.
        self.token_proj = nn.Linear(d_model, config.scan_dim)

        self.combiner = MultiEpochCombiner(config.combiner)

        z_grid = torch.linspace(
            config.z_min, config.z_max, config.n_z_bins, dtype=torch.float32,
        )
        feature_rest_waves = torch.from_numpy(bank.feature_rest_waves).to(torch.float32)
        gather_idx, gather_valid = self._build_gather_grid(
            z_grid, feature_rest_waves,
            config.window_radius_patches, config.patch_size,
        )
        phase_bin_centers = torch.from_numpy(bank.phase_bin_centers).to(torch.float32)

        self.register_buffer('z_grid', z_grid, persistent=False)
        self.register_buffer('_gather_idx', gather_idx, persistent=False)
        self.register_buffer('_gather_valid', gather_valid, persistent=False)
        self.register_buffer('phase_bin_centers', phase_bin_centers, persistent=False)


    def forward(
        self,
        tokens: torch.Tensor,                            # (B, T*P, D)
        phases: torch.Tensor,                            # (B, T) normalized phase
        n_patches: int,
        epoch_mask: torch.Tensor | None = None,
        ivar_per_patch_te: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the per-(class, z) match surface from backbone tokens.

        Returns a dict with:
          match_surface: (B, n_classes, n_z_bins) — primary output
          class_log_probs: (B, n_classes) — class marginal log P (already normalized)
          z_log_posterior: (B, n_z_bins) — z marginal log P
          joint_log_posterior: (B, n_classes, n_z_bins) — full joint
          match_entropy: scalar — H(p(class)) averaged across batch (diagnostic)
          signature_active_fraction: scalar — fraction of cells with valid contributions
        """
        return self._forward_signatures(
            tokens, phases, n_patches, epoch_mask, ivar_per_patch_te,
        )

    def _forward_signatures(
        self,
        tokens: torch.Tensor,
        phases: torch.Tensor,
        n_patches: int,
        epoch_mask: torch.Tensor | None,
        ivar_per_patch_te: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        B, TP, D = tokens.shape
        if n_patches != self.n_patches:
            raise ValueError(f'expected n_patches={self.n_patches}, got {n_patches}')
        if TP % n_patches != 0:
            raise ValueError(f'tokens length {TP} not divisible by n_patches={n_patches}')
        T = TP // n_patches
        if phases.shape != (B, T):
            raise ValueError(f'phases shape {tuple(phases.shape)} != ({B}, {T})')

        # A backbone under autocast can hand us float16 against float32 weights.
        if tokens.dtype != self.token_proj.weight.dtype:
            tokens = tokens.to(self.token_proj.weight.dtype)

        energy = self.token_proj(tokens).reshape(B, T, n_patches, self.config.scan_dim)
        energy_flat = energy.reshape(B * T, n_patches, self.config.scan_dim)
        phase_days_flat = (phases * PHASE_NORM).reshape(B * T)

        # Gather, phase-interpolate and correlate in one chunked loop: the
        # intermediates run to several GB if built for all of B*T at once.
        BT = B * T
        chunk = max(1, int(self.config.ncc_chunk_size))
        match_pieces: list[torch.Tensor] = []
        active_eff_accum: list[torch.Tensor] = []
        for start in range(0, BT, chunk):
            end = min(start + chunk, BT)
            energy_chunk = energy_flat[start:end]
            phase_chunk = phase_days_flat[start:end]

            gathered_chunk = gather_windows(
                energy_chunk, self._gather_idx, self._gather_valid,
            )  # (chunk, Z, F, W, S)

            sig_eff_chunk, active_chunk = self._phase_interpolated_signatures(
                phase_chunk,
            )  # (chunk, C, F, K, W, S), (chunk, C, F, K)

            match_chunk = self._vectorized_ncc(
                gathered_chunk, sig_eff_chunk, active_chunk,
            )  # (chunk, Z, F, C)
            match_pieces.append(match_chunk)
            active_eff_accum.append(active_chunk)

        match_per_epoch_per_feature = torch.cat(match_pieces, dim=0)
        active_eff = torch.cat(active_eff_accum, dim=0)

        match_per_epoch = match_per_epoch_per_feature.mean(dim=2)  # (B*T, Z, C)
        epoch_match_surface = match_per_epoch.reshape(
            B, T, self.config.n_z_bins, self.n_classes,
        ).permute(0, 1, 3, 2).contiguous()

        match_surface = self.combiner(
            epoch_match_surface, epoch_mask=epoch_mask,
            ivar_per_patch_te=ivar_per_patch_te,
        )  # (B, C, Z)

        M = self.temperature * match_surface
        joint_log_post = M - torch.logsumexp(
            M.reshape(B, -1), dim=-1,
        ).reshape(B, 1, 1)
        class_log_probs = torch.logsumexp(joint_log_post, dim=2)
        z_log_posterior = torch.logsumexp(joint_log_post, dim=1)

        class_probs = class_log_probs.exp()
        match_entropy = -(class_probs * class_log_probs).sum(dim=-1).mean()
        # Signature availability, not evidence coverage: with phase-neighbor
        # fallback this trends to ~1.0 by construction.
        signature_active_fraction = active_eff.float().mean()

        out = {
            'match_surface': match_surface,                  # (B, C, Z) primary
            'class_log_probs': class_log_probs,              # (B, C) normalized
            'z_log_posterior': z_log_posterior,              # (B, Z) normalized
            'joint_log_posterior': joint_log_post,           # (B, C, Z)
            'temperature': self.temperature.detach(),
            'match_entropy': match_entropy,
            'signature_active_fraction': signature_active_fraction,
        }
        return out


    def _phase_interpolated_signatures(
        self,
        phase_days: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Phase-blend signatures per epoch.

        Returns:
          sig_eff:    (B*T, C, F, K, W, S)
          active_eff: (B*T, C, F, K)
        """
        sigs, active = self.signatures()
        # sigs: (C, P, F, K, W, S); active: (C, P, F, K)

        lo, hi, alpha = phase_interpolate_weights(phase_days, self.phase_bin_centers)
        s_lo = sigs[:, lo].permute(1, 0, 2, 3, 4, 5)
        s_hi = sigs[:, hi].permute(1, 0, 2, 3, 4, 5)
        a_lo = active[:, lo].permute(1, 0, 2, 3)
        a_hi = active[:, hi].permute(1, 0, 2, 3)
        a = alpha.reshape(-1, 1, 1, 1, 1, 1)
        # Slot-level NaN rescue: if one bin is
        # NaN at a slot, use the neighbor's slot rather than blending NaN.
        s_lo_nan = torch.isnan(s_lo).any(dim=(-1, -2), keepdim=True)
        s_hi_nan = torch.isnan(s_hi).any(dim=(-1, -2), keepdim=True)
        s_lo_safe = torch.where(s_lo_nan, s_hi, s_lo)
        s_hi_safe = torch.where(s_hi_nan, s_lo, s_hi)
        sig_eff = (1.0 - a) * s_lo_safe + a * s_hi_safe
        sig_eff = torch.nan_to_num(sig_eff, nan=0.0)
        active_eff = a_lo | a_hi
        return sig_eff, active_eff

    def _vectorized_ncc(
        self,
        g: torch.Tensor,         # (n, Z, F, W, S)
        t: torch.Tensor,         # (n, C, F, K, W, S)
        active: torch.Tensor,    # (n, C, F, K)
        eps: float = 1e-8,
    ) -> torch.Tensor:           # (n, Z, F, C)
        n, Z, nF, W, S = g.shape
        _, C, _, K, _, _ = t.shape
        g_flat = g.reshape(n, Z, nF, W * S)
        t_flat = t.reshape(n, C, nF, K, W * S)
        num = torch.einsum('nzfa,ncfka->nzcfk', g_flat, t_flat)
        g_norm = (g_flat * g_flat).sum(dim=-1).clamp(min=eps).sqrt()      # (n, Z, F)
        t_norm = (t_flat * t_flat).sum(dim=-1).clamp(min=eps).sqrt()      # (n, C, F, K)
        score = num / (g_norm.unsqueeze(2).unsqueeze(-1)
                       * t_norm.unsqueeze(1) + eps)                       # (n, Z, C, F, K)
        score = score.permute(0, 1, 3, 2, 4).contiguous()                 # (n, Z, F, C, K)

        active_b = active.permute(0, 2, 1, 3).unsqueeze(1)                # (n, 1, F, C, K)
        masked = score.masked_fill(~active_b, float('-inf'))
        n_active = active_b.sum(dim=-1)                                   # (n, 1, F, C)

        # Mean of the top-k prototype scores, ignoring inactive slots.
        k_eff = min(int(self.config.top_k), K)
        top = masked.topk(k=k_eff, dim=-1).values
        valid = torch.isfinite(top)
        top_clean = torch.where(valid, top, torch.zeros_like(top))
        denom = valid.sum(dim=-1).clamp(min=1).to(top.dtype)
        out = top_clean.sum(dim=-1) / denom

        return torch.where(n_active.expand_as(out) > 0, out, torch.zeros_like(out))


    @staticmethod
    def _build_gather_grid(
        z_grid: torch.Tensor,
        feature_rest_waves: torch.Tensor,
        window_radius_patches: int,
        patch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_patches = N_WAVE // patch_size
        obs_wave = feature_rest_waves.unsqueeze(0) * (1.0 + z_grid.unsqueeze(1))
        log_obs = torch.log(obs_wave.clamp(min=1.0))
        frac_bin = (log_obs - LOG_WAVE_MIN) / LOG_WAVE_RANGE * (N_WAVE - 1)
        patch_center = (frac_bin / patch_size).round().long()
        offsets = torch.arange(-window_radius_patches, window_radius_patches + 1)
        raw_idx = patch_center.unsqueeze(-1) + offsets
        valid = (raw_idx >= 0) & (raw_idx < n_patches)
        idx = raw_idx.clamp(0, n_patches - 1)
        return idx, valid


    @staticmethod
    def _build_gather_grid(
        z_grid: torch.Tensor,
        feature_rest_waves: torch.Tensor,
        window_radius_patches: int,
        patch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_patches = N_WAVE // patch_size
        obs_wave = feature_rest_waves.unsqueeze(0) * (1.0 + z_grid.unsqueeze(1))
        log_obs = torch.log(obs_wave.clamp(min=1.0))
        frac_bin = (log_obs - LOG_WAVE_MIN) / LOG_WAVE_RANGE * (N_WAVE - 1)
        patch_center = (frac_bin / patch_size).round().long()
        offsets = torch.arange(-window_radius_patches, window_radius_patches + 1)
        raw_idx = patch_center.unsqueeze(-1) + offsets
        valid = (raw_idx >= 0) & (raw_idx < n_patches)
        idx = raw_idx.clamp(0, n_patches - 1)
        return idx, valid

    def cosine_drift(self) -> torch.Tensor:
        """Cosine drift of the signatures from their bank initialization."""
        if self.signatures is None:
            return torch.tensor(0.0, device=self.token_proj.weight.device)
        return self.signatures.cosine_drift()
