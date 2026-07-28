"""Spectrotemporal transformer backbone with factored attention."""

from __future__ import annotations

import torch
import torch.nn as nn

from strider.network.config import BackboneConfig
from strider.network.layers import (
    CNNStemEmbedding,
    FourierPositionalEncoding,
    FiLMLayer,
)


class SpectralAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        normed = self.norm(x)
        x = x + self.attn(
            normed, normed, normed,
            key_padding_mask=key_padding_mask, need_weights=False,
        )[0]
        x = x + self.ff(x)
        return x


class TemporalAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, n_patches: int, epoch_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T_times_P, D = x.shape
        T = T_times_P // n_patches
        x = x.reshape(B, T, n_patches, D)
        x = x.permute(0, 2, 1, 3).reshape(B * n_patches, T, D)

        kpm = None
        if epoch_mask is not None:
            kpm = (~epoch_mask).unsqueeze(1).expand(-1, n_patches, -1).reshape(B * n_patches, T)

        normed = self.norm(x)
        x = x + self.attn(normed, normed, normed, key_padding_mask=kpm, need_weights=False)[0]
        x = x + self.ff(x)
        x = x.reshape(B, n_patches, T, D).permute(0, 2, 1, 3).reshape(B, T_times_P, D)
        return x


class SpectrotemporalBackbone(nn.Module):
    """Factored spectral/temporal attention backbone.

    Input: (B, T, C, W) multi-epoch spectra
    Output: (B, T*P, D) token embeddings + (B, D) CLS summary
    """

    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config
        self.n_patches = config.n_patches

        self.patch_embed = CNNStemEmbedding(
            config.n_channels,
            config.patch_size,
            config.d_model,
            config.cnn_stem_channels,
            config.cnn_stem_kernels,
        )
        self.pos_enc = FourierPositionalEncoding(config.n_freqs, config.d_model)
        self.phase_film = FiLMLayer(1, config.d_model)

        self.blocks = nn.ModuleList()
        for _ in range(config.n_pairs):
            self.blocks.append(
                SpectralAttentionBlock(config.d_model, config.n_heads, config.dropout)
            )
            self.blocks.append(TemporalAttentionBlock(config.d_model, config.n_heads, config.dropout))

        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        spectra: torch.Tensor,
        phases: torch.Tensor,
        wave_grid_frac: torch.Tensor | None = None,
        wave_mask: torch.Tensor | None = None,
        epoch_mask: torch.Tensor | None = None,
    ) -> dict:
        B, T, C, W = spectra.shape
        P = self.n_patches

        x = self.patch_embed(spectra.reshape(B * T, C, W))
        x = x.reshape(B, T * P, -1)

        if wave_grid_frac is not None:
            if wave_grid_frac.dim() == 2:
                wave_grid_frac = wave_grid_frac[0]
            pos = self.pos_enc(wave_grid_frac).unsqueeze(0).expand(B, -1, -1)
            x = x + pos.repeat(1, T, 1)

        # Phase conditioning, one epoch at a time.
        phase_cond = phases.unsqueeze(-1)
        x_by_epoch = x.reshape(B, T, P, -1)
        x = torch.stack(
            [self.phase_film(x_by_epoch[:, t], phase_cond[:, t]) for t in range(T)],
            dim=1,
        ).reshape(B, T * P, -1)

        # Per-patch wave mask: (B*T, P) True=valid, False=masked
        wave_patch_mask = None
        spectral_kpm = None
        if wave_mask is not None:
            wm = wave_mask.reshape(B * T, W)
            wave_patch_mask = wm.reshape(B * T, P, -1).any(dim=-1)
            spectral_kpm = ~wave_patch_mask
            x = (x.reshape(B * T, P, -1) * wave_patch_mask.unsqueeze(-1).to(x.dtype)).reshape(
                B, T * P, -1
            )

        for pair_idx in range(self.config.n_pairs):
            spectral_block = self.blocks[2 * pair_idx]
            temporal_block = self.blocks[2 * pair_idx + 1]

            x = x.reshape(B * T, P, -1)
            x = spectral_block(x, key_padding_mask=spectral_kpm)
            x = x.reshape(B, T * P, -1)

            x = temporal_block(x, P, epoch_mask)

        x = self.norm(x)

        if epoch_mask is not None:
            valid_mask = epoch_mask.unsqueeze(-1).unsqueeze(-1).expand(B, T, P, x.shape[-1])
            valid_mask = valid_mask.reshape(B, T * P, -1)
            cls = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
        else:
            cls = x.mean(dim=1)

        return {'tokens': x, 'cls': cls}
