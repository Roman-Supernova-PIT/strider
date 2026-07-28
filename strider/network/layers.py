"""Reusable layers: positional encoding, FiLM conditioning, patch embedding."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierPositionalEncoding(nn.Module):
    """Learnable Fourier features for continuous log-wavelength positions."""

    def __init__(self, n_freqs: int, d_model: int):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(n_freqs) * 0.1)
        self.proj = nn.Linear(2 * n_freqs, d_model)

    def forward(self, log_wave_frac: torch.Tensor) -> torch.Tensor:
        angles = log_wave_frac.unsqueeze(-1) * self.freqs * 2 * math.pi
        features = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return self.proj(features)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: scale and shift conditioned on input."""

    def __init__(self, cond_dim: int, d_model: int):
        super().__init__()
        self.net = nn.Linear(cond_dim, 2 * d_model)
        nn.init.zeros_(self.net.weight)
        with torch.no_grad():
            self.net.bias[:d_model].fill_(1.0)
            self.net.bias[d_model:].zero_()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(cond).chunk(2, dim=-1)
        if gamma.dim() < x.dim():
            gamma = gamma.unsqueeze(-2)
            beta = beta.unsqueeze(-2)
        return gamma * x + beta


class CNNStemEmbedding(nn.Module):
    """Full-resolution CNN feature extractor followed by patch striding.

    This preserves the PatchEmbedding output contract, (B*T, P, d_model), but
    gives each token a wider local receptive field before the transformer sees
    it. The final convolution owns the stride, so total downsampling remains
    exactly `patch_size`.

    Note: because the stride-1 convolutions mix neighbouring wavelength bins
    before the backbone applies patch-level wave masking, bad-bin masks are no
    longer pixel-surgical inside the local receptive field. This is intentional
    for the CNN-stem ablation: the mask still excludes affected patch tokens
    from attention, while the stem learns robust local spectral filters.
    """

    def __init__(
        self,
        n_channels: int,
        patch_size: int,
        d_model: int,
        stem_channels: list[int],
        stem_kernels: list[int],
    ):
        super().__init__()
        if len(stem_channels) != len(stem_kernels):
            raise ValueError("stem_channels and stem_kernels must have the same length")

        layers: list[nn.Module] = []
        in_ch = n_channels
        for out_ch, kernel in zip(stem_channels, stem_kernels):
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2),
                nn.GELU(),
            ])
            in_ch = out_ch
        layers.append(nn.Conv1d(in_ch, d_model, kernel_size=patch_size, stride=patch_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).transpose(1, 2)
