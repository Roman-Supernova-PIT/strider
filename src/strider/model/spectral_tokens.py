"""Mask-aware observer-frame spectral tokens."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import nn


class MaskAwareContinuumRemoval(nn.Module):
    """Remove structure broader than a fixed width on the log-wavelength grid."""

    def __init__(self, sigma_bins: float, truncate: float = 4.0) -> None:
        super().__init__()
        if sigma_bins <= 0.0:
            raise ValueError("Continuum smoothing width must be positive")
        radius = max(1, int(round(float(truncate) * float(sigma_bins))))
        position = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (position / float(sigma_bins)).square())
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel.reshape(1, 1, -1))
        self.padding = radius
        self.sigma_bins = float(sigma_bins)

    def forward(
        self,
        flux: torch.Tensor,
        mask: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if flux.shape != mask.shape:
            raise ValueError("Flux and wavelength mask must have the same shape")
        if weight is not None and weight.shape != flux.shape:
            raise ValueError("Continuum weight must match flux")
        leading = flux.shape[:-1]
        values = flux.reshape(-1, 1, flux.shape[-1])
        measured = mask.reshape(-1, 1, mask.shape[-1]).to(values.dtype)
        smoothing_weight = measured
        if weight is not None:
            reliability = weight.reshape(-1, 1, weight.shape[-1]).to(values.dtype)
            smoothing_weight = measured * reliability.clamp_min(0.0)
        kernel = self.kernel.to(values.dtype)
        numerator = functional.conv1d(
            values * smoothing_weight,
            kernel,
            padding=self.padding,
        )
        denominator = functional.conv1d(
            smoothing_weight,
            kernel,
            padding=self.padding,
        )
        continuum = numerator / denominator.clamp_min(1.0e-6)
        flattened = (values - continuum) * measured
        return flattened.reshape(*leading, flux.shape[-1])


def velocity_sigma_to_log_bins(
    wavelength_min: float,
    wavelength_max: float,
    wavelength_bins: int,
    sigma_km_s: float,
) -> float:
    """Convert a velocity width to bins on a uniform log-wavelength grid."""
    if wavelength_min <= 0.0 or wavelength_max <= wavelength_min:
        raise ValueError("Wavelength limits must be positive and ordered")
    if wavelength_bins < 2:
        raise ValueError("At least two wavelength bins are required")
    if sigma_km_s <= 0.0:
        raise ValueError("Continuum velocity width must be positive")
    speed_of_light_km_s = 299_792.458
    log_step = math.log(wavelength_max / wavelength_min) / (wavelength_bins - 1)
    return math.log1p(sigma_km_s / speed_of_light_km_s) / log_step


def normalize_valid_bins(
    flux: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Centre and scale each spectrum using only measured wavelength bins."""
    support = mask.to(flux.dtype)
    measured = support
    if weight is not None:
        if weight.shape != flux.shape:
            raise ValueError("Normalization weight must match flux")
        measured = measured * weight.to(flux.dtype).clamp_min(0.0)
    count = measured.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(flux.dtype).tiny
    )
    mean = (flux * measured).sum(dim=-1, keepdim=True) / count
    difference = flux - mean
    centred = difference * support
    variance = (difference.square() * measured).sum(dim=-1, keepdim=True) / count
    return centred / torch.sqrt(variance.clamp_min(1e-6))


def relative_visit_amplitude(
    flux: torch.Tensor,
    wavelength_mask: torch.Tensor,
    visit_mask: torch.Tensor,
) -> torch.Tensor:
    """Return the visit brightness trajectory after removing one object scale."""
    if flux.shape != wavelength_mask.shape:
        raise ValueError("Flux and wavelength mask must have the same shape")
    if visit_mask.shape != flux.shape[:2]:
        raise ValueError("Visit mask must match the object and visit axes")
    measured = wavelength_mask.to(flux.dtype) * visit_mask[..., None]
    count = measured.sum(dim=-1).clamp_min(1.0)
    root_mean_square = torch.sqrt(
        (flux.square() * measured).sum(dim=-1) / count
    )
    valid = (measured.sum(dim=-1) > 0) & visit_mask.bool()
    tiny = torch.finfo(flux.dtype).tiny
    log_amplitude = torch.log(root_mean_square.clamp_min(tiny))
    centre = (log_amplitude * valid).sum(dim=1, keepdim=True)
    centre = centre / valid.sum(dim=1, keepdim=True).clamp_min(1)
    return (log_amplitude - centre) * valid.to(flux.dtype)


def object_normalized_flux_amplitude(
    flux: torch.Tensor,
    wavelength_mask: torch.Tensor,
    visit_mask: torch.Tensor,
    visit_flux_scale: torch.Tensor,
    wavelength_weight: torch.Tensor,
    wavelength_reliability: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return signed spectral-amplitude evolution with one scale per object.

    ``wavelength_reliability`` may continuously reduce the influence of noisy
    measurements without turning reported uncertainty into a hard support cut.
    """
    if flux.shape != wavelength_mask.shape:
        raise ValueError("Flux and wavelength mask must have the same shape")
    if visit_mask.shape != flux.shape[:2]:
        raise ValueError("Visit mask must match the object and visit axes")
    if visit_flux_scale.shape != flux.shape[:2]:
        raise ValueError("Visit flux scale must match the object and visit axes")
    if wavelength_weight.shape != (flux.shape[-1],):
        raise ValueError("Wavelength weight must match the spectral axis")
    if wavelength_reliability is not None and wavelength_reliability.shape != flux.shape:
        raise ValueError("Wavelength reliability must match flux")

    measured = wavelength_mask.to(flux.dtype) * visit_mask[..., None]
    weight = wavelength_weight.to(flux.dtype)[None, None, :] * measured
    if wavelength_reliability is not None:
        reliability = wavelength_reliability.to(flux.dtype)
        reliability = torch.where(
            torch.isfinite(reliability) & (reliability > 0.0),
            reliability,
            torch.zeros_like(reliability),
        )
        weight = weight * reliability
    raw_flux = flux * visit_flux_scale[..., None]
    band_flux = (raw_flux * weight).sum(dim=-1)
    band_flux = band_flux / weight.sum(dim=-1).clamp_min(
        torch.finfo(flux.dtype).tiny
    )
    valid = (weight.sum(dim=-1) > 0) & visit_mask.bool()
    object_scale = torch.sqrt(
        (band_flux.square() * valid).sum(dim=1, keepdim=True)
        / valid.sum(dim=1, keepdim=True).clamp_min(1)
    )
    relative = band_flux / object_scale.clamp_min(torch.finfo(flux.dtype).tiny)
    return relative * valid.to(flux.dtype)


class MaskAwareTokenEncoder(nn.Module):
    """Encode flux without treating missing wavelength bins as measured zeros."""

    def __init__(
        self,
        token_dim: int,
        dropout: float,
        minimum_support: float = 0.5,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if activation not in {"gelu", "linear"}:
            raise ValueError("Token activation must be 'gelu' or 'linear'")
        self.first = _MaskedConv(
            1,
            token_dim,
            kernel_size=7,
            dilation=1,
            minimum_support=minimum_support,
            activation=activation,
        )
        self.second = _MaskedConv(
            token_dim,
            token_dim,
            kernel_size=5,
            dilation=2,
            minimum_support=minimum_support,
            activation=activation,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        flux: torch.Tensor,
        mask: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if weight is not None and weight.shape != flux.shape:
            raise ValueError("Encoder reliability weight must match flux")
        leading = flux.shape[:-1]
        values = flux.reshape(-1, 1, flux.shape[-1])
        support = mask.reshape(-1, 1, mask.shape[-1]).to(values.dtype)
        reliability = (
            None
            if weight is None
            else weight.reshape(-1, 1, weight.shape[-1]).to(values.dtype)
        )
        values, support = self.first(values, support, reliability)
        residual = values
        values, support = self.second(values, support)
        values = self.dropout(values + residual * support)
        tokens = values.transpose(1, 2)
        tokens = self.output_norm(tokens) * support.transpose(1, 2)
        return (
            tokens.reshape(*leading, tokens.shape[-2], tokens.shape[-1]),
            support.reshape(*leading, support.shape[-1]),
        )


class MaskAwareMultiscaleAttentionEncoder(nn.Module):
    """Encode local line structure, then connect separated wavelength regions.

    The convolution branches operate at three physically useful widths before
    the wavelength axis is compressed.  A single self-attention block then
    relates the compressed regions.  Keeping attention after compression makes
    this suitable for the explicit class--redshift scan while retaining masks
    throughout the calculation.
    """

    def __init__(
        self,
        token_dim: int,
        attention_heads: int,
        dropout: float,
        pool_size: int = 4,
        minimum_support: float = 0.5,
    ) -> None:
        super().__init__()
        if token_dim < 1:
            raise ValueError("Token dimension must be positive")
        if attention_heads < 1 or token_dim % attention_heads:
            raise ValueError("Token dimension must be divisible by the attention heads")
        if pool_size < 1:
            raise ValueError("Token pool size must be positive")
        self.pool_size = int(pool_size)
        self.minimum_support = float(minimum_support)
        self.branches = nn.ModuleList(
            [
                _MaskedConv(
                    1,
                    token_dim,
                    kernel_size=kernel_size,
                    dilation=1,
                    minimum_support=minimum_support,
                    activation="gelu",
                )
                for kernel_size in (3, 9, 17)
            ]
        )
        self.multiscale_mix = nn.Sequential(
            nn.Linear(3 * token_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(token_dim),
        )
        self.attention_norm = nn.LayerNorm(token_dim)
        self.value_norm = nn.LayerNorm(token_dim)
        self.attention = nn.MultiheadAttention(
            token_dim,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, 2 * token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * token_dim, token_dim),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        flux: torch.Tensor,
        mask: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if flux.shape != mask.shape:
            raise ValueError("Flux and wavelength mask must have the same shape")
        if weight is not None and weight.shape != flux.shape:
            raise ValueError("Encoder reliability weight must match flux")
        if flux.shape[-1] % self.pool_size:
            raise ValueError("Wavelength bins must be divisible by the token pool size")
        leading = flux.shape[:-1]
        values = flux.reshape(-1, 1, flux.shape[-1])
        measured = mask.reshape(-1, 1, mask.shape[-1]).to(values.dtype)
        reliability = (
            None
            if weight is None
            else weight.reshape(-1, 1, weight.shape[-1]).to(values.dtype)
        )
        branch_values = []
        branch_support = []
        for branch in self.branches:
            encoded, supported = branch(values, measured, reliability)
            branch_values.append(encoded.transpose(1, 2))
            branch_support.append(supported.transpose(1, 2))

        support_count = torch.stack(branch_support, dim=0).sum(dim=0)
        mixed = torch.cat(branch_values, dim=-1)
        mixed = self.multiscale_mix(mixed)
        mixed = mixed * (support_count > 0).to(mixed.dtype)

        pooled_bins = flux.shape[-1] // self.pool_size
        mixed = mixed.reshape(
            mixed.shape[0], pooled_bins, self.pool_size, mixed.shape[-1]
        )
        centre_support = measured.transpose(1, 2).reshape(
            measured.shape[0], pooled_bins, self.pool_size
        )
        count = centre_support.sum(dim=-1)
        pooled_support = count >= self.minimum_support * self.pool_size
        pooled = (mixed * centre_support[..., None]).sum(dim=-2)
        pooled = pooled / count[..., None].clamp_min(1.0)
        pooled = pooled * pooled_support[..., None].to(pooled.dtype)

        position = _fourier_position(
            pooled_bins, pooled.shape[-1], pooled.device, pooled.dtype
        )
        tokens = pooled + position[None, :, :] * pooled_support[..., None]
        active = pooled_support.any(dim=1)
        contextual = torch.zeros_like(tokens)
        if active.any():
            active_pooled = pooled[active]
            active_tokens = tokens[active]
            active_support = pooled_support[active]
            normalized = self.attention_norm(active_tokens)
            attended = self.attention(
                normalized,
                normalized,
                self.value_norm(active_pooled),
                key_padding_mask=~active_support,
                need_weights=False,
            )[0]
            weight = active_support[..., None].to(active_tokens.dtype)
            # Position controls which regions exchange information but is not
            # copied into the representation used for template similarity.
            active_tokens = (active_pooled + attended) * weight
            active_tokens = (active_tokens + self.feed_forward(active_tokens)) * weight
            contextual[active] = self.output_norm(active_tokens) * weight

        return (
            contextual.reshape(*leading, pooled_bins, contextual.shape[-1]),
            pooled_support.reshape(*leading, pooled_bins),
        )


def _fourier_position(
    length: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a fixed sinusoidal wavelength position encoding."""
    position = torch.linspace(0.0, 1.0, length, device=device)
    pair_count = (width + 1) // 2
    frequency = torch.exp(
        torch.linspace(0.0, math.log(64.0), pair_count, device=device)
    )
    angle = 2.0 * math.pi * position[:, None] * frequency[None, :]
    encoded = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1).flatten(1)
    return encoded[:, :width].to(dtype)


class _MaskedConv(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int,
        minimum_support: float,
        activation: str,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.padding = int(padding)
        self.minimum_support = float(minimum_support)
        self.activation = activation
        self.register_buffer("support_kernel", torch.ones(1, 1, kernel_size))

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if weight is not None and weight.shape != mask.shape:
            raise ValueError("Convolution reliability weight must match mask")
        influence = mask
        if weight is not None:
            reliability = torch.where(
                torch.isfinite(weight) & (weight > 0.0),
                weight.to(values.dtype),
                torch.zeros_like(weight, dtype=values.dtype),
            )
            # Reliability is relative inverse variance. Capping it at one
            # prevents unusually precise bins from being amplified while
            # allowing increasingly uncertain bins to approach zero smoothly.
            influence = mask * reliability.clamp_max(1.0)
        convolved = self.conv(values * influence)
        support_count = functional.conv1d(
            mask,
            self.support_kernel.to(mask.dtype),
            padding=self.padding,
            dilation=self.dilation,
        )
        influence_count = functional.conv1d(
            influence,
            self.support_kernel.to(influence.dtype),
            padding=self.padding,
            dilation=self.dilation,
        )
        support_fraction = support_count / float(self.kernel_size)
        valid = (
            (support_fraction >= self.minimum_support)
            & (influence_count > 0.0)
        ).to(convolved.dtype)
        bias = self.conv.bias
        if bias is not None:
            bias_view = bias[None, :, None]
            convolved = (convolved - bias_view) / support_fraction.clamp_min(
                1.0 / self.kernel_size
            )
            convolved = convolved + bias_view
        if self.activation == "gelu":
            convolved = functional.gelu(convolved)
        convolved = convolved * valid
        return convolved, valid
