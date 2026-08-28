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

    def forward(self, flux: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if flux.shape != mask.shape:
            raise ValueError("Flux and wavelength mask must have the same shape")
        leading = flux.shape[:-1]
        values = flux.reshape(-1, 1, flux.shape[-1])
        measured = mask.reshape(-1, 1, mask.shape[-1]).to(values.dtype)
        kernel = self.kernel.to(values.dtype)
        numerator = functional.conv1d(
            values * measured,
            kernel,
            padding=self.padding,
        )
        denominator = functional.conv1d(
            measured,
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


def normalize_valid_bins(flux: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Centre and scale each spectrum using only measured wavelength bins."""
    measured = mask.to(flux.dtype)
    count = measured.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (flux * measured).sum(dim=-1, keepdim=True) / count
    centred = (flux - mean) * measured
    variance = centred.square().sum(dim=-1, keepdim=True) / count
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
) -> torch.Tensor:
    """Return signed broadband FLAM evolution with one scale per object."""
    if flux.shape != wavelength_mask.shape:
        raise ValueError("Flux and wavelength mask must have the same shape")
    if visit_mask.shape != flux.shape[:2]:
        raise ValueError("Visit mask must match the object and visit axes")
    if visit_flux_scale.shape != flux.shape[:2]:
        raise ValueError("Visit flux scale must match the object and visit axes")
    if wavelength_weight.shape != (flux.shape[-1],):
        raise ValueError("Wavelength weight must match the spectral axis")

    measured = wavelength_mask.to(flux.dtype) * visit_mask[..., None]
    weight = wavelength_weight.to(flux.dtype)[None, None, :] * measured
    raw_flux = flux * visit_flux_scale[..., None]
    band_flux = (raw_flux * weight).sum(dim=-1)
    band_flux = band_flux / weight.sum(dim=-1).clamp_min(
        torch.finfo(flux.dtype).tiny
    )
    valid = (measured.sum(dim=-1) > 0) & visit_mask.bool()
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        leading = flux.shape[:-1]
        values = flux.reshape(-1, 1, flux.shape[-1])
        support = mask.reshape(-1, 1, mask.shape[-1]).to(values.dtype)
        values, support = self.first(values, support)
        residual = values
        values, support = self.second(values, support)
        values = self.dropout(values + residual * support)
        tokens = values.transpose(1, 2)
        tokens = self.output_norm(tokens) * support.transpose(1, 2)
        return (
            tokens.reshape(*leading, tokens.shape[-2], tokens.shape[-1]),
            support.reshape(*leading, support.shape[-1]),
        )


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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        convolved = self.conv(values * mask)
        count = functional.conv1d(
            mask,
            self.support_kernel.to(mask.dtype),
            padding=self.padding,
            dilation=self.dilation,
        )
        fraction = count / float(self.kernel_size)
        valid = (fraction >= self.minimum_support).to(convolved.dtype)
        bias = self.conv.bias
        if bias is not None:
            bias_view = bias[None, :, None]
            convolved = (convolved - bias_view) / fraction.clamp_min(1.0 / self.kernel_size)
            convolved = convolved + bias_view
        if self.activation == "gelu":
            convolved = functional.gelu(convolved)
        convolved = convolved * valid
        return convolved, valid
