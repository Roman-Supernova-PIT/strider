"""The spectral encoder and the continuum removal, both aware of which pixels are measured.

Spectra here are on a uniform grid in log wavelength. A mask is 1 where a pixel is measured and 0 where it is
not, and a weight is a pixel's relative inverse variance.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["ContinuumRemoval", "SpectralEncoder", "velocity_to_log_bins", "normalise", "scale_free_normalise"]

SPEED_OF_LIGHT_KM_S = 299_792.458


def velocity_to_log_bins(minimum, maximum, bins, velocity_km_s):
    """Return a velocity width in pixels of a log-uniform wavelength grid."""
    log_step = math.log(maximum / minimum) / (bins - 1)
    return math.log1p(velocity_km_s / SPEED_OF_LIGHT_KM_S) / log_step


class ContinuumRemoval(nn.Module):
    """Subtract a spectrum's continuum: its weighted Gaussian smoothing over the measured pixels.

    Parameters
    ----------
    sigma_bins : float
        Width of the Gaussian in pixels.
    """

    def __init__(self, sigma_bins):
        super().__init__()
        radius = max(1, round(4.0 * sigma_bins))
        position = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (position / sigma_bins).square())
        self.register_buffer("kernel", (kernel / kernel.sum()).reshape(1, 1, -1), persistent=False)
        self.radius = radius

    def forward(self, flux, mask, weight=None):
        shape = flux.shape
        flux = flux.reshape(-1, 1, shape[-1])
        mask = mask.reshape(-1, 1, shape[-1]).to(flux.dtype)
        smoothing = mask if weight is None else mask * weight.reshape(-1, 1, shape[-1]).clamp_min(0.0)
        numerator = F.conv1d(flux * smoothing, self.kernel, padding=self.radius)
        denominator = F.conv1d(smoothing, self.kernel, padding=self.radius)
        return ((flux - numerator / denominator.clamp_min(1.0e-6)) * mask).reshape(shape)


def normalise(flux, mask):
    """Centre and scale each spectrum by its mean and standard deviation over the measured pixels."""
    measured = mask.to(flux.dtype)
    count = measured.sum(-1, keepdim=True).clamp_min(torch.finfo(flux.dtype).tiny)
    difference = flux - (flux * measured).sum(-1, keepdim=True) / count
    variance = (difference.square() * measured).sum(-1, keepdim=True) / count
    return difference * measured / torch.sqrt(variance.clamp_min(1e-6))


def scale_free_normalise(flux, mask):
    """Divide each spectrum by its largest measured absolute value, then `normalise` it."""
    scale = (flux.abs() * mask.to(flux.dtype)).amax(-1, keepdim=True)
    return normalise(flux / scale.clamp_min(torch.finfo(flux.dtype).tiny), mask)


class MaskedConv(nn.Module):
    """A 1-d convolution that counts only measured pixels, rescaled for the fraction of the kernel they fill.

    A pixel's influence is its mask times its weight, capped at one, so that noisy pixels count less and no pixel
    counts more than a measured one of median noise. Outputs with too few measured pixels in reach are zero.
    """

    def __init__(self, channels_in, channels_out, kernel_size, minimum_support):
        super().__init__()
        self.conv = nn.Conv1d(channels_in, channels_out, kernel_size, padding=(kernel_size - 1) // 2)
        self.kernel_size = kernel_size
        self.minimum_support = minimum_support

    def forward(self, values, mask, weight=None):
        influence = mask
        if weight is not None:
            weight = torch.where(torch.isfinite(weight) & (weight > 0.0), weight, torch.zeros_like(weight))
            influence = mask * weight.clamp_max(1.0)
        convolved = self.conv(values * influence)
        ones = torch.ones(1, 1, self.kernel_size, dtype=mask.dtype, device=mask.device)
        padding = self.conv.padding
        fraction = F.conv1d(mask, ones, padding=padding) / self.kernel_size
        valid = (fraction >= self.minimum_support) & (F.conv1d(influence, ones, padding=padding) > 0.0)
        bias = self.conv.bias[None, :, None]
        convolved = (convolved - bias) / fraction.clamp_min(1.0 / self.kernel_size) + bias
        return F.gelu(convolved) * valid.to(convolved.dtype), valid.to(convolved.dtype)


def fourier_position(length, width, dtype, device):
    """Fixed sinusoidal features of position along the wavelength axis, (length, width)."""
    position = torch.linspace(0.0, 1.0, length, device=device)
    frequency = torch.exp(torch.linspace(0.0, math.log(64.0), (width + 1) // 2, device=device))
    angle = 2.0 * math.pi * position[:, None] * frequency[None, :]
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1).flatten(1)[:, :width].to(dtype)


class SpectralEncoder(nn.Module):
    """Encode a spectrum as tokens: local line structure at three widths, then attention between regions.

    The front (three masked convolutions and their mix) acts alike at every pixel; the back pools every
    ``pool_size`` pixels into a token and relates the tokens by one masked self-attention block.

    Parameters
    ----------
    token_dim : int
        Width of each token.
    attention_heads : int
        Attention heads.
    dropout : float
        Dropout probability (training only).
    pool_size : int
        Pixels per token.
    minimum_support : float
        Least fraction of measured pixels a convolution window or token needs.
    feedforward_multiplier : int
        Width of the feed-forward layer, in tokens widths.
    """

    def __init__(self, token_dim, attention_heads, dropout, pool_size, minimum_support, feedforward_multiplier):
        super().__init__()
        self.pool_size = pool_size
        self.minimum_support = minimum_support
        self.branches = nn.ModuleList(MaskedConv(1, token_dim, size, minimum_support) for size in (3, 9, 17))
        self.multiscale_mix = nn.Sequential(
            nn.Linear(3 * token_dim, token_dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(token_dim)
        )
        self.attention_norm = nn.LayerNorm(token_dim)
        self.value_norm = nn.LayerNorm(token_dim)
        self.attention = nn.MultiheadAttention(token_dim, attention_heads, dropout=dropout, batch_first=True)
        width = feedforward_multiplier * token_dim
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, token_dim),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(token_dim)

    def forward(self, flux, mask, weight=None):
        """Tokens (N, T, d) and their support (N, T) of spectra (N, L)."""
        return self.back(*self.front(flux, mask, weight))

    def front(self, flux, mask, weight=None):
        """Front features (N, L, d), zero where no branch is supported, and the mask as (N, 1, L)."""
        values = flux.reshape(-1, 1, flux.shape[-1])
        mask = mask.reshape(-1, 1, mask.shape[-1]).to(values.dtype)
        weight = None if weight is None else weight.reshape(-1, 1, weight.shape[-1]).to(values.dtype)
        encoded, supported = zip(*(branch(values, mask, weight) for branch in self.branches))
        mixed = self.multiscale_mix(torch.cat([value.transpose(1, 2) for value in encoded], dim=-1))
        count = torch.stack([value.transpose(1, 2) for value in supported]).sum(0)
        return mixed * (count > 0).to(mixed.dtype), mask

    def back(self, features, mask):
        """Tokens (N, T, d) and their support (N, T) from front features (N, L, d) and the mask (N, 1, L)."""
        count, width = features.shape[1] // self.pool_size, features.shape[-1]
        features = features.reshape(features.shape[0], count, self.pool_size, width)
        measured = mask.transpose(1, 2).reshape(mask.shape[0], count, self.pool_size)
        measured_count = measured.sum(-1)
        support = measured_count >= self.minimum_support * self.pool_size
        pooled = (features * measured[..., None]).sum(-2) / measured_count[..., None].clamp_min(1.0)
        pooled = pooled * support[..., None].to(pooled.dtype)
        # position steers which regions exchange information, but is not copied into the tokens
        position = fourier_position(count, width, pooled.dtype, pooled.device)
        tokens = pooled + position[None] * support[..., None]
        contextual = torch.zeros_like(tokens)
        active = support.any(dim=1)
        if active.any():
            pooled, tokens, support_active = pooled[active], tokens[active], support[active]
            normalised = self.attention_norm(tokens)
            attended = self.attention(
                normalised, normalised, self.value_norm(pooled), key_padding_mask=~support_active, need_weights=False
            )[0]
            weight = support_active[..., None].to(tokens.dtype)
            tokens = (pooled + attended) * weight
            tokens = (tokens + self.feed_forward(tokens)) * weight
            contextual[active] = self.output_norm(tokens) * weight
        return contextual, support
