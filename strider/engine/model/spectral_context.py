"""Class context from the complete observed-frame spectral series.

Inputs are background-scaled flux and wavelength masks with shape ``(B,V,L)``.
The branch alternates attention across wavelength and visits, then returns class
logits ``(B,C)``. It never receives dates, phase, redshift labels or per-bin
uncertainties. ONIR remains the redshift anchor.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from .spectral_tokens import normalize_valid_bins


class FullSpectrumContext(nn.Module):
    """Encode broad spectral shape without assigning redshift on its own."""

    def __init__(
        self,
        wavelength_bins: int,
        token_dim: int,
        class_count: int,
        patch_size: int,
        attention_heads: int,
        attention_layers: int,
        dropout: float,
        initial_scale: float,
        input_normalization: str,
        use_visit_attention: bool = False,
        minimum_support: float = 0.5,
    ) -> None:
        super().__init__()
        if wavelength_bins % patch_size:
            raise ValueError("context_patch_size must divide wavelength_bins")
        if token_dim % attention_heads:
            raise ValueError("context_attention_heads must divide the token dimension")
        if attention_layers < 1:
            raise ValueError("context_attention_layers must be positive")
        if input_normalization not in {"none", "per_visit"}:
            raise ValueError("Context input_normalization must be 'none' or 'per_visit'")
        self.input_normalization = input_normalization
        self.use_visit_attention = bool(use_visit_attention)
        self.stem = MaskAwarePatchStem(token_dim, patch_size, minimum_support)
        patch_count = wavelength_bins // patch_size
        self.position = nn.Parameter(torch.zeros(1, patch_count, token_dim))
        nn.init.normal_(self.position, std=0.02)
        self.blocks = nn.ModuleList(
            [
                SpectralAttentionBlock(token_dim, attention_heads, dropout)
                for _ in range(attention_layers)
            ]
        )
        self.visit_blocks = nn.ModuleList(
            [
                VisitAttentionBlock(token_dim, attention_heads, dropout)
                for _ in range(attention_layers)
            ]
            if self.use_visit_attention
            else []
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, class_count),
        )
        self.scale = nn.Parameter(torch.tensor(float(initial_scale)))

    def forward(
        self,
        flux: torch.Tensor,
        wavelength_mask: torch.Tensor,
        visit_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mask = wavelength_mask * visit_mask[:, :, None]  # (B,V,L)
        encoder_flux = (
            normalize_valid_bins(flux, mask)
            if self.input_normalization == "per_visit"
            else flux
        )  # (B,V,L)
        tokens, patch_support = self.stem(encoder_flux, mask)  # (B,V,P,D), (B,V,P)

        batch, visits, patches, token_dim = tokens.shape
        encoded = tokens + self.position[:, None]  # (B,V,P,D)
        encoded = encoded * patch_support[..., None]
        flat_support = patch_support.reshape(batch * visits, patches).bool()
        active_visit = flat_support.any(dim=1)
        for layer, block in enumerate(self.blocks):
            flat = encoded.reshape(batch * visits, patches, token_dim)
            updated = torch.zeros_like(flat)
            if active_visit.any():
                updated[active_visit] = block(
                    flat[active_visit], flat_support[active_visit]
                )
            encoded = updated.reshape(batch, visits, patches, token_dim)
            if self.use_visit_attention:
                encoded = self.visit_blocks[layer](encoded, patch_support)
        weight = patch_support[..., None]
        visit_features = (encoded * weight).sum(dim=2) / weight.sum(dim=2).clamp_min(1.0)
        valid_visit = (patch_support.sum(dim=2) > 0) & visit_mask.bool()
        visit_weight = valid_visit[..., None].to(visit_features.dtype)
        object_features = (visit_features * visit_weight).sum(dim=1)
        object_features = object_features / visit_weight.sum(dim=1).clamp_min(1.0)
        object_valid = valid_visit.any(dim=1).to(object_features.dtype)
        class_logits = self.classifier(object_features) * object_valid[:, None]
        return {
            "context_class_logits": class_logits,
            "scaled_context_class_logits": torch.tanh(self.scale) * class_logits,
        }


class MaskAwarePatchStem(nn.Module):
    """Reduce the wavelength grid while keeping missing bins out of the CNN."""

    def __init__(
        self,
        token_dim: int,
        patch_size: int,
        minimum_support: float,
    ) -> None:
        super().__init__()
        if not 0.0 < minimum_support <= 1.0:
            raise ValueError("context_minimum_support must lie in (0, 1]")
        broad_kernel = 2 * patch_size + 1
        self.patch = _MaskedConvolution(
            1,
            token_dim,
            kernel_size=broad_kernel,
            stride=patch_size,
            padding=patch_size,
            minimum_support=minimum_support,
        )
        self.local = _MaskedConvolution(
            token_dim,
            token_dim,
            kernel_size=5,
            stride=1,
            padding=2,
            minimum_support=minimum_support,
        )
        self.norm = nn.LayerNorm(token_dim)

    def forward(
        self, flux: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        leading = flux.shape[:-1]
        values = flux.reshape(-1, 1, flux.shape[-1])
        support = mask.reshape(-1, 1, mask.shape[-1]).to(values.dtype)
        values, support = self.patch(values, support)
        residual = values
        values, support = self.local(values, support)
        values = functional.gelu(values + residual * support)
        tokens = self.norm(values.transpose(1, 2)) * support.transpose(1, 2)
        return (
            tokens.reshape(*leading, tokens.shape[-2], tokens.shape[-1]),
            support.reshape(*leading, support.shape[-1]),
        )


class _MaskedConvolution(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        minimum_support: float,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.minimum_support = float(minimum_support)
        self.register_buffer("support_kernel", torch.ones(1, 1, kernel_size))

    def forward(
        self, values: torch.Tensor, support: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        convolved = self.conv(values * support)
        measured = functional.conv1d(
            support,
            self.support_kernel.to(support.dtype),
            stride=self.stride,
            padding=self.padding,
        )
        fraction = measured / float(self.kernel_size)
        valid = (fraction >= self.minimum_support).to(convolved.dtype)
        if self.conv.bias is not None:
            bias = self.conv.bias[None, :, None]
            convolved = (convolved - bias) / fraction.clamp_min(1.0 / self.kernel_size)
            convolved = convolved + bias
        return convolved * valid, valid


class SpectralAttentionBlock(nn.Module):
    def __init__(self, token_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(token_dim)
        self.attention = nn.MultiheadAttention(
            token_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, 4 * token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * token_dim, token_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        valid = support[..., None].to(tokens.dtype)
        normalized = self.attention_norm(tokens)
        attended = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~support,
            need_weights=False,
        )[0]
        tokens = (tokens + attended) * valid
        return (tokens + self.feed_forward(tokens)) * valid


class VisitAttentionBlock(nn.Module):
    """Share information across visits at each wavelength patch."""

    def __init__(self, token_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(token_dim)
        self.attention = nn.MultiheadAttention(
            token_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, 4 * token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * token_dim, token_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        batch, visits, patches, token_dim = tokens.shape
        sequences = tokens.permute(0, 2, 1, 3).reshape(
            batch * patches, visits, token_dim
        )  # (B*P,V,D)
        valid = support.permute(0, 2, 1).reshape(batch * patches, visits).bool()
        active = valid.any(dim=1)
        result = torch.zeros_like(sequences)
        if active.any():
            values = sequences[active]
            measured = valid[active]
            normalized = self.attention_norm(values)
            attended = self.attention(
                normalized,
                normalized,
                normalized,
                key_padding_mask=~measured,
                need_weights=False,
            )[0]
            weight = measured[..., None].to(values.dtype)
            values = (values + attended) * weight
            result[active] = (values + self.feed_forward(values)) * weight
        return result.reshape(batch, patches, visits, token_dim).permute(0, 2, 1, 3)
