"""Attention over visits and named spectral regions.

Inputs have shape ``(B, V, Z, F, D)`` for objects, visits, trial redshifts,
ONIR regions and learned features. Observer dates enter only through visit
differences divided by ``1 + z``. They never score a class on their own.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class FactoredOnirEvidence(nn.Module):
    """Add full-shape and spectral-evolution evidence to the ONIR match."""

    def __init__(
        self,
        hidden_dim: int,
        class_count: int,
        attention_heads: int,
        dropout: float,
        shape_initial_scale: float = 0.5,
        temporal_initial_scale: float = 0.0,
        relative_brightness: bool = False,
        brightness_initial_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.shape_pool = ClassQueryAttention(
            hidden_dim, class_count, attention_heads, dropout
        )
        self.time_pool = AxisAttentionPool(hidden_dim, attention_heads, dropout)
        self.temporal_pool = ClassQueryAttention(
            hidden_dim, class_count, attention_heads, dropout
        )
        self.time_modulation = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.brightness_projection = (
            nn.Linear(1, hidden_dim, bias=False) if relative_brightness else None
        )
        self.shape_scale = nn.Parameter(torch.tensor(float(shape_initial_scale)))
        self.temporal_scale = nn.Parameter(torch.tensor(float(temporal_initial_scale)))
        self.brightness_scale = (
            nn.Parameter(torch.tensor(float(brightness_initial_scale)))
            if relative_brightness
            else None
        )
        self.class_count = int(class_count)

    def forward(
        self,
        features: torch.Tensor,
        feature_support: torch.Tensor,
        observer_days: torch.Tensor,
        visit_mask: torch.Tensor,
        redshift_grid: torch.Tensor,
        relative_brightness: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return spectral and temporal logits with shape ``(B, C, Z)``."""
        batch, visits, redshift_count, feature_count, hidden_dim = features.shape
        expected_support = (batch, visits, redshift_count, feature_count)
        if feature_support.shape != expected_support:
            raise ValueError(
                "Factored ONIR support has the wrong shape: "
                f"expected {expected_support}, got {tuple(feature_support.shape)}"
            )
        if observer_days.shape != (batch, visits):
            raise ValueError("Observer days must have shape (objects, visits)")
        if visit_mask.shape != (batch, visits):
            raise ValueError("Visit mask must have shape (objects, visits)")
        if len(redshift_grid) != redshift_count:
            raise ValueError("Redshift grid does not match the ONIR representation")
        if self.brightness_projection is not None:
            if relative_brightness is None:
                raise ValueError("Relative brightness is enabled but was not supplied")
            if relative_brightness.shape != (batch, visits):
                raise ValueError("Relative brightness must have shape (objects, visits)")

        valid = feature_support.bool() & visit_mask[:, :, None, None].bool()
        per_visit_shape, shape_attention = self.shape_pool(features, valid)
        visit_valid = valid.any(dim=-1)
        raw_shape = _masked_mean(per_visit_shape, visit_valid[..., None], dim=1)
        raw_shape = raw_shape.permute(0, 2, 1).contiguous()  # (B,C,Z)
        shape_logits = torch.tanh(self.shape_scale) * raw_shape
        mean_shape_attention = _masked_mean(
            shape_attention,
            visit_valid[..., None, None],
            dim=1,
        ).permute(0, 2, 1, 3).contiguous()  # (B,C,Z,F)

        if visits < 2:
            raw_temporal = features.new_zeros((batch, self.class_count, redshift_count))
            temporal_attention = features.new_zeros(
                (batch, self.class_count, redshift_count, feature_count)
            )
        else:
            raw_temporal, temporal_attention = self._temporal_evidence(
                features,
                valid,
                observer_days,
                redshift_grid,
                relative_brightness,
            )
        temporal_logits = torch.tanh(self.temporal_scale) * raw_temporal
        return {
            "shape_joint_logits": shape_logits,
            "raw_shape_joint_logits": raw_shape,
            "temporal_joint_logits": temporal_logits,
            "raw_temporal_joint_logits": raw_temporal,
            "shape_feature_attention": mean_shape_attention.detach(),
            "temporal_feature_attention": temporal_attention.detach(),
        }

    def _temporal_evidence(
        self,
        features: torch.Tensor,
        valid: torch.Tensor,
        observer_days: torch.Tensor,
        redshift_grid: torch.Tensor,
        relative_brightness: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        change = features[:, 1:] - features[:, :-1]  # (B,V-1,Z,F,D)
        pair_valid = valid[:, 1:] & valid[:, :-1]  # (B,V-1,Z,F)
        observed_interval = observer_days[:, 1:] - observer_days[:, :-1]
        rest_interval = observed_interval[:, :, None] / (
            1.0 + redshift_grid[None, None, :]
        )
        interval_scale = torch.log1p(rest_interval.abs()) / torch.log(
            features.new_tensor(101.0)
        )
        time_features = torch.stack(
            [
                interval_scale,
                torch.sign(rest_interval),
                torch.tanh(rest_interval / 30.0),
            ],
            dim=-1,
        )
        modulation = self.time_modulation(time_features)  # (B,V-1,Z,D)
        pair_change = change
        if self.brightness_projection is not None:
            if relative_brightness is None or self.brightness_scale is None:
                raise RuntimeError("Relative-brightness components are incomplete")
            brightness_change = relative_brightness[:, 1:] - relative_brightness[:, :-1]
            brightness = self.brightness_projection(brightness_change[..., None])
            brightness = torch.tanh(self.brightness_scale) * brightness
            pair_change = pair_change + brightness[:, :, None, None, :]
        pair_features = _bounded_change(
            pair_change * modulation[:, :, :, None, :]
        )

        # Pool visits for each named region before asking which regions support
        # each class. This is the temporal-then-spectral factorization.
        pair_features = pair_features.permute(0, 2, 3, 1, 4)
        pair_valid = pair_valid.permute(0, 2, 3, 1)
        evolved = self.time_pool(pair_features, pair_valid)  # (B,Z,F,D)
        evolved_valid = pair_valid.any(dim=-1)  # (B,Z,F)
        temporal, attention = self.temporal_pool(evolved, evolved_valid)
        return (
            temporal.permute(0, 2, 1).contiguous(),
            attention.permute(0, 2, 1, 3).contiguous(),
        )


class ClassQueryAttention(nn.Module):
    """Use one multi-head query per class to combine named regions."""

    def __init__(
        self,
        hidden_dim: int,
        class_count: int,
        attention_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        _require_head_divisibility(hidden_dim, attention_heads)
        self.heads = int(attention_heads)
        self.head_dim = int(hidden_dim) // self.heads
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query = nn.Parameter(
            torch.empty(class_count, self.heads, self.head_dim)
        )
        self.readout = nn.Parameter(
            torch.empty(class_count, self.heads, self.head_dim)
        )
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.query, std=self.head_dim**-0.5)
        nn.init.normal_(self.readout, std=hidden_dim**-0.5)

    def forward(
        self,
        features: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool the penultimate feature axis and return class scores."""
        keys = self.key(features).unflatten(-1, (self.heads, self.head_dim))
        values = self.value(features).unflatten(-1, (self.heads, self.head_dim))
        scores = torch.einsum("...fhd,chd->...chf", keys, self.query)
        scores = scores / math.sqrt(self.head_dim)
        weights = _masked_softmax(scores, valid[..., None, None, :], dim=-1)
        pooled = torch.einsum("...chf,...fhd->...chd", weights, values)
        pooled = self.dropout(pooled)
        logits = torch.einsum("...chd,chd->...c", pooled, self.readout)
        return logits, weights.mean(dim=-2)


class AxisAttentionPool(nn.Module):
    """Multi-head attention pooling over one sequence axis."""

    def __init__(self, hidden_dim: int, attention_heads: int, dropout: float) -> None:
        super().__init__()
        _require_head_divisibility(hidden_dim, attention_heads)
        self.hidden_dim = int(hidden_dim)
        self.heads = int(attention_heads)
        self.head_dim = self.hidden_dim // self.heads
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query = nn.Parameter(torch.empty(self.heads, self.head_dim))
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        # A learned offset would create temporal evidence when every visit is
        # identical. Keep zero spectral change exactly zero.
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        nn.init.normal_(self.query, std=self.head_dim**-0.5)

    def forward(self, sequence: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        keys = self.key(sequence).unflatten(-1, (self.heads, self.head_dim))
        values = self.value(sequence).unflatten(-1, (self.heads, self.head_dim))
        scores = torch.einsum("...ahd,hd->...ha", keys, self.query)
        scores = scores / math.sqrt(self.head_dim)
        weights = _masked_softmax(scores, valid[..., None, :], dim=-1)
        pooled = torch.einsum("...ha,...ahd->...hd", weights, values)
        pooled = self.dropout(pooled).flatten(-2)
        return self.norm(self.output(pooled)) * valid.any(dim=-1, keepdim=True)


def _masked_softmax(
    scores: torch.Tensor,
    valid: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    valid = valid.bool()
    masked = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    weights = torch.softmax(masked, dim=dim) * valid.to(scores.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1.0e-8)


def _masked_mean(
    values: torch.Tensor,
    valid: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    weight = valid.to(values.dtype)
    return (values * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)


def _bounded_change(change: torch.Tensor) -> torch.Tensor:
    mean_square = change.square().mean(dim=-1, keepdim=True)
    return change / torch.sqrt(1.0 + mean_square)


def _require_head_divisibility(hidden_dim: int, attention_heads: int) -> None:
    if attention_heads <= 0 or hidden_dim % attention_heads:
        raise ValueError("Attention heads must divide the ONIR token dimension")
