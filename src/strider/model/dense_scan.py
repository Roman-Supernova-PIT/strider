"""Whole-spectrum matching across candidate redshifts."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn


class DenseRestFrameEvidence(nn.Module):
    """Match observer-frame tokens to learned class templates at every redshift."""

    def __init__(
        self,
        observed_wavelength: np.ndarray,
        rest_wavelength: np.ndarray,
        redshift_grid: np.ndarray,
        hidden_dim: int,
        token_dim: int,
        class_count: int,
        patch_size: int,
        rest_bins: int,
        initial_scale: float,
        evidence_scale: float,
        redshift_chunk_size: int,
        minimum_overlap: float,
        overlap_exponent: float,
    ) -> None:
        super().__init__()
        if len(observed_wavelength) % patch_size:
            raise ValueError("dense_scan_patch_size must divide the wavelength bins")
        if token_dim < 1 or rest_bins < 2:
            raise ValueError("Dense scan token and rest-grid sizes must be positive")
        if redshift_chunk_size < 1:
            raise ValueError("dense_scan_chunk_size must be positive")
        if not 0.0 < minimum_overlap <= 1.0:
            raise ValueError("dense_scan_minimum_overlap must lie in (0, 1]")
        if overlap_exponent < 0.0:
            raise ValueError("dense_scan_overlap_exponent cannot be negative")

        observed = np.asarray(observed_wavelength, dtype=np.float64)
        observed = np.exp(np.log(observed).reshape(-1, patch_size).mean(axis=1))
        rest_input = np.asarray(rest_wavelength, dtype=np.float64)
        rest = np.geomspace(rest_input[0], rest_input[-1], rest_bins)
        target = observed[None, :] / (1.0 + redshift_grid[:, None])
        upper = np.searchsorted(rest, target, side="left")
        valid = (upper > 0) & (upper < len(rest))
        upper = np.clip(upper, 1, len(rest) - 1)
        lower = upper - 1
        log_rest = np.log(rest)
        log_target = np.log(target)
        denominator = log_rest[upper] - log_rest[lower]
        weight = np.divide(
            log_target - log_rest[lower],
            denominator,
            out=np.zeros_like(target),
            where=denominator > 0,
        )
        self.register_buffer("lower_index", torch.from_numpy(lower.astype(np.int64)))
        self.register_buffer("upper_index", torch.from_numpy(upper.astype(np.int64)))
        self.register_buffer("upper_weight", torch.from_numpy(weight.astype(np.float32)))
        self.register_buffer(
            "template_valid",
            torch.from_numpy(valid.astype(np.float32)),
        )

        self.projection = nn.Linear(hidden_dim, token_dim, bias=False)
        self.templates = nn.Parameter(torch.empty(class_count, len(rest), token_dim))
        nn.init.normal_(self.templates, std=token_dim**-0.5)
        self.scale = nn.Parameter(torch.tensor(float(initial_scale)))
        self.evidence_scale = float(evidence_scale)
        self.redshift_chunk_size = int(redshift_chunk_size)
        self.minimum_overlap = float(minimum_overlap)
        self.overlap_exponent = float(overlap_exponent)
        self.patch_size = int(patch_size)

    def forward(
        self,
        observer_tokens: torch.Tensor,
        token_mask: torch.Tensor,
        visit_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        raw = self.raw_evidence(observer_tokens, token_mask, visit_mask)
        return {
            "dense_scan_joint_logits": torch.tanh(self.scale)
            * raw["raw_dense_scan_joint_logits"],
            **raw,
        }

    def raw_evidence(
        self,
        observer_tokens: torch.Tensor,
        token_mask: torch.Tensor,
        visit_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if observer_tokens.ndim != 4:
            raise ValueError(
                "Observer tokens must have shape "
                "(objects, visits, wavelength, features)"
            )
        if token_mask.shape != observer_tokens.shape[:-1]:
            raise ValueError("Dense scan token mask does not match the observer tokens")
        if visit_mask.shape != observer_tokens.shape[:2]:
            raise ValueError("Dense scan visit mask does not match the observer tokens")

        tokens, measured = self._reduce_tokens(observer_tokens, token_mask)
        tokens = functional.normalize(tokens, dim=-1)
        templates = functional.normalize(self.templates, dim=-1)
        raw_chunks = []
        overlap_chunks = []
        support_chunks = []
        redshift_count = self.lower_index.shape[0]

        for start in range(0, redshift_count, self.redshift_chunk_size):
            stop = min(start + self.redshift_chunk_size, redshift_count)
            shifted, template_support = self._shift_templates(
                templates,
                start,
                stop,
            )
            common_count = torch.einsum(
                "bvl,kl->bvk",
                measured,
                template_support,
            )
            numerator = torch.einsum(
                "bvld,kcld,bvl,kl->bvkc",
                tokens,
                shifted,
                measured,
                template_support,
            )
            similarity = numerator / common_count[..., None].clamp_min(1.0)
            overlap = common_count / float(tokens.shape[2])
            usable = (overlap >= self.minimum_overlap) & visit_mask[:, :, None].bool()
            score = similarity * overlap[..., None].pow(self.overlap_exponent)
            raw = (score * usable[..., None]).sum(dim=1)
            raw = raw / usable.sum(dim=1)[..., None].clamp_min(1.0)
            raw_chunks.append(raw.permute(0, 2, 1).contiguous())
            support_chunks.append(
                usable.any(dim=1)[:, None, :].expand(-1, self.templates.shape[0], -1)
            )
            overlap_chunks.append(
                (overlap * visit_mask[:, :, None]).sum(dim=1)
                / visit_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            )

        raw_joint = self.evidence_scale * torch.cat(raw_chunks, dim=-1)
        overlap_fraction = torch.cat(overlap_chunks, dim=-1)
        return {
            "raw_dense_scan_joint_logits": raw_joint,
            "dense_scan_overlap_fraction": overlap_fraction.detach(),
            "dense_scan_support": torch.cat(support_chunks, dim=-1),
        }

    def _reduce_tokens(
        self,
        observer_tokens: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.projection(observer_tokens)
        shape = projected.shape
        values = projected.reshape(
            shape[0],
            shape[1],
            shape[2] // self.patch_size,
            self.patch_size,
            shape[3],
        )
        support = token_mask.reshape(
            shape[0],
            shape[1],
            shape[2] // self.patch_size,
            self.patch_size,
        ).to(values.dtype)
        count = support.sum(dim=3)
        pooled = (values * support[..., None]).sum(dim=3)
        pooled = pooled / count[..., None].clamp_min(1.0)
        measured = count >= 0.5 * self.patch_size
        return pooled * measured[..., None], measured.to(pooled.dtype)

    def _shift_templates(
        self,
        templates: torch.Tensor,
        start: int,
        stop: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lower = self.lower_index[start:stop]
        upper = self.upper_index[start:stop]
        weight = self.upper_weight[start:stop][None, :, :, None]
        shifted = templates[:, lower] * (1.0 - weight)
        shifted = shifted + templates[:, upper] * weight
        shifted = functional.normalize(shifted.permute(1, 0, 2, 3), dim=-1)
        return shifted, self.template_valid[start:stop].to(shifted.dtype)


class WholeDetailRestFrameEvidence(nn.Module):
    """Scan continuum detail alone or blend it with the broad spectrum.

    ``detail`` keeps the same parameters as ``blend`` so an existing checkpoint
    can seed the cheaper one-pass ablation without changing parameter names.
    """

    def __init__(
        self,
        observed_wavelength: np.ndarray,
        rest_wavelength: np.ndarray,
        redshift_grid: np.ndarray,
        hidden_dim: int,
        token_dim: int,
        class_count: int,
        patch_size: int,
        rest_bins: int,
        initial_scale: float,
        evidence_scale: float,
        redshift_chunk_size: int,
        minimum_overlap: float,
        overlap_exponent: float,
        initial_detail_weight: float = 0.5,
        minimum_whole_weight: float = 0.0,
        scan_view: str = "blend",
    ) -> None:
        super().__init__()
        if scan_view not in {"detail", "blend"}:
            raise ValueError("Continuum scan view must be 'detail' or 'blend'")
        if not 0.0 <= minimum_whole_weight < 1.0:
            raise ValueError("Minimum whole-spectrum weight must lie in [0, 1)")
        maximum_detail_weight = 1.0 - minimum_whole_weight
        if not 0.0 < initial_detail_weight < maximum_detail_weight:
            raise ValueError(
                "Initial continuum-detail weight must lie between zero and "
                "one minus the minimum whole-spectrum weight"
            )
        arguments = {
            "observed_wavelength": observed_wavelength,
            "rest_wavelength": rest_wavelength,
            "redshift_grid": redshift_grid,
            "hidden_dim": hidden_dim,
            "token_dim": token_dim,
            "class_count": class_count,
            "patch_size": patch_size,
            "rest_bins": rest_bins,
            "initial_scale": 0.0,
            "evidence_scale": evidence_scale,
            "redshift_chunk_size": redshift_chunk_size,
            "minimum_overlap": minimum_overlap,
            "overlap_exponent": overlap_exponent,
        }
        self.matcher = DenseRestFrameEvidence(**arguments)
        self.matcher.scale.requires_grad_(False)
        self.scale = nn.Parameter(torch.tensor(float(initial_scale)))
        initial_fraction = initial_detail_weight / maximum_detail_weight
        initial_logit = np.log(initial_fraction / (1.0 - initial_fraction))
        self.detail_intercept = nn.Parameter(torch.tensor(float(initial_logit)))
        self.detail_redshift_slope = nn.Parameter(torch.tensor(0.0))
        self.maximum_detail_weight = float(maximum_detail_weight)
        self.scan_view = scan_view
        redshift = torch.from_numpy(np.asarray(redshift_grid, dtype=np.float32))
        redshift_coordinate = (redshift - 1.0) / max(float(redshift.max()), 1.0)
        self.register_buffer("redshift_coordinate", redshift_coordinate)

    def forward(
        self,
        observer_tokens: torch.Tensor,
        token_mask: torch.Tensor,
        detail_tokens: torch.Tensor,
        detail_mask: torch.Tensor,
        visit_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        detail = self.matcher.raw_evidence(detail_tokens, detail_mask, visit_mask)
        detail_raw = detail["raw_dense_scan_joint_logits"]
        scale = torch.tanh(self.scale)
        detail_logits = scale * detail_raw
        if self.scan_view == "detail":
            detail_weight = torch.ones_like(self.redshift_coordinate)
            return {
                "dense_scan_joint_logits": detail_logits,
                "raw_dense_scan_joint_logits": detail_raw,
                "dense_detail_joint_logits": detail_logits,
                "dense_detail_contribution": detail_logits,
                "dense_detail_weight": detail_weight,
                "dense_scan_overlap_fraction": detail[
                    "dense_scan_overlap_fraction"
                ],
                "dense_scan_support": detail["dense_scan_support"],
            }

        whole = self.matcher.raw_evidence(observer_tokens, token_mask, visit_mask)
        detail_weight = self.maximum_detail_weight * torch.sigmoid(
            self.detail_intercept
            + self.detail_redshift_slope * self.redshift_coordinate
        )
        whole_weight = 1.0 - detail_weight
        whole_raw = whole["raw_dense_scan_joint_logits"]
        combined_raw = (
            whole_raw * whole_weight[None, None, :]
            + detail_raw * detail_weight[None, None, :]
        )
        whole_logits = scale * whole_raw
        whole_contribution = whole_logits * whole_weight[None, None, :]
        detail_contribution = detail_logits * detail_weight[None, None, :]
        return {
            "dense_scan_joint_logits": scale * combined_raw,
            "raw_dense_scan_joint_logits": combined_raw,
            "dense_whole_joint_logits": whole_logits,
            "dense_detail_joint_logits": detail_logits,
            "dense_whole_contribution": whole_contribution,
            "dense_detail_contribution": detail_contribution,
            "dense_detail_weight": detail_weight.detach(),
            "dense_scan_overlap_fraction": whole[
                "dense_scan_overlap_fraction"
            ],
            "dense_scan_support": whole["dense_scan_support"]
            & detail["dense_scan_support"],
        }
