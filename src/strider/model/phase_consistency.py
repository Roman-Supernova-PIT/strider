"""Candidate-redshift consistency between spectral phase estimates and dates."""

from __future__ import annotations

import math

import torch
from torch import nn


class CandidatePhaseConsistency(nn.Module):
    """Compare spectral phase estimates with dates at every trial redshift.

    Without an external peak date, the unknown phase of the first visit is
    marginalized over the phase grid.  When a measured peak-date offset is
    supplied, the module instead evaluates the corresponding phase trajectory
    and marginalizes a Gaussian uncertainty on that measurement.  Simulation
    peak dates are never inputs to this module.
    """

    def __init__(
        self,
        initial_scale: float = 0.0,
        minimum_visits: int = 2,
        use_peak_date: bool = False,
        peak_uncertainty_days: float = 0.0,
        peak_quadrature_points: int = 1,
        peak_outlier_fraction: float = 0.0,
        minimum_coverage_fraction: float = 0.5,
    ) -> None:
        super().__init__()
        if minimum_visits < 2:
            raise ValueError("Candidate phase consistency requires at least two visits")
        if peak_uncertainty_days < 0.0:
            raise ValueError("Peak-date uncertainty cannot be negative")
        if peak_quadrature_points < 1 or peak_quadrature_points % 2 == 0:
            raise ValueError("Peak-date quadrature requires a positive odd point count")
        if not 0.0 <= peak_outlier_fraction < 1.0:
            raise ValueError("Peak-date outlier fraction must lie in [0, 1)")
        if not 0.0 < minimum_coverage_fraction <= 1.0:
            raise ValueError("Minimum phase coverage must lie in (0, 1]")
        self.scale = nn.Parameter(torch.tensor(float(initial_scale)))
        self.minimum_visits = int(minimum_visits)
        self.use_peak_date = bool(use_peak_date)
        self.peak_uncertainty_days = float(peak_uncertainty_days)
        self.peak_quadrature_points = int(peak_quadrature_points)
        self.peak_outlier_fraction = float(peak_outlier_fraction)
        self.minimum_coverage_fraction = float(minimum_coverage_fraction)

    def forward(
        self,
        phase_logits: torch.Tensor,
        observer_days: torch.Tensor,
        visit_mask: torch.Tensor,
        redshift_grid: torch.Tensor,
        phase_grid: torch.Tensor,
        peak_day_offset: torch.Tensor | None = None,
        peak_date_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return scaled and raw class-redshift logits with shape ``(B,C,Z)``."""
        batch, visits, classes, redshifts, phase_bins = phase_logits.shape
        if observer_days.shape != (batch, visits):
            raise ValueError("Observer days must have shape (objects, visits)")
        if visit_mask.shape != (batch, visits):
            raise ValueError("Visit mask must have shape (objects, visits)")
        if len(redshift_grid) != redshifts or len(phase_grid) != phase_bins:
            raise ValueError("Candidate phase grids do not match the phase logits")
        if phase_bins < 2:
            raise ValueError("Candidate phase consistency requires at least two phase bins")
        phase_step = phase_grid[1:] - phase_grid[:-1]
        if not torch.allclose(phase_step, phase_step[0].expand_as(phase_step)):
            raise ValueError("Candidate phase consistency requires a uniform phase grid")

        if self.use_peak_date:
            raw = self._measured_peak_score(
                phase_logits,
                observer_days,
                visit_mask,
                redshift_grid,
                phase_grid,
                phase_step[0],
                peak_day_offset,
                peak_date_valid,
            )
            return torch.tanh(self.scale) * raw, raw

        valid_visit = visit_mask.bool()
        first_day = observer_days.masked_fill(~valid_visit, torch.inf).amin(dim=1)
        first_day = torch.where(valid_visit.any(dim=1), first_day, torch.zeros_like(first_day))
        relative_days = observer_days - first_day[:, None]
        rest_days = relative_days[:, :, None] / (
            1.0 + redshift_grid[None, None, :]
        )
        initial_phase = phase_grid[None, None, None, :]
        target_phase = rest_days[:, :, :, None] + initial_phase
        inside = (target_phase >= phase_grid[0]) & (target_phase <= phase_grid[-1])

        position = (target_phase - phase_grid[0]) / phase_step[0]
        lower = position.floor().long().clamp(0, phase_bins - 1)
        upper = (lower + 1).clamp(max=phase_bins - 1)
        weight = (position - lower.to(position.dtype)).clamp(0.0, 1.0)

        probability = torch.softmax(phase_logits, dim=-1)
        gather_shape = (batch, visits, classes, redshifts, phase_bins)
        lower_index = lower[:, :, None].expand(gather_shape)
        upper_index = upper[:, :, None].expand(gather_shape)
        lower_probability = torch.gather(probability, -1, lower_index)
        upper_probability = torch.gather(probability, -1, upper_index)
        interpolated = lower_probability * (1.0 - weight[:, :, None])
        interpolated = interpolated + upper_probability * weight[:, :, None]
        relative_log_probability = torch.log(
            interpolated.clamp_min(torch.finfo(interpolated.dtype).tiny)
        ) + math.log(phase_bins)

        valid = inside & valid_visit[:, :, None, None]
        valid = valid[:, :, None]
        count = valid.sum(dim=1)
        score = (
            relative_log_probability * valid.to(relative_log_probability.dtype)
        ).sum(dim=1) / count.clamp_min(1).to(relative_log_probability.dtype)
        visit_count = visit_mask.sum(dim=1).clamp_min(1.0)
        coverage = count.to(score.dtype) / visit_count[:, None, None, None]
        score = score * coverage

        usable = count >= self.minimum_visits
        score = score.masked_fill(~usable, torch.finfo(score.dtype).min)
        usable_count = usable.sum(dim=-1)
        raw = torch.logsumexp(score, dim=-1) - torch.log(
            usable_count.clamp_min(1).to(score.dtype)
        )
        raw = torch.where(usable_count > 0, raw, torch.zeros_like(raw))
        return torch.tanh(self.scale) * raw, raw

    def _measured_peak_score(
        self,
        phase_logits: torch.Tensor,
        observer_days: torch.Tensor,
        visit_mask: torch.Tensor,
        redshift_grid: torch.Tensor,
        phase_grid: torch.Tensor,
        phase_step: torch.Tensor,
        peak_day_offset: torch.Tensor | None,
        peak_date_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        """Score a measured observer-frame peak date without exposing it to a net."""
        batch = phase_logits.shape[0]
        if peak_day_offset is None or peak_date_valid is None:
            raise ValueError("Peak-informed phase consistency requires a peak date and validity")
        if peak_day_offset.shape != (batch,) or peak_date_valid.shape != (batch,):
            raise ValueError("Peak-date inputs must have shape (objects,)")

        nodes, log_weights = self._peak_quadrature(phase_logits)
        phase_probability = torch.softmax(phase_logits, dim=-1)
        raw_by_offset = []
        for node in nodes:
            candidate_peak = peak_day_offset + node * self.peak_uncertainty_days
            target_phase = (
                observer_days[:, :, None] - candidate_peak[:, None, None]
            ) / (1.0 + redshift_grid[None, None, :])
            raw_by_offset.append(
                self._trajectory_score(
                    phase_probability,
                    target_phase,
                    visit_mask,
                    phase_grid,
                    phase_step,
                )
            )
        stacked = torch.stack(raw_by_offset, dim=-1)
        raw = torch.logsumexp(stacked + log_weights, dim=-1)

        if self.peak_outlier_fraction > 0.0:
            outlier = phase_logits.new_tensor(self.peak_outlier_fraction)
            raw = torch.logaddexp(
                torch.log1p(-outlier) + raw,
                torch.log(outlier).expand_as(raw),
            )
        valid_object = peak_date_valid.bool()
        return torch.where(valid_object[:, None, None], raw, torch.zeros_like(raw))

    def _trajectory_score(
        self,
        phase_probability: torch.Tensor,
        target_phase: torch.Tensor,
        visit_mask: torch.Tensor,
        phase_grid: torch.Tensor,
        phase_step: torch.Tensor,
    ) -> torch.Tensor:
        """Return evidence relative to a uniform phase prediction."""
        phase_bins = phase_probability.shape[-1]
        inside = (target_phase >= phase_grid[0]) & (target_phase <= phase_grid[-1])
        position = (target_phase - phase_grid[0]) / phase_step
        lower = position.floor().long().clamp(0, phase_bins - 1)
        upper = (lower + 1).clamp(max=phase_bins - 1)
        weight = (position - lower.to(position.dtype)).clamp(0.0, 1.0)

        index_shape = (
            *lower.shape[:2],
            phase_probability.shape[2],
            lower.shape[2],
            1,
        )
        lower_index = lower[:, :, None, :, None].expand(index_shape)
        upper_index = upper[:, :, None, :, None].expand(index_shape)
        lower_probability = torch.gather(
            phase_probability, -1, lower_index
        ).squeeze(-1)
        upper_probability = torch.gather(
            phase_probability, -1, upper_index
        ).squeeze(-1)
        interpolated = lower_probability * (1.0 - weight[:, :, None])
        interpolated = interpolated + upper_probability * weight[:, :, None]
        relative_log_probability = torch.log(
            interpolated.clamp_min(torch.finfo(interpolated.dtype).tiny)
        ) + math.log(phase_bins)

        valid = inside & visit_mask.bool()[:, :, None]
        count = valid.sum(dim=1)
        score = (
            relative_log_probability * valid[:, :, None].to(relative_log_probability.dtype)
        ).sum(dim=1) / count[:, None].clamp_min(1).to(relative_log_probability.dtype)
        visit_count = visit_mask.sum(dim=1).clamp_min(1.0)
        coverage = count.to(score.dtype) / visit_count[:, None]
        usable = (count >= self.minimum_visits) & (
            coverage >= self.minimum_coverage_fraction
        )
        return torch.where(usable[:, None], score, torch.zeros_like(score))

    def _peak_quadrature(
        self, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.peak_quadrature_points == 1 or self.peak_uncertainty_days == 0.0:
            return reference.new_zeros(1), reference.new_zeros(1)
        nodes = torch.linspace(
            -2.0,
            2.0,
            self.peak_quadrature_points,
            device=reference.device,
            dtype=reference.dtype,
        )
        log_weights = -0.5 * nodes.square()
        log_weights = log_weights - torch.logsumexp(log_weights, dim=0)
        return nodes, log_weights
