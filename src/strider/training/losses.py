"""Losses for conditional class-redshift evidence and evidence sufficiency."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from strider.model.posterior import posterior_logits
from strider.model.coadd import final_inverse_variance_coadd
from strider.model.spectral_tokens import normalize_valid_bins


def joint_targets(
    class_index: torch.Tensor,
    redshift: torch.Tensor,
    redshift_grid: torch.Tensor,
) -> torch.Tensor:
    below = redshift < redshift_grid[0]
    above = redshift > redshift_grid[-1]
    if (below | above).any():
        invalid = redshift[below | above].detach().cpu().tolist()
        raise ValueError(
            "Training redshifts fall outside the candidate grid "
            f"[{float(redshift_grid[0]):.6g}, {float(redshift_grid[-1]):.6g}]: "
            f"{invalid[:8]}"
        )
    redshift_index = (redshift[:, None] - redshift_grid[None, :]).abs().argmin(dim=1)
    return class_index * redshift_grid.numel() + redshift_index


def training_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    redshift_grid: torch.Tensor,
    redshift_cell_width: torch.Tensor,
    redshift_prior: str,
    evidence_sufficiency_weight: float,
    no_source_redshift_weight: float = 0.0,
    no_source_class_weight: float = 0.0,
    profile_drift_weight: float = 0.0,
    class_weight: torch.Tensor | None = None,
    phase_loss_weight: float = 0.0,
    coadd_reconstruction_loss_weight: float = 0.0,
    alias_ranking_loss_weight: float = 0.0,
    alias_ranking_minimum_delta_z: float = 0.1,
    alias_ranking_margin: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    measured_source = batch["has_source"] > 0.5
    joint_source = measured_source.clone()
    unsupported_source = torch.zeros_like(measured_source)
    if measured_source.any():
        source_support = outputs.get("joint_support")
        if source_support is not None:
            source_classes = batch["class_index"][measured_source]
            source_redshift_index = (
                batch["redshift"][measured_source, None] - redshift_grid[None, :]
            ).abs().argmin(dim=1)
            target_supported = source_support[measured_source][
                torch.arange(source_classes.shape[0], device=source_classes.device),
                source_classes,
                source_redshift_index,
            ]
            if not target_supported.all():
                source_positions = torch.nonzero(
                    measured_source, as_tuple=False
                ).flatten()
                unsupported_source[source_positions[~target_supported]] = True
                joint_source = measured_source & ~unsupported_source
    if joint_source.any():
        source_logits = posterior_logits(
            outputs["joint_logits"][joint_source], redshift_cell_width, redshift_prior
        )
        logits = source_logits.reshape(joint_source.sum(), -1)
        targets = joint_targets(
            batch["class_index"][joint_source],
            batch["redshift"][joint_source],
            redshift_grid,
        )
        per_object_joint = functional.cross_entropy(logits, targets, reduction="none")
        if class_weight is None:
            joint = per_object_joint.mean()
        else:
            object_weight = class_weight[batch["class_index"][joint_source]]
            joint = (per_object_joint * object_weight).sum() / object_weight.sum().clamp_min(1e-12)
    else:
        joint = outputs["joint_logits"].sum() * 0.0
    if alias_ranking_loss_weight > 0.0:
        alias_ranking, alias_margin_success = _alias_ranking_loss(
            source_logits if joint_source.any() else None,
            batch["class_index"][joint_source],
            batch["redshift"][joint_source],
            redshift_grid,
            class_weight,
            float(alias_ranking_minimum_delta_z),
            float(alias_ranking_margin),
            outputs["joint_logits"],
        )
    else:
        alias_ranking = outputs["joint_logits"].sum() * 0.0
        alias_margin_success = 0.0
    broad_target = ~measured_source
    if broad_target.any():
        # With no source present, observing dates must not produce a preferred
        # redshift. Match the marginal to the declared prior measure.
        measured_logits = posterior_logits(
            outputs["joint_logits"][broad_target], redshift_cell_width, redshift_prior
        )
        redshift_logits = torch.logsumexp(measured_logits, dim=1)  # (B_no_source, Z)
        redshift_log_probability = functional.log_softmax(redshift_logits, dim=-1)
        if redshift_prior == "flat_z":
            base_target = redshift_cell_width / redshift_cell_width.sum()
        else:
            base_target = torch.full_like(
                redshift_cell_width, 1.0 / redshift_cell_width.numel()
            )
        joint_support = outputs.get("joint_support")
        if joint_support is None:
            redshift_support = torch.ones_like(redshift_logits, dtype=torch.bool)
        else:
            redshift_support = joint_support[broad_target].any(dim=1)
        target = base_target[None, :] * redshift_support.to(base_target.dtype)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        target_log_probability = torch.log(target.clamp_min(1e-12))
        no_source_redshift = (
            target * (target_log_probability - redshift_log_probability)
        ).sum(dim=-1).mean()
    else:
        no_source_redshift = outputs["joint_logits"].sum() * 0.0
    if broad_target.any():
        class_logits = torch.logsumexp(measured_logits, dim=2)
        class_log_probability = functional.log_softmax(class_logits, dim=-1)
        if joint_support is None:
            class_support = torch.ones_like(class_logits, dtype=torch.bool)
        else:
            class_support = joint_support[broad_target].any(dim=2)
        class_target = class_support.to(class_logits.dtype)
        class_target = class_target / class_target.sum(dim=-1, keepdim=True).clamp_min(1.0)
        no_source_class = (
            class_target
            * (
                torch.log(class_target.clamp_min(1e-12))
                - class_log_probability
            )
        ).sum(dim=-1).mean()
    else:
        no_source_class = outputs["joint_logits"].sum() * 0.0
    sufficiency_valid = ~unsupported_source
    if sufficiency_valid.any():
        sufficiency_target = batch["evidence_sufficiency_target"][sufficiency_valid]
        per_object_sufficiency = functional.binary_cross_entropy_with_logits(
            outputs["evidence_sufficiency_logit"][sufficiency_valid],
            sufficiency_target,
            reduction="none",
        )
        source_count = sufficiency_target.sum()
        no_source_count = (1.0 - sufficiency_target).sum()
        if source_count > 0 and no_source_count > 0:
            weights = torch.where(
                sufficiency_target > 0.5,
                0.5 / source_count,
                0.5 / no_source_count,
            )
            evidence_sufficiency = (per_object_sufficiency * weights).sum()
        else:
            evidence_sufficiency = per_object_sufficiency.mean()
    else:
        evidence_sufficiency = outputs["evidence_sufficiency_logit"].sum() * 0.0
    profile_drift = outputs.get("profile_drift")
    if profile_drift is None:
        profile_drift = outputs["joint_logits"].sum() * 0.0
    phase_loss, phase_error, phase_fraction = _phase_loss(
        outputs,
        batch,
        redshift_grid,
        float(phase_loss_weight),
    )
    coadd_reconstruction = _coadd_reconstruction_loss(
        outputs,
        batch,
        float(coadd_reconstruction_loss_weight),
    )
    total = (
        joint
        + float(evidence_sufficiency_weight) * evidence_sufficiency
        + float(no_source_redshift_weight) * no_source_redshift
        + float(no_source_class_weight) * no_source_class
        + float(profile_drift_weight) * profile_drift
        + float(phase_loss_weight) * phase_loss
        + float(coadd_reconstruction_loss_weight) * coadd_reconstruction
        + float(alias_ranking_loss_weight) * alias_ranking
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "joint_loss": float(joint.detach().cpu()),
        "evidence_sufficiency_loss": float(evidence_sufficiency.detach().cpu()),
        "no_source_redshift_loss": float(no_source_redshift.detach().cpu()),
        "no_source_class_loss": float(no_source_class.detach().cpu()),
        "profile_drift_loss": float(profile_drift.detach().cpu()),
        "phase_loss": float(phase_loss.detach().cpu()),
        "phase_median_absolute_error_days": phase_error,
        "phase_supervised_visit_fraction": phase_fraction,
        "coadd_reconstruction_loss": float(
            coadd_reconstruction.detach().cpu()
        ),
        "alias_ranking_loss": float(alias_ranking.detach().cpu()),
        "alias_ranking_margin_success_fraction": alias_margin_success,
        "unsupported_source_fraction": float(
            unsupported_source.sum().detach().cpu()
            / measured_source.sum().clamp_min(1).detach().cpu()
        ),
    }


def _coadd_reconstruction_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weight: float,
) -> torch.Tensor:
    prediction = outputs.get("coadd_reconstruction")
    if prediction is None:
        if weight > 0.0:
            raise ValueError(
                "coadd_reconstruction_loss_weight is positive but the model "
                "has no coadd reconstruction head"
            )
        return outputs["joint_logits"].sum() * 0.0
    required = ("clean_flux_target", "flux_error_shape", "visit_flux_scale")
    missing = [name for name in required if name not in batch]
    if missing:
        raise KeyError(
            "Coadd reconstruction target is missing: " + ", ".join(missing)
        )
    target, _, target_support = final_inverse_variance_coadd(
        batch["clean_flux_target"],
        batch["wavelength_mask"],
        batch["visit_mask"],
        batch["flux_error_shape"],
        batch["visit_flux_scale"],
    )
    support = target_support * outputs["coadd_reconstruction_mask"]
    target = normalize_valid_bins(target, support)
    source = batch["has_source"] > 0.5
    valid_object = source & support.bool().any(dim=-1)
    if not valid_object.any():
        return prediction.sum() * 0.0
    per_bin = functional.smooth_l1_loss(
        prediction,
        target,
        reduction="none",
    )
    per_object = (per_bin * support).sum(dim=-1) / support.sum(dim=-1).clamp_min(1.0)
    return per_object[valid_object].mean()


def _alias_ranking_loss(
    source_logits: torch.Tensor | None,
    source_class: torch.Tensor,
    source_redshift: torch.Tensor,
    redshift_grid: torch.Tensor,
    class_weight: torch.Tensor | None,
    minimum_delta_z: float,
    margin: float,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Rank the true cell above the strongest distant class--redshift alias.

    Ordinary joint cross-entropy already suppresses every incorrect cell. This
    optional term is intentionally narrower: it only compares the target with
    the strongest supported solution at least ``minimum_delta_z`` away. It is
    therefore suitable as a controlled alias experiment, not a default loss.
    """
    if minimum_delta_z <= 0.0:
        raise ValueError("alias_ranking_minimum_delta_z must be positive")
    if margin < 0.0:
        raise ValueError("alias_ranking_margin must be non-negative")
    if source_logits is None or source_logits.shape[0] == 0:
        return reference.sum() * 0.0, 0.0

    redshift_index = (
        source_redshift[:, None] - redshift_grid[None, :]
    ).abs().argmin(dim=1)
    object_index = torch.arange(source_logits.shape[0], device=source_logits.device)
    target_score = source_logits[object_index, source_class, redshift_index]
    distant = (
        source_redshift[:, None] - redshift_grid[None, :]
    ).abs() >= minimum_delta_z
    supported = source_logits > -1.0e3
    alias_mask = distant[:, None, :] & supported
    alias_score = (
        source_logits.masked_fill(~alias_mask, -torch.inf)
        .flatten(1)
        .max(dim=1)
        .values
    )
    valid = torch.isfinite(alias_score)
    if not valid.any():
        return source_logits.sum() * 0.0, 0.0

    per_object = functional.softplus(
        float(margin) + alias_score[valid] - target_score[valid]
    )
    if class_weight is None:
        loss = per_object.mean()
    else:
        weight = class_weight[source_class[valid]]
        loss = (per_object * weight).sum() / weight.sum().clamp_min(1.0e-12)
    success = float(
        (target_score[valid] >= alias_score[valid] + float(margin))
        .float()
        .mean()
        .detach()
        .cpu()
    )
    return loss, success


def _phase_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    redshift_grid: torch.Tensor,
    phase_loss_weight: float,
) -> tuple[torch.Tensor, float, float]:
    phase_logits = outputs.get("phase_logits")
    if phase_logits is None:
        if phase_loss_weight > 0.0:
            raise ValueError(
                "phase_loss_weight is positive but the model has no phase head"
            )
        zero = outputs["joint_logits"].sum() * 0.0
        return zero, 0.0, 0.0

    phase_grid = outputs["phase_grid"]
    targets = batch["simulation_rest_phase_days"]
    visit_mask = batch["visit_mask"] > 0.5
    source = batch["has_source"] > 0.5
    in_range = (targets >= phase_grid[0]) & (targets <= phase_grid[-1])
    valid = visit_mask & source[:, None] & in_range

    batch_size, visits = targets.shape
    object_index = torch.arange(batch_size, device=targets.device)[:, None]
    visit_index = torch.arange(visits, device=targets.device)[None, :]
    class_index = batch["class_index"][:, None]
    redshift_index = (
        batch["redshift"][:, None] - redshift_grid[None, :]
    ).abs().argmin(dim=1)[:, None]
    selected = phase_logits[
        object_index,
        visit_index,
        class_index,
        redshift_index,
    ]  # (B,V,P)
    target_index = (
        targets[..., None] - phase_grid[None, None, :]
    ).abs().argmin(dim=-1)

    if valid.any():
        per_visit = functional.cross_entropy(
            selected.reshape(-1, selected.shape[-1]),
            target_index.reshape(-1),
            reduction="none",
        ).reshape(batch_size, visits)
        weight = valid.to(per_visit.dtype)
        per_object = (per_visit * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        supervised_object = valid.any(dim=1)
        loss = per_object[supervised_object].mean()
        predicted_phase = phase_grid[selected.argmax(dim=-1)]
        median_error = float(
            (predicted_phase - targets).abs()[valid].median().detach().cpu()
        )
    else:
        loss = selected.sum() * 0.0
        median_error = 0.0
    source_visit = visit_mask & source[:, None]
    supervised_fraction = float(
        valid.sum().detach().cpu()
        / source_visit.sum().clamp_min(1).detach().cpu()
    )
    return loss, median_error, supervised_fraction
