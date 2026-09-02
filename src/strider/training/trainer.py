"""Train STRIDER with deterministic epoch-boundary continuation.

Inputs are prepared object records and native-bin spectra. The loop writes the
resolved configuration, environment, best model and complete training state.
Interrupted partial epochs restart from their beginning rather than being mixed
with a new data order or noise realisation.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from strider.config import project_path
from strider.data.dataset import SundialDataset, collate_objects
from strider.data.template_support import template_support_policy
from strider.data.wavelength_support import wavelength_support_report
from strider.model import Strider, measurement_inputs
from strider.model.posterior import joint_probability
from strider.reporting import training_end, training_epoch, training_start

from .device import choose_device
from .losses import training_loss
from .precision import autocast_context
from .resume import (
    atomic_torch_save,
    capture_random_state,
    load_training_state,
    restore_random_state,
)
from .run_record import write_run_record
from .visit_batches import VisitCountBatchSampler


TRAINING_REPORT_REDSHIFT_MAX = 2.0
MACRO_REDSHIFT_MINIMUM_CLASS_COUNT = 100


def _job_epoch_limit() -> int | None:
    """Return an optional epoch-boundary limit for this scheduler job.

    This is deliberately an environment setting rather than part of the model
    configuration: changing how a long run is split across Slurm allocations
    must not invalidate an otherwise compatible continuation checkpoint.
    """
    raw = os.environ.get("STRIDER_MAX_EPOCHS_THIS_JOB")
    if raw is None or not raw.strip():
        return None
    try:
        limit = int(raw)
    except ValueError as error:
        raise ValueError(
            "STRIDER_MAX_EPOCHS_THIS_JOB must be a positive integer"
        ) from error
    if limit <= 0:
        raise ValueError("STRIDER_MAX_EPOCHS_THIS_JOB must be a positive integer")
    return limit


def train(config: dict[str, Any], resume: bool = False) -> dict[str, Any]:
    seed = int(config["project"]["seed"])
    _set_seeds(seed)
    device = choose_device()
    output_dir = project_path(config, config["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    config_digest = write_run_record(config, output_dir, device)
    checkpoint_path = output_dir / "best_model.pt"
    science_checkpoint_path = output_dir / "best_science_model.pt"
    redshift_checkpoint_path = output_dir / "best_redshift_model.pt"
    macro_redshift_checkpoint_path = output_dir / "best_macro_redshift_model.pt"
    training_state_path = output_dir / "training_state.pt"
    history_path = output_dir / "training_history.json"
    if template_support_policy(config["observation"]) == "retain":
        support_report = wavelength_support_report(config)
        _write_json(output_dir / "wavelength_support.json", support_report)
        if not support_report["passed"]:
            raise ValueError(
                "Training stopped because simulation-template wavelength support "
                "is associated with class or redshift"
            )
    if not resume and (checkpoint_path.exists() or training_state_path.exists()):
        raise FileExistsError(
            f"Run products already exist in {output_dir}; use --resume or a new output_dir"
        )

    training_data = SundialDataset(config, "train", "generated", training=True)
    settings = config["training"]
    mixed_precision = str(settings.get("mixed_precision", "float32"))
    max_gradient_norm = float(settings.get("max_gradient_norm", 1.0))
    if max_gradient_norm <= 0.0:
        raise ValueError("max_gradient_norm must be positive")
    class_weight = _class_weights(
        training_data, settings, len(config["model"]["classes"])
    ).to(device)
    training_loader = _loader(training_data, settings, shuffle=True)
    validation_view_weights = _validation_view_weights(settings)
    checkpoint_metric_view = _checkpoint_metric_view(settings, validation_view_weights)
    selection_split = str(settings.get("selection_split", "validation"))
    validation_loaders = {
        view: _loader(
            SundialDataset(
                config,
                selection_split,
                view,
                training=False,
                pair_no_source=(
                    bool(settings.get("paired_no_source", False))
                    if view == "generated"
                    else False
                ),
            ),
            settings,
            shuffle=False,
        )
        for view in validation_view_weights
    }

    model = Strider(config).to(device)
    if not resume and settings.get("initial_checkpoint"):
        _load_initial_checkpoint(model, config, str(settings["initial_checkpoint"]))
    if bool(settings.get("temporal_only", False)):
        if model.temporal_evidence is None:
            raise ValueError(
                "temporal_only requires model.temporal_mode: spectral_evolution"
            )
        if not resume and not settings.get("initial_checkpoint"):
            raise ValueError("temporal_only requires initial_checkpoint for a new run")
        _freeze_spectral_model(model)
    optimizer = torch.optim.AdamW(
        _optimizer_parameter_groups(model, float(settings["weight_decay"])),
        lr=float(settings["learning_rate"]),
    )
    scheduler = _learning_rate_scheduler(optimizer, settings)
    best_selection_score = float("inf")
    best_science_score = float("-inf")
    best_science_epoch = 0
    best_redshift_score = float("inf")
    best_redshift_epoch = 0
    best_macro_redshift_score = float("inf")
    best_macro_redshift_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    start_epoch = 0
    if resume:
        if not training_state_path.exists():
            raise FileNotFoundError(
                f"No training state to resume: {training_state_path}"
            )
        state = load_training_state(training_state_path)
        if state.get("config_sha256") != config_digest:
            raise ValueError(
                "The saved training state was produced by a different config"
            )
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        best_selection_score = float(state["best_selection_score"])
        epochs_without_improvement = int(state["epochs_without_improvement"])
        history = list(state["history"])
        start_epoch = int(state["completed_epoch"])
        (
            best_science_score,
            best_science_epoch,
            best_redshift_score,
            best_redshift_epoch,
        ) = _resume_candidate_metrics(state, history, checkpoint_metric_view)
        (
            best_macro_redshift_score,
            best_macro_redshift_epoch,
        ) = _resume_macro_redshift_metrics(state, history, checkpoint_metric_view)
        # Model construction consumes random values, so restore only after all
        # objects needed by the loop have been created and loaded.
        restore_random_state(state["random_state"])

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    validation_object_count = len(next(iter(validation_loaders.values())).dataset)
    job_epoch_limit = _job_epoch_limit()
    training_start(
        device=str(device),
        parameters=parameter_count,
        trainable_parameters=trainable_parameter_count,
        train_objects=len(training_data),
        validation_objects=validation_object_count,
        validation_views=list(validation_loaders),
        epochs=int(settings["epochs"]),
        batch_size=int(settings["batch_size"]),
        workers=int(settings["num_workers"]),
        output_dir=output_dir,
        start_epoch=start_epoch,
        classes=list(config["model"]["classes"]),
        redshift_bins=len(model.redshift_grid),
        visit_limit=config["data"].get("max_visits"),
        full_history_fraction=float(
            settings.get("full_visit_training_fraction", 0.0)
        ),
        job_epoch_limit=job_epoch_limit,
    )
    if resume and history:
        recovered_science = _recover_candidate_checkpoint(
            science_checkpoint_path,
            checkpoint_path,
            model,
            history,
            best_science_epoch,
            start_epoch,
            validation_view_weights,
            checkpoint_metric_view,
            config_digest,
            list(config["model"]["classes"]),
            model.redshift_grid,
            model.redshift_prior,
            role="science",
        )
        if not recovered_science:
            best_science_score = float("-inf")
            best_science_epoch = 0
        recovered_redshift = _recover_candidate_checkpoint(
            redshift_checkpoint_path,
            checkpoint_path,
            model,
            history,
            best_redshift_epoch,
            start_epoch,
            validation_view_weights,
            checkpoint_metric_view,
            config_digest,
            list(config["model"]["classes"]),
            model.redshift_grid,
            model.redshift_prior,
            role="redshift",
        )
        if not recovered_redshift:
            best_redshift_score = float("inf")
            best_redshift_epoch = 0
        recovered_macro_redshift = _recover_candidate_checkpoint(
            macro_redshift_checkpoint_path,
            checkpoint_path,
            model,
            history,
            best_macro_redshift_epoch,
            start_epoch,
            validation_view_weights,
            checkpoint_metric_view,
            config_digest,
            list(config["model"]["classes"]),
            model.redshift_grid,
            model.redshift_prior,
            role="macro_redshift",
        )
        if not recovered_macro_redshift:
            # Older runs did not retain this candidate. Start tracking it from
            # the first newly completed epoch instead of pretending that the
            # weights for a historical leader can be reconstructed.
            best_macro_redshift_score = float("inf")
            best_macro_redshift_epoch = 0

    epochs_completed_this_job = 0
    stop_reason = "configured_epochs"
    for epoch in range(start_epoch, int(settings["epochs"])):
        epoch_started = time.perf_counter()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        training_data.set_epoch(epoch)
        training_metrics = _run_epoch(
            model,
            training_loader,
            device,
            model.redshift_cell_width,
            model.redshift_prior,
            float(settings["evidence_sufficiency_loss_weight"]),
            float(settings.get("no_source_redshift_loss_weight", 0.0)),
            float(settings.get("no_source_class_loss_weight", 0.0)),
            _profile_drift_weight(config),
            class_weight,
            float(settings.get("phase_loss_weight", 0.0)),
            optimizer,
            mixed_precision,
            max_gradient_norm,
            int(settings["batch_size"]),
            coadd_reconstruction_loss_weight=float(
                settings.get("coadd_reconstruction_loss_weight", 0.0)
            ),
            alias_ranking_loss_weight=float(
                settings.get("alias_ranking_loss_weight", 0.0)
            ),
            alias_ranking_minimum_delta_z=float(
                settings.get("alias_ranking_minimum_delta_z", 0.1)
            ),
            alias_ranking_margin=float(settings.get("alias_ranking_margin", 0.0)),
        )
        validation_by_view = {
            view: _run_epoch(
                model,
                loader,
                device,
                model.redshift_cell_width,
                model.redshift_prior,
                float(settings["evidence_sufficiency_loss_weight"]),
                float(settings.get("no_source_redshift_loss_weight", 0.0)),
                float(settings.get("no_source_class_loss_weight", 0.0)),
                _profile_drift_weight(config),
                class_weight,
                float(settings.get("phase_loss_weight", 0.0)),
                optimizer=None,
                mixed_precision=mixed_precision,
                max_gradient_norm=max_gradient_norm,
                optimizer_step_objects=None,
                coadd_reconstruction_loss_weight=float(
                    settings.get("coadd_reconstruction_loss_weight", 0.0)
                ),
                alias_ranking_loss_weight=float(
                    settings.get("alias_ranking_loss_weight", 0.0)
                ),
                alias_ranking_minimum_delta_z=float(
                    settings.get("alias_ranking_minimum_delta_z", 0.1)
                ),
                alias_ranking_margin=float(
                    settings.get("alias_ranking_margin", 0.0)
                ),
            )
            for view, loader in validation_loaders.items()
        }
        selection_score = _weighted_validation_score(
            validation_by_view,
            validation_view_weights,
        )
        science_score = _science_score(validation_by_view, checkpoint_metric_view)
        redshift_score = _ia_outlier_score(validation_by_view, checkpoint_metric_view)
        macro_redshift_score = _macro_redshift_outlier_score(
            validation_by_view, checkpoint_metric_view
        )
        # Keep the original field for readers written before multi-view
        # validation. It is always the generated-observation result.
        validation_metrics = validation_by_view["generated"]
        record = {
            "epoch": epoch + 1,
            "train": training_metrics,
            "validation": validation_metrics,
            "validation_views": validation_by_view,
            "selection_score": selection_score,
            "science_score": science_score,
            "redshift_outlier_score": redshift_score,
            "macro_redshift_outlier_score": macro_redshift_score,
            "learning_rate": learning_rate,
            "learned_scales": _learned_scales(model),
            # Retain the training-plus-validation wall time so matched
            # architecture pilots can compare cost without parsing logs.
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        saved_best = selection_score < best_selection_score
        saved_science = science_score > best_science_score
        saved_redshift = redshift_score < best_redshift_score
        saved_macro_redshift = macro_redshift_score < best_macro_redshift_score
        if saved_best:
            best_selection_score = selection_score
            epochs_without_improvement = 0
            _save_candidate_checkpoint(
                checkpoint_path,
                model,
                record,
                validation_view_weights,
                checkpoint_metric_view,
                config_digest,
                list(config["model"]["classes"]),
                model.redshift_grid,
                model.redshift_prior,
                role="posterior",
            )
        else:
            epochs_without_improvement += 1
        if saved_science:
            best_science_score = science_score
            best_science_epoch = epoch + 1
            _save_candidate_checkpoint(
                science_checkpoint_path,
                model,
                record,
                validation_view_weights,
                checkpoint_metric_view,
                config_digest,
                list(config["model"]["classes"]),
                model.redshift_grid,
                model.redshift_prior,
                role="science",
            )
        if saved_redshift:
            best_redshift_score = redshift_score
            best_redshift_epoch = epoch + 1
            _save_candidate_checkpoint(
                redshift_checkpoint_path,
                model,
                record,
                validation_view_weights,
                checkpoint_metric_view,
                config_digest,
                list(config["model"]["classes"]),
                model.redshift_grid,
                model.redshift_prior,
                role="redshift",
            )
        if saved_macro_redshift:
            best_macro_redshift_score = macro_redshift_score
            best_macro_redshift_epoch = epoch + 1
            _save_candidate_checkpoint(
                macro_redshift_checkpoint_path,
                model,
                record,
                validation_view_weights,
                checkpoint_metric_view,
                config_digest,
                list(config["model"]["classes"]),
                model.redshift_grid,
                model.redshift_prior,
                role="macro_redshift",
            )
        scheduler.step()
        atomic_torch_save(
            {
                "format_version": "strider-training-state-v1",
                "completed_epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_selection_score": best_selection_score,
                "best_science_score": best_science_score,
                "best_science_epoch": best_science_epoch,
                "best_redshift_score": best_redshift_score,
                "best_redshift_epoch": best_redshift_epoch,
                "best_macro_redshift_score": best_macro_redshift_score,
                "best_macro_redshift_epoch": best_macro_redshift_epoch,
                "checkpoint_metric_view": checkpoint_metric_view,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
                "random_state": capture_random_state(),
                "config_sha256": config_digest,
            },
            training_state_path,
        )
        _write_json(history_path, history)
        training_epoch(
            record,
            total_epochs=int(settings["epochs"]),
            seconds=time.perf_counter() - epoch_started,
            saved=saved_best,
            saved_science=saved_science,
            saved_redshift=saved_redshift,
            saved_macro_redshift=saved_macro_redshift,
        )
        epochs_completed_this_job += 1
        if epochs_without_improvement >= int(settings["early_stopping_patience"]):
            stop_reason = "early_stopping"
            break
        if (
            job_epoch_limit is not None
            and epochs_completed_this_job >= job_epoch_limit
            and epoch + 1 < int(settings["epochs"])
        ):
            stop_reason = "job_epoch_limit"
            break

    if not checkpoint_path.exists():
        raise RuntimeError("Training completed without writing a best-model checkpoint")
    report = {
        "device": str(device),
        "epochs_completed": len(history),
        "epochs_completed_this_job": epochs_completed_this_job,
        "configured_epochs": int(settings["epochs"]),
        "stop_reason": stop_reason,
        "best_validation_loss": best_selection_score,
        "best_selection_score": best_selection_score,
        "validation_view_weights": validation_view_weights,
        "checkpoint_metric_view": checkpoint_metric_view,
        "checkpoint": str(checkpoint_path),
        "science_checkpoint": str(science_checkpoint_path),
        "redshift_checkpoint": str(redshift_checkpoint_path),
        "macro_redshift_checkpoint": str(macro_redshift_checkpoint_path),
        "best_science_score": best_science_score,
        "best_science_epoch": best_science_epoch,
        "best_redshift_score": best_redshift_score,
        "best_redshift_epoch": best_redshift_epoch,
        "best_macro_redshift_score": best_macro_redshift_score,
        "best_macro_redshift_epoch": best_macro_redshift_epoch,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "best_epoch": int(
            min(history, key=lambda item: item["selection_score"])["epoch"]
        ),
    }
    _write_json(output_dir / "training_summary.json", report)
    training_end(report)
    return report


def _validation_view_weights(settings: dict[str, Any]) -> dict[str, float]:
    configured = settings.get("validation_view_weights", {"generated": 1.0})
    weights = {str(view): float(weight) for view, weight in configured.items()}
    if "generated" not in weights:
        raise ValueError("validation_view_weights must include generated")
    if any(weight < 0.0 for weight in weights.values()):
        raise ValueError("validation view weights cannot be negative")
    if sum(weights.values()) <= 0.0:
        raise ValueError("at least one validation view weight must be positive")
    return weights


def _checkpoint_metric_view(
    settings: dict[str, Any], validation_view_weights: dict[str, float]
) -> str:
    """Return the validation view used for science-facing checkpoints."""
    configured = settings.get("checkpoint_metric_view")
    view = str(
        configured
        if configured is not None
        else ("original" if "original" in validation_view_weights else "generated")
    )
    if view not in validation_view_weights:
        raise ValueError(
            "checkpoint_metric_view must be present in validation_view_weights"
        )
    return view


def _learned_scales(model: Strider) -> dict[str, float]:
    roman_reference = getattr(model, "roman_reference", None)
    if roman_reference is not None:
        return {
            "coadd": float(
                torch.tanh(roman_reference.coadd_scale).detach().cpu()
            ),
            "sequence": float(
                torch.tanh(roman_reference.sequence_scale).detach().cpu()
            ),
            "continuum_removed": float(
                torch.sigmoid(
                    roman_reference.continuum_removed_intercept
                ).detach().cpu()
            ),
        }
    if model.factored_evidence is not None:
        scales = {
            "shape": float(
                torch.tanh(model.factored_evidence.shape_scale).detach().cpu()
            ),
            "temporal": float(
                torch.tanh(model.factored_evidence.temporal_scale).detach().cpu()
            ),
        }
        if model.full_spectrum_context is not None:
            scales["context"] = float(
                torch.tanh(model.full_spectrum_context.scale).detach().cpu()
            )
        if model.dense_scan is not None:
            scales["dense"] = float(torch.tanh(model.dense_scan.scale).detach().cpu())
            if hasattr(model.dense_scan, "detail_intercept"):
                if getattr(model, "dense_scan_view", "blend") == "detail":
                    scales["dense_detail"] = 1.0
                else:
                    scales["dense_detail"] = float(
                        (
                            float(
                                getattr(
                                    model.dense_scan,
                                    "maximum_detail_weight",
                                    1.0,
                                )
                            )
                            * torch.sigmoid(model.dense_scan.detail_intercept)
                        )
                        .detach()
                        .cpu()
                    )
        if model.factored_evidence.brightness_scale is not None:
            amplitude_name = (
                "flux_evolution"
                if model.relative_amplitude_mode == "object_normalized_flux"
                else "background_scaled_amplitude"
            )
            scales[amplitude_name] = float(
                torch.tanh(model.factored_evidence.brightness_scale).detach().cpu()
            )
        if model.phase_consistency is not None:
            scales["phase"] = float(
                torch.tanh(model.phase_consistency.scale).detach().cpu()
            )
        return scales
    if model.temporal_evidence is not None:
        return {
            "temporal": float(torch.tanh(model.temporal_evidence.scale).detach().cpu())
        }
    if model.phase_consistency is not None:
        return {
            "phase": float(torch.tanh(model.phase_consistency.scale).detach().cpu())
        }
    return {}


def _profile_drift_weight(config: dict[str, Any]) -> float:
    if str(config["model"].get("architecture", "")) == "roman_reference":
        return 0.0
    return float(config.get("onir", {}).get("drift_loss_weight", 0.0))


def _weighted_validation_score(
    validation_by_view: dict[str, dict[str, float]],
    view_weights: dict[str, float],
) -> float:
    total_weight = sum(view_weights.values())
    return (
        sum(
            view_weights[view] * validation_by_view[view]["loss"]
            for view in view_weights
        )
        / total_weight
    )


def _science_score(validation_by_view: dict[str, dict[str, Any]], view: str) -> float:
    """Return z<2 macro F1 on the deployment-facing validation view."""
    return float(validation_by_view[view]["z_lt_2_macro_f1_present"])


def _ia_outlier_score(
    validation_by_view: dict[str, dict[str, Any]], view: str
) -> float:
    """Return the z<2 Ia |dz|>0.1 fraction on the selected view."""
    return float(
        validation_by_view[view]["z_lt_2_metrics_by_class"][0][
            "outlier_fraction_abs_delta_z_gt_0_1"
        ]
    )


def _macro_redshift_outlier_score(
    validation_by_view: dict[str, dict[str, Any]], view: str
) -> float:
    """Return a stable, class-balanced z<2 redshift outlier fraction.

    Classes with fewer than 100 validation objects are excluded because the
    worst-case binomial standard error of an outlier fraction is then above
    five percentage points. If every present class is smaller, use the present
    classes as a fallback. The existing Ia-only score remains available.
    """
    rows = validation_by_view[view]["z_lt_2_metrics_by_class"]
    present = [
        row
        for row in rows
        if int(row.get("N", 0)) > 0
        and math.isfinite(float(row["outlier_fraction_abs_delta_z_gt_0_1"]))
    ]
    eligible = [
        row
        for row in present
        if int(row["N"]) >= MACRO_REDSHIFT_MINIMUM_CLASS_COUNT
    ]
    selected = eligible or present
    if not selected:
        return float("inf")
    return float(
        np.mean(
            [
                float(row["outlier_fraction_abs_delta_z_gt_0_1"])
                for row in selected
            ]
        )
    )


def _resume_candidate_metrics(
    state: dict[str, Any],
    history: list[dict[str, Any]],
    checkpoint_metric_view: str,
) -> tuple[float, int, float, int]:
    """Read new resume fields or reconstruct them from an older history."""
    science = [
        (
            float(
                record.get(
                    "science_score",
                    _science_score(record["validation_views"], checkpoint_metric_view),
                )
            ),
            int(record["epoch"]),
        )
        for record in history
    ]
    redshift = [
        (
            float(
                record.get(
                    "redshift_outlier_score",
                    _ia_outlier_score(
                        record["validation_views"], checkpoint_metric_view
                    ),
                )
            ),
            int(record["epoch"]),
        )
        for record in history
    ]
    science_score, science_epoch = (
        max(science, key=lambda item: item[0]) if science else (float("-inf"), 0)
    )
    redshift_score, redshift_epoch = (
        min(redshift, key=lambda item: item[0]) if redshift else (float("inf"), 0)
    )
    matching_state = state.get("checkpoint_metric_view") == checkpoint_metric_view
    return (
        float(state.get("best_science_score", science_score))
        if matching_state
        else science_score,
        int(state.get("best_science_epoch", science_epoch))
        if matching_state
        else science_epoch,
        float(state.get("best_redshift_score", redshift_score))
        if matching_state
        else redshift_score,
        int(state.get("best_redshift_epoch", redshift_epoch))
        if matching_state
        else redshift_epoch,
    )


def _resume_macro_redshift_metrics(
    state: dict[str, Any],
    history: list[dict[str, Any]],
    checkpoint_metric_view: str,
) -> tuple[float, int]:
    """Read or reconstruct the class-balanced redshift checkpoint leader."""
    candidates = [
        (
            float(
                record.get(
                    "macro_redshift_outlier_score",
                    _macro_redshift_outlier_score(
                        record["validation_views"], checkpoint_metric_view
                    ),
                )
            ),
            int(record["epoch"]),
        )
        for record in history
    ]
    score, epoch = (
        min(candidates, key=lambda item: item[0])
        if candidates
        else (float("inf"), 0)
    )
    matching_state = state.get("checkpoint_metric_view") == checkpoint_metric_view
    return (
        float(state.get("best_macro_redshift_score", score))
        if matching_state
        else score,
        int(state.get("best_macro_redshift_epoch", epoch))
        if matching_state
        else epoch,
    )


def _recover_candidate_checkpoint(
    candidate_path: Path,
    posterior_path: Path,
    model: Strider,
    history: list[dict[str, Any]],
    candidate_epoch: int,
    completed_epoch: int,
    view_weights: dict[str, float],
    checkpoint_metric_view: str,
    config_digest: str,
    classes: list[str],
    redshift_grid: torch.Tensor,
    redshift_prior: str,
    *,
    role: str,
) -> bool:
    """Migrate an old continuation when its candidate weights still exist."""
    if candidate_path.exists():
        return True
    record = next(
        (item for item in history if int(item["epoch"]) == candidate_epoch),
        None,
    )
    if record is None:
        return False
    if candidate_epoch == completed_epoch:
        _save_candidate_checkpoint(
            candidate_path,
            model,
            record,
            view_weights,
            checkpoint_metric_view,
            config_digest,
            classes,
            redshift_grid,
            redshift_prior,
            role=role,
        )
        return True
    if posterior_path.exists():
        posterior = load_training_state(posterior_path)
        if int(posterior.get("epoch", -1)) == candidate_epoch:
            migrated = dict(posterior)
            migrated["checkpoint_role"] = role
            migrated["science_score"] = float(
                record.get(
                    "science_score",
                    _science_score(record["validation_views"], checkpoint_metric_view),
                )
            )
            migrated["redshift_outlier_score"] = float(
                record.get(
                    "redshift_outlier_score",
                    _ia_outlier_score(
                        record["validation_views"], checkpoint_metric_view
                    ),
                )
            )
            migrated["macro_redshift_outlier_score"] = float(
                record.get(
                    "macro_redshift_outlier_score",
                    _macro_redshift_outlier_score(
                        record["validation_views"], checkpoint_metric_view
                    ),
                )
            )
            migrated["checkpoint_metric_view"] = checkpoint_metric_view
            atomic_torch_save(migrated, candidate_path)
            return True
    return False


def _save_candidate_checkpoint(
    path: Path,
    model: Strider,
    record: dict[str, Any],
    view_weights: dict[str, float],
    checkpoint_metric_view: str,
    config_digest: str,
    classes: list[str],
    redshift_grid: torch.Tensor,
    redshift_prior: str,
    *,
    role: str,
) -> None:
    validation_by_view = record["validation_views"]
    generated = validation_by_view["generated"]
    science_score = float(
        record.get(
            "science_score",
            _science_score(validation_by_view, checkpoint_metric_view),
        )
    )
    redshift_outlier_score = float(
        record.get(
            "redshift_outlier_score",
            _ia_outlier_score(validation_by_view, checkpoint_metric_view),
        )
    )
    macro_redshift_outlier_score = float(
        record.get(
            "macro_redshift_outlier_score",
            _macro_redshift_outlier_score(
                validation_by_view, checkpoint_metric_view
            ),
        )
    )
    atomic_torch_save(
        {
            "format_version": "strider-checkpoint-v2",
            "checkpoint_role": role,
            "model_state": model.state_dict(),
            "epoch": int(record["epoch"]),
            "validation_loss": generated["loss"],
            "validation_views": validation_by_view,
            "validation_view_weights": view_weights,
            "checkpoint_metric_view": checkpoint_metric_view,
            "selection_score": float(record["selection_score"]),
            "science_score": science_score,
            "redshift_outlier_score": redshift_outlier_score,
            "macro_redshift_outlier_score": macro_redshift_outlier_score,
            "classes": classes,
            "redshift_grid": redshift_grid.detach().cpu(),
            "redshift_prior": redshift_prior,
            "config_sha256": config_digest,
        },
        path,
    )


def _optimizer_parameter_groups(
    model: Strider,
    weight_decay: float,
) -> list[dict[str, Any]]:
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _loader(
    dataset: SundialDataset, settings: dict[str, Any], shuffle: bool
) -> DataLoader:
    workers = int(settings["num_workers"])
    persistent = (
        bool(settings.get("persistent_workers", workers > 0)) if workers else False
    )
    options: dict[str, Any] = {}
    if workers:
        options["prefetch_factor"] = int(settings.get("prefetch_factor", 2))
    common = {
        "num_workers": workers,
        "collate_fn": collate_objects,
        "persistent_workers": persistent,
        "pin_memory": torch.cuda.is_available(),
        **options,
    }
    visit_batching = bool(settings.get("batch_by_visit_count", False)) and (
        bool(dataset.training_visit_counts) or dataset.max_visits is None
    )
    if visit_batching:
        return DataLoader(
            dataset,
            batch_sampler=VisitCountBatchSampler(
                dataset,
                batch_size=int(settings["batch_size"]),
                shuffle=shuffle,
                maximum_visits_per_batch=settings.get(
                    "maximum_visits_per_batch"
                ),
                maximum_squared_visits_per_batch=settings.get(
                    "maximum_squared_visits_per_batch"
                ),
            ),
            **common,
        )
    return DataLoader(
        dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=shuffle,
        drop_last=False,
        **common,
    )


def _learning_rate_scheduler(
    optimizer: torch.optim.Optimizer,
    settings: dict[str, Any],
) -> torch.optim.lr_scheduler.LambdaLR:
    schedule = str(settings.get("learning_rate_schedule", "constant"))
    if schedule not in {"constant", "cosine"}:
        raise ValueError("learning_rate_schedule must be 'constant' or 'cosine'")
    total_epochs = int(settings["epochs"])
    warmup_epochs = int(settings.get("warmup_epochs", 0))
    if not 0 <= warmup_epochs < total_epochs:
        raise ValueError("warmup_epochs must be nonnegative and smaller than epochs")
    minimum = float(settings.get("minimum_learning_rate_fraction", 0.0))
    if not 0.0 <= minimum <= 1.0:
        raise ValueError("minimum_learning_rate_fraction must lie in [0, 1]")

    def factor(epoch: int) -> float:
        if warmup_epochs and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        if schedule == "constant" or total_epochs - warmup_epochs <= 1:
            return 1.0
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs - 1)
        progress = min(max(progress, 0.0), 1.0)
        return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _class_weights(
    dataset: SundialDataset,
    settings: dict[str, Any],
    class_count: int,
) -> torch.Tensor:
    mode = str(settings.get("class_weight_mode", "none"))
    if mode == "none":
        return torch.ones(class_count, dtype=torch.float32)
    if mode != "inverse_frequency":
        raise ValueError("class_weight_mode must be 'none' or 'inverse_frequency'")
    counts = np.bincount(
        dataset.objects["class_index"].to_numpy(dtype=np.int64),
        minlength=class_count,
    )
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"Training split has no examples for class indices {missing}")
    power = float(settings.get("class_weight_power", 0.5))
    weights = np.power(counts.astype(np.float64), -power)
    weights /= weights.mean()
    return torch.from_numpy(weights.astype(np.float32))


def _run_epoch(
    model: Strider,
    loader: DataLoader,
    device: torch.device,
    redshift_cell_width: torch.Tensor,
    redshift_prior: str,
    evidence_sufficiency_weight: float,
    no_source_redshift_weight: float,
    no_source_class_weight: float,
    profile_drift_weight: float,
    class_weight: torch.Tensor,
    phase_loss_weight: float,
    optimizer: torch.optim.Optimizer | None,
    mixed_precision: str,
    max_gradient_norm: float,
    optimizer_step_objects: int | None,
    *,
    coadd_reconstruction_loss_weight: float = 0.0,
    alias_ranking_loss_weight: float = 0.0,
    alias_ranking_minimum_delta_z: float = 0.1,
    alias_ranking_margin: float = 0.0,
) -> dict[str, float]:
    training = optimizer is not None
    if training and (optimizer_step_objects is None or optimizer_step_objects < 1):
        raise ValueError("optimizer_step_objects must be positive during training")
    _set_training_mode(model, training)
    totals = {
        "loss": 0.0,
        "joint_loss": 0.0,
        "evidence_sufficiency_loss": 0.0,
        "no_source_redshift_loss": 0.0,
        "no_source_class_loss": 0.0,
        "profile_drift_loss": 0.0,
        "phase_loss": 0.0,
        "phase_median_absolute_error_days": 0.0,
        "phase_supervised_visit_fraction": 0.0,
        "coadd_reconstruction_loss": 0.0,
        "alias_ranking_loss": 0.0,
        "alias_ranking_margin_success_fraction": 0.0,
        "unsupported_source_fraction": 0.0,
    }
    object_count = 0
    accumulated_objects = 0
    progress = _empty_validation_progress(model.class_names)
    if training:
        optimizer.zero_grad(set_to_none=True)
    for batch in loader:
        batch = {
            name: value.to(device, non_blocking=device.type == "cuda")
            for name, value in batch.items()
        }
        batch_size = int(batch["flux"].shape[0])
        with torch.set_grad_enabled(training):
            with autocast_context(device, mixed_precision):
                outputs = model(measurement_inputs(batch))
                loss, metrics = training_loss(
                    outputs,
                    batch,
                    model.redshift_grid,
                    redshift_cell_width,
                    redshift_prior,
                    evidence_sufficiency_weight,
                    no_source_redshift_weight,
                    no_source_class_weight,
                    profile_drift_weight,
                    class_weight,
                    phase_loss_weight,
                    coadd_reconstruction_loss_weight=(
                        coadd_reconstruction_loss_weight
                    ),
                    alias_ranking_loss_weight=alias_ranking_loss_weight,
                    alias_ranking_minimum_delta_z=(
                        alias_ranking_minimum_delta_z
                    ),
                    alias_ranking_margin=alias_ranking_margin,
                )
            if not training:
                _update_validation_progress(progress, outputs, batch, model)
            if training:
                # Sum object losses across memory-sized microbatches. Dividing
                # gradients by the accumulated object count makes long-visit
                # objects carry the same optimization weight as short ones.
                (loss * batch_size).backward()
                accumulated_objects += batch_size
                if accumulated_objects >= optimizer_step_objects:
                    _optimizer_step(
                        model,
                        optimizer,
                        accumulated_objects,
                        max_gradient_norm,
                    )
                    accumulated_objects = 0
        object_count += batch_size
        for name in totals:
            totals[name] += metrics[name] * batch_size
    if training and accumulated_objects:
        _optimizer_step(
            model,
            optimizer,
            accumulated_objects,
            max_gradient_norm,
        )
    result = {name: value / max(object_count, 1) for name, value in totals.items()}
    if not training:
        result.update(_finish_validation_progress(progress))
    return result


def _optimizer_step(
    model: Strider,
    optimizer: torch.optim.Optimizer,
    accumulated_objects: int,
    max_gradient_norm: float,
) -> None:
    scale = float(accumulated_objects)
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(scale)
    torch.nn.utils.clip_grad_norm_(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        max_norm=max_gradient_norm,
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _empty_validation_progress(
    class_names: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    class_count = len(class_names)
    return {
        "source_count": 0,
        "source_class_correct": 0,
        "ia_true": 0,
        "ia_predicted": 0,
        "ia_correct": 0,
        "ia_absolute_delta_z": [],
        "source_evidence_sum": 0.0,
        "no_source_count": 0,
        "no_source_evidence_sum": 0.0,
        "z_lt_2_class_names": tuple(class_names),
        "z_lt_2_confusion": np.zeros((class_count, class_count), dtype=np.int64),
        "z_lt_2_delta_z_by_class": [[] for _ in range(class_count)],
    }


def _update_validation_progress(
    progress: dict[str, Any],
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    model: Strider,
) -> None:
    probability = joint_probability(
        outputs["joint_logits"],
        model.redshift_cell_width,
        model.redshift_prior,
    )  # (B,C,Z)
    predicted_class = probability.sum(dim=2).argmax(dim=1)
    predicted_redshift = _posterior_median_redshift(
        probability.sum(dim=1),
        model.redshift_grid,
    )
    evidence = torch.sigmoid(outputs["evidence_sufficiency_logit"])
    source = batch["has_source"] > 0.5
    no_source = ~source
    true_class = batch["class_index"].long()

    progress["source_count"] += int(source.sum())
    progress["source_class_correct"] += int(
        ((predicted_class == true_class) & source).sum()
    )
    progress["source_evidence_sum"] += float(evidence[source].sum())
    progress["no_source_count"] += int(no_source.sum())
    progress["no_source_evidence_sum"] += float(evidence[no_source].sum())

    ia_true = source & (true_class == 0)
    ia_predicted = source & (predicted_class == 0)
    progress["ia_true"] += int(ia_true.sum())
    progress["ia_predicted"] += int(ia_predicted.sum())
    progress["ia_correct"] += int((ia_true & ia_predicted).sum())
    if ia_true.any():
        absolute_delta = torch.abs(
            predicted_redshift[ia_true] - batch["redshift"][ia_true]
        )
        progress["ia_absolute_delta_z"].extend(absolute_delta.detach().cpu().tolist())

    z_lt_2 = source & (batch["redshift"] < TRAINING_REPORT_REDSHIFT_MAX)
    if not z_lt_2.any():
        return
    class_count = len(progress["z_lt_2_class_names"])
    pairs = true_class[z_lt_2] * class_count + predicted_class[z_lt_2]
    confusion = torch.bincount(pairs, minlength=class_count * class_count)
    progress["z_lt_2_confusion"] += (
        confusion.reshape(class_count, class_count).detach().cpu().numpy()
    )
    delta_z = predicted_redshift - batch["redshift"]
    for class_index in range(class_count):
        selected = z_lt_2 & (true_class == class_index)
        if selected.any():
            progress["z_lt_2_delta_z_by_class"][class_index].extend(
                delta_z[selected].detach().cpu().tolist()
            )


def _finish_validation_progress(progress: dict[str, Any]) -> dict[str, Any]:
    def fraction(numerator: int | float, denominator: int) -> float:
        return float(numerator / denominator) if denominator else float("nan")

    precision = fraction(progress["ia_correct"], progress["ia_predicted"])
    recall = fraction(progress["ia_correct"], progress["ia_true"])
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if math.isfinite(precision)
        and math.isfinite(recall)
        and precision + recall > 0.0
        else 0.0
    )
    absolute_delta = progress["ia_absolute_delta_z"]
    result: dict[str, Any] = {
        "source_class_accuracy": fraction(
            progress["source_class_correct"], progress["source_count"]
        ),
        "ia_precision": precision,
        "ia_recall": recall,
        "ia_f1": float(f1),
        "ia_median_absolute_delta_z": (
            float(np.median(absolute_delta)) if absolute_delta else float("nan")
        ),
        "source_mean_evidence_sufficiency": fraction(
            progress["source_evidence_sum"], progress["source_count"]
        ),
        "no_source_mean_evidence_sufficiency": fraction(
            progress["no_source_evidence_sum"], progress["no_source_count"]
        ),
    }
    result.update(_finish_z_lt_2_progress(progress))
    return result


def _finish_z_lt_2_progress(progress: dict[str, Any]) -> dict[str, Any]:
    confusion = np.asarray(progress["z_lt_2_confusion"], dtype=np.int64)
    class_names = tuple(progress["z_lt_2_class_names"])
    rows = []
    present_f1 = []
    present_recall = []
    for class_index, class_name in enumerate(class_names):
        true_count = int(confusion[class_index].sum())
        predicted_count = int(confusion[:, class_index].sum())
        true_positive = int(confusion[class_index, class_index])
        precision = _safe_fraction(true_positive, predicted_count)
        recall = _safe_fraction(true_positive, true_count)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if true_count
            and math.isfinite(precision)
            and math.isfinite(recall)
            and precision + recall > 0.0
            else (0.0 if true_count else float("nan"))
        )
        delta_z = np.asarray(
            progress["z_lt_2_delta_z_by_class"][class_index],
            dtype=np.float64,
        )
        row = {
            "class_index": class_index,
            "class_name": class_name,
            "N": true_count,
            "predicted_N": predicted_count,
            "precision": precision,
            "recall": recall,
            "f1": float(f1),
            "median_delta_z": (
                float(np.median(delta_z)) if delta_z.size else float("nan")
            ),
            "population_scatter_delta_z": (
                float(np.std(delta_z, ddof=1)) if delta_z.size > 1 else 0.0
            ),
            "median_absolute_delta_z": (
                float(np.median(np.abs(delta_z))) if delta_z.size else float("nan")
            ),
            "outlier_fraction_abs_delta_z_gt_0_1": (
                float(np.mean(np.abs(delta_z) > 0.1)) if delta_z.size else float("nan")
            ),
        }
        rows.append(row)
        if true_count:
            present_f1.append(float(f1) if math.isfinite(f1) else 0.0)
            present_recall.append(float(recall))

    ia = rows[0]
    return {
        "z_lt_2_redshift_max": TRAINING_REPORT_REDSHIFT_MAX,
        "z_lt_2_N": int(confusion.sum()),
        "z_lt_2_balanced_accuracy_present": (
            float(np.mean(present_recall)) if present_recall else float("nan")
        ),
        "z_lt_2_macro_f1_present": (
            float(np.mean(present_f1)) if present_f1 else float("nan")
        ),
        "ia_z_lt_2_precision": ia["precision"],
        "ia_z_lt_2_recall": ia["recall"],
        "ia_z_lt_2_f1": ia["f1"],
        "ia_z_lt_2_median_absolute_delta_z": ia["median_absolute_delta_z"],
        "ia_z_lt_2_population_scatter_delta_z": ia["population_scatter_delta_z"],
        "z_lt_2_metrics_by_class": rows,
    }


def _safe_fraction(numerator: int | float, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _posterior_median_redshift(
    probability_mass: torch.Tensor,
    redshift_grid: torch.Tensor,
) -> torch.Tensor:
    """Match the catalog's interpolated posterior-median point estimate."""
    midpoint_cdf = probability_mass.cumsum(dim=-1) - 0.5 * probability_mass
    upper = (midpoint_cdf >= 0.5).to(torch.int64).argmax(dim=-1)
    lower = (upper - 1).clamp_min(0)
    batch = torch.arange(probability_mass.shape[0], device=probability_mass.device)
    lower_cdf = midpoint_cdf[batch, lower]
    upper_cdf = midpoint_cdf[batch, upper]
    denominator = (upper_cdf - lower_cdf).clamp_min(1e-12)
    weight = ((0.5 - lower_cdf) / denominator).clamp(0.0, 1.0)
    interpolated = redshift_grid[lower] + weight * (
        redshift_grid[upper] - redshift_grid[lower]
    )
    return torch.where(upper == 0, redshift_grid[0], interpolated)


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_initial_checkpoint(
    model: Strider,
    config: dict[str, Any],
    checkpoint_setting: str,
) -> None:
    path = project_path(config, checkpoint_setting)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("classes") != list(config["model"]["classes"]):
        raise ValueError("Initial checkpoint classes do not match this run")
    checkpoint_grid = torch.as_tensor(checkpoint.get("redshift_grid"))
    if checkpoint_grid.shape != model.redshift_grid.shape or not torch.allclose(
        checkpoint_grid, model.redshift_grid.cpu()
    ):
        raise ValueError("Initial checkpoint redshift grid does not match this run")
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    allowed_missing = ["temporal_evidence."]
    if bool(config["model"].get("dense_rest_frame_scan", False)):
        allowed_missing.append("dense_scan.")
    if bool(config["model"].get("dense_continuum_detail", False)):
        allowed_missing.append("continuum_removal.")
    if bool(config["model"].get("coadd_reconstruction", False)):
        allowed_missing.append("coadd_reconstruction_head.")
    if bool(config["model"].get("relative_brightness_evolution", False)):
        allowed_missing.extend(
            (
                "factored_evidence.brightness_projection.",
                "factored_evidence.brightness_scale",
            )
        )
    if int(config["model"].get("phase_auxiliary_bins", 0)):
        allowed_missing.extend(("phase_head.", "phase_consistency."))
        missing = [name for name in missing if name != "phase_grid"]
    allowed_unexpected = []
    if not bool(config["model"].get("full_spectrum_context", False)):
        allowed_unexpected.append("full_spectrum_context.")
    incompatible_unexpected = [
        name for name in unexpected if not name.startswith(tuple(allowed_unexpected))
    ]
    if incompatible_unexpected or any(
        not name.startswith(tuple(allowed_missing)) for name in missing
    ):
        raise ValueError(
            "Initial checkpoint is incompatible: "
            f"missing={missing}, unexpected={incompatible_unexpected}"
        )


def _freeze_spectral_model(model: Strider) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("temporal_evidence.")


def _set_training_mode(model: Strider, training: bool) -> None:
    model.train(training)
    if not training or model.temporal_evidence is None:
        return
    spectral_is_frozen = all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("temporal_evidence.")
    )
    if spectral_is_frozen:
        model.eval()
        model.temporal_evidence.train()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    temporary.replace(path)
