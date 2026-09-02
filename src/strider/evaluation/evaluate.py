"""Evaluate original, regenerated, clean, and no-source observations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from strider.config import project_path, resolved_config_sha256
from strider.data.dataset import SundialDataset
from strider.data.external_test import (
    config_for_prepared_external_test,
    prepared_external_test_provenance,
)
from strider.model import Strider, measurement_inputs
from strider.model.posterior import joint_probability
from strider.reporting import evaluation_end, evaluation_start, evaluation_view

from .checkpoint import load_trained_model
from .controls import blank_redshift_metrics, sufficiency_auc
from .loader import inference_loader
from .metrics import metrics_by_redshift, source_metrics


def evaluate(
    config: dict[str, Any],
    split: str | None = None,
    views: list[str] | None = None,
    external_prepared_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    if external_prepared_dir is None and output_dir is not None:
        raise ValueError("output_dir is reserved for an external-test evaluation")
    if external_prepared_dir is not None and output_dir is None:
        raise ValueError("An external-test evaluation requires a separate output_dir")
    external_provenance: dict[str, str] | None = None
    data_config = config
    if external_prepared_dir is not None:
        external_provenance = prepared_external_test_provenance(
            external_prepared_dir
        )
        data_config = config_for_prepared_external_test(
            config,
            external_prepared_dir,
        )
        evaluation_output_dir = Path(output_dir).expanduser().resolve()
        checkpoint_output_dir = project_path(
            config,
            config["project"]["output_dir"],
        ).resolve()
        if evaluation_output_dir == checkpoint_output_dir:
            raise ValueError(
                "External-test output_dir must not overwrite the checkpoint run"
            )
    else:
        evaluation_output_dir = project_path(config, config["project"]["output_dir"])
    evaluation_output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint, device = load_trained_model(config)
    evaluation_split = split or str(data_config["evaluation"].get("split", "test"))
    if external_provenance is not None and evaluation_split != "test":
        raise ValueError("A prepared external-test store can only evaluate the test split")
    requested_views = (
        list(views) if views is not None else list(config["evaluation"]["views"])
    )
    extra_views = (
        "clean",
        "no_source",
        "residual",
        "reported_error_with_source",
        "reported_error_no_source",
    )
    evaluation_views = (
        requested_views
        if views is not None
        else requested_views
        + [view for view in extra_views if view not in requested_views]
    )
    report: dict[str, Any] = {
        "device": str(device),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "config_sha256": resolved_config_sha256(config),
        "split": evaluation_split,
        "redshift_point_estimator": "primary_basin_peak",
        "redshift_interval": "primary_basin_conditional_68",
        "views": {},
    }
    if external_provenance is not None:
        report["external_test"] = {
            **external_provenance,
            "model_config_sha256": resolved_config_sha256(config),
            "data_config_sha256": resolved_config_sha256(data_config),
        }
    threshold = float(config["evaluation"]["outlier_delta_z"])
    predictions_by_view: dict[str, pd.DataFrame] = {}
    observed_snr_catalog: pd.DataFrame | None = None
    evaluation_start(
        split=evaluation_split,
        checkpoint_epoch=report["checkpoint_epoch"],
        device=report["device"],
    )
    for view in evaluation_views:
        dataset = SundialDataset(
            data_config,
            evaluation_split,
            view,
            training=False,
            include_observed_signal_to_noise=observed_snr_catalog is None,
        )
        loader = inference_loader(dataset, data_config)
        predictions = _predict(
            model,
            loader,
            device,
            save_redshift_probability=bool(
                config["evaluation"].get("save_redshift_probability", False)
            ),
        )
        observed_snr_columns = [
            "snid",
            "median_coadded_observed_signal_to_noise",
            "observed_snr_valid_wavelength_bins",
        ]
        if observed_snr_catalog is None:
            observed_snr_catalog = predictions[observed_snr_columns].copy()
        else:
            predictions = predictions.merge(
                observed_snr_catalog,
                on="snid",
                how="left",
                validate="one_to_one",
            )
        predictions = _attach_prediction_provenance(
            predictions,
            split=evaluation_split,
            view=view,
            checkpoint_epoch=int(checkpoint["epoch"]),
            config_sha256=resolved_config_sha256(config),
            data_config_sha256=(
                resolved_config_sha256(data_config)
                if external_provenance is not None
                else None
            ),
            dataset_tag=(
                external_provenance["dataset_tag"]
                if external_provenance is not None
                else None
            ),
        )
        predictions.to_parquet(
            evaluation_output_dir / f"{evaluation_split}_predictions_{view}.parquet",
            index=False,
        )
        predictions_by_view[view] = predictions
        if view in {"no_source", "residual", "reported_error_no_source"}:
            view_report = {
                **blank_redshift_metrics(
                    predictions,
                    int(config["project"]["seed"]),
                ),
                "N": int(len(predictions)),
                "mean_evidence_sufficiency": float(
                    predictions["evidence_score"].mean()
                ),
                "fraction_sufficient_evidence_above_0_5": float(
                    (predictions["evidence_score"] > 0.5).mean()
                ),
                "mean_largest_joint_probability": float(
                    predictions["largest_joint_probability"].mean()
                ),
                "mean_largest_class_probability": float(
                    predictions["largest_class_probability"].mean()
                ),
                "fraction_largest_class_probability_above_0_5": float(
                    (predictions["largest_class_probability"] > 0.5).mean()
                ),
                "median_absolute_delta_z_to_simulation": float(
                    np.median(np.abs(predictions["predicted_redshift"] - predictions["true_redshift"]))
                ),
                "fraction_within_delta_z_0_1_of_simulation": float(
                    (np.abs(predictions["predicted_redshift"] - predictions["true_redshift"]) <= 0.1).mean()
                ),
            }
            if "mean_runtime_phase_largest_probability" in predictions:
                view_report["mean_runtime_phase_largest_probability"] = float(
                    predictions["mean_runtime_phase_largest_probability"].mean()
                )
        else:
            view_report = source_metrics(predictions, threshold)
            view_report["redshift_groups"] = metrics_by_redshift(
                predictions, threshold, data_config["data"]["redshift_edges"]
            )
            pd.DataFrame(view_report["metrics_by_class"]).to_csv(
                evaluation_output_dir / f"{evaluation_split}_metrics_by_class_{view}.csv",
                index=False,
            )
            ia_edges = config["evaluation"].get(
                "ia_redshift_edges", data_config["data"]["redshift_edges"]
            )
            ia_report = dict(view_report)
            ia_report["redshift_groups"] = metrics_by_redshift(
                predictions,
                threshold,
                ia_edges,
            )
            ia_rows = _ia_metric_rows(ia_report)
            view_report["ia_redshift_groups"] = ia_rows[1:]
            pd.DataFrame(ia_rows).to_csv(
                evaluation_output_dir / f"{evaluation_split}_metrics_Ia_{view}.csv",
                index=False,
            )
        report["views"][view] = view_report
        evaluation_view(view, view_report, threshold)
    blank = predictions_by_view.get("no_source")
    report["control_summary"] = (
        {
            f"{view}_versus_no_source_sufficiency_auc": sufficiency_auc(
                predictions, blank
            )
            for view, predictions in predictions_by_view.items()
            if view not in {"no_source", "residual", "reported_error_no_source"}
        }
        if blank is not None
        else {}
    )
    report["data_selection"] = _data_selection_summary(data_config)
    summary_name = _evaluation_summary_name(
        evaluation_split,
        split_overridden=split is not None,
        selected_views=views,
    )
    with (evaluation_output_dir / summary_name).open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    evaluation_end(evaluation_output_dir)
    return report


def _attach_prediction_provenance(
    predictions: pd.DataFrame,
    *,
    split: str,
    view: str,
    checkpoint_epoch: int,
    config_sha256: str,
    data_config_sha256: str | None = None,
    dataset_tag: str | None = None,
) -> pd.DataFrame:
    """Make every saved prediction table self-identifying for calibration."""
    result = predictions.copy()
    provenance = {
        "data_split": str(split),
        "data_view": str(view),
        "checkpoint_epoch": int(checkpoint_epoch),
        "config_sha256": str(config_sha256),
    }
    if data_config_sha256 is not None:
        provenance["data_config_sha256"] = str(data_config_sha256)
    if dataset_tag is not None:
        provenance["dataset_tag"] = str(dataset_tag)
    for position, (name, value) in enumerate(provenance.items()):
        if name in result:
            if not (result[name] == value).all():
                raise ValueError(f"Prediction provenance conflict in {name}")
            continue
        result.insert(position, name, value)
    return result


def _evaluation_summary_name(
    evaluation_split: str,
    *,
    split_overridden: bool,
    selected_views: list[str] | None,
) -> str:
    """Keep focused evaluations from replacing the canonical full summary."""
    split_prefix = f"{evaluation_split}_" if split_overridden else ""
    if selected_views is None:
        return f"{split_prefix}evaluation_summary.json"
    view_suffix = "_".join(selected_views)
    return f"{split_prefix}evaluation_summary_views_{view_suffix}.json"


def _data_selection_summary(config: dict[str, Any]) -> dict[str, Any]:
    prepared = project_path(config, config["data"]["prepared_dir"])
    path = prepared / "dataset_summary.json"
    if not path.is_file():
        return {"available": False}
    with path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    excluded = summary.get(
        "objects_excluded_incomplete_template_support",
        summary.get("objects_excluded_incomplete_wavelength_coverage", 0),
    )
    exclusions = summary.get(
        "incomplete_template_support",
        summary.get("incomplete_wavelength_coverage", {}),
    )
    return {
        "available": True,
        "template_support_policy": summary.get("template_support_policy", "complete"),
        "retained_objects": summary.get("objects"),
        "excluded_objects": excluded,
        "excluded_by_split_class_and_redshift": exclusions,
    }


@torch.no_grad()
def _predict(
    model: Strider,
    loader: DataLoader,
    device: torch.device,
    time_transform: str = "none",
    time_reassignment: dict[int, torch.Tensor] | None = None,
    save_redshift_probability: bool = False,
    logit_source: str = "combined",
) -> pd.DataFrame:
    rows = []
    grid = model.redshift_grid.detach().cpu().numpy()
    for batch in loader:
        batch = _transform_times(batch, time_transform, time_reassignment)
        device_batch = {name: value.to(device) for name, value in batch.items()}
        outputs = model(measurement_inputs(device_batch))
        logits = _selected_joint_logits(outputs, logit_source)
        joint = joint_probability(
            logits,
            model.redshift_cell_width,
            model.redshift_prior,
        )  # (B,C,Z)
        class_probability = joint.sum(dim=2)
        redshift_probability = joint.sum(dim=1)
        predicted_class = class_probability.argmax(dim=1)
        true_class = device_batch["class_index"].long()
        object_index = torch.arange(len(true_class), device=device)
        redshift_given_true_class = joint[object_index, true_class]
        redshift_given_true_class = redshift_given_true_class / redshift_given_true_class.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-12)
        class_given_true_redshift = _class_probability_at_redshift(
            joint,
            device_batch["redshift"],
            model.redshift_grid,
            model.redshift_cell_width,
        )
        predicted_class_given_true_redshift = class_given_true_redshift.argmax(dim=1)
        mean_redshift = (redshift_probability * model.redshift_grid).sum(dim=1)
        evidence_sufficiency = torch.sigmoid(outputs["evidence_sufficiency_logit"])
        spectral_strength = _logit_strength(outputs.get("spectral_joint_logits"), joint)
        temporal_strength = _logit_strength(outputs.get("temporal_joint_logits"), joint)
        context_strength = _logit_strength(outputs.get("context_joint_logits"), joint)
        dense_strength = _logit_strength(outputs.get("dense_scan_joint_logits"), joint)
        dense_detail_strength = _logit_strength(
            outputs.get("dense_detail_joint_logits"), joint
        )
        dense_overlap = outputs.get("dense_scan_overlap_fraction")
        route_logits = _candidate_route_logits(outputs)
        phase_probability = None
        phase_grid = None
        phase_logits = None
        if "phase_logits" in outputs:
            phase_logits = outputs["phase_logits"]  # (B,V,C,Z,P)
            batch_size, visits = phase_logits.shape[:2]
            object_index = torch.arange(batch_size, device=device)[:, None]
            visit_index = torch.arange(visits, device=device)[None, :]
            true_redshift_index = (
                device_batch["redshift"][:, None] - model.redshift_grid[None, :]
            ).abs().argmin(dim=1)[:, None]
            selected_phase_logits = phase_logits[
                object_index,
                visit_index,
                true_class[:, None],
                true_redshift_index,
            ]  # (B,V,P)
            phase_probability = torch.softmax(selected_phase_logits, dim=-1)
            phase_grid = outputs["phase_grid"].detach().cpu().numpy()
        for index in range(len(batch["snid"])):
            # Use float64 here because posterior tails can be below the smallest
            # positive float32 value. These summaries must remain finite even
            # when the posterior contains exact zero-mass cells.
            redshift_distribution = (
                redshift_probability[index].detach().cpu().numpy().astype(np.float64)
            )
            conditional_redshift_distribution = (
                redshift_given_true_class[index].detach().cpu().numpy().astype(np.float64)
            )
            cell_width = (
                model.redshift_cell_width.detach().cpu().numpy().astype(np.float64)
            )
            lower, upper = _central_interval(grid, redshift_distribution, 0.68)
            conditional_lower, conditional_upper = _central_interval(
                grid, conditional_redshift_distribution, 0.68
            )
            peak_summary = _posterior_peak_summary(
                grid,
                redshift_distribution,
                cell_width,
            )
            peak_candidates = _posterior_basin_candidates(
                grid,
                redshift_distribution,
                cell_width,
            )
            joint_candidates = _joint_posterior_basin_candidates(
                grid,
                joint[index].detach().cpu().numpy().astype(np.float64),
                cell_width,
                model.class_names,
            )
            joint_primary = joint_candidates[0]
            primary_candidate = peak_candidates[0]
            primary_left = int(primary_candidate["left_index"])
            primary_right = int(primary_candidate["right_index"])
            primary_class_probability = joint[
                index, :, primary_left:primary_right
            ].sum(dim=1)
            primary_class_probability = primary_class_probability / (
                primary_class_probability.sum().clamp_min(1.0e-12)
            )
            primary_class = int(primary_class_probability.argmax().cpu())
            candidate_classes = []
            candidate_class_indices = []
            candidate_class_probabilities = []
            for candidate in peak_candidates:
                left = int(candidate["left_index"])
                right = int(candidate["right_index"])
                candidate_probability = joint[index, :, left:right].sum(dim=1)
                candidate_probability = candidate_probability / (
                    candidate_probability.sum().clamp_min(1.0e-12)
                )
                candidate_class = int(candidate_probability.argmax().cpu())
                candidate_class_indices.append(candidate_class)
                candidate_classes.append(model.class_names[candidate_class])
                candidate_class_probabilities.append(
                    float(candidate_probability[candidate_class].cpu())
                )
            candidate_route_support = {
                name: _candidate_route_support(
                    component[index].detach().cpu().numpy(),
                    peak_candidates,
                    candidate_class_indices,
                )
                for name, component in route_logits.items()
            }
            candidate_route_support_mean = np.asarray(
                list(candidate_route_support.values()), dtype=np.float64
            ).mean(axis=0)
            density_mode = peak_summary["dominant_redshift"]
            posterior_median = _posterior_quantile(grid, redshift_distribution, 0.5)
            primary_peak = float(primary_candidate["peak_redshift"])
            differential_entropy, information_gain = _posterior_information(
                redshift_distribution,
                cell_width,
                model.redshift_prior,
            )
            row = {
                "snid": int(batch["snid"][index]),
                "true_class": int(batch["class_index"][index]),
                "true_class_name": model.class_names[int(batch["class_index"][index])],
                "predicted_class": int(predicted_class[index].cpu()),
                "predicted_class_name": model.class_names[
                    int(predicted_class[index].cpu())
                ],
                "p_Ia": float(class_probability[index, 0].cpu()),
                "true_redshift": float(batch["redshift"][index]),
                "visit_count": int(batch["visit_mask"][index].sum()),
                "coadded_clean_signal_to_noise": float(
                    batch["coadded_clean_signal_to_noise"][index]
                ),
                # The catalog point estimate follows the highest-density
                # posterior basin.  The full-posterior median is retained
                # below because it remains useful for diagnosing broad tails.
                "predicted_redshift": primary_peak,
                "z_strider": primary_peak,
                "redshift_point_estimator": "primary_basin_peak",
                "posterior_median_redshift": posterior_median,
                "posterior_density_mode_redshift": density_mode,
                "posterior_primary_redshift": primary_candidate["median_redshift"],
                "posterior_primary_peak_redshift": primary_candidate["peak_redshift"],
                "posterior_primary_lower_68": primary_candidate["lower_68"],
                "posterior_primary_upper_68": primary_candidate["upper_68"],
                "posterior_primary_basin_lower": primary_candidate["basin_lower"],
                "posterior_primary_basin_upper": primary_candidate["basin_upper"],
                "posterior_primary_basin_mass": primary_candidate["mass"],
                "posterior_primary_is_largest_mass_basin": primary_candidate[
                    "is_largest_mass_basin"
                ],
                "posterior_primary_competitor_peak_redshift": primary_candidate[
                    "strongest_competitor_peak_redshift"
                ],
                "posterior_primary_log_peak_to_competitor_saddle_ratio": (
                    primary_candidate[
                        "log_peak_to_strongest_competitor_saddle_ratio"
                    ]
                ),
                "posterior_primary_to_competitor_peak_height_ratio": (
                    primary_candidate["primary_to_strongest_competitor_height_ratio"]
                ),
                "posterior_primary_predicted_class": primary_class,
                "posterior_primary_predicted_class_name": model.class_names[
                    primary_class
                ],
                "posterior_primary_largest_class_probability": float(
                    primary_class_probability[primary_class].cpu()
                ),
                "posterior_dominant_peak_mass": peak_summary["dominant_mass"],
                "posterior_primary_peak_mass": primary_candidate["mass"],
                "posterior_secondary_peak_redshift": peak_summary[
                    "secondary_redshift"
                ],
                "posterior_secondary_peak_mass": peak_summary["secondary_mass"],
                "posterior_secondary_to_dominant_peak_mass_ratio": peak_summary[
                    "secondary_to_dominant_mass_ratio"
                ],
                "posterior_alternate_peak_redshift": (
                    peak_candidates[1]["peak_redshift"]
                    if len(peak_candidates) > 1
                    else float("nan")
                ),
                "posterior_alternate_basin_mass": (
                    peak_candidates[1]["mass"] if len(peak_candidates) > 1 else 0.0
                ),
                "posterior_alternate_to_primary_basin_mass_ratio": (
                    float(peak_candidates[1]["mass"])
                    / max(float(primary_candidate["mass"]), 1.0e-300)
                    if len(peak_candidates) > 1
                    else 0.0
                ),
                "posterior_distinct_peak_count": peak_summary["distinct_peak_count"],
                "posterior_candidate_count": len(peak_candidates),
                "posterior_candidate_redshifts": [
                    float(candidate["median_redshift"])
                    for candidate in peak_candidates
                ],
                "posterior_candidate_peak_redshifts": [
                    float(candidate["peak_redshift"])
                    for candidate in peak_candidates
                ],
                "posterior_candidate_masses": [
                    float(candidate["mass"]) for candidate in peak_candidates
                ],
                "posterior_candidate_is_largest_mass_basin": [
                    bool(candidate["is_largest_mass_basin"])
                    for candidate in peak_candidates
                ],
                "posterior_candidate_lower_68": [
                    float(candidate["lower_68"]) for candidate in peak_candidates
                ],
                "posterior_candidate_upper_68": [
                    float(candidate["upper_68"]) for candidate in peak_candidates
                ],
                "posterior_candidate_class_names": candidate_classes,
                "posterior_candidate_class_indices": candidate_class_indices,
                "posterior_candidate_class_probabilities": (
                    candidate_class_probabilities
                ),
                "posterior_candidate_route_support_mean": (
                    candidate_route_support_mean.tolist()
                ),
                "posterior_primary_route_support_mean": float(
                    candidate_route_support_mean[0]
                ),
                "joint_primary_redshift": joint_primary["median_redshift"],
                "joint_primary_peak_redshift": joint_primary["peak_redshift"],
                "joint_primary_lower_68": joint_primary["lower_68"],
                "joint_primary_upper_68": joint_primary["upper_68"],
                "joint_primary_predicted_class": joint_primary["class_index"],
                "joint_primary_predicted_class_name": joint_primary["class_name"],
                "joint_primary_basin_mass": joint_primary["mass"],
                "joint_primary_class_probability": joint_primary[
                    "class_probability"
                ],
                "joint_candidate_count": len(joint_candidates),
                "joint_candidate_redshifts": [
                    float(candidate["median_redshift"])
                    for candidate in joint_candidates
                ],
                "joint_candidate_peak_redshifts": [
                    float(candidate["peak_redshift"])
                    for candidate in joint_candidates
                ],
                "joint_candidate_lower_68": [
                    float(candidate["lower_68"]) for candidate in joint_candidates
                ],
                "joint_candidate_upper_68": [
                    float(candidate["upper_68"]) for candidate in joint_candidates
                ],
                "joint_candidate_masses": [
                    float(candidate["mass"]) for candidate in joint_candidates
                ],
                "joint_candidate_class_indices": [
                    int(candidate["class_index"]) for candidate in joint_candidates
                ],
                "joint_candidate_class_names": [
                    str(candidate["class_name"]) for candidate in joint_candidates
                ],
                "joint_candidate_log_peak_to_saddle_ratios": [
                    float(candidate["log_peak_to_saddle_ratio"])
                    for candidate in joint_candidates
                ],
                "posterior_median_minus_mode": posterior_median - density_mode,
                "posterior_mean_redshift": float(mean_redshift[index].cpu()),
                "predicted_redshift_given_true_class": _posterior_quantile(
                    grid, conditional_redshift_distribution, 0.5
                ),
                "true_class_redshift_lower_68": conditional_lower,
                "true_class_redshift_upper_68": conditional_upper,
                "predicted_class_given_true_redshift": int(
                    predicted_class_given_true_redshift[index].cpu()
                ),
                # The headline interval is conditional on the same primary
                # basin as z_strider.  Keep the complete-posterior interval
                # under explicit names for multimodality and calibration work.
                "redshift_lower_68": primary_candidate["lower_68"],
                "redshift_upper_68": primary_candidate["upper_68"],
                "redshift_68_interval_width": (
                    float(primary_candidate["upper_68"])
                    - float(primary_candidate["lower_68"])
                ),
                "full_posterior_lower_68": lower,
                "full_posterior_upper_68": upper,
                "full_posterior_68_interval_width": upper - lower,
                "redshift_differential_entropy_nats": differential_entropy,
                "redshift_information_gain_nats": information_gain,
                "evidence_score": float(evidence_sufficiency[index].cpu()),
                "largest_joint_probability": float(joint[index].max().cpu()),
                "largest_class_probability": float(class_probability[index].max().cpu()),
                "spectral_evidence_strength": float(spectral_strength[index].cpu()),
                "temporal_evidence_strength": float(temporal_strength[index].cpu()),
                "context_evidence_strength": float(context_strength[index].cpu()),
                "dense_scan_evidence_strength": float(dense_strength[index].cpu()),
                "dense_detail_evidence_strength": float(
                    dense_detail_strength[index].cpu()
                ),
            }
            if "median_coadded_observed_signal_to_noise" in batch:
                row["median_coadded_observed_signal_to_noise"] = float(
                    batch["median_coadded_observed_signal_to_noise"][index]
                )
                row["observed_snr_valid_wavelength_bins"] = int(
                    batch["observed_snr_valid_wavelength_bins"][index]
                )
            for route_name, support in candidate_route_support.items():
                row[f"posterior_candidate_{route_name}_relative_support"] = support
                row[f"posterior_primary_{route_name}_relative_support"] = float(
                    support[0]
                )
            if dense_overlap is not None:
                predicted_index = int(np.argmin(np.abs(grid - primary_peak)))
                true_index = int(
                    np.argmin(np.abs(grid - float(batch["redshift"][index])))
                )
                row["dense_scan_overlap_at_prediction"] = float(
                    dense_overlap[index, predicted_index].cpu()
                )
                row["dense_scan_overlap_at_true_redshift"] = float(
                    dense_overlap[index, true_index].cpu()
                )
            if phase_logits is not None:
                predicted_redshift_index = int(
                    np.argmin(np.abs(grid - posterior_median))
                )
                runtime_phase = torch.softmax(
                    phase_logits[
                        index,
                        :,
                        predicted_class[index],
                        predicted_redshift_index,
                    ],
                    dim=-1,
                )
                valid_visit = device_batch["visit_mask"][index] > 0.5
                row["mean_runtime_phase_largest_probability"] = float(
                    runtime_phase[valid_visit].max(dim=-1).values.mean().cpu()
                )
            if phase_probability is not None and phase_grid is not None:
                row.update(
                    _phase_summary(
                        phase_grid,
                        phase_probability[index].detach().cpu().numpy(),
                        batch["simulation_rest_phase_days"][index].numpy(),
                        batch["visit_mask"][index].numpy(),
                    )
                )
            for class_index, class_name in enumerate(model.class_names):
                row[f"class_probability_{class_name}"] = float(
                    class_probability[index, class_index].cpu()
                )
                row[f"primary_basin_class_probability_{class_name}"] = float(
                    primary_class_probability[class_index].cpu()
                )
            if save_redshift_probability:
                row["redshift_probability"] = redshift_distribution.astype(
                    np.float32
                ).tolist()
            rows.append(row)
    return pd.DataFrame(rows)


def _selected_joint_logits(
    outputs: dict[str, torch.Tensor], source: str
) -> torch.Tensor:
    if source == "combined":
        return outputs["joint_logits"]
    if source == "spectral":
        required = ("spectral_joint_logits",)
    elif source == "onir":
        required = ("onir_joint_logits",)
    elif source == "onir_shape":
        required = ("onir_joint_logits", "shape_joint_logits")
    elif source == "named_shape":
        required = ("shape_joint_logits",)
    elif source == "dense":
        required = ("dense_scan_joint_logits",)
    elif source == "dense_whole":
        required = ("dense_whole_joint_logits",)
    elif source == "dense_detail":
        required = ("dense_detail_joint_logits",)
    elif source == "without_dense":
        required = ("spectral_joint_logits", "dense_scan_joint_logits")
    elif source == "with_onir_masked":
        required = ("onir_joint_logits", "shape_joint_logits")
    elif source in {
        "without_onir_masked",
        "without_onir_spectral",
        "without_onir",
    }:
        required = ("shape_joint_logits",)
    elif source == "global_spectrum":
        required = ("dense_scan_joint_logits",)
    else:
        raise ValueError(
            "logit_source must be 'onir', 'named_shape', 'onir_shape', "
            "'dense', 'dense_whole', 'dense_detail', 'without_dense', "
            "'with_onir_masked', 'without_onir_masked', "
            "'without_onir_spectral', 'without_onir', 'global_spectrum', "
            "'spectral', or 'combined'"
        )
    missing = [name for name in required if name not in outputs]
    if missing:
        raise ValueError(
            f"This model does not expose {source} logits: " + ", ".join(missing)
        )
    if source == "onir_shape":
        return outputs["onir_joint_logits"] + outputs["shape_joint_logits"]
    if source == "without_dense":
        return outputs["spectral_joint_logits"] - outputs["dense_scan_joint_logits"]
    if source in {
        "with_onir_masked",
        "without_onir_masked",
        "without_onir_spectral",
        "without_onir",
    }:
        names = [
            "shape_joint_logits",
            "context_joint_logits",
            "dense_scan_joint_logits",
        ]
        if source != "without_onir_spectral":
            names.append("temporal_joint_logits")
        if source == "with_onir_masked":
            names.insert(0, "onir_joint_logits")
        logits = _sum_exposed_logits(
            outputs,
            tuple(names),
        )
        if source in {"with_onir_masked", "without_onir_masked"}:
            if "joint_support" not in outputs:
                raise ValueError(
                    f"This model does not expose {source} logits: "
                    "joint_support"
                )
            logits = logits.masked_fill(~outputs["joint_support"], -1.0e4)
        return logits
    if source == "global_spectrum":
        return _sum_exposed_logits(
            outputs,
            ("dense_scan_joint_logits", "context_joint_logits"),
        )
    return outputs[required[0]]


def _candidate_route_logits(
    outputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return independent evidence components used for candidate diagnostics.

    The combined dense route is deliberately omitted when its whole-spectrum
    and continuum-subtracted components are available, so the same evidence is
    not counted three times.  Candidate phase consistency is separated from
    the base temporal route for the same reason.
    """
    route_names = {
        "reference_coadd": "reference_coadd_joint_logits",
        "reference_sequence": "reference_sequence_joint_logits",
        "onir": "onir_joint_logits",
        "shape": "shape_joint_logits",
        "dense_whole": "dense_whole_joint_logits",
        "dense_detail": "dense_detail_joint_logits",
        "context": "context_joint_logits",
    }
    routes = {
        route: outputs[key]
        for route, key in route_names.items()
        if key in outputs
    }
    if "dense_whole" not in routes and "dense_detail" not in routes:
        dense = outputs.get("dense_scan_joint_logits")
        if dense is not None:
            routes["dense"] = dense
    temporal = (
        None
        if "reference_sequence" in routes
        else outputs.get("temporal_joint_logits")
    )
    phase = outputs.get("phase_consistency_joint_logits")
    if temporal is not None:
        routes["temporal"] = temporal - phase if phase is not None else temporal
    if phase is not None:
        routes["phase_consistency"] = phase
    if not routes:
        routes["spectral"] = outputs["spectral_joint_logits"]
    return routes


def _candidate_route_support(
    route_logits: np.ndarray,
    candidates: list[dict[str, float | int]],
    candidate_class_indices: list[int],
) -> list[float]:
    """Score how strongly one route supports each reported candidate.

    This is a diagnostic, not a calibrated probability.  A candidate receives
    high support when its best class--redshift cell is close to that route's
    own global maximum.  The score is downweighted when the route is nearly
    flat, so an uninformative component cannot appear to agree with every
    candidate.
    """
    logits = np.asarray(route_logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError("Route logits must have shape (class, redshift)")
    if len(candidates) != len(candidate_class_indices):
        raise ValueError("Candidate basins and class indices do not align")
    supported = np.isfinite(logits) & (logits > -1.0e3)
    values = logits[supported]
    if values.size == 0:
        return [0.0] * len(candidates)
    global_peak = float(values.max())
    contrast = max(global_peak - float(np.median(values)), 0.0)
    informativeness = float(-np.expm1(-contrast))
    support: list[float] = []
    for candidate, class_index in zip(candidates, candidate_class_indices):
        left = int(candidate["left_index"])
        right = int(candidate["right_index"])
        valid_class = 0 <= class_index < logits.shape[0]
        valid_basin = 0 <= left < right <= logits.shape[1]
        if not valid_class or not valid_basin:
            raise ValueError("Candidate support indices are outside the route grid")
        local_supported = supported[class_index, left:right]
        if not np.any(local_supported):
            support.append(0.0)
            continue
        local_peak = float(logits[class_index, left:right][local_supported].max())
        relative_peak = float(np.exp(np.clip(local_peak - global_peak, -50.0, 0.0)))
        support.append(informativeness * relative_peak)
    return support


def _sum_exposed_logits(
    outputs: dict[str, torch.Tensor], names: tuple[str, ...]
) -> torch.Tensor:
    """Sum available route logits while requiring the first component."""
    logits = outputs[names[0]]
    for name in names[1:]:
        component = outputs.get(name)
        if component is not None:
            logits = logits + component
    return logits


def _phase_summary(
    phase_grid: np.ndarray,
    probability: np.ndarray,
    true_phase: np.ndarray,
    visit_mask: np.ndarray,
) -> dict[str, float]:
    valid = (
        (visit_mask > 0.5)
        & (true_phase >= phase_grid[0])
        & (true_phase <= phase_grid[-1])
    )
    if not np.any(valid):
        return {
            "true_class_redshift_phase_median_delta_days": float("nan"),
            "true_class_redshift_phase_median_absolute_error_days": float("nan"),
            "true_class_redshift_phase_68_interval_coverage": float("nan"),
            "true_class_redshift_phase_order_accuracy": float("nan"),
        }
    probability = probability[valid]
    true_phase = true_phase[valid]
    predicted = np.asarray(
        [_posterior_quantile(phase_grid, row, 0.5) for row in probability]
    )
    lower = np.asarray(
        [_posterior_quantile(phase_grid, row, 0.16) for row in probability]
    )
    upper = np.asarray(
        [_posterior_quantile(phase_grid, row, 0.84) for row in probability]
    )
    delta = predicted - true_phase
    if len(predicted) > 1:
        order_accuracy = float(
            np.mean(np.sign(np.diff(predicted)) == np.sign(np.diff(true_phase)))
        )
    else:
        order_accuracy = float("nan")
    return {
        "true_class_redshift_phase_median_delta_days": float(np.median(delta)),
        "true_class_redshift_phase_median_absolute_error_days": float(
            np.median(np.abs(delta))
        ),
        "true_class_redshift_phase_68_interval_coverage": float(
            np.mean((true_phase >= lower) & (true_phase <= upper))
        ),
        "true_class_redshift_phase_order_accuracy": order_accuracy,
    }


def _ia_metric_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    def values(source: dict[str, Any]) -> dict[str, Any]:
        return {
            name: value
            for name, value in source.items()
            if name.startswith("Ia_") and np.isscalar(value)
        }

    rows = [{"redshift_min": np.nan, "redshift_max": np.nan, **values(report)}]
    for group in report.get("redshift_groups", []):
        lower, upper = group["redshift_range"]
        rows.append(
            {
                "redshift_min": lower,
                "redshift_max": upper,
                **values(group),
            }
        )
    return rows


def _class_probability_at_redshift(
    joint_probability_grid: torch.Tensor,
    true_redshift: torch.Tensor,
    redshift_grid: torch.Tensor,
    cell_width: torch.Tensor,
) -> torch.Tensor:
    """Interpolate class density at each object's true redshift."""
    batch, class_count, redshift_count = joint_probability_grid.shape
    upper = torch.searchsorted(redshift_grid, true_redshift).clamp(1, redshift_count - 1)
    lower = upper - 1
    lower_z = redshift_grid[lower]
    upper_z = redshift_grid[upper]
    weight = ((true_redshift - lower_z) / (upper_z - lower_z)).clamp(0.0, 1.0)
    density = joint_probability_grid / cell_width[None, None, :]
    lower_value = density.gather(2, lower[:, None, None].expand(batch, class_count, 1)).squeeze(2)
    upper_value = density.gather(2, upper[:, None, None].expand(batch, class_count, 1)).squeeze(2)
    probability = lower_value + weight[:, None] * (upper_value - lower_value)
    return probability / probability.sum(dim=1, keepdim=True).clamp_min(1.0e-12)


def _logit_strength(
    logits: torch.Tensor | None,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Root-mean-square centred logit amplitude for component comparisons."""
    if logits is None:
        return reference.new_zeros(reference.shape[0])
    centred = logits - logits.mean(dim=(1, 2), keepdim=True)
    return torch.sqrt(centred.square().mean(dim=(1, 2)))


def _posterior_information(
    probability: np.ndarray,
    cell_width: np.ndarray,
    prior: str,
) -> tuple[float, float]:
    """Return differential entropy and information gain in natural-log units."""
    mass = np.asarray(probability, dtype=np.float64)
    width = np.asarray(cell_width, dtype=np.float64)
    positive = mass > 0.0
    density = mass[positive] / width[positive]
    differential_entropy = float(-(mass[positive] * np.log(density)).sum())
    if prior == "flat_z":
        prior_mass = width / width.sum()
    elif prior == "flat_log1p":
        prior_mass = np.full(len(mass), 1.0 / len(mass), dtype=np.float64)
    else:
        raise ValueError("redshift_prior must be 'flat_z' or 'flat_log1p'")
    information_gain = float(
        (mass[positive] * np.log(mass[positive] / prior_mass[positive])).sum()
    )
    return differential_entropy, information_gain


def _transform_times(
    batch: dict[str, torch.Tensor],
    transform: str,
    time_reassignment: dict[int, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    if transform == "none":
        return batch
    changed = dict(batch)
    times = batch["observer_days"].clone()
    if transform == "zero":
        times.zero_()
    elif transform == "reverse_within_object":
        flux = batch["flux"].clone()
        wavelength_mask = batch["wavelength_mask"].clone()
        for index in range(len(times)):
            count = int(batch["visit_mask"][index].sum())
            flux[index, :count] = torch.flip(flux[index, :count], dims=(0,))
            wavelength_mask[index, :count] = torch.flip(
                wavelength_mask[index, :count], dims=(0,)
            )
        changed["flux"] = flux
        changed["wavelength_mask"] = wavelength_mask
    elif transform == "reassigned_within_redshift":
        if time_reassignment is None:
            raise ValueError("Date reassignment must be made across the full dataset")
        for index, snid in enumerate(batch["snid"].tolist()):
            replacement = time_reassignment.get(int(snid))
            if replacement is None:
                continue
            count = int(batch["visit_mask"][index].sum())
            times[index, :count] = replacement[:count].to(times)
    else:
        raise ValueError(f"Unsupported time transform: {transform}")
    changed["observer_days"] = times
    return changed


def _central_interval(grid: np.ndarray, probability: np.ndarray, mass: float) -> tuple[float, float]:
    tail = 0.5 * (1.0 - mass)
    return (
        _posterior_quantile(grid, probability, tail),
        _posterior_quantile(grid, probability, 1.0 - tail),
    )


def _posterior_quantile(
    grid: np.ndarray,
    probability: np.ndarray,
    quantile: float,
) -> float:
    """Interpolate a quantile through probability masses on any ordered grid."""
    mass = np.asarray(probability, dtype=np.float64)
    mass = mass / mass.sum()
    midpoint_cdf = np.cumsum(mass) - 0.5 * mass
    return float(
        np.interp(
            quantile,
            midpoint_cdf,
            np.asarray(grid, dtype=np.float64),
            left=float(grid[0]),
            right=float(grid[-1]),
        )
    )


def _interpolated_peak(grid: np.ndarray, probability: np.ndarray) -> float:
    """Refine an interior grid maximum with a local quadratic in log(1+z)."""
    grid = np.asarray(grid, dtype=np.float64)
    probability = np.asarray(probability, dtype=np.float64)
    index = int(np.argmax(probability))
    return _interpolated_peak_at_index(grid, probability, index)


def _interpolated_peak_at_index(
    grid: np.ndarray,
    probability: np.ndarray,
    index: int,
) -> float:
    """Refine one selected grid maximum without changing which peak was selected."""
    grid = np.asarray(grid, dtype=np.float64)
    probability = np.asarray(probability, dtype=np.float64)
    if index == 0 or index == len(grid) - 1:
        return float(grid[index])
    x = np.log1p(grid[index - 1:index + 2])
    y = np.log(np.clip(probability[index - 1:index + 2], 1e-300, None))
    curvature, slope, _ = np.polyfit(x, y, 2)
    if not np.isfinite(curvature) or curvature >= 0.0:
        return float(grid[index])
    peak = float(np.clip(-slope / (2.0 * curvature), x[0], x[-1]))
    return float(np.expm1(peak))


def _posterior_peak_summary(
    grid: np.ndarray,
    probability_mass: np.ndarray,
    cell_width: np.ndarray,
    *,
    smoothing_sigma_bins: float = 2.0,
    minimum_peak_height_ratio: float = 0.10,
    minimum_peak_mass: float = 0.05,
) -> dict[str, float | int]:
    """Describe dominant and competing modes without altering the posterior.

    Peak locations are selected from posterior density, not grid-cell mass. A
    small Gaussian smoothing is used only to identify distinct peak basins; all
    reported basin masses are integrated from the original posterior.
    """
    grid = np.asarray(grid, dtype=np.float64)
    mass = np.asarray(probability_mass, dtype=np.float64)
    width = np.asarray(cell_width, dtype=np.float64)
    if grid.ndim != 1 or mass.shape != grid.shape or width.shape != grid.shape:
        raise ValueError("Posterior peak inputs must be one-dimensional and aligned")
    if np.any(width <= 0.0) or not np.isfinite(width).all():
        raise ValueError("Posterior cell widths must be finite and positive")
    total = float(mass.sum())
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("Posterior probability must have positive finite mass")
    mass = mass / total
    density = mass / width
    smoothed = _gaussian_smooth_1d(density, smoothing_sigma_bins)
    peak_indices = _local_peak_indices(smoothed)
    dominant_topology_index = int(np.argmax(smoothed))
    if dominant_topology_index not in peak_indices:
        peak_indices = np.sort(
            np.append(peak_indices, dominant_topology_index)
        ).astype(int)

    boundaries = [0]
    for left, right in zip(peak_indices[:-1], peak_indices[1:]):
        valley = int(left + np.argmin(smoothed[left : right + 1]))
        boundaries.append(valley + 1)
    boundaries.append(len(grid))
    basin_mass = np.asarray(
        [
            mass[lower:upper].sum()
            for lower, upper in zip(boundaries[:-1], boundaries[1:])
        ],
        dtype=np.float64,
    )
    peak_height = smoothed[peak_indices]
    dominant_position = int(np.argmax(peak_height))
    dominant_mass = float(basin_mass[dominant_position])
    dominant_height = max(float(peak_height[dominant_position]), 1e-300)

    qualifying = [
        position
        for position in range(len(peak_indices))
        if position != dominant_position
        and peak_height[position] / dominant_height >= minimum_peak_height_ratio
        and basin_mass[position] >= minimum_peak_mass
    ]
    if qualifying:
        secondary_position = max(
            qualifying,
            key=lambda position: (peak_height[position], basin_mass[position]),
        )
        secondary_index = int(peak_indices[secondary_position])
        secondary_redshift = _interpolated_peak_at_index(
            grid,
            smoothed,
            secondary_index,
        )
        secondary_mass = float(basin_mass[secondary_position])
        secondary_ratio = secondary_mass / max(dominant_mass, 1e-300)
    else:
        secondary_redshift = float("nan")
        secondary_mass = 0.0
        secondary_ratio = 0.0

    distinct_count = 1 + len(qualifying)
    return {
        "dominant_redshift": _interpolated_peak(grid, density),
        "dominant_mass": dominant_mass,
        "secondary_redshift": secondary_redshift,
        "secondary_mass": secondary_mass,
        "secondary_to_dominant_mass_ratio": float(secondary_ratio),
        "distinct_peak_count": int(distinct_count),
    }


def _posterior_basin_candidates(
    grid: np.ndarray,
    probability_mass: np.ndarray,
    cell_width: np.ndarray,
    *,
    maximum_candidates: int = 3,
    smoothing_sigma_bins: float = 2.0,
    minimum_peak_height_ratio: float = 0.10,
    minimum_peak_mass: float = 0.05,
) -> list[dict[str, float | int | bool]]:
    """Return the strongest distinct posterior basins without merging them.

    Candidate discovery uses a lightly smoothed posterior density so unequal
    grid-cell widths and single-cell noise do not choose the topology. Reported
    masses, quantiles and intervals always use the original posterior mass.
    The first entry is the basin containing the highest posterior-density peak.
    """
    if maximum_candidates < 1:
        raise ValueError("maximum_candidates must be positive")
    grid = np.asarray(grid, dtype=np.float64)
    mass = np.asarray(probability_mass, dtype=np.float64)
    width = np.asarray(cell_width, dtype=np.float64)
    if grid.ndim != 1 or mass.shape != grid.shape or width.shape != grid.shape:
        raise ValueError("Posterior candidate inputs must be one-dimensional and aligned")
    if np.any(width <= 0.0) or not np.isfinite(width).all():
        raise ValueError("Posterior cell widths must be finite and positive")
    total = float(mass.sum())
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("Posterior probability must have positive finite mass")
    mass = mass / total
    density = mass / width
    smoothed = _gaussian_smooth_1d(density, smoothing_sigma_bins)
    peak_indices = _local_peak_indices(smoothed)
    dominant_index = int(np.argmax(smoothed))
    if dominant_index not in peak_indices:
        peak_indices = np.sort(np.append(peak_indices, dominant_index)).astype(int)

    boundaries = [0]
    for left, right in zip(peak_indices[:-1], peak_indices[1:]):
        valley = int(left + np.argmin(smoothed[left : right + 1]))
        boundaries.append(valley + 1)
    boundaries.append(len(grid))
    basin_mass = np.asarray(
        [
            mass[left:right].sum()
            for left, right in zip(boundaries[:-1], boundaries[1:])
        ],
        dtype=np.float64,
    )
    peak_height = smoothed[peak_indices]
    log_prominence = _log_peak_to_saddle_ratios(smoothed, peak_indices)
    dominant_position = int(np.argmax(peak_height))
    dominant_height = max(float(peak_height[dominant_position]), 1.0e-300)
    qualifying = [dominant_position]
    qualifying.extend(
        position
        for position in range(len(peak_indices))
        if position != dominant_position
        and peak_height[position] / dominant_height >= minimum_peak_height_ratio
        and basin_mass[position] >= minimum_peak_mass
    )
    secondary = sorted(
        qualifying[1:],
        key=lambda position: (peak_height[position], basin_mass[position]),
        reverse=True,
    )
    ordered = [dominant_position, *secondary][:maximum_candidates]
    largest_mass_position = int(np.argmax(basin_mass))

    strongest_competitor_position = secondary[0] if secondary else None
    if strongest_competitor_position is None:
        strongest_competitor_redshift = float("nan")
        primary_competitor_saddle_contrast = float("nan")
        primary_to_competitor_height_ratio = float("nan")
    else:
        primary_index = int(peak_indices[dominant_position])
        competitor_index = int(peak_indices[strongest_competitor_position])
        lower_peak = min(primary_index, competitor_index)
        upper_peak = max(primary_index, competitor_index)
        connecting_saddle = float(smoothed[lower_peak : upper_peak + 1].min())
        primary_competitor_saddle_contrast = float(
            np.log(dominant_height)
            - np.log(max(connecting_saddle, 1.0e-300))
        )
        primary_to_competitor_height_ratio = float(
            dominant_height
            / max(float(peak_height[strongest_competitor_position]), 1.0e-300)
        )
        strongest_competitor_redshift = _interpolated_peak_at_index(
            grid,
            smoothed,
            competitor_index,
        )

    candidates: list[dict[str, float | int | bool]] = []
    for position in ordered:
        left = int(boundaries[position])
        right = int(boundaries[position + 1])
        conditional_mass = mass[left:right]
        conditional_mass = conditional_mass / conditional_mass.sum()
        local_grid = grid[left:right]
        lower_68, upper_68 = _central_interval(local_grid, conditional_mass, 0.68)
        candidates.append(
            {
                "peak_redshift": _interpolated_peak_at_index(
                    grid,
                    smoothed,
                    int(peak_indices[position]),
                ),
                "median_redshift": _posterior_quantile(
                    local_grid,
                    conditional_mass,
                    0.5,
                ),
                "lower_68": lower_68,
                "upper_68": upper_68,
                "mass": float(basin_mass[position]),
                "is_largest_mass_basin": bool(position == largest_mass_position),
                "height_ratio": float(peak_height[position] / dominant_height),
                "peak_density": float(peak_height[position]),
                "log_peak_to_saddle_ratio": float(log_prominence[position]),
                "basin_lower": float(local_grid[0]),
                "basin_upper": float(local_grid[-1]),
                "left_index": left,
                "right_index": right,
            }
        )
    candidates[0].update(
        {
            "strongest_competitor_peak_redshift": float(
                strongest_competitor_redshift
            ),
            "log_peak_to_strongest_competitor_saddle_ratio": float(
                primary_competitor_saddle_contrast
            ),
            "primary_to_strongest_competitor_height_ratio": float(
                primary_to_competitor_height_ratio
            ),
        }
    )
    return candidates


def _joint_posterior_basin_candidates(
    grid: np.ndarray,
    joint_probability_mass: np.ndarray,
    cell_width: np.ndarray,
    class_names: list[str] | tuple[str, ...],
    *,
    maximum_candidates: int = 3,
    maximum_candidates_per_class: int = 3,
    smoothing_sigma_bins: float = 2.0,
    minimum_peak_height_ratio: float = 0.10,
    minimum_joint_basin_mass: float = 0.02,
) -> list[dict[str, float | int | str]]:
    """Decompose the joint class--redshift posterior without inventing a class axis.

    Redshift basins are found independently within each physical class.  They
    are then ranked together by their absolute joint posterior density.  This
    preserves hypotheses that occupy the same redshift basin but correspond to
    different classes, while avoiding an arbitrary adjacency between nominal
    class labels.
    """
    if maximum_candidates < 1 or maximum_candidates_per_class < 1:
        raise ValueError("Joint posterior candidate limits must be positive")
    grid = np.asarray(grid, dtype=np.float64)
    joint = np.asarray(joint_probability_mass, dtype=np.float64)
    width = np.asarray(cell_width, dtype=np.float64)
    if joint.ndim != 2 or joint.shape[1] != len(grid):
        raise ValueError("Joint posterior must have shape (class, redshift)")
    if joint.shape[0] != len(class_names):
        raise ValueError("Joint posterior classes and class names do not align")
    if not np.isfinite(joint).all() or np.any(joint < 0.0):
        raise ValueError("Joint posterior probability must be finite and non-negative")
    total = float(joint.sum())
    if total <= 0.0:
        raise ValueError("Joint posterior probability must have positive mass")
    joint = joint / total

    candidates: list[dict[str, float | int | str]] = []
    class_probability = joint.sum(axis=1)
    for class_index, class_mass in enumerate(class_probability):
        if class_mass <= 0.0:
            continue
        conditional = joint[class_index] / class_mass
        class_candidates = _posterior_basin_candidates(
            grid,
            conditional,
            width,
            maximum_candidates=maximum_candidates_per_class,
            smoothing_sigma_bins=smoothing_sigma_bins,
            minimum_peak_height_ratio=0.0,
            minimum_peak_mass=0.0,
        )
        for candidate in class_candidates:
            enriched: dict[str, float | int | str] = dict(candidate)
            enriched.update(
                {
                    "class_index": int(class_index),
                    "class_name": str(class_names[class_index]),
                    "class_probability": float(class_mass),
                    "mass": float(class_mass * float(candidate["mass"])),
                    "peak_density": float(
                        class_mass * float(candidate["peak_density"])
                    ),
                }
            )
            candidates.append(enriched)

    candidates.sort(
        key=lambda candidate: (
            float(candidate["peak_density"]),
            float(candidate["mass"]),
        ),
        reverse=True,
    )
    dominant_density = max(float(candidates[0]["peak_density"]), 1.0e-300)
    selected = [candidates[0]]
    selected.extend(
        candidate
        for candidate in candidates[1:]
        if float(candidate["peak_density"]) / dominant_density
        >= minimum_peak_height_ratio
        and float(candidate["mass"]) >= minimum_joint_basin_mass
    )
    return selected[:maximum_candidates]


def _log_peak_to_saddle_ratios(
    density: np.ndarray,
    peak_indices: np.ndarray,
) -> np.ndarray:
    """Return 1D superlevel-set prominence for every peak.

    A secondary peak dies at the highest valley connecting it to any higher
    peak.  The dominant peak has no finite death saddle and is reported as
    infinity.  This is a topology diagnostic, not a calibrated probability.
    """
    values = np.asarray(density, dtype=np.float64)
    peaks = np.asarray(peak_indices, dtype=np.int64)
    heights = values[peaks]
    result = np.full(len(peaks), np.inf, dtype=np.float64)
    for position, peak in enumerate(peaks):
        higher = np.flatnonzero(heights > heights[position])
        if not len(higher):
            continue
        saddle = max(
            float(values[min(peak, peaks[other]) : max(peak, peaks[other]) + 1].min())
            for other in higher
        )
        result[position] = float(
            np.log(max(float(heights[position]), 1.0e-300))
            - np.log(max(saddle, 1.0e-300))
        )
    return result


def _gaussian_smooth_1d(values: np.ndarray, sigma_bins: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if sigma_bins <= 0.0:
        return values.copy()
    radius = max(1, int(np.ceil(4.0 * sigma_bins)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(offsets / sigma_bins))
    kernel /= kernel.sum()
    padded = np.pad(values, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _local_peak_indices(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 1:
        return np.asarray([0], dtype=int)
    interior = np.flatnonzero(
        (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])
    ) + 1
    edges = []
    if values[0] > values[1]:
        edges.append(0)
    if values[-1] > values[-2]:
        edges.append(len(values) - 1)
    peaks = np.asarray([*edges[:1], *interior, *edges[1:]], dtype=int)
    if peaks.size == 0:
        peaks = np.asarray([int(np.argmax(values))], dtype=int)
    return np.unique(peaks)
