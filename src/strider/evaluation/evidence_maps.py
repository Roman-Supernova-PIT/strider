"""Write per-object spectra, class-redshift and ONIR-region evidence figures.

Inputs are one prepared evaluation object and one trained STRIDER model. The
figure uses observer-frame wavelength and days since the first spectrum.
Simulation labels are drawn only as evaluation references, never model inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from strider.atlas import load_onir_bank
from strider.config import project_path
from strider.data.dataset import SundialDataset, collate_objects
from strider.model import measurement_inputs
from strider.model.posterior import joint_probability

from .checkpoint import load_trained_model
from .evaluate import (
    _central_interval,
    _posterior_basin_candidates,
    _posterior_peak_summary,
    _posterior_quantile,
)


PREDICTED_COLOR = "#007c91"
TRUTH_COLOR = "#2ca25f"
SECONDARY_COLOR = "#9aa0a6"


ROUTE_LOGIT_KEYS = (
    ("Coadded spectrum", "reference_coadd_joint_logits"),
    ("Spectral evolution", "reference_sequence_joint_logits"),
    ("ONIR profiles", "onir_joint_logits"),
    ("Named-feature shape", "shape_joint_logits"),
    ("Whole spectrum", "dense_whole_contribution"),
    ("Continuum-subtracted", "dense_detail_contribution"),
    ("Context", "context_joint_logits"),
    ("Temporal", "temporal_joint_logits"),
)


def evidence_grade(
    score: float, thresholds: tuple[float, float, float] = (0.25, 0.5, 0.75)
) -> str:
    """Return a concise evidence grade using calibrated score boundaries."""
    if len(thresholds) != 3 or not all(
        lower < upper for lower, upper in zip(thresholds, thresholds[1:])
    ):
        raise ValueError("Evidence grade thresholds must contain three increasing values")
    return ("LIMITED", "LOW", "MEDIUM", "HIGH")[
        sum(score >= threshold for threshold in thresholds)
    ]


@torch.no_grad()
def write_evidence_maps(
    config: dict[str, Any],
    *,
    split: str | None = None,
    view: str | None = None,
    object_list: str | Path | None = None,
    objects_per_redshift: int | None = None,
    competing_peak_ratio: float | None = None,
    layout: str | None = None,
) -> dict[str, Any]:
    """Render representative objects near the configured redshifts."""
    import matplotlib.pyplot as plt

    output_dir = project_path(config, config["project"]["output_dir"])
    layout = str(layout or config["evaluation"].get("evidence_map_layout", "summary"))
    if layout not in {"summary", "diagnostic"}:
        raise ValueError("evidence_map_layout must be 'summary' or 'diagnostic'")
    if object_list is None:
        object_list = config["evaluation"].get("evidence_map_object_list")
    figure_dir = output_dir / ("alias_audit" if object_list else "evidence_maps")
    figure_dir.mkdir(parents=True, exist_ok=True)
    model, _, device = load_trained_model(config)
    split = str(split or config["evaluation"].get("split", "calibration"))
    view = str(view or config["evaluation"].get("evidence_map_view", "original"))
    targets = [
        float(value)
        for value in config["evaluation"].get(
            "evidence_map_redshifts", [0.75, 1.5, 2.5]
        )
    ]
    configured_count = int(
        config["evaluation"].get("evidence_map_objects_per_redshift", 4)
    )
    count = int(
        configured_count if objects_per_redshift is None else objects_per_redshift
    )
    if configured_count < 1 or count < 1:
        raise ValueError("The evidence-map object count must be positive")
    configured_ia_count = int(
        config["evaluation"].get("ia_examples_per_redshift", 0)
    )
    ia_count = (
        configured_ia_count
        if objects_per_redshift is None
        else int(round(configured_ia_count * count / configured_count))
    )
    ia_count = min(ia_count, count)
    if not 0 <= ia_count <= count:
        raise ValueError("ia_examples_per_redshift must lie between zero and the example count")
    grade_thresholds = tuple(
        float(value)
        for value in config["evaluation"].get(
            "evidence_grade_thresholds", [0.25, 0.5, 0.75]
        )
    )
    if any(value < 0.0 or value > 1.0 for value in grade_thresholds):
        raise ValueError("Evidence grade thresholds must lie between zero and one")
    competing_peak_ratio = float(
        competing_peak_ratio
        if competing_peak_ratio is not None
        else config["evaluation"].get("competing_peak_mass_ratio", 0.25)
    )
    if not 0.0 <= competing_peak_ratio <= 1.0:
        raise ValueError("competing_peak_mass_ratio must lie between zero and one")
    class_names = list(model.class_names)
    ia_index = class_names.index("Ia") if ia_count else None
    dataset = SundialDataset(config, split, view, training=False)
    bank = (
        load_onir_bank(project_path(config, config["onir"]["bank_path"]))
        if layout == "diagnostic"
        else None
    )
    if object_list:
        manifest = pd.read_csv(project_path(config, object_list))
        selected_objects = _manifest_indices(dataset.objects, manifest)
    else:
        selected_objects = [
            (f"z{target:.2f}", index)
            for target, index in _representative_indices(
                dataset.objects,
                targets,
                count,
                preferred_class_index=ia_index,
                preferred_count=ia_count,
            )
        ]
    paths = []
    audit_rows = []
    for cohort, index in selected_objects:
        item = dataset[index]
        if layout == "summary":
            _, epoch_snr_records = dataset.observed_signal_to_noise_records(
                index,
                include_epoch_history=True,
            )
            if len(epoch_snr_records) == len(item["flux"]):
                item = {
                    **item,
                    "epoch_observed_signal_to_noise": torch.tensor(
                        [
                            float(record["median_epoch_observed_signal_to_noise"])
                            for record in epoch_snr_records
                        ],
                        dtype=torch.float32,
                    ),
                }
        batch = {
            name: value.to(device)
            for name, value in collate_objects([item]).items()
        }
        output = model(measurement_inputs(batch))
        if layout == "diagnostic" and (
            "feature_evidence" not in output or "feature_evidence_support" not in output
        ):
            raise ValueError("Evidence maps require an ONIR model with region support output")
        joint = joint_probability(
            output["joint_logits"], model.redshift_cell_width, model.redshift_prior
        )[0].cpu().numpy()  # (C,Z)
        predicted_class = int(joint.sum(axis=1).argmax())
        redshift_probability = joint.sum(axis=0)
        redshift_grid = model.redshift_grid.cpu().numpy()
        redshift_cell_width = model.redshift_cell_width.cpu().numpy()
        peak_summary = _posterior_peak_summary(
            redshift_grid,
            redshift_probability,
            redshift_cell_width,
        )
        posterior_median_redshift = _posterior_quantile(
            redshift_grid, redshift_probability, 0.5
        )
        basin_candidates = _posterior_basin_candidates(
            redshift_grid,
            redshift_probability,
            redshift_cell_width,
        )
        primary_basin = basin_candidates[0]
        predicted_redshift = float(primary_basin["peak_redshift"])
        alternate_basin = basin_candidates[1] if len(basin_candidates) > 1 else None
        audit_peak_summary = {
            **peak_summary,
            "dominant_redshift": float(primary_basin["peak_redshift"]),
            "dominant_mass": float(primary_basin["mass"]),
            "secondary_redshift": (
                float(alternate_basin["peak_redshift"])
                if alternate_basin is not None
                else float("nan")
            ),
            "secondary_mass": (
                float(alternate_basin["mass"])
                if alternate_basin is not None
                else 0.0
            ),
            "secondary_to_dominant_mass_ratio": (
                float(alternate_basin["mass"])
                / max(float(primary_basin["mass"]), 1.0e-300)
                if alternate_basin is not None
                else 0.0
            ),
        }
        route_evidence = _route_evidence(output, predicted_class)
        audit_rows.extend(
            _route_audit_rows(
                cohort=cohort,
                item=item,
                class_name=class_names[predicted_class],
                predicted_class_probability=float(joint.sum(axis=1)[predicted_class]),
                redshift_grid=redshift_grid,
                redshift_probability=redshift_probability,
                redshift_cell_width=redshift_cell_width,
                peak_summary=audit_peak_summary,
                route_evidence=route_evidence,
            )
        )
        common_figure_arguments = {
            "item": item,
            "wavelength_angstrom": dataset.output_wavelength,
            "class_names": class_names,
            "redshift_grid": redshift_grid,
            "redshift_cell_width": redshift_cell_width,
            "joint_probability_mass": joint,
            "predicted_class": predicted_class,
            "predicted_redshift": predicted_redshift,
            "posterior_median_redshift": posterior_median_redshift,
            "evidence_sufficiency": float(
                torch.sigmoid(output["evidence_sufficiency_logit"])[0].cpu()
            ),
            "evidence_grade_thresholds": grade_thresholds,
            "competing_peak_mass_ratio": competing_peak_ratio,
            "split": split,
            "view": view,
        }
        if layout == "summary":
            figure = plt.figure(figsize=(11.8, 7.6), constrained_layout=True)
            draw_evidence_summary(figure, **common_figure_arguments)
        else:
            assert bank is not None
            figure = plt.figure(figsize=(13.0, 15.5), constrained_layout=True)
            draw_evidence_map(
                figure,
                feature_names=list(bank.feature_names),
                feature_evidence=output["feature_evidence"][
                    0, :, predicted_class
                ].cpu().numpy(),
                feature_support=output["feature_evidence_support"][
                    0, :, predicted_class
                ].cpu().numpy(),
                route_evidence=route_evidence,
                route_support=output["joint_support"][
                    0, predicted_class
                ].cpu().numpy(),
                **common_figure_arguments,
            )
        true_class = class_names[int(item["class_index"])]
        safe_cohort = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in cohort
        ).strip("_")
        layout_suffix = "_summary" if layout == "summary" else "_diagnostic"
        path = figure_dir / (
            f"{split}_{view}_{safe_cohort}_{true_class}_snid"
            f"{int(item['snid'])}{layout_suffix}.png"
        )
        figure.savefig(path, dpi=170, facecolor="white")
        plt.close(figure)
        paths.append(str(path))
    audit_path = figure_dir / f"{split}_{view}_route_audit.csv"
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False, float_format="%.6g")
    return {
        "device": str(device),
        "split": split,
        "view": view,
        "layout": layout,
        "figures": paths,
        "route_audit": str(audit_path),
    }


def draw_evidence_summary(
    figure,
    *,
    item: dict[str, torch.Tensor],
    wavelength_angstrom: np.ndarray,
    class_names: list[str],
    redshift_grid: np.ndarray,
    redshift_cell_width: np.ndarray,
    joint_probability_mass: np.ndarray,
    predicted_class: int,
    predicted_redshift: float,
    posterior_median_redshift: float | None = None,
    evidence_sufficiency: float,
    evidence_grade_thresholds: tuple[float, float, float] = (0.25, 0.5, 0.75),
    competing_peak_mass_ratio: float = 0.25,
    split: str,
    view: str,
    spectrum_selection: str = "best",
) -> None:
    """Draw the concise, public-facing STRIDER evidence summary."""
    import cmasher as cmr
    from matplotlib.colors import LogNorm

    del (
        predicted_redshift,
        posterior_median_redshift,
        evidence_sufficiency,
        evidence_grade_thresholds,
        split,
        view,
    )
    outer_grid = figure.add_gridspec(
        2,
        1,
        height_ratios=(0.78, 1.84),
        hspace=0.12,
    )
    evidence_grid = outer_grid[1].subgridspec(
        2,
        2,
        height_ratios=(0.94, 0.82),
        width_ratios=(5.15, 1.35),
        hspace=0.025,
        wspace=0.07,
    )
    spectra_axis = figure.add_subplot(outer_grid[0])
    joint_axis = figure.add_subplot(evidence_grid[0, 0])
    redshift_axis = figure.add_subplot(evidence_grid[1, 0], sharex=joint_axis)
    class_axis = figure.add_subplot(evidence_grid[0, 1], sharey=joint_axis)
    solution_axis = figure.add_subplot(evidence_grid[1, 1])

    _draw_spectra_summary(
        spectra_axis,
        item,
        wavelength_angstrom,
        selection=spectrum_selection,
    )

    edges = _cell_edges(redshift_grid)
    density = joint_probability_mass / redshift_cell_width[None, :]
    redshift_probability = joint_probability_mass.sum(axis=0)
    candidates = _posterior_basin_candidates(
        redshift_grid,
        redshift_probability,
        redshift_cell_width,
    )
    primary = candidates[0]
    secondary = candidates[1] if len(candidates) > 1 else None
    primary_redshift = float(primary["peak_redshift"])
    secondary_redshift = (
        float(secondary["peak_redshift"])
        if secondary is not None
        else float("nan")
    )
    secondary_ratio = (
        float(secondary["mass"]) / max(float(primary["mass"]), 1.0e-300)
        if secondary is not None
        else 0.0
    )
    show_secondary = np.isfinite(secondary_redshift) and (
        secondary_ratio >= competing_peak_mass_ratio
    )

    relative_density = density / max(float(np.nanmax(density)), 1.0e-30)
    joint_axis.pcolormesh(
        edges,
        np.arange(len(class_names) + 1) - 0.5,
        np.clip(relative_density, 1.0e-3, 1.0),
        cmap=cmr.ember,
        norm=LogNorm(vmin=1.0e-3, vmax=1.0),
        shading="flat",
    )
    joint_axis.set_yticks(np.arange(len(class_names)))
    joint_axis.set_yticklabels(class_names)
    joint_axis.set_ylabel("class")
    joint_axis.tick_params(labelbottom=False)
    joint_axis.set_xlim(float(redshift_grid[0]), float(redshift_grid[-1]))
    joint_axis.scatter(
        [primary_redshift],
        [predicted_class],
        marker="D",
        color=PREDICTED_COLOR,
        s=52,
        label="STRIDER",
        zorder=4,
    )
    if show_secondary:
        joint_axis.scatter(
            [secondary_redshift],
            [predicted_class],
            marker="^",
            facecolors="none",
            edgecolors=SECONDARY_COLOR,
            linewidths=1.4,
            s=55,
            label="alternate",
            zorder=4,
        )
    joint_axis.scatter(
        [float(item["redshift"])],
        [int(item["class_index"])],
        marker="o",
        facecolors="none",
        edgecolors=TRUTH_COLOR,
        linewidths=1.8,
        s=70,
        label="truth",
        zorder=4,
    )
    joint_axis.legend(
        frameon=False,
        loc="upper right",
        fontsize=9.5,
        ncol=3 if show_secondary else 2,
        labelcolor="white",
        handletextpad=0.4,
        columnspacing=0.9,
    )

    class_probability = joint_probability_mass.sum(axis=1)
    bar_colors = [
        PREDICTED_COLOR if index == predicted_class else SECONDARY_COLOR
        for index in range(len(class_names))
    ]
    class_rows = np.arange(len(class_names))
    class_axis.barh(
        class_rows,
        np.ones(len(class_names)),
        color="#E8ECEF",
        height=0.36,
        zorder=0,
    )
    class_axis.barh(
        class_rows,
        class_probability,
        color=bar_colors,
        height=0.36,
        zorder=1,
    )
    class_axis.set_yticks(np.arange(len(class_names)))
    class_axis.set_yticklabels(class_names)
    class_axis.tick_params(axis="y", labelleft=True, left=False, pad=3)
    class_axis.set_xlim(0.0, 1.03)
    class_axis.set_xticks((0.0, 0.5, 1.0))
    class_axis.set_xlabel(r"$P(\mathrm{class})$")
    class_labels = [
        f"{probability:.0%}" if probability >= 0.005 else "<1%"
        for probability in class_probability
    ]
    for row, probability, label, color in zip(
        class_rows,
        class_probability,
        class_labels,
        bar_colors,
        strict=True,
    ):
        inside = probability >= 0.20
        class_axis.text(
            float(probability - 0.025 if inside else probability + 0.025),
            row,
            label,
            ha="right" if inside else "left",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white" if inside and color == PREDICTED_COLOR else "0.18",
            zorder=2,
        )

    redshift_density = density.sum(axis=0)
    redshift_axis.fill_between(
        redshift_grid,
        redshift_density,
        color=PREDICTED_COLOR,
        alpha=0.16,
        linewidth=0,
    )
    redshift_axis.plot(
        redshift_grid,
        redshift_density,
        color=PREDICTED_COLOR,
        lw=1.8,
    )
    redshift_axis.axvspan(
        float(primary["lower_68"]),
        float(primary["upper_68"]),
        color=PREDICTED_COLOR,
        alpha=0.12,
    )
    redshift_axis.axvline(primary_redshift, color=PREDICTED_COLOR, lw=1.2)
    if show_secondary:
        redshift_axis.axvline(
            secondary_redshift,
            color=SECONDARY_COLOR,
            ls=":",
            lw=1.1,
        )
    redshift_axis.axvline(
        float(item["redshift"]),
        color=TRUTH_COLOR,
        lw=1.1,
    )
    redshift_axis.set_xlim(float(redshift_grid[0]), float(redshift_grid[-1]))
    redshift_axis.set_ylim(
        0.0,
        1.05 * max(float(np.nanmax(redshift_density)), 1.0e-30),
    )
    redshift_axis.set_xlabel("redshift")
    redshift_axis.set_ylabel("posterior density")

    solution_axis.axis("off")
    solution_axis.set_xlim(0.0, 1.0)
    solution_axis.set_ylim(0.0, 1.0)
    primary_row = 0.84
    solution_axis.plot(
        [0.0, 0.12],
        [primary_row, primary_row],
        transform=solution_axis.transAxes,
        color=PREDICTED_COLOR,
        lw=1.8,
        clip_on=False,
    )
    solution_axis.text(
        0.17,
        primary_row,
        rf"STRIDER  $z={primary_redshift:.3f}$",
        transform=solution_axis.transAxes,
        ha="left",
        va="center",
        fontsize=10.5,
        color="0.16",
    )
    solution_axis.text(
        0.17,
        0.66,
        f"68% interval  {float(primary['lower_68']):.3f}–"
        f"{float(primary['upper_68']):.3f}",
        transform=solution_axis.transAxes,
        ha="left",
        va="center",
        fontsize=9,
        color="0.38",
    )
    if show_secondary:
        alternate_row = 0.39
        solution_axis.plot(
            [0.0, 0.12],
            [alternate_row, alternate_row],
            transform=solution_axis.transAxes,
            color=SECONDARY_COLOR,
            lw=1.3,
            ls=":",
            clip_on=False,
        )
        solution_axis.text(
            0.17,
            alternate_row,
            rf"alternate  $z={secondary_redshift:.3f}$",
            transform=solution_axis.transAxes,
            ha="left",
            va="center",
            fontsize=10,
            color="0.28",
        )
    truth_row = 0.16 if show_secondary else 0.34
    solution_axis.plot(
        [0.0, 0.12],
        [truth_row, truth_row],
        transform=solution_axis.transAxes,
        color=TRUTH_COLOR,
        lw=1.3,
        clip_on=False,
    )
    solution_axis.text(
        0.17,
        truth_row,
        rf"truth  $z={float(item['redshift']):.3f}$",
        transform=solution_axis.transAxes,
        ha="left",
        va="center",
        fontsize=10,
        color="0.28",
    )

    predicted_name = class_names[predicted_class]
    true_name = class_names[int(item["class_index"])]
    figure.suptitle(
        f"Truth: {true_name}, z={float(item['redshift']):.3f}\n"
        f"STRIDER: {predicted_name}, z={primary_redshift:.3f}",
        fontsize=15,
        fontweight="bold",
        linespacing=1.05,
    )
    figure.text(
        0.055,
        0.982,
        f"SNID {int(item['snid'])}",
        ha="left",
        va="top",
        fontsize=11,
        color="0.38",
    )
    for axis in (spectra_axis, joint_axis, redshift_axis, class_axis):
        axis.tick_params(axis="both", labelsize=10.5)
        axis.xaxis.label.set_size(12)
        axis.yaxis.label.set_size(12)


def draw_evidence_map(
    figure,
    *,
    item: dict[str, torch.Tensor],
    wavelength_angstrom: np.ndarray,
    class_names: list[str],
    feature_names: list[str],
    redshift_grid: np.ndarray,
    redshift_cell_width: np.ndarray,
    joint_probability_mass: np.ndarray,
    feature_evidence: np.ndarray,
    feature_support: np.ndarray,
    route_evidence: dict[str, np.ndarray] | None = None,
    route_support: np.ndarray | None = None,
    predicted_class: int,
    predicted_redshift: float,
    posterior_median_redshift: float | None = None,
    evidence_sufficiency: float,
    evidence_grade_thresholds: tuple[float, float, float] = (0.25, 0.5, 0.75),
    competing_peak_mass_ratio: float = 0.25,
    split: str,
    view: str,
) -> None:
    """Draw one evidence map into an existing Matplotlib figure."""
    import cmasher as cmr
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    show_routes = bool(route_evidence)
    height_ratios = (0.9, 1.25, 0.72, 1.15, 0.75) if show_routes else (0.9, 1.25, 1.15, 0.75)
    grid = figure.add_gridspec(len(height_ratios), 2, height_ratios=height_ratios)
    spectra_axis = figure.add_subplot(grid[0, :])
    joint_axis = figure.add_subplot(grid[1, :])
    route_axis = figure.add_subplot(grid[2, :]) if show_routes else None
    feature_row = 3 if show_routes else 2
    marginal_row = 4 if show_routes else 3
    feature_axis = figure.add_subplot(grid[feature_row, :])
    class_axis = figure.add_subplot(grid[marginal_row, 0])
    redshift_axis = figure.add_subplot(grid[marginal_row, 1])

    _draw_spectra(spectra_axis, item, wavelength_angstrom)
    edges = _cell_edges(redshift_grid)
    density = joint_probability_mass / redshift_cell_width[None, :]
    redshift_probability = joint_probability_mass.sum(axis=0)
    basin_candidates = _posterior_basin_candidates(
        redshift_grid,
        redshift_probability,
        redshift_cell_width,
    )
    primary_basin = basin_candidates[0]
    secondary_basin = basin_candidates[1] if len(basin_candidates) > 1 else None
    peak_redshift = float(primary_basin["peak_redshift"])
    # Recompute the released point estimate from the plotted posterior so a
    # standalone figure can never display a marker from a different summary.
    predicted_redshift = peak_redshift
    posterior_median_redshift = (
        float(posterior_median_redshift)
        if posterior_median_redshift is not None
        else _posterior_quantile(redshift_grid, redshift_probability, 0.5)
    )
    secondary_redshift = (
        float(secondary_basin["peak_redshift"])
        if secondary_basin is not None
        else float("nan")
    )
    secondary_ratio = (
        float(secondary_basin["mass"])
        / max(float(primary_basin["mass"]), 1.0e-300)
        if secondary_basin is not None
        else 0.0
    )
    show_secondary = np.isfinite(secondary_redshift) and (
        secondary_ratio >= competing_peak_mass_ratio
    )
    full_lower_68, full_upper_68 = _central_interval(
        redshift_grid,
        redshift_probability,
        0.68,
    )
    primary_lower_68 = float(primary_basin["lower_68"])
    primary_upper_68 = float(primary_basin["upper_68"])
    relative_density = density / max(float(np.nanmax(density)), 1e-30)
    image = joint_axis.pcolormesh(
        edges,
        np.arange(len(class_names) + 1) - 0.5,
        np.clip(relative_density, 1e-3, 1.0),
        cmap=cmr.ember,
        norm=LogNorm(vmin=1e-3, vmax=1.0),
        shading="flat",
    )
    joint_axis.set_yticks(np.arange(len(class_names)))
    joint_axis.set_yticklabels(class_names)
    joint_axis.set_ylabel("class")
    joint_axis.set_xlabel("redshift")
    joint_axis.set_title(
        "Class–redshift evidence", loc="left", fontweight="bold"
    )
    joint_axis.scatter(
        [peak_redshift], [predicted_class], marker="D", color=PREDICTED_COLOR, s=55,
        label="STRIDER peak",
    )
    if show_secondary:
        joint_axis.scatter(
            [secondary_redshift],
            [predicted_class],
            marker="^",
            facecolors="none",
            edgecolors=SECONDARY_COLOR,
            linewidths=1.4,
            s=55,
            label="Competing peak",
        )
    joint_axis.scatter(
        [float(item["redshift"])],
        [int(item["class_index"])],
        marker="o",
        facecolors="none",
        edgecolors=TRUTH_COLOR,
        linewidths=1.8,
        s=75,
        label="True",
    )
    joint_axis.legend(
        frameon=False,
        loc="upper right",
        fontsize=8,
        labelcolor="white",
    )
    figure.colorbar(image, ax=joint_axis, pad=0.01, label="relative density")

    if route_axis is not None and route_evidence is not None:
        route_names = list(route_evidence)
        route_values = np.stack(
            [np.asarray(route_evidence[name], dtype=np.float64) for name in route_names]
        )
        if route_support is None:
            route_support = np.ones(route_values.shape[1], dtype=bool)
        support = np.broadcast_to(
            np.asarray(route_support, dtype=bool)[None, :], route_values.shape
        )
        masked_routes = np.ma.masked_where(~support | ~np.isfinite(route_values), route_values)
        finite_routes = np.abs(masked_routes.compressed())
        route_limit = max(
            float(np.quantile(finite_routes, 0.98)) if finite_routes.size else 0.0,
            0.05,
        )
        route_image = route_axis.pcolormesh(
            edges,
            np.arange(len(route_names) + 1) - 0.5,
            masked_routes,
            cmap=cmr.fusion_r,
            norm=TwoSlopeNorm(vmin=-route_limit, vcenter=0.0, vmax=route_limit),
            shading="flat",
        )
        route_axis.set_facecolor("#e6e6e6")
        route_axis.set_yticks(np.arange(len(route_names)))
        route_axis.set_yticklabels(route_names, fontsize=8)
        route_axis.set_ylabel("model route")
        route_axis.set_xlabel("redshift")
        route_axis.set_title(
            f"Route contributions for {class_names[predicted_class]}",
            loc="left",
            fontweight="bold",
        )
        figure.colorbar(
            route_image,
            ax=route_axis,
            pad=0.01,
            label="signed logit contribution",
        )

    masked_feature = np.ma.masked_where(~feature_support, feature_evidence)
    finite = np.abs(masked_feature.compressed())
    limit = max(float(np.quantile(finite, 0.98)) if finite.size else 0.0, 0.05)
    region_image = feature_axis.pcolormesh(
        edges,
        np.arange(len(feature_names) + 1) - 0.5,
        masked_feature,
        # Warm colors denote positive profile agreement; cool colors denote
        # anti-correlation. This matches the visual convention used by the
        # class-redshift evidence panel above.
        cmap=cmr.fusion_r,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        shading="flat",
    )
    feature_axis.set_facecolor("#e6e6e6")
    feature_axis.set_yticks(np.arange(len(feature_names)))
    feature_axis.set_yticklabels(feature_names, fontsize=8)
    feature_axis.set_ylabel("named spectral region")
    feature_axis.set_xlabel("redshift")
    feature_axis.set_title(
        f"Named-feature diagnostic for {class_names[predicted_class]} (ONIR route)",
        loc="left",
        fontweight="bold",
    )
    figure.colorbar(
        region_image,
        ax=feature_axis,
        pad=0.01,
        label="signed profile similarity",
    )

    class_probability = joint_probability_mass.sum(axis=1)
    top = np.argsort(class_probability)[::-1][: min(6, len(class_names))]
    bar_colors = [
        PREDICTED_COLOR if index == predicted_class else SECONDARY_COLOR
        for index in top
    ]
    bars = class_axis.barh(np.arange(len(top)), class_probability[top], color=bar_colors)
    class_axis.set_yticks(np.arange(len(top)))
    class_axis.set_yticklabels(np.asarray(class_names)[top])
    class_axis.invert_yaxis()
    class_axis.set_xlim(0.0, 1.05)
    class_axis.set_xlabel("P(class)")
    class_axis.set_title("Class probabilities", loc="left", fontweight="bold")
    class_axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

    redshift_density = density.sum(axis=0)
    redshift_axis.plot(redshift_grid, redshift_density, color=PREDICTED_COLOR, lw=1.8)
    redshift_axis.axvspan(
        primary_lower_68,
        primary_upper_68,
        color=PREDICTED_COLOR,
        alpha=0.13,
        label="Primary 68% interval",
    )
    redshift_axis.axvline(
        peak_redshift,
        color=PREDICTED_COLOR,
        lw=1.2,
        label="Peak",
    )
    redshift_axis.axvline(
        posterior_median_redshift,
        color=SECONDARY_COLOR,
        ls="--",
        lw=1.0,
        label="Median",
    )
    if show_secondary:
        redshift_axis.axvline(
            secondary_redshift,
            color=SECONDARY_COLOR,
            ls=":",
            lw=1.0,
            label="Competing peak",
        )
    redshift_axis.axvline(
        float(item["redshift"]),
        color=TRUTH_COLOR,
        lw=1.0,
        label="True",
    )
    redshift_axis.set_xlabel("redshift")
    redshift_axis.set_ylabel("posterior density")
    redshift_axis.set_title(
        "Redshift posterior · primary 68% "
        f"[{primary_lower_68:.3f}, {primary_upper_68:.3f}]",
        loc="left",
        fontweight="bold",
    )
    if show_secondary:
        redshift_axis.text(
            0.98,
            0.96,
            f"competing z={secondary_redshift:.3f}\n"
            f"alternate/primary mass={secondary_ratio:.2f}\n"
            f"full 68%=[{full_lower_68:.3f}, {full_upper_68:.3f}]",
            transform=redshift_axis.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="0.35",
        )
    redshift_axis.legend(frameon=False, fontsize=7, ncol=2)

    grade = evidence_grade(evidence_sufficiency, evidence_grade_thresholds)
    figure.suptitle(
        f"SNID {int(item['snid'])} · {class_names[predicted_class]} "
        f"P={class_probability[predicted_class]:.2f} · "
        f"z_STRIDER={predicted_redshift:.3f} · "
        f"posterior median={posterior_median_redshift:.3f} · "
        f"{grade}\n"
        f"true: {class_names[int(item['class_index'])]}, "
        f"z={float(item['redshift']):.3f} · "
        f"primary basin mass {float(primary_basin['mass']):.2f} · "
        f"evidence score {evidence_sufficiency:.2f}",
        fontsize=14,
        fontweight="bold",
    )


def _draw_spectra_summary(
    axis,
    item: dict[str, torch.Tensor],
    wavelength: np.ndarray,
    *,
    selection: str = "best",
) -> None:
    """Show either the best-S/N or latest visit as one interpretable spectrum."""
    if selection not in {"best", "latest"}:
        raise ValueError("Spectrum selection must be 'best' or 'latest'")
    flux = item["flux"].numpy()
    mask = item["wavelength_mask"].numpy() > 0
    days = item["observer_days"].numpy()
    retained = (
        item["visit_mask"].numpy() > 0
        if "visit_mask" in item
        else np.ones(len(flux), dtype=bool)
    )
    retained_indices = np.flatnonzero(retained)
    if not len(retained_indices):
        raise ValueError("Evidence summaries require at least one retained visit")

    epoch_snr = item.get("epoch_observed_signal_to_noise")
    epoch_snr_values = (
        epoch_snr.numpy().astype(np.float64)
        if epoch_snr is not None and len(epoch_snr) == len(flux)
        else np.full(len(flux), np.nan, dtype=np.float64)
    )
    retained_snr = epoch_snr_values[retained_indices]
    has_observed_snr = bool(np.isfinite(retained_snr).any())
    if selection == "latest":
        selected_index = int(retained_indices[-1])
    elif has_observed_snr:
        selected_index = int(retained_indices[np.nanargmax(retained_snr)])
    else:
        selected_index = int(retained_indices[len(retained_indices) // 2])

    # Match the edge exclusion used by the observed-S/N measurement. The
    # detector extremes are commonly much noisier and obscure the informative
    # interior when a single visit is shown.
    log_wavelength = np.log(np.asarray(wavelength, dtype=np.float64))
    log_span = float(log_wavelength[-1] - log_wavelength[0])
    display_minimum = float(np.exp(log_wavelength[0] + 0.05 * log_span))
    display_maximum = float(np.exp(log_wavelength[-1] - 0.05 * log_span))
    display_wavelength = (wavelength >= display_minimum) & (
        wavelength <= display_maximum
    )
    selected_valid = mask[selected_index] & display_wavelength
    measured = np.abs(flux[selected_index, selected_valid])
    scale = max(
        float(np.quantile(measured, 0.99)) if measured.size else 0.0,
        1e-8,
    )
    selected_snr = float(epoch_snr_values[selected_index])
    selected_values = flux[selected_index, selected_valid] / scale
    axis.plot(
        wavelength[selected_valid],
        selected_values,
        color=PREDICTED_COLOR,
        lw=1.35,
        zorder=3,
    )
    axis.axhline(0.0, color="0.25", lw=0.7, alpha=0.65, zorder=0)

    display_values = selected_values[np.isfinite(selected_values)]
    if display_values.size:
        low, high = np.quantile(display_values, (0.005, 0.995))
        padding = max(0.08 * float(high - low), 0.08)
        axis.set_ylim(float(low - padding), float(high + padding))
    axis.set_xlim(display_minimum, display_maximum)
    total_visits = int(item.get("total_visit_count", len(retained_indices)))
    selected_visit = int(np.searchsorted(retained_indices, selected_index)) + 1
    if selection == "latest":
        status = (
            f"Evidence after {len(retained_indices)} of {total_visits} visits    "
            f"new spectrum: visit {selected_visit}    "
            f"day {days[selected_index]:+.0f}"
        )
    else:
        spectrum_name = (
            "Best observed spectrum" if has_observed_snr else "Representative spectrum"
        )
        status = (
            f"{spectrum_name} — visit {selected_visit} of {total_visits}    "
            f"day {days[selected_index]:+.0f}"
        )
    if np.isfinite(selected_snr):
        status += f"    S/N {selected_snr:.2f}"
    axis.set_title(status, loc="right", fontsize=10.5, color="0.28")
    axis.set_ylabel("relative flux")
    axis.set_xlabel("observed wavelength (Å)")


def _draw_spectra(axis, item: dict[str, torch.Tensor], wavelength: np.ndarray) -> None:
    import cmasher as cmr

    flux = item["flux"].numpy()
    mask = item["wavelength_mask"].numpy() > 0
    days = item["observer_days"].numpy()
    retained = (
        item["visit_mask"].numpy() > 0
        if "visit_mask" in item
        else np.ones(len(flux), dtype=bool)
    )
    measured = np.abs(flux[mask])
    scale = max(float(np.quantile(measured, 0.99)) if measured.size else 0.0, 1e-8)
    colors = cmr.guppy_r(np.linspace(0.1, 0.9, max(int(retained.sum()), 1)))
    for color, values, valid, day in zip(colors, flux[retained], mask[retained], days[retained]):
        axis.plot(
            wavelength[valid],
            values[valid] / scale,
            lw=0.8,
            color=color,
            label=f"{day:.0f} d",
        )
    handles, labels = axis.get_legend_handles_labels()
    if len(handles) > 8:
        selected = np.unique(np.linspace(0, len(handles) - 1, 6, dtype=int))
        handles = [handles[index] for index in selected]
        labels = [labels[index] for index in selected]
    axis.legend(handles, labels, frameon=False, ncol=min(len(handles), 6), fontsize=8)
    axis.set_ylabel("relative flux")
    axis.set_xlabel("observed wavelength (Å)")
    axis.set_title(
        "Input spectra · days since first spectrum",
        loc="left",
        fontweight="bold",
    )


def _route_evidence(
    output: dict[str, torch.Tensor], predicted_class: int
) -> dict[str, np.ndarray]:
    """Return the additive model-route logits for one predicted class."""
    routes: dict[str, np.ndarray] = {}
    for label, key in ROUTE_LOGIT_KEYS:
        if key not in output:
            continue
        routes[label] = output[key][0, predicted_class].detach().cpu().numpy()
    if "dense_scan_joint_logits" in output and not any(
        label in routes for label in ("Whole spectrum", "Continuum-subtracted")
    ):
        routes["Whole-spectrum scan"] = (
            output["dense_scan_joint_logits"][0, predicted_class].detach().cpu().numpy()
        )
    return routes


def _route_audit_rows(
    *,
    cohort: str,
    item: dict[str, torch.Tensor],
    class_name: str,
    predicted_class_probability: float,
    redshift_grid: np.ndarray,
    redshift_probability: np.ndarray,
    redshift_cell_width: np.ndarray,
    peak_summary: dict[str, float | int],
    route_evidence: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Compare each model route at the primary, alternate and true redshifts."""
    dominant_redshift = float(peak_summary["dominant_redshift"])
    secondary_redshift = float(peak_summary["secondary_redshift"])
    true_redshift = float(item["redshift"])
    density = np.asarray(redshift_probability) / np.asarray(redshift_cell_width)
    relative_density = density / max(float(np.max(density)), 1e-30)

    def index_at(redshift: float) -> int | None:
        if not np.isfinite(redshift):
            return None
        return int(np.abs(redshift_grid - redshift).argmin())

    dominant_index = index_at(dominant_redshift)
    secondary_index = index_at(secondary_redshift)
    true_index = index_at(true_redshift)
    assert dominant_index is not None and true_index is not None

    def value_at(values: np.ndarray, index: int | None) -> float:
        return float(values[index]) if index is not None else float("nan")

    common = {
        "cohort": cohort,
        "snid": int(item["snid"]),
        "true_class_index": int(item["class_index"]),
        "predicted_class": class_name,
        "predicted_class_probability": predicted_class_probability,
        "true_redshift": true_redshift,
        "dominant_redshift": dominant_redshift,
        "primary_peak_redshift": dominant_redshift,
        "secondary_redshift": secondary_redshift,
        "alternate_peak_redshift": secondary_redshift,
        "dominant_peak_mass": float(peak_summary["dominant_mass"]),
        "primary_basin_mass": float(peak_summary["dominant_mass"]),
        "secondary_peak_mass": float(peak_summary["secondary_mass"]),
        "alternate_basin_mass": float(peak_summary["secondary_mass"]),
        "secondary_to_dominant_mass_ratio": float(
            peak_summary["secondary_to_dominant_mass_ratio"]
        ),
        "relative_density_at_dominant": value_at(relative_density, dominant_index),
        "relative_density_at_primary": value_at(relative_density, dominant_index),
        "relative_density_at_secondary": value_at(relative_density, secondary_index),
        "relative_density_at_alternate": value_at(relative_density, secondary_index),
        "relative_density_at_true": value_at(relative_density, true_index),
    }
    rows = []
    for route, values in route_evidence.items():
        values = np.asarray(values, dtype=np.float64)
        dominant_value = value_at(values, dominant_index)
        secondary_value = value_at(values, secondary_index)
        true_value = value_at(values, true_index)
        rows.append(
            {
                **common,
                "route": route,
                "contribution_at_dominant": dominant_value,
                "contribution_at_primary": dominant_value,
                "contribution_at_secondary": secondary_value,
                "contribution_at_alternate": secondary_value,
                "contribution_at_true": true_value,
                "true_minus_dominant": true_value - dominant_value,
                "secondary_minus_dominant": secondary_value - dominant_value,
            }
        )
    return rows


def _manifest_indices(
    objects: pd.DataFrame,
    manifest: pd.DataFrame,
) -> list[tuple[str, int]]:
    """Resolve an explicit SNID manifest to dataset indices in manifest order."""
    if "snid" not in manifest:
        raise ValueError("Evidence-map object list must contain a 'snid' column")
    if manifest["snid"].duplicated().any():
        duplicates = manifest.loc[manifest["snid"].duplicated(), "snid"].tolist()
        raise ValueError(f"Evidence-map object list repeats SNIDs: {duplicates}")
    object_indices = {
        int(snid): int(index)
        for index, snid in objects["snid"].items()
    }
    requested = [int(value) for value in manifest["snid"]]
    missing = [snid for snid in requested if snid not in object_indices]
    if missing:
        raise ValueError(f"Evidence-map SNIDs are absent from this split: {missing}")
    if "cohort" in manifest:
        cohorts = manifest["cohort"].fillna("selected").astype(str).tolist()
    else:
        cohorts = ["selected"] * len(manifest)
    return [
        (cohort, object_indices[snid])
        for cohort, snid in zip(cohorts, requested, strict=True)
    ]


def _cell_edges(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.float64)
    edges = np.empty(len(grid) + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (grid[:-1] + grid[1:])
    edges[0] = grid[0]
    edges[-1] = grid[-1]
    return edges


def _representative_indices(
    objects,
    targets: list[float],
    count: int,
    preferred_class_index: int | None = None,
    preferred_count: int = 0,
) -> list[tuple[float, int]]:
    selected = []
    for target in targets:
        ordered = objects.assign(
            distance=np.abs(objects["redshift"] - target)
        ).sort_values("distance")
        chosen = []
        if preferred_count:
            preferred = ordered[
                ordered["class_index"].eq(preferred_class_index)
            ].head(preferred_count)
            chosen.extend(int(index) for index in preferred.index)
        used_classes = set(
            int(objects.loc[index, "class_index"]) for index in chosen
        )
        for row in ordered.itertuples():
            if int(row.Index) in chosen:
                continue
            class_index = int(row.class_index)
            if class_index in used_classes:
                continue
            chosen.append(int(row.Index))
            used_classes.add(class_index)
            if len(chosen) == count:
                break
        if len(chosen) < count:
            chosen_set = set(chosen)
            chosen.extend(
                int(index)
                for index in ordered.index
                if int(index) not in chosen_set
            )
        selected.extend((target, index) for index in chosen[:count])
    return selected
