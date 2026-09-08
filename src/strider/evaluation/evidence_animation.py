"""Animate the cumulative class and redshift evidence as visits arrive."""

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
from .evaluate import _posterior_basin_candidates, _posterior_quantile
from .evidence_maps import (
    _manifest_indices,
    _representative_indices,
    _route_evidence,
    draw_evidence_map,
    draw_evidence_summary,
)


@torch.no_grad()
def write_evidence_gifs(
    config: dict[str, Any],
    *,
    split: str | None = None,
    view: str | None = None,
    object_list: str | Path | None = None,
    layout: str | None = None,
    maximum_frames: int | None = None,
) -> dict[str, Any]:
    """Write cumulative evidence animations for selected evaluation objects."""
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    output_dir = project_path(config, config["project"]["output_dir"])
    figure_dir = output_dir / "evidence_gifs"
    figure_dir.mkdir(parents=True, exist_ok=True)
    model, _, device = load_trained_model(config)
    layout = str(
        layout or config["evaluation"].get("evidence_gif_layout", "summary")
    )
    if layout not in {"summary", "diagnostic"}:
        raise ValueError("evidence_gif_layout must be 'summary' or 'diagnostic'")
    split = str(split or config["evaluation"].get("split", "calibration"))
    view = str(view or config["evaluation"].get("evidence_map_view", "original"))
    targets = [
        float(value)
        for value in config["evaluation"].get(
            "evidence_map_redshifts", [0.75, 1.5, 2.5]
        )
    ]
    maximum_frames = int(
        config["evaluation"].get(
            "evidence_gif_max_frames", config["data"].get("max_visits", 32)
        )
        if maximum_frames is None
        else maximum_frames
    )
    if maximum_frames < 1:
        raise ValueError("evidence_gif_max_frames must be positive")
    grade_thresholds = tuple(
        float(value)
        for value in config["evaluation"].get(
            "evidence_grade_thresholds", [0.25, 0.5, 0.75]
        )
    )
    competing_peak_ratio = float(
        config["evaluation"].get("competing_peak_mass_ratio", 0.25)
    )
    dataset = SundialDataset(config, split, view, training=False)
    bank = (
        load_onir_bank(project_path(config, config["onir"]["bank_path"]))
        if layout == "diagnostic"
        else None
    )
    class_names = list(model.class_names)
    ia_index = class_names.index("Ia") if "Ia" in class_names else None
    paths = []
    trajectory_paths = []
    trajectory_summaries = []

    if object_list is not None:
        manifest = pd.read_csv(project_path(config, object_list))
        selected_objects = _manifest_indices(dataset.objects, manifest)
    else:
        selected_objects = [
            (f"z{target:.2f}", index)
            for target, index in _representative_indices(
                dataset.objects,
                targets,
                1,
                preferred_class_index=ia_index,
                preferred_count=1 if ia_index is not None else 0,
            )
        ]

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
        item = {
            **item,
            "total_visit_count": torch.tensor(int(item["flux"].shape[0])),
        }
        visit_counts = _animation_visit_counts(item["flux"].shape[0], maximum_frames)
        prefixes = [_visit_prefix(item, count) for count in visit_counts]
        snapshots = [
            _evidence_snapshot(
                model,
                prefix,
                device,
                include_diagnostics=layout == "diagnostic",
            )
            for prefix in prefixes
        ]
        trajectory = [
            _basin_trajectory_row(
                snapshot,
                count,
                model.redshift_grid.cpu().numpy(),
                model.redshift_cell_width.cpu().numpy(),
                previous=snapshots[frame - 1] if frame else None,
            )
            for frame, (count, snapshot) in enumerate(zip(visit_counts, snapshots))
        ]
        trajectory_summary = _basin_trajectory_summary(trajectory)
        trajectory_summary.update(
            {
                "snid": int(item["snid"]),
                "true_class": class_names[int(item["class_index"])],
                "true_redshift": float(item["redshift"]),
            }
        )
        trajectory_summaries.append(trajectory_summary)
        figure_size = (11.8, 7.6) if layout == "summary" else (13.0, 15.5)
        figure = plt.figure(figsize=figure_size, constrained_layout=True)

        def draw(frame: int) -> tuple[()]:
            figure.clear()
            count = visit_counts[frame]
            prefix = prefixes[frame]
            snapshot = snapshots[frame]
            common_arguments = {
                "item": prefix,
                "wavelength_angstrom": dataset.output_wavelength,
                "class_names": class_names,
                "redshift_grid": model.redshift_grid.cpu().numpy(),
                "redshift_cell_width": model.redshift_cell_width.cpu().numpy(),
                "joint_probability_mass": snapshot["joint"],
                "predicted_class": snapshot["predicted_class"],
                "predicted_redshift": snapshot["predicted_redshift"],
                "posterior_median_redshift": snapshot["posterior_median_redshift"],
                "evidence_sufficiency": snapshot["evidence_score"],
                "evidence_grade_thresholds": grade_thresholds,
                "competing_peak_mass_ratio": competing_peak_ratio,
                "split": split,
                "view": view,
            }
            if layout == "summary":
                draw_evidence_summary(
                    figure,
                    spectrum_selection="latest",
                    **common_arguments,
                )
            else:
                assert bank is not None
                draw_evidence_map(
                    figure,
                    feature_names=list(bank.feature_names),
                    feature_evidence=snapshot["feature_evidence"],
                    feature_support=snapshot["feature_support"],
                    route_evidence=snapshot["route_evidence"],
                    route_support=snapshot["route_support"],
                    **common_arguments,
                )
            if layout == "diagnostic":
                figure.text(
                    0.99,
                    0.995,
                    f"{count} of {item['flux'].shape[0]} visits",
                    ha="right",
                    va="top",
                    fontsize=10,
                    color="0.3",
                )
                basin_row = trajectory[frame]
                figure.text(
                    0.99,
                    0.979,
                    f"{int(basin_row['basin_count'])} basin"
                    f"{'s' if int(basin_row['basin_count']) != 1 else ''} · "
                    f"primary mass {basin_row['primary_mass']:.2f} · "
                    "alternate/primary mass "
                    f"{basin_row['secondary_to_primary_mass_ratio']:.2f}"
                    + (
                        " · MODE SWITCH"
                        if bool(basin_row["primary_mode_switch_from_previous"])
                        else ""
                    ),
                    ha="right",
                    va="top",
                    fontsize=9,
                    color="0.3",
                )
            return ()

        frames = list(range(len(visit_counts)))
        if len(frames) > 1:
            frames.append(frames[-1])
        movie = animation.FuncAnimation(
            figure,
            draw,
            frames=frames,
            interval=1100,
            repeat_delay=1600,
            blit=False,
        )
        true_class = class_names[int(item["class_index"])]
        safe_cohort = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in cohort
        ).strip("_")
        path = figure_dir / (
            f"{split}_{view}_{safe_cohort}_{true_class}_snid{int(item['snid'])}.gif"
        )
        movie.save(path, writer=animation.PillowWriter(fps=0.9), dpi=105)
        trajectory_path = path.with_suffix(".csv")
        pd.DataFrame(trajectory).to_csv(
            trajectory_path,
            index=False,
            float_format="%.6g",
        )
        plt.close(figure)
        paths.append(str(path))
        trajectory_paths.append(str(trajectory_path))
    trajectory_summary_path = figure_dir / f"{split}_{view}_basin_summary.csv"
    pd.DataFrame(trajectory_summaries).to_csv(
        trajectory_summary_path,
        index=False,
        float_format="%.6g",
    )
    return {
        "device": str(device),
        "split": split,
        "view": view,
        "layout": layout,
        "gifs": paths,
        "basin_trajectories": trajectory_paths,
        "basin_summary": str(trajectory_summary_path),
    }


def _animation_visit_counts(total: int, maximum_frames: int) -> list[int]:
    if total < 1:
        raise ValueError("Evidence animation needs at least one visit")
    if total <= maximum_frames:
        return list(range(1, total + 1))
    return np.unique(
        np.rint(np.linspace(1, total, maximum_frames)).astype(int)
    ).tolist()


def _visit_prefix(item: dict[str, torch.Tensor], count: int) -> dict[str, torch.Tensor]:
    prefix = dict(item)
    for name in (
        "flux",
        "flux_error_shape",
        "wavelength_mask",
        "observer_days",
        "visit_flux_scale",
        "simulation_rest_phase_days",
        "epoch_observed_signal_to_noise",
    ):
        if name in item:
            prefix[name] = item[name][:count]
    return prefix


@torch.no_grad()
def _evidence_snapshot(
    model,
    item: dict[str, torch.Tensor],
    device: torch.device,
    *,
    include_diagnostics: bool = True,
) -> dict[str, Any]:
    batch = {
        name: value.to(device)
        for name, value in collate_objects([item]).items()
    }
    output = model(measurement_inputs(batch))
    joint = joint_probability(
        output["joint_logits"], model.redshift_cell_width, model.redshift_prior
    )[0].cpu().numpy()
    predicted_class = int(joint.sum(axis=1).argmax())
    redshift_grid = model.redshift_grid.cpu().numpy()
    redshift_mass = joint.sum(axis=0)
    posterior_median_redshift = _posterior_quantile(
        redshift_grid, redshift_mass, 0.5
    )
    predicted_redshift = float(
        _posterior_basin_candidates(
            redshift_grid,
            redshift_mass,
            model.redshift_cell_width.cpu().numpy(),
        )[0]["peak_redshift"]
    )
    snapshot = {
        "joint": joint,
        "predicted_class": predicted_class,
        "predicted_redshift": predicted_redshift,
        "posterior_median_redshift": posterior_median_redshift,
        "evidence_score": float(
            torch.sigmoid(output["evidence_sufficiency_logit"])[0].cpu()
        ),
    }
    if include_diagnostics:
        if (
            "feature_evidence" not in output
            or "feature_evidence_support" not in output
        ):
            raise ValueError("Diagnostic evidence animations require ONIR output")
        snapshot.update(
            {
                "feature_evidence": output["feature_evidence"][
                    0, :, predicted_class
                ].cpu().numpy(),
                "feature_support": output["feature_evidence_support"][
                    0, :, predicted_class
                ].cpu().numpy(),
                "route_evidence": _route_evidence(output, predicted_class),
                "route_support": output["joint_support"][
                    0, predicted_class
                ].cpu().numpy(),
            }
        )
    return snapshot


def _basin_trajectory_row(
    snapshot: dict[str, Any],
    visit_count: int,
    redshift_grid: np.ndarray,
    redshift_cell_width: np.ndarray,
    *,
    previous: dict[str, Any] | None = None,
) -> dict[str, float | int | str | bool]:
    """Summarize how redshift basins change as spectra accumulate."""
    joint = np.asarray(snapshot["joint"], dtype=np.float64)
    redshift_mass = joint.sum(axis=0)
    candidates = _posterior_basin_candidates(
        redshift_grid,
        redshift_mass,
        redshift_cell_width,
        maximum_candidates=len(redshift_grid),
    )
    primary = candidates[0]
    left = int(primary["left_index"])
    right = int(primary["right_index"])
    class_mass = joint[:, left:right].sum(axis=1)
    primary_class = int(class_mass.argmax())
    primary_class_probability = float(
        class_mass[primary_class] / max(float(class_mass.sum()), 1.0e-300)
    )
    secondary = candidates[1] if len(candidates) > 1 else None
    previous_peak = float("nan")
    peak_shift = float("nan")
    if previous is not None:
        previous_candidates = _posterior_basin_candidates(
            redshift_grid,
            np.asarray(previous["joint"], dtype=np.float64).sum(axis=0),
            redshift_cell_width,
            maximum_candidates=len(redshift_grid),
        )
        previous_peak = float(previous_candidates[0]["peak_redshift"])
        peak_shift = float(primary["peak_redshift"]) - previous_peak
    secondary_mass = float(secondary["mass"]) if secondary is not None else 0.0
    mode_switch = bool(np.isfinite(peak_shift) and abs(peak_shift) > 0.1)
    competitor_saddle_contrast = float(
        primary["log_peak_to_strongest_competitor_saddle_ratio"]
    )
    return {
        "visit_count": int(visit_count),
        "basin_count": int(len(candidates)),
        "primary_peak_redshift": float(primary["peak_redshift"]),
        "primary_median_redshift": float(primary["median_redshift"]),
        "primary_lower_68": float(primary["lower_68"]),
        "primary_upper_68": float(primary["upper_68"]),
        "primary_mass": float(primary["mass"]),
        # Compatibility name retained, now using the finite contrast to the
        # strongest credible competitor rather than infinite global prominence.
        "primary_log_peak_to_saddle_ratio": competitor_saddle_contrast,
        "primary_log_peak_to_competitor_saddle_ratio": competitor_saddle_contrast,
        "primary_to_competitor_peak_height_ratio": float(
            primary["primary_to_strongest_competitor_height_ratio"]
        ),
        "primary_is_largest_mass_basin": bool(primary["is_largest_mass_basin"]),
        "primary_class_index": primary_class,
        "primary_class_probability": primary_class_probability,
        "secondary_peak_redshift": (
            float(secondary["peak_redshift"])
            if secondary is not None
            else float("nan")
        ),
        "secondary_mass": secondary_mass,
        "secondary_to_primary_mass_ratio": secondary_mass
        / max(float(primary["mass"]), 1.0e-300),
        "previous_primary_peak_redshift": previous_peak,
        "primary_peak_shift_from_previous": peak_shift,
        "primary_mode_switch_from_previous": mode_switch,
        "evidence_score": float(snapshot["evidence_score"]),
    }


def _basin_trajectory_summary(
    trajectory: list[dict[str, float | int | str | bool]],
) -> dict[str, float | int | bool]:
    """Summarize whether the primary redshift solution remains stable."""
    if not trajectory:
        raise ValueError("A basin trajectory must contain at least one visit prefix")
    shifts = np.asarray(
        [row["primary_peak_shift_from_previous"] for row in trajectory],
        dtype=np.float64,
    )
    switches = np.asarray(
        [bool(row["primary_mode_switch_from_previous"]) for row in trajectory],
        dtype=bool,
    )
    late_start = max(1, int(np.floor(0.75 * len(trajectory))))
    finite_shifts = np.abs(shifts[np.isfinite(shifts)])
    final = trajectory[-1]
    return {
        "visit_prefix_count": int(len(trajectory)),
        "final_visit_count": int(final["visit_count"]),
        "primary_mode_switch_count": int(switches.sum()),
        "late_primary_mode_switch": bool(switches[late_start:].any()),
        "maximum_primary_peak_shift": float(
            finite_shifts.max() if finite_shifts.size else 0.0
        ),
        "final_primary_peak_redshift": float(final["primary_peak_redshift"]),
        "final_primary_mass": float(final["primary_mass"]),
        "final_alternate_to_primary_mass_ratio": float(
            final["secondary_to_primary_mass_ratio"]
        ),
        "final_evidence_score": float(final["evidence_score"]),
    }
