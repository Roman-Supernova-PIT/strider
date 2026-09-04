#!/usr/bin/env python3
"""Render local Sundial evidence maps and true visit-prefix diagnostics.

The downloaded checkpoint is first checked against its untouched resolved
configuration.  Only filesystem locations and data-loader settings are then
relocated for local, read-only inference; no scientific model setting changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from strider.atlas import load_onir_bank
from strider.config import resolved_config_sha256
from strider.data.dataset import SundialDataset, collate_objects
from strider.evaluation.evidence_animation import (
    _basin_trajectory_row,
    _evidence_snapshot,
    _visit_prefix,
)
from strider.evaluation.evidence_maps import draw_evidence_map
from strider.evaluation.evaluate import _central_interval, _posterior_quantile
from strider.model import Strider, measurement_inputs
from strider.model.posterior import joint_probability


DEFAULT_PREFIXES = (1, 2, 4, 8, 16, 24, 26, 32)
REDSHIFT_EDGES = (0.0, 0.75, 1.25, 1.75, 2.25, 3.0)
PALETTE = ("#482878", "#355f8d", "#238a8d", "#35b779", "#b8de29")


class PrefixItems(Dataset):
    """Expose the chronological first ``k`` spectra of a fixed object cohort."""

    def __init__(self, items: list[dict[str, torch.Tensor]], count: int) -> None:
        self.items = items
        self.count = int(count)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.items[index]
        return _visit_prefix(item, min(self.count, int(item["flux"].shape[0])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--model-package",
        type=Path,
        default=root / "local_models/20260818/runs_detector/ia_binary_full_main",
    )
    parser.add_argument(
        "--onir-bank",
        type=Path,
        default=root / "local_models/20260818/data_detector/onir/ia_binary_full.npz",
    )
    parser.add_argument(
        "--prepared-data",
        type=Path,
        default=root / "data/local/sundial_pilot",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results/sundial_latest_20260818/local_evidence_accumulation",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--view", default="original")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--maximum-objects", type=int, default=0)
    parser.add_argument(
        "--examples-per-redshift",
        type=int,
        default=1,
        help="Number of deterministic Ia evidence-map examples near each target redshift.",
    )
    parser.add_argument(
        "--maps-only",
        action="store_true",
        help="Render maps/GIFs from the packaged full-test predictions without rerunning prefixes.",
    )
    parser.add_argument(
        "--predictions-only",
        action="store_true",
        help="Save per-object prefix predictions without rebuilding plots or evidence maps.",
    )
    parser.add_argument(
        "--prefixes",
        type=int,
        nargs="+",
        default=list(DEFAULT_PREFIXES),
    )
    parser.add_argument(
        "--v2-reference",
        type=Path,
        default=Path(
            "/Users/mdixon/Documents/Dixon_2026/strider-v2/analysis/plots/"
            "strider_v2_11c_50ep/prefix_epoch_diagnostics_quicklook/"
            "prefix_epoch_metrics.csv"
        ),
    )
    return parser.parse_args()


def runtime_config(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = args.model_package.resolve() / "config.resolved.yaml"
    checkpoint_path = args.model_package.resolve() / "best_model.pt"
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_project_root"] = str(Path(__file__).resolve().parents[1])
    config["_config_path"] = str(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    recorded = checkpoint.get("config_sha256")
    actual = resolved_config_sha256(config)
    if recorded != actual:
        raise ValueError(
            "Downloaded checkpoint and untouched resolved configuration differ: "
            f"{recorded} != {actual}"
        )

    root = Path(__file__).resolve().parents[1]
    config["data"]["prepared_dir"] = str(args.prepared_data.resolve())
    config["onir"]["bank_path"] = str(args.onir_bank.resolve())
    config["onir"]["catalog_path"] = str((root / "configs/onir_features.yaml").resolve())
    config["project"]["output_dir"] = str(args.output_dir.resolve())
    # The pilot predates stored native-support columns. Its actual wavelength
    # mask is retained, so local evaluation uses the older equivalent policy.
    config["observation"]["template_support_policy"] = "retain"
    config["training"]["num_workers"] = 0
    config["training"]["persistent_workers"] = False
    return config, checkpoint


def load_model(
    config: dict[str, Any], checkpoint: dict[str, Any]
) -> tuple[Strider, torch.device]:
    device = torch.device("cpu")
    model = Strider(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, device


@torch.inference_mode()
def predict_prefix(
    model: Strider,
    device: torch.device,
    items: list[dict[str, torch.Tensor]],
    count: int,
    batch_size: int,
) -> pd.DataFrame:
    loader = DataLoader(
        PrefixItems(items, count),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_objects,
    )
    rows: list[dict[str, float | int]] = []
    grid = model.redshift_grid.detach().cpu().numpy()
    cell_width = model.redshift_cell_width.detach().cpu().numpy()
    for batch in loader:
        device_batch = {name: value.to(device) for name, value in batch.items()}
        output = model(measurement_inputs(device_batch))
        joint = joint_probability(
            output["joint_logits"],
            model.redshift_cell_width,
            model.redshift_prior,
        )
        class_probability = joint.sum(dim=2).cpu().numpy()
        redshift_probability = joint.sum(dim=1).cpu().numpy()
        evidence = torch.sigmoid(output["evidence_sufficiency_logit"]).cpu().numpy()
        for index in range(len(batch["snid"])):
            probability = redshift_probability[index].astype(np.float64)
            median = _posterior_quantile(grid, probability, 0.5)
            lower, upper = _central_interval(grid, probability, 0.68)
            density_mode = float(grid[np.argmax(probability / cell_width)])
            rows.append(
                {
                    "snid": int(batch["snid"][index]),
                    "true_class": int(batch["class_index"][index]),
                    "true_redshift": float(batch["redshift"][index]),
                    "spectra_used": int(batch["visit_mask"][index].sum()),
                    "p_Ia": float(class_probability[index, 0]),
                    "predicted_redshift": float(median),
                    "posterior_mode_redshift": density_mode,
                    "redshift_68_width": float(upper - lower),
                    "evidence_score": float(evidence[index]),
                }
            )
    return pd.DataFrame(rows)


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def metric_row(predictions: pd.DataFrame, count: int, regime: str) -> dict[str, Any]:
    if regime == "z<2":
        values = predictions[predictions["true_redshift"] < 2.0].copy()
    else:
        values = predictions.copy()
    true_ia = values["true_class"].eq(0).to_numpy()
    selected = values["p_Ia"].ge(0.9).to_numpy()
    tp = int((true_ia & selected).sum())
    ia = values.loc[true_ia]
    residual = (
        ia["predicted_redshift"].to_numpy() - ia["true_redshift"].to_numpy()
    )
    normalized = residual / (1.0 + ia["true_redshift"].to_numpy())
    center = float(np.median(normalized)) if len(normalized) else float("nan")
    true_p = values.loc[true_ia, "p_Ia"].to_numpy()
    other_p = values.loc[~true_ia, "p_Ia"].to_numpy()
    return {
        "regime": regime,
        "prefix_spectra": int(count),
        "n_objects": int(len(values)),
        "n_true_ia": int(true_ia.sum()),
        "n_selected": int(selected.sum()),
        "n_with_at_least_prefix": int((values["spectra_used"] >= count).sum()),
        "purity_pia_ge_0p9": safe_ratio(tp, int(selected.sum())),
        "completeness_pia_ge_0p9": safe_ratio(tp, int(true_ia.sum())),
        "true_ia_p16": float(np.quantile(true_p, 0.16)),
        "true_ia_p50": float(np.quantile(true_p, 0.50)),
        "true_ia_p84": float(np.quantile(true_p, 0.84)),
        "non_ia_p16": float(np.quantile(other_p, 0.16)),
        "non_ia_p50": float(np.quantile(other_p, 0.50)),
        "non_ia_p84": float(np.quantile(other_p, 0.84)),
        "true_ia_median_absolute_delta_z": float(np.median(np.abs(residual))),
        "true_ia_normalized_bias": center,
        "true_ia_nmad": float(1.4826 * np.median(np.abs(normalized - center))),
        "true_ia_fraction_abs_delta_z_lt_0p1": float((np.abs(residual) < 0.1).mean()),
        "true_ia_median_68_width": float(np.median(ia["redshift_68_width"])),
        "mean_evidence_score": float(values["evidence_score"].mean()),
    }


def metrics_by_redshift(predictions: pd.DataFrame, count: int) -> list[dict[str, Any]]:
    rows = []
    for lower, upper in zip(REDSHIFT_EDGES[:-1], REDSHIFT_EDGES[1:]):
        values = predictions[
            predictions["true_redshift"].ge(lower)
            & predictions["true_redshift"].lt(upper)
        ]
        true_ia = values["true_class"].eq(0).to_numpy()
        selected = values["p_Ia"].ge(0.9).to_numpy()
        tp = int((true_ia & selected).sum())
        ia = values.loc[true_ia]
        residual = ia["predicted_redshift"].to_numpy() - ia["true_redshift"].to_numpy()
        rows.append(
            {
                "prefix_spectra": int(count),
                "redshift_bin": f"{lower:.2f}–{upper:.2f}",
                "n_objects": int(len(values)),
                "n_true_ia": int(true_ia.sum()),
                "n_selected": int(selected.sum()),
                "purity_pia_ge_0p9": safe_ratio(tp, int(selected.sum())),
                "completeness_pia_ge_0p9": safe_ratio(tp, int(true_ia.sum())),
                "true_ia_median_absolute_delta_z": (
                    float(np.median(np.abs(residual))) if len(residual) else float("nan")
                ),
                "true_ia_fraction_abs_delta_z_lt_0p1": (
                    float((np.abs(residual) < 0.1).mean())
                    if len(residual)
                    else float("nan")
                ),
            }
        )
    return rows


def style_axis(axis, x: np.ndarray) -> None:
    axis.set_xscale("log", base=2)
    axis.set_xticks(x)
    crowded = {24, 26, 32}.issubset({int(value) for value in x})
    axis.set_xticklabels(
        ["" if crowded and int(value) == 26 else str(int(value)) for value in x]
    )
    axis.grid(color="0.90", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def plot_accumulation(
    metrics: pd.DataFrame,
    output_dir: Path,
    regime: str = "all",
) -> Path:
    if regime not in {"all", "z<2"}:
        raise ValueError("regime must be 'all' or 'z<2'")
    values = metrics[metrics["regime"].eq(regime)].sort_values("prefix_spectra")
    if values.empty:
        raise ValueError(f"No accumulation rows found for regime {regime!r}")
    x = values["prefix_spectra"].to_numpy(dtype=float)
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.8), sharex=True)
    probability_axis, selection_axis, redshift_axis, resolution_axis = axes.ravel()

    probability_axis.plot(x, values["true_ia_p50"], color="#238b45", lw=2.2, label="true Ia")
    probability_axis.fill_between(
        x, values["true_ia_p16"], values["true_ia_p84"],
        color="#238b45", alpha=0.18, linewidth=0,
    )
    probability_axis.plot(x, values["non_ia_p50"], color="#5f6368", lw=2.2, label="non-Ia")
    probability_axis.fill_between(
        x, values["non_ia_p16"], values["non_ia_p84"],
        color="#5f6368", alpha=0.15, linewidth=0,
    )
    probability_axis.axhline(0.9, color="#b88700", ls="--", lw=1.1)
    probability_axis.set_ylim(-0.02, 1.02)
    probability_axis.set_ylabel(r"$P(\mathrm{Ia})$")
    probability_axis.set_title("Classification confidence", loc="left", fontweight="bold")
    probability_axis.legend(frameon=False)

    selection_axis.plot(
        x, values["purity_pia_ge_0p9"], color="#1f6f43", marker="o", lw=2.1,
        label="purity",
    )
    selection_axis.plot(
        x, values["completeness_pia_ge_0p9"], color="#6c5aa7", marker="o", lw=2.1,
        label="completeness",
    )
    selection_axis.set_ylim(0.0, 1.03)
    selection_axis.set_ylabel(r"$P(\mathrm{Ia})\geq0.9$ sample")
    selection_axis.set_title("Ia selection", loc="left", fontweight="bold")
    selection_axis.legend(frameon=False)

    redshift_axis.plot(
        x, values["true_ia_median_absolute_delta_z"], color="#2f6f8f",
        marker="o", lw=2.1, label=r"median $|\Delta z|$",
    )
    redshift_axis.plot(
        x, values["true_ia_nmad"], color="#b88700", marker="o", lw=2.1,
        label="normalized NMAD",
    )
    redshift_axis.set_ylabel("redshift error")
    redshift_axis.set_title("Ia redshift accuracy", loc="left", fontweight="bold")
    redshift_axis.legend(frameon=False)

    resolution_axis.plot(
        x, values["true_ia_fraction_abs_delta_z_lt_0p1"], color="#007c91",
        marker="o", lw=2.1, label=r"fraction $|\Delta z|<0.1$",
    )
    resolution_axis.set_ylim(0.0, 1.03)
    resolution_axis.set_ylabel("fraction of true Ia")
    resolution_axis.set_title("Usable redshift fraction", loc="left", fontweight="bold")
    width_axis = resolution_axis.twinx()
    width_axis.plot(
        x, values["true_ia_median_68_width"], color="#9c3d54", marker="s",
        ls=":", lw=1.7, label="median 68% width",
    )
    width_axis.set_ylabel("posterior 68% width", color="#9c3d54")
    width_axis.tick_params(axis="y", colors="#9c3d54")

    for axis in axes.ravel():
        style_axis(axis, x)
    for axis in axes[-1]:
        axis.set_xlabel("spectra available to STRIDER")
    sample_label = "full redshift range" if regime == "all" else "z < 2"
    figure.suptitle(
        f"STRIDER evidence accumulation ({sample_label})",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5, 0.005,
        f"Fixed {int(values['n_objects'].iloc[0]):,}-object pilot cohort; "
        "chronological prefixes; current binary checkpoint; stored observed flux.",
        ha="center", fontsize=8.5, color="0.35",
    )
    figure.tight_layout(rect=(0, 0.025, 1, 0.96))
    suffix = "full" if regime == "all" else "zlt2"
    path = output_dir / f"v3_sundial_visit_accumulation_{suffix}.png"
    figure.savefig(path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return path


def plot_by_redshift(metrics: pd.DataFrame, output_dir: Path) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.8), sharex=True)
    columns = (
        ("purity_pia_ge_0p9", "Ia purity", (0.0, 1.03)),
        ("completeness_pia_ge_0p9", "Ia completeness", (0.0, 1.03)),
        ("true_ia_median_absolute_delta_z", r"median $|\Delta z|$", None),
        ("true_ia_fraction_abs_delta_z_lt_0p1", r"fraction $|\Delta z|<0.1$", (0.0, 1.03)),
    )
    x = np.sort(metrics["prefix_spectra"].unique()).astype(float)
    for axis, (column, title, limits) in zip(axes.ravel(), columns):
        for color, (label, group) in zip(PALETTE, metrics.groupby("redshift_bin", sort=False)):
            group = group.sort_values("prefix_spectra")
            axis.plot(
                group["prefix_spectra"], group[column], color=color,
                marker="o", markersize=4, lw=1.8, label=label,
            )
        axis.set_title(title, loc="left", fontweight="bold")
        if limits is not None:
            axis.set_ylim(*limits)
        style_axis(axis, x)
    axes[0, 0].legend(title="true redshift", frameon=False, ncol=2, fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("spectra available to STRIDER")
    figure.suptitle(
        "STRIDER accumulation by true redshift",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path = output_dir / "v3_sundial_visit_accumulation_by_redshift.png"
    figure.savefig(path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return path


def plot_v2_reference(
    current: pd.DataFrame, reference_path: Path, output_dir: Path
) -> Path | None:
    if not reference_path.is_file():
        return None
    v2 = pd.read_csv(reference_path).sort_values("prefix_epochs")
    v3 = current[current["regime"].eq("z<2")].sort_values("prefix_spectra")
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), sharex=False)
    panels = (
        ("purity_pia_ge_0p9", "gold_purity", "purity", (0.0, 1.03)),
        ("completeness_pia_ge_0p9", "gold_completeness", "completeness", (0.0, 1.03)),
        ("true_ia_nmad", "true_ia_nmad", "true-Ia normalized NMAD", None),
    )
    for axis, (v3_name, v2_name, title, limits) in zip(axes, panels):
        axis.plot(
            v3["prefix_spectra"], v3[v3_name], color="#007c91", marker="o",
            lw=2.2, label="v3 · Sundial · no z prior",
        )
        axis.plot(
            v2["prefix_epochs"], v2[v2_name], color="#8c6bb1", marker="s",
            ls="--", lw=1.8, label="v2 legacy reference",
        )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("spectra available")
        if limits is not None:
            axis.set_ylim(*limits)
        all_x = np.unique(np.r_[v3["prefix_spectra"], v2["prefix_epochs"]]).astype(float)
        style_axis(axis, all_x)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Visit-prefix context: current and legacy diagnostics", fontweight="bold")
    figure.text(
        0.5, -0.02,
        "Context only—not a controlled model comparison: v2 used 120 different objects and a broad photo-z prior; v3 uses 500 Sundial objects with no redshift prior.",
        ha="center", fontsize=8.2, color="0.35",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    path = output_dir / "v2_v3_visit_accumulation_context.png"
    figure.savefig(path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return path


def select_examples(
    predictions: pd.DataFrame,
    objects: pd.DataFrame,
    targets: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5),
    count_per_target: int = 1,
) -> list[tuple[float, int]]:
    table = predictions.merge(
        objects[["snid", "observation_count"]], on="snid", how="left"
    )
    table = table[table["true_class"].eq(0)].copy()
    table["absolute_delta_z"] = np.abs(
        table["predicted_redshift"] - table["true_redshift"]
    )
    chosen: list[tuple[float, int]] = []
    used: set[int] = set()
    for target in targets:
        candidates = table[~table["snid"].isin(used)].copy()
        candidates["score"] = (
            np.abs(candidates["true_redshift"] - target)
            + 0.15 * candidates["absolute_delta_z"]
            + 0.05 * (1.0 - candidates["p_Ia"])
            + 0.15 * candidates["observation_count"].lt(32)
        )
        for row in candidates.sort_values("score").head(count_per_target).itertuples():
            snid = int(row.snid)
            used.add(snid)
            chosen.append((target, snid))
    return chosen


def render_maps_and_gifs(
    config: dict[str, Any],
    model: Strider,
    device: torch.device,
    dataset: SundialDataset,
    items: list[dict[str, torch.Tensor]],
    full_predictions: pd.DataFrame,
    output_dir: Path,
    examples_per_redshift: int = 1,
) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    map_dir = output_dir / "evidence_maps"
    gif_dir = output_dir / "evidence_gifs"
    map_dir.mkdir(parents=True, exist_ok=True)
    gif_dir.mkdir(parents=True, exist_ok=True)
    bank = load_onir_bank(Path(config["onir"]["bank_path"]))
    class_names = list(model.class_names)
    index_by_snid = {int(item["snid"]): index for index, item in enumerate(items)}
    examples = select_examples(
        full_predictions,
        dataset.objects,
        count_per_target=examples_per_redshift,
    )
    map_paths: list[Path] = []
    gif_paths: list[Path] = []
    trajectory_paths: list[Path] = []
    trajectory_plot_paths: list[Path] = []

    for target, snid in examples:
        item = items[index_by_snid[snid]]
        snapshot = _evidence_snapshot(model, item, device)
        figure = plt.figure(figsize=(13.0, 15.5), constrained_layout=True)
        draw_evidence_map(
            figure,
            item=item,
            wavelength_angstrom=dataset.output_wavelength,
            class_names=class_names,
            feature_names=list(bank.feature_names),
            redshift_grid=model.redshift_grid.cpu().numpy(),
            redshift_cell_width=model.redshift_cell_width.cpu().numpy(),
            joint_probability_mass=snapshot["joint"],
            feature_evidence=snapshot["feature_evidence"],
            feature_support=snapshot["feature_support"],
            route_evidence=snapshot["route_evidence"],
            route_support=snapshot["route_support"],
            predicted_class=snapshot["predicted_class"],
            predicted_redshift=snapshot["predicted_redshift"],
            evidence_sufficiency=snapshot["evidence_score"],
            split="Sundial test",
            view="stored observed flux",
        )
        path = map_dir / f"v3_sundial_z{target:.2f}_Ia_snid{snid}.png"
        figure.savefig(path, dpi=170, facecolor="white")
        plt.close(figure)
        map_paths.append(path)

    representative = examples[::examples_per_redshift]
    gif_examples = (representative[0], representative[len(representative) // 2], representative[-1])
    for target, snid in gif_examples:
        full_item = items[index_by_snid[snid]]
        total = int(full_item["flux"].shape[0])
        counts = sorted(set(min(value, total) for value in (1, 2, 4, 8, 16, 24, 32)))
        frames = []
        for count in counts:
            prefix = _visit_prefix(full_item, count)
            frames.append((count, prefix, _evidence_snapshot(model, prefix, device)))
        redshift_grid = model.redshift_grid.cpu().numpy()
        redshift_cell_width = model.redshift_cell_width.cpu().numpy()
        trajectory = [
            _basin_trajectory_row(
                snapshot,
                count,
                redshift_grid,
                redshift_cell_width,
                previous=frames[index - 1][2] if index else None,
            )
            for index, (count, _, snapshot) in enumerate(frames)
        ]
        figure = plt.figure(figsize=(13.0, 15.5), constrained_layout=True)

        def draw(frame_index: int) -> tuple[()]:
            figure.clear()
            count, prefix, snapshot = frames[frame_index]
            draw_evidence_map(
                figure,
                item=prefix,
                wavelength_angstrom=dataset.output_wavelength,
                class_names=class_names,
                feature_names=list(bank.feature_names),
                redshift_grid=model.redshift_grid.cpu().numpy(),
                redshift_cell_width=model.redshift_cell_width.cpu().numpy(),
                joint_probability_mass=snapshot["joint"],
                feature_evidence=snapshot["feature_evidence"],
                feature_support=snapshot["feature_support"],
                route_evidence=snapshot["route_evidence"],
                route_support=snapshot["route_support"],
                predicted_class=snapshot["predicted_class"],
                predicted_redshift=snapshot["predicted_redshift"],
                evidence_sufficiency=snapshot["evidence_score"],
                split="Sundial test",
                view="stored observed flux",
            )
            figure.text(
                0.99, 0.995, f"{count} of {total} spectra", ha="right", va="top",
                fontsize=10, color="0.3",
            )
            basin = trajectory[frame_index]
            figure.text(
                0.99,
                0.979,
                f"{int(basin['basin_count'])} basin"
                f"{'s' if int(basin['basin_count']) != 1 else ''}  ·  "
                f"primary mass {basin['primary_mass']:.2f}  ·  "
                "alternate/primary mass "
                f"{basin['secondary_to_primary_mass_ratio']:.2f}",
                ha="right",
                va="top",
                fontsize=9,
                color="0.3",
            )
            return ()

        movie = animation.FuncAnimation(
            figure,
            draw,
            frames=list(range(len(frames))) + [len(frames) - 1],
            interval=1200,
            repeat_delay=1600,
            blit=False,
        )
        path = gif_dir / f"v3_sundial_accumulation_z{target:.2f}_Ia_snid{snid}.gif"
        movie.save(path, writer=animation.PillowWriter(fps=0.85), dpi=95)
        plt.close(figure)
        gif_paths.append(path)
        trajectory_path = path.with_suffix(".csv")
        trajectory_frame = pd.DataFrame(trajectory)
        trajectory_frame.to_csv(trajectory_path, index=False, float_format="%.6g")
        trajectory_paths.append(trajectory_path)

        trajectory_plot = path.with_name(path.stem + "_basins.png")
        plot_basin_trajectory(
            trajectory_frame,
            true_redshift=float(full_item["redshift"]),
            output=trajectory_plot,
        )
        trajectory_plot_paths.append(trajectory_plot)
    return map_paths, gif_paths, trajectory_paths, trajectory_plot_paths


def plot_basin_trajectory(
    trajectory: pd.DataFrame,
    *,
    true_redshift: float,
    output: Path,
) -> None:
    """Show how the dominant and competing redshift basins evolve by visit."""
    visits = trajectory["visit_count"].to_numpy()
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.25))

    axes[0].axhline(true_redshift, color="0.2", lw=1.2, label="true redshift")
    axes[0].plot(
        visits,
        trajectory["primary_peak_redshift"],
        color="#007c91",
        marker="o",
        lw=2.0,
        label="primary",
    )
    secondary = trajectory["secondary_peak_redshift"].to_numpy(dtype=float)
    if np.isfinite(secondary).any():
        axes[0].plot(
            visits,
            secondary,
            color="#e76f51",
            marker="o",
            ls="--",
            lw=1.5,
            label="runner-up",
        )
    axes[0].set_ylabel("redshift")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(
        visits,
        trajectory["primary_mass"],
        color="#007c91",
        marker="o",
        lw=2.0,
        label="primary mass",
    )
    axes[1].plot(
        visits,
        trajectory["secondary_to_primary_mass_ratio"],
        color="#e76f51",
        marker="o",
        lw=1.6,
        label="alternate/primary mass",
    )
    ratio_maximum = float(
        np.nanmax(trajectory["secondary_to_primary_mass_ratio"].to_numpy())
    )
    axes[1].set_ylim(-0.02, max(1.02, 1.08 * ratio_maximum))
    axes[1].set_ylabel("posterior fraction")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].step(
        visits,
        trajectory["basin_count"],
        where="mid",
        color="#6a4c93",
        lw=2.0,
        label="basins",
    )
    axes[2].set_ylabel("basin count")
    axes[2].yaxis.get_major_locator().set_params(integer=True)
    axes[2].set_ylim(0.5, max(1.5, float(trajectory["basin_count"].max()) + 0.5))
    evidence_axis = axes[2].twinx()
    evidence_axis.plot(
        visits,
        trajectory["evidence_score"],
        color="#2a9d8f",
        marker="o",
        lw=1.6,
        label="evidence",
    )
    evidence_axis.set_ylim(-0.02, 1.02)
    evidence_axis.set_ylabel("evidence score", color="#2a9d8f")
    evidence_axis.tick_params(axis="y", colors="#2a9d8f")

    for axis in axes:
        axis.set_xlabel("spectra accumulated")
        axis.spines[["top", "right"]].set_visible(False)
    axes[2].spines["right"].set_visible(True)
    figure.tight_layout()
    figure.savefig(output, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.maximum_objects < 0 or args.examples_per_redshift < 1:
        raise ValueError("batch size must be positive and maximum objects nonnegative")
    if args.maps_only and args.predictions_only:
        raise ValueError("--maps-only and --predictions-only are mutually exclusive")
    prefixes = sorted(set(int(value) for value in args.prefixes))
    if not prefixes or prefixes[0] < 1:
        raise ValueError("prefixes must contain positive integers")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config, checkpoint = runtime_config(args)
    model, device = load_model(config, checkpoint)
    dataset = SundialDataset(config, args.split, args.view, training=False)
    positions = np.arange(len(dataset))
    if args.maximum_objects:
        positions = positions[: args.maximum_objects]
    print(f"Loading {len(positions):,} fixed Sundial objects...", flush=True)
    items = [dataset[int(position)] for position in positions]

    if args.predictions_only:
        for count in prefixes:
            print(f"Running chronological prefix {count}...", flush=True)
            predictions = predict_prefix(model, device, items, count, args.batch_size)
            path = args.output_dir / f"v3_sundial_prefix_{count:02d}_predictions.csv"
            predictions.to_csv(path, index=False, float_format="%.5g")
            print(path, flush=True)
        return

    if args.maps_only:
        packaged_predictions = args.model_package / "test_predictions_original.parquet"
        predictions = pd.read_parquet(packaged_predictions)
        predictions = predictions[predictions["snid"].isin(
            [int(item["snid"]) for item in items]
        )].copy()
        maps, gifs, trajectories, trajectory_plots = render_maps_and_gifs(
            config,
            model,
            device,
            dataset,
            items,
            predictions,
            args.output_dir,
            examples_per_redshift=args.examples_per_redshift,
        )
        report_path = args.output_dir / "local_evidence_maps_manifest.json"
        report_path.write_text(
            json.dumps(
                {
                    "checkpoint_epoch": int(checkpoint["epoch"]),
                    "objects_available": len(items),
                    "evidence_maps": [str(path) for path in maps],
                    "evidence_gifs": [str(path) for path in gifs],
                    "basin_trajectories": [str(path) for path in trajectories],
                    "basin_trajectory_plots": [
                        str(path) for path in trajectory_plots
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Evidence maps: {len(maps)}", flush=True)
        print(f"Evidence GIFs: {len(gifs)}", flush=True)
        print(f"Manifest: {report_path}", flush=True)
        return

    metric_rows: list[dict[str, Any]] = []
    redshift_rows: list[dict[str, Any]] = []
    predictions_by_prefix: dict[int, pd.DataFrame] = {}
    for count in prefixes:
        print(f"Running chronological prefix {count}...", flush=True)
        predictions = predict_prefix(model, device, items, count, args.batch_size)
        predictions_by_prefix[count] = predictions
        metric_rows.extend(
            metric_row(predictions, count, regime) for regime in ("z<2", "all")
        )
        redshift_rows.extend(metrics_by_redshift(predictions, count))

    metrics = pd.DataFrame(metric_rows)
    by_redshift = pd.DataFrame(redshift_rows)
    metrics_path = args.output_dir / "v3_sundial_visit_accumulation_summary.csv"
    redshift_path = args.output_dir / "v3_sundial_visit_accumulation_by_redshift.csv"
    metrics.to_csv(metrics_path, index=False, float_format="%.4f")
    by_redshift.to_csv(redshift_path, index=False, float_format="%.4f")
    plot_paths = [
        plot_accumulation(metrics, args.output_dir, "all"),
        plot_accumulation(metrics, args.output_dir, "z<2"),
        plot_by_redshift(by_redshift, args.output_dir),
    ]
    comparison = plot_v2_reference(metrics, args.v2_reference, args.output_dir)
    if comparison is not None:
        plot_paths.append(comparison)

    full_count = max(prefixes)
    maps, gifs, trajectories, trajectory_plots = render_maps_and_gifs(
        config,
        model,
        device,
        dataset,
        items,
        predictions_by_prefix[full_count],
        args.output_dir,
        examples_per_redshift=args.examples_per_redshift,
    )
    report = {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_role": checkpoint.get("checkpoint_role"),
        "model_package": str(args.model_package.resolve()),
        "prepared_data": str(args.prepared_data.resolve()),
        "split": args.split,
        "view": args.view,
        "objects": len(items),
        "prefixes": prefixes,
        "selection_threshold": 0.9,
        "plots": [str(path) for path in plot_paths],
        "evidence_maps": [str(path) for path in maps],
        "evidence_gifs": [str(path) for path in gifs],
        "basin_trajectories": [str(path) for path in trajectories],
        "basin_trajectory_plots": [str(path) for path in trajectory_plots],
        "notes": [
            "Every prefix uses the same object cohort and chronological first-k spectra.",
            "Objects with fewer than k spectra retain all spectra available to them.",
            "The v2 overlay is contextual only because it used different objects and a broad redshift prior.",
            "Simulation truth is used only to score and annotate the held-out examples.",
        ],
    }
    report_path = args.output_dir / "local_evidence_accumulation_manifest.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Summary: {metrics_path}", flush=True)
    print(f"By redshift: {redshift_path}", flush=True)
    print(f"Evidence maps: {len(maps)}", flush=True)
    print(f"Evidence GIFs: {len(gifs)}", flush=True)
    print(f"Manifest: {report_path}", flush=True)


if __name__ == "__main__":
    main()
