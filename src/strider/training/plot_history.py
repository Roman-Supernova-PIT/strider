"""Plot the scientific progress of a STRIDER training run.

The input is the JSON history written after every completed epoch.  The figure
separates optimisation, Ia performance, evidence sufficiency, loss components,
and the learned information-route scales.  It does not read training data or
modify a checkpoint.
"""

from __future__ import annotations

import json
import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from strider.config import project_path


def plot_training_history(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = project_path(config, config["project"]["output_dir"])
    history_path = output_dir / "training_history.json"
    if not history_path.is_file():
        raise FileNotFoundError(f"Training history not found: {history_path}")

    with history_path.open("r", encoding="utf-8") as handle:
        history = json.load(handle)
    if not isinstance(history, list) or not history:
        raise ValueError(f"Training history is empty: {history_path}")

    epochs = np.asarray([int(row["epoch"]) for row in history])
    selection = _series(history, "selection_score")
    finite_selection = np.flatnonzero(np.isfinite(selection))
    if not len(finite_selection):
        raise ValueError(f"Training history has no finite selection score: {history_path}")
    best_index = int(finite_selection[np.argmin(selection[finite_selection])])
    checkpoint_epochs = _checkpoint_epochs(epochs, history, best_index)

    figure, axes = plt.subplots(3, 2, figsize=(12.5, 10.5), constrained_layout=True)
    _plot_objective(axes[0, 0], epochs, history, selection, best_index)
    _plot_ia_metrics(axes[0, 1], epochs, history)
    _plot_evidence(axes[1, 0], epochs, history)
    _plot_loss_components(axes[1, 1], epochs, history)
    _plot_route_scales(axes[2, 0], epochs, history)
    _plot_learning_rate(axes[2, 1], epochs, history)

    name = _display_name(str(config["project"].get("name", "run")))
    checkpoint_text = " · ".join(
        f"{role} {epoch}"
        for role, epoch in checkpoint_epochs.items()
        if epoch is not None
    )
    title = f"STRIDER · {name}"
    if checkpoint_text:
        title += f"\nCheckpoints · {checkpoint_text}"
    figure.suptitle(title, fontsize=15)
    figure_path = output_dir / "training_progress.png"
    figure.savefig(figure_path, dpi=170)
    plt.close(figure)
    return {
        "epochs": int(len(history)),
        "best_epoch": int(epochs[best_index]),
        "best_selection_score": float(selection[best_index]),
        "checkpoint_epochs": checkpoint_epochs,
        "figure": str(figure_path),
    }


def _plot_objective(
    axis: plt.Axes,
    epochs: np.ndarray,
    history: list[dict[str, Any]],
    selection: np.ndarray,
    best_index: int,
) -> None:
    axis.plot(epochs, _series(history, "train", "loss"), label="training")
    axis.plot(epochs, selection, label="selection")
    axis.scatter(
        epochs[best_index],
        selection[best_index],
        marker="*",
        s=130,
        color="#d95f02",
        zorder=3,
        label=f"lowest selection loss · epoch {epochs[best_index]}",
    )
    axis.set_title("Loss")
    axis.set_ylabel("loss per object")
    axis.legend(frameon=False)
    _finish_epoch_axis(axis)


def _plot_ia_metrics(
    axis: plt.Axes, epochs: np.ndarray, history: list[dict[str, Any]]
) -> None:
    found = False
    for preferred, fallback, label in (
        ("ia_z_lt_2_precision", "ia_precision", "Ia purity"),
        ("ia_z_lt_2_recall", "ia_recall", "Ia completeness"),
        ("ia_z_lt_2_f1", "ia_f1", "Ia F1"),
        ("z_lt_2_macro_f1_present", None, "macro F1"),
    ):
        values = _validation_series(history, preferred, fallback)
        if np.isfinite(values).any():
            axis.plot(epochs, values, label=label)
            found = True
    redshift_error = _validation_series(
        history,
        "ia_z_lt_2_median_absolute_delta_z",
        "ia_median_absolute_delta_z",
    )
    redshift_scatter = _validation_series(
        history,
        "ia_z_lt_2_population_scatter_delta_z",
        None,
    )
    finite_error = np.isfinite(redshift_error)
    finite_scatter = np.isfinite(redshift_scatter)
    error_axis = None
    if finite_error.any() or finite_scatter.any():
        error_axis = axis.twinx()
        if finite_error.any():
            error_axis.plot(
                epochs[finite_error],
                redshift_error[finite_error],
                color="#d95f02",
                linestyle="--",
                label=r"median $|\Delta z|$",
            )
        if finite_scatter.any():
            error_axis.plot(
                epochs[finite_scatter],
                redshift_scatter[finite_scatter],
                color="#7570b3",
                linestyle=":",
                label=r"$\Delta z$ scatter",
            )
        error_axis.set_yscale("log")
        error_axis.set_ylabel("redshift error")
        found = True
    axis.set_title(r"Validation at $z < 2$")
    axis.set_ylabel("score")
    axis.set_ylim(-0.03, 1.03)
    if found:
        handles, labels = axis.get_legend_handles_labels()
        if error_axis is not None:
            error_handles, error_labels = error_axis.get_legend_handles_labels()
            handles.extend(error_handles)
            labels.extend(error_labels)
        axis.legend(handles, labels, frameon=False, ncol=2, fontsize=8)
    else:
        _unavailable(axis)
    _finish_epoch_axis(axis)


def _plot_evidence(
    axis: plt.Axes, epochs: np.ndarray, history: list[dict[str, Any]]
) -> None:
    found = False
    for key, label in (
        ("source_mean_evidence_sufficiency", "transient spectra"),
        ("no_source_mean_evidence_sufficiency", "blank controls"),
    ):
        values = _series(history, "validation", key)
        if np.isfinite(values).any():
            axis.plot(epochs, values, label=label)
            found = True
    axis.set_title("Evidence")
    axis.set_ylabel("mean evidence score")
    axis.set_ylim(-0.03, 1.03)
    if found:
        axis.legend(frameon=False)
    else:
        _unavailable(axis)
    _finish_epoch_axis(axis)


def _plot_loss_components(
    axis: plt.Axes, epochs: np.ndarray, history: list[dict[str, Any]]
) -> None:
    found = False
    for key, label in (
        ("joint_loss", "class + redshift"),
        ("evidence_sufficiency_loss", "evidence score"),
        ("no_source_redshift_loss", "blank redshift"),
        ("no_source_class_loss", "blank class"),
        ("phase_loss", "phase"),
        ("alias_ranking_loss", "distant alias ranking"),
    ):
        values = _series(history, "validation", key)
        positive = np.isfinite(values) & (values > 0.0)
        if positive.any():
            axis.plot(epochs[positive], values[positive], label=label)
            found = True
    axis.set_title("Validation loss terms")
    axis.set_ylabel("loss per object")
    if found:
        axis.set_yscale("log")
        axis.legend(frameon=False, fontsize=8)
    else:
        _unavailable(axis)
    _finish_epoch_axis(axis)


def _plot_route_scales(
    axis: plt.Axes, epochs: np.ndarray, history: list[dict[str, Any]]
) -> None:
    names = sorted(
        {
            str(name)
            for row in history
            for name in row.get("learned_scales", {})
        }
    )
    labels = {
        "context": "spectral context gate",
        "dense": "complete-spectrum scan gate",
        "dense_detail": "continuum-subtracted fraction",
        "shape": "feature-shape gate",
        "temporal": "temporal gate",
        "flux_evolution": "relative-flux gate",
        "background_scaled_amplitude": "amplitude gate",
    }
    for name in names:
        values = np.asarray(
            [float(row.get("learned_scales", {}).get(name, np.nan)) for row in history]
        )
        style = "--" if name == "dense_detail" else "-"
        axis.plot(
            epochs,
            values,
            linestyle=style,
            label=labels.get(name, name.replace("_", " ")),
        )
    axis.axhline(0.0, color="0.75", linewidth=0.8, zorder=0)
    axis.set_title("Route settings")
    axis.set_ylabel("gate / mixture weight")
    if names:
        axis.legend(frameon=False)
    else:
        _unavailable(axis)
    _finish_epoch_axis(axis)


def _plot_learning_rate(
    axis: plt.Axes, epochs: np.ndarray, history: list[dict[str, Any]]
) -> None:
    learning_rate = _series(history, "learning_rate")
    positive = np.isfinite(learning_rate) & (learning_rate > 0.0)
    axis.set_title("Learning rate")
    axis.set_ylabel("learning rate")
    if positive.any():
        axis.plot(epochs[positive], learning_rate[positive], color="#7570b3")
        axis.set_yscale("log")
    else:
        _unavailable(axis)
    _finish_epoch_axis(axis)


def _series(history: list[dict[str, Any]], *keys: str) -> np.ndarray:
    values = []
    for row in history:
        value: Any = row
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                value = float("nan")
                break
            value = value[key]
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float("nan")
        values.append(number if math.isfinite(number) else float("nan"))
    return np.asarray(values, dtype=np.float64)


def _validation_series(
    history: list[dict[str, Any]],
    preferred: str,
    fallback: str | None,
) -> np.ndarray:
    preferred_values = _series(history, "validation", preferred)
    if fallback is None:
        return preferred_values
    fallback_values = _series(history, "validation", fallback)
    use_fallback = ~np.isfinite(preferred_values)
    preferred_values[use_fallback] = fallback_values[use_fallback]
    return preferred_values


def _checkpoint_epochs(
    epochs: np.ndarray,
    history: list[dict[str, Any]],
    posterior_index: int,
) -> dict[str, int | None]:
    """Recover the three checkpoint roles directly from the saved history."""

    def optimum(key: str, *, maximize: bool) -> int | None:
        values = _series(history, key)
        finite = np.flatnonzero(np.isfinite(values))
        if not len(finite):
            return None
        local = np.argmax(values[finite]) if maximize else np.argmin(values[finite])
        return int(epochs[int(finite[int(local)])])

    return {
        "posterior": int(epochs[posterior_index]),
        "science": optimum("science_score", maximize=True),
        "redshift": optimum("redshift_outlier_score", maximize=False),
    }


def _display_name(raw_name: str) -> str:
    normalized = raw_name.lower().strip()
    if normalized.startswith("ia_binary"):
        return "Ia binary"
    if normalized.startswith("grouped_7"):
        return "7 classes"
    if normalized.startswith("multiclass_15"):
        return "15 classes"
    return raw_name.replace("_", " ").strip().title()


def _finish_epoch_axis(axis: plt.Axes) -> None:
    axis.set_xlabel("completed epoch")


def _unavailable(axis: plt.Axes) -> None:
    axis.text(
        0.5,
        0.5,
        "available in newer training histories",
        ha="center",
        va="center",
        transform=axis.transAxes,
        color="0.45",
    )
