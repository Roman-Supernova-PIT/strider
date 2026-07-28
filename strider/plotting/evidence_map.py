"""Compact STRIDER evidence map."""

from __future__ import annotations

from typing import Any

import numpy as np


STRIDER_COLOR = "#00eaff"
SECONDARY_COLOR = "#666666"
TRUTH_COLOR = "#39ff14"
PHASE_CMAP = "cmr.guppy_r"
EVIDENCE_CMAP = "cmr.ember"

def _evidence_grid(joint_log_probs: np.ndarray) -> np.ndarray:
    joint = np.asarray(joint_log_probs, dtype=float)
    rel = np.exp(joint - np.nanmax(joint))
    peak = np.nanmax(rel)
    return rel / peak if peak > 0 else rel


def _spectra_matrix(
    spectra: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if "wavelength_aa" not in spectra or "flux" not in spectra:
        return None
    wave = np.asarray(spectra["wavelength_aa"], dtype=float)
    flux = np.atleast_2d(np.asarray(spectra["flux"], dtype=float))
    phases = np.atleast_1d(np.asarray(spectra.get("phase_days", []), dtype=float))
    if phases.size != flux.shape[0] or wave.size != flux.shape[1]:
        return None
    return wave, flux, phases


def _plot_window(z_grid: np.ndarray, values: list[float | None]) -> tuple[float, float]:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not finite:
        return float(z_grid[0]), float(z_grid[-1])
    lo, hi = min(finite), max(finite)
    if hi - lo < 0.08:
        center = 0.5 * (lo + hi)
        lo, hi = center - 0.04, center + 0.04
    span = hi - lo
    pad = 0.18 * span
    return max(float(z_grid[0]), lo - pad), min(float(z_grid[-1]), hi + pad)


def _context_title(controls: dict[str, Any]) -> str:
    if controls.get("z_prior"):
        return "With prior"
    if controls.get("z_window"):
        return "With redshift window"
    return "Spectra only"


def _draw_spectra(ax, spectra: dict[str, Any]) -> None:
    import cmasher  # noqa: F401
    import matplotlib as mpl
    import matplotlib.colors as mcolors

    panel = _spectra_matrix(spectra)
    if panel is None:
        ax.text(0.5, 0.5, "Spectra not retained", ha="center", va="center")
        ax.axis("off")
        return

    wave, flux, phases = panel
    if np.nanmax(wave) < 100.0:
        wave = wave * 1.0e4
    order = np.argsort(phases)
    norm = mcolors.Normalize(
        vmin=float(np.nanmin(phases)),
        vmax=max(float(np.nanmax(phases)), float(np.nanmin(phases)) + 1.0),
    )
    cmap = mpl.colormaps[PHASE_CMAP]
    finite_flux = flux[np.isfinite(flux)]
    positive_flux = finite_flux[finite_flux > 0]
    if positive_flux.size:
        scale = float(np.nanpercentile(positive_flux, 99.5))
    elif finite_flux.size:
        scale = float(np.nanpercentile(np.abs(finite_flux), 99.5))
    else:
        scale = 1.0
    scale = max(scale, 1.0e-30)
    lines = []

    for epoch_index in order:
        epoch = flux[epoch_index]
        valid = np.isfinite(epoch)
        if valid.sum() < 2:
            continue
        line, = ax.plot(
            wave[valid],
            epoch[valid] / scale,
            color=cmap(norm(phases[epoch_index])),
            linewidth=1.25,
            label=f"{phases[epoch_index]:+.0f} d",
        )
        lines.append(line)

    if lines:
        legend_lines = lines
        if len(lines) > 8:
            keep = np.unique(np.linspace(0, len(lines) - 1, 6, dtype=int))
            legend_lines = [lines[index] for index in keep]
        ax.legend(
            handles=legend_lines,
            loc="upper right",
            ncols=min(len(legend_lines), 6),
            fontsize=8,
            frameon=False,
            handlelength=2.0,
            columnspacing=1.2,
        )
    ax.set_yticks([])
    ax.set_xlabel(r"observed wavelength [$\AA$]")
    ax.set_ylabel("relative flux")
    ax.set_title("Spectra", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)


def _draw_class_probabilities(ax, out: dict[str, Any]) -> None:
    names = np.asarray(out["class_names"])
    probabilities = np.asarray(out["p_class"], dtype=float)
    ranking = np.asarray(out["p_class_uncal_with_controls"], dtype=float)
    top = np.argsort(ranking)[::-1][:3]
    labels = names[top]
    values = probabilities[top]
    predicted = str(out["strider_class"])
    colors = [STRIDER_COLOR if label == predicted else "#9aa0a6" for label in labels]

    rows = np.arange(top.size)
    ax.barh(rows, values, color=colors, height=0.62)
    ax.set_yticks(rows)
    ax.set_yticklabels(labels, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(r"$P(\mathrm{class})$")
    ax.set_title("Class probabilities", loc="left", fontweight="bold")
    for row, value in zip(rows, values):
        label = "<0.001" if 0 < value < 0.001 else f"{value:.3f}"
        ax.text(
            min(value + 0.015, 0.94),
            row,
            label,
            va="center",
            fontsize=8.5,
            fontweight="bold" if row == 0 else "normal",
        )
    ax.grid(axis="x", color="#eeeeee", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def _draw_redshift_posterior(
    ax,
    out: dict[str, Any],
    meta: dict[str, Any],
    xlim: tuple[float, float],
) -> None:
    redshift = out["redshift"]
    z_grid = np.asarray(redshift["z_grid"], dtype=float)
    posterior = np.asarray(redshift["posterior"], dtype=float)
    posterior /= max(float(np.nanmax(posterior)), 1.0e-30)
    controls_applied = bool(out["controls"]["prior_or_window_supplied"])

    label = (
        _context_title(out["controls"]).lower()
        if controls_applied
        else "spectra only"
    )
    ax.plot(z_grid, posterior, color=STRIDER_COLOR, linewidth=2.0, label=label)
    if controls_applied:
        spectra_only = np.asarray(out["z_marginal_uncal_spectra_only"], dtype=float)
        spectra_only /= max(float(np.nanmax(spectra_only)), 1.0e-30)
        ax.plot(
            z_grid,
            spectra_only,
            color=SECONDARY_COLOR,
            linewidth=1.4,
            linestyle="--",
            label="spectra only",
        )
    ax.axvspan(
        redshift["z_p16"],
        redshift["z_p84"],
        color=STRIDER_COLOR,
        alpha=0.13,
        label="68%",
    )
    ax.axvline(
        redshift["z_p05"],
        color=STRIDER_COLOR,
        alpha=0.7,
        linestyle=":",
        linewidth=1.1,
        label="90%",
    )
    ax.axvline(
        redshift["z_p95"],
        color=STRIDER_COLOR,
        alpha=0.7,
        linestyle=":",
        linewidth=1.1,
    )
    ax.axvline(
        redshift["z_STRIDER"],
        color="#111111",
        linewidth=1.2,
        label=r"$z_\mathrm{STRIDER}$",
    )
    if meta.get("z_true") is not None:
        ax.axvline(
            float(meta["z_true"]),
            color=TRUTH_COLOR,
            linewidth=1.3,
            linestyle="-.",
            label="truth",
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([])
    ax.set_xlabel("redshift")
    ax.set_ylabel(r"relative $P(z)$")
    ax.set_title("Redshift posterior", loc="left", fontweight="bold")
    ax.legend(loc="upper right", fontsize=7.5, frameon=False, ncols=2)
    ax.spines[["top", "right"]].set_visible(False)


def _draw_joint_evidence(
    ax,
    joint_log_probs: np.ndarray,
    class_names: list[str],
    z_grid: np.ndarray,
    pred_idx: int,
    z_pred: float,
    meta: dict[str, Any],
    xlim: tuple[float, float],
    title: str,
):
    import cmasher  # noqa: F401
    from matplotlib.colors import LogNorm

    evidence = _evidence_grid(joint_log_probs)
    n_classes = len(class_names)

    im = ax.imshow(
        np.clip(evidence[::-1], 1.0e-3, 1.0),
        origin="lower",
        aspect="auto",
        extent=[float(z_grid[0]), float(z_grid[-1]), -0.5, n_classes - 0.5],
        cmap=EVIDENCE_CMAP,
        norm=LogNorm(vmin=1.0e-3, vmax=1.0),
    )
    ax.set_xlim(*xlim)
    ax.set_yticks(np.arange(n_classes))
    ax.set_yticklabels(class_names[::-1], fontsize=8, fontweight="bold")
    ax.set_xlabel("redshift")
    ax.set_ylabel("class")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.scatter(
        [z_pred],
        [n_classes - 1 - pred_idx],
        marker="D",
        s=105,
        facecolors=STRIDER_COLOR,
        edgecolors="white",
        linewidths=1.2,
        zorder=4,
    )
    true_class = meta.get("true_class")
    if true_class in class_names and meta.get("z_true") is not None:
        ax.scatter(
            [float(meta["z_true"])],
            [n_classes - 1 - class_names.index(true_class)],
            marker="o",
            s=130,
            facecolors="none",
            edgecolors=TRUTH_COLOR,
            linewidths=2.0,
            zorder=5,
        )
    return im


def evidence_map(fig, *, out: dict[str, Any], meta: dict[str, Any]) -> None:
    """Render one deployment-safe STRIDER evidence summary."""
    if "joint_log_probs" not in out:
        raise ValueError("Evidence map requires classify(..., return_joint=True).")
    if "spectra" not in out:
        raise ValueError("Evidence map requires classify(..., return_inputs=True).")

    redshift = out["redshift"]
    z_grid = np.asarray(redshift["z_grid"], dtype=float)
    suffix = "cal" if out["calibrated"] else "uncal"
    xlim = _plot_window(
        z_grid,
        [
            redshift["z_p05"],
            redshift["z_p95"],
            out[f"z_p05_{suffix}_spectra_only"],
            out[f"z_p95_{suffix}_spectra_only"],
            meta.get("z_true"),
        ],
    )
    controls_applied = bool(out["controls"]["prior_or_window_supplied"])
    class_names = list(out["class_names"])

    fig.clear()
    fig.suptitle(
        str(meta.get("object", "STRIDER object")),
        x=0.07,
        y=0.965,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.93,
        0.965,
        f"Class: {out['strider_class']}\n"
        f"Redshift: {redshift['z_STRIDER']:.4f}",
        ha="right",
        va="top",
        fontsize=11,
        fontweight="bold",
        linespacing=1.35,
    )
    if meta.get("frame_label"):
        fig.text(
            0.07,
            0.925,
            str(meta["frame_label"]),
            ha="left",
            va="top",
            fontsize=8.5,
            color=SECONDARY_COLOR,
        )
    grid_top = 0.86 if meta.get("frame_label") else 0.90
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=(1.20, 0.60, 0.58),
        width_ratios=(1.0, 1.0),
        left=0.07,
        right=0.93,
        top=grid_top,
        bottom=0.07,
        hspace=0.50,
        wspace=0.28,
    )
    spectra_ax = fig.add_subplot(grid[1, :])
    if controls_applied:
        spectra_joint_ax = fig.add_subplot(grid[0, 0])
        reported_joint_ax = fig.add_subplot(grid[0, 1])
        joint_axes = [spectra_joint_ax, reported_joint_ax]
    else:
        spectra_joint_ax = fig.add_subplot(grid[0, :])
        reported_joint_ax = None
        joint_axes = [spectra_joint_ax]
    class_ax = fig.add_subplot(grid[2, 0])
    redshift_ax = fig.add_subplot(grid[2, 1])

    _draw_spectra(spectra_ax, out["spectra"])
    spectra_pred_idx = int(np.argmax(out["p_class_uncal_spectra_only"]))
    evidence_image = _draw_joint_evidence(
        spectra_joint_ax,
        np.asarray(out["joint_log_probs_spectra_only"], dtype=float),
        class_names,
        z_grid,
        spectra_pred_idx,
        float(redshift["spectra_only"]["z_S"]),
        meta,
        xlim,
        "Spectra only",
    )
    if reported_joint_ax is not None:
        _draw_joint_evidence(
            reported_joint_ax,
            np.asarray(out["joint_log_probs"], dtype=float),
            class_names,
            z_grid,
            int(out["strider_class_index"]),
            float(redshift["with_controls"]["z_S"]),
            meta,
            xlim,
            _context_title(out["controls"]),
        )
        reported_joint_ax.set_ylabel("")
    colorbar = fig.colorbar(evidence_image, ax=joint_axes, fraction=0.025, pad=0.025)
    colorbar.set_ticks([1.0e-3, 1.0])
    colorbar.set_ticklabels(["low", "high"])
    colorbar.set_label("raw evidence", fontsize=9, fontweight="bold")
    colorbar.ax.tick_params(length=0)
    colorbar.minorticks_off()
    _draw_class_probabilities(class_ax, out)
    _draw_redshift_posterior(redshift_ax, out, meta, xlim)
