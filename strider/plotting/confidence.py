"""Class-probability growth with cumulative spectral signal-to-noise."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def epoch_snr(flux: np.ndarray, flux_err: np.ndarray) -> np.ndarray:
    """Median |flux| / flux_err for each spectrum."""
    values = np.atleast_2d(np.asarray(flux, dtype=float))
    errors = np.atleast_2d(np.asarray(flux_err, dtype=float))
    if values.shape != errors.shape:
        raise ValueError("flux and flux_err must have the same shape")

    result = np.zeros(values.shape[0], dtype=float)
    for index, (spectrum, error) in enumerate(zip(values, errors)):
        valid = np.isfinite(spectrum) & np.isfinite(error) & (error > 0)
        if valid.any():
            result[index] = float(np.median(np.abs(spectrum[valid]) / error[valid]))
    return result


def cumulative_snr(flux: np.ndarray, flux_err: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Quadrature sum of per-spectrum S/N in phase order."""
    order = np.argsort(np.asarray(phase, dtype=float))
    per_epoch = epoch_snr(flux, flux_err)[order]
    return np.sqrt(np.cumsum(np.square(per_epoch)))


def confidence_curve(
    ax: Any,
    *,
    outputs: Sequence[dict[str, Any]],
    flux: np.ndarray,
    flux_err: np.ndarray,
    phase: np.ndarray,
    true_class: str | None = None,
) -> None:
    """Plot cumulative S/N against binary Ia probability."""
    if not outputs:
        raise ValueError("confidence plot needs at least one classification")

    names = list(outputs[-1]["class_names"])
    if "Ia" not in names:
        raise ValueError("confidence plot needs an Ia class")

    snr = cumulative_snr(flux, flux_err, phase)
    phases = np.sort(np.asarray(phase, dtype=float))
    ia_index = names.index("Ia")
    p_ia = np.asarray(
        [
            float(
                output.get("classification", {}).get(
                    "p_Ia",
                    np.asarray(output["p_class"], dtype=float)[ia_index],
                )
            )
            for output in outputs
        ],
        dtype=float,
    )
    probabilities = {"Ia": p_ia, "non-Ia": 1.0 - p_ia}
    truth_group = None if true_class is None else ("Ia" if true_class == "Ia" else "non-Ia")
    for name, color in (("Ia", "#10cbd4"), ("non-Ia", "#6b42dc")):
        is_truth = name == truth_group
        label = f"{name} (true)" if is_truth else name
        ax.plot(
            snr,
            probabilities[name],
            marker="o",
            markersize=5 if is_truth else 4,
            linewidth=2.4 if is_truth else 1.6,
            color=color,
            alpha=1.0 if is_truth or truth_group is None else 0.8,
            label=label,
        )

    annotation_class = truth_group or ("Ia" if p_ia[-1] >= 0.5 else "non-Ia")
    annotated_probability = probabilities[annotation_class]
    label_indices = [0] if len(phases) == 1 else [0, len(phases) - 1]
    for point_index in label_indices:
        first = point_index == 0
        count_label = "1 spectrum" if first else f"{point_index + 1} spectra"
        ax.annotate(
            f"{count_label}\n{phases[point_index]:+.0f} d",
            (snr[point_index], annotated_probability[point_index]),
            xytext=(5 if first else -5, 6 if first else -8),
            textcoords="offset points",
            fontsize=8,
            color="#4f5660",
            ha="left" if first else "right",
            va="bottom" if first else "top",
        )

    ax.set_xlim(left=0)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("cumulative spectral S/N")
    ax.set_ylabel("probability")
    ax.grid(axis="both", color="#d9dde3", linewidth=0.8, alpha=0.65)
    ax.legend(frameon=False, loc="best")
