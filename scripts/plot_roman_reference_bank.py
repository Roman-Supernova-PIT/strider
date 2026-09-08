#!/usr/bin/env python3
"""Plot the compact physical content and redshift coverage of a reference bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from strider.atlas.roman_reference import RomanReferenceBank


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _normalise(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    measured = np.asarray(mask, dtype=bool) & np.isfinite(values)
    output = np.full(values.shape, np.nan, dtype=np.float64)
    if measured.sum() < 2:
        return output
    selected = np.asarray(values, dtype=np.float64)[measured]
    selected -= np.median(selected)
    scale = np.sqrt(np.mean(np.square(selected)))
    if not np.isfinite(scale) or scale <= 0.0:
        return output
    output[measured] = selected / scale
    return output


def _profile_rows(
    values: np.ndarray,
    masks: np.ndarray,
    support: np.ndarray,
) -> list[np.ndarray]:
    return [
        _normalise(values[index], masks[index])
        for index in range(len(values))
        if support[index] > 0 and np.any(masks[index])
    ]


def _save_figure(figure: plt.Figure, output: Path) -> list[str]:
    paths = []
    for suffix in (".png", ".pdf"):
        path = output.with_suffix(suffix)
        figure.savefig(path, dpi=220 if suffix == ".png" else None)
        paths.append(str(path))
    plt.close(figure)
    return paths


def plot_coadded_profiles(bank: RomanReferenceBank, output_dir: Path) -> list[str]:
    wavelength = bank.rest_wavelength / 10_000.0
    figure, axes = plt.subplots(5, 3, figsize=(12.0, 14.0), sharex=True)
    full_color = "#2563a6"
    removed_color = "#c05a24"
    for class_index, (name, axis) in enumerate(
        zip(bank.class_names, axes.flat, strict=True)
    ):
        full = _profile_rows(
            bank.coadd_full_profiles[class_index],
            bank.coadd_profile_masks[class_index],
            bank.coadd_support_counts[class_index],
        )
        removed = _profile_rows(
            bank.coadd_continuum_removed_profiles[class_index],
            bank.coadd_profile_masks[class_index],
            bank.coadd_support_counts[class_index],
        )
        for profile in full:
            axis.plot(wavelength, profile, color=full_color, alpha=0.16, linewidth=0.7)
        if full:
            axis.plot(
                wavelength,
                np.nanmedian(np.stack(full), axis=0),
                color=full_color,
                linewidth=1.5,
                label="Complete spectrum",
            )
        if removed:
            axis.plot(
                wavelength,
                np.nanmedian(np.stack(removed), axis=0),
                color=removed_color,
                linewidth=1.3,
                linestyle="--",
                label="Continuum removed",
            )
        axis.axhline(0.0, color="0.75", linewidth=0.6)
        axis.set_title(f"{name}  ·  {len(full)} profiles", loc="left", fontsize=10)
        axis.set_xlim(float(wavelength[0]), float(wavelength[-1]))
        axis.tick_params(labelsize=8)
    figure.supxlabel("Rest wavelength (μm)")
    figure.supylabel("Normalized spectral shape")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.974),
        ncol=2,
        frameon=False,
    )
    figure.suptitle("Coadded spectral references", y=0.995)
    figure.tight_layout(rect=(0.02, 0.02, 1.0, 0.935))
    return _save_figure(figure, output_dir / "coadded_profiles")


def plot_phase_profiles(bank: RomanReferenceBank, output_dir: Path) -> list[str]:
    wavelength = bank.rest_wavelength / 10_000.0
    phase_labels = [
        f"{left:g} to {right:g} d"
        for left, right in zip(
            bank.phase_edges_days[:-1],
            bank.phase_edges_days[1:],
            strict=True,
        )
    ]
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(phase_labels)))
    figure, axes = plt.subplots(5, 3, figsize=(12.0, 14.0), sharex=True)
    for class_index, (name, axis) in enumerate(
        zip(bank.class_names, axes.flat, strict=True)
    ):
        supported_phases = 0
        for phase_index, (label, color) in enumerate(
            zip(phase_labels, colors, strict=True)
        ):
            profiles = _profile_rows(
                bank.phase_continuum_removed_profiles[class_index, phase_index],
                bank.phase_profile_masks[class_index, phase_index],
                bank.phase_support_counts[class_index, phase_index],
            )
            if not profiles:
                continue
            supported_phases += 1
            axis.plot(
                wavelength,
                np.nanmedian(np.stack(profiles), axis=0),
                color=color,
                linewidth=1.15,
                label=label,
            )
        axis.axhline(0.0, color="0.75", linewidth=0.6)
        axis.set_title(
            f"{name}  ·  {supported_phases}/{len(phase_labels)} phases",
            loc="left",
            fontsize=10,
        )
        axis.set_xlim(float(wavelength[0]), float(wavelength[-1]))
        axis.tick_params(labelsize=8)
    figure.supxlabel("Rest wavelength (μm)")
    figure.supylabel("Normalized continuum-removed shape")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.974),
        ncol=5,
        frameon=False,
    )
    figure.suptitle("Spectral evolution references", y=0.995)
    figure.tight_layout(rect=(0.02, 0.02, 1.0, 0.935))
    return _save_figure(figure, output_dir / "phase_profiles")


def plot_redshift_coverage(
    bank: RomanReferenceBank,
    audit: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    edges = np.asarray(audit["redshift_edges"], dtype=np.float64)
    counts = np.asarray(
        [
            audit["training_objects_used_by_fine_class_and_redshift"][name]
            for name in bank.class_names
        ],
        dtype=np.int64,
    )
    totals = counts.sum(axis=1, keepdims=True)
    fractions = np.divide(
        counts,
        totals,
        out=np.zeros_like(counts, dtype=np.float64),
        where=totals > 0,
    )
    figure, axis = plt.subplots(figsize=(9.2, 7.2))
    image = axis.imshow(fractions, aspect="auto", cmap="viridis", vmin=0.0)
    axis.set_xticks(np.arange(len(edges) - 1))
    axis.set_xticklabels(
        [
            f"{left:g}–{right:g}"
            for left, right in zip(edges[:-1], edges[1:], strict=True)
        ]
    )
    axis.set_yticks(np.arange(len(bank.class_names)))
    axis.set_yticklabels(bank.class_names)
    axis.set_xlabel("Simulation redshift")
    axis.set_ylabel("Reference class")
    axis.set_title("Training references contributing across redshift")
    for row in range(counts.shape[0]):
        for column in range(counts.shape[1]):
            color = (
                "white" if fractions[row, column] > 0.45 * fractions.max() else "black"
            )
            axis.text(
                column,
                row,
                f"{counts[row, column]:,}",
                ha="center",
                va="center",
                fontsize=7,
                color=color,
            )
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Fraction of each class's contributing objects")
    figure.tight_layout()
    return _save_figure(figure, output_dir / "redshift_coverage")


def make_gallery(
    bank_path: Path,
    output_dir: Path,
    audit_path: Path | None = None,
) -> dict[str, Any]:
    bank_path = bank_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bank = RomanReferenceBank.load(bank_path)
    files = {
        "coadded_profiles": plot_coadded_profiles(bank, output_dir),
        "phase_profiles": plot_phase_profiles(bank, output_dir),
    }
    resolved_audit = (
        audit_path.expanduser().resolve()
        if audit_path is not None
        else bank_path.with_suffix(".audit.json")
    )
    if resolved_audit.is_file():
        audit = json.loads(resolved_audit.read_text(encoding="utf-8"))
        required = {
            "redshift_edges",
            "training_objects_used_by_fine_class_and_redshift",
        }
        if required <= audit.keys():
            files["redshift_coverage"] = plot_redshift_coverage(bank, audit, output_dir)
    summary = {
        "bank": str(bank_path),
        "audit": str(resolved_audit) if resolved_audit.is_file() else None,
        "classes": list(bank.class_names),
        "coadded_profile_shape": list(bank.coadd_full_profiles.shape),
        "phase_profile_shape": list(bank.phase_full_profiles.shape),
        "files": files,
    }
    summary_path = output_dir / "gallery_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary


def main() -> None:
    arguments = _arguments()
    result = make_gallery(arguments.bank, arguments.output_dir, arguments.audit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
