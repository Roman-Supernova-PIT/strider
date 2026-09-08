"""Plot held-out spectra and redshift results at three representative redshifts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from strider.config import project_path
from strider.data.dataset import SundialDataset, collate_objects
from strider.model import measurement_inputs
from strider.model.posterior import joint_probability

from .checkpoint import load_trained_model
from .evidence_maps import _representative_indices, evidence_grade

if TYPE_CHECKING:
    from strider.model.strider import Strider


@torch.no_grad()
def plot_examples(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = project_path(config, config["project"]["output_dir"])
    figure_dir = output_dir / "example_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    model, _, device = load_trained_model(config)
    split = str(config["evaluation"].get("split", "test"))
    original = SundialDataset(config, split, "original", training=False)
    clean = SundialDataset(config, split, "clean", training=False)
    generated = SundialDataset(config, split, "generated", training=False)
    wavelength = original.output_wavelength
    ia_count = int(config["evaluation"].get("ia_examples_per_redshift", 0))
    if not 0 <= ia_count <= 4:
        raise ValueError("ia_examples_per_redshift must lie between zero and four")
    ia_index = list(model.class_names).index("Ia") if ia_count else None
    grade_thresholds = tuple(
        float(value)
        for value in config["evaluation"].get(
            "evidence_grade_thresholds", [0.25, 0.5, 0.75]
        )
    )
    written = []
    for target in (0.75, 1.5, 2.5):
        indices = [
            index
            for _, index in _representative_indices(
                original.objects,
                [target],
                4,
                preferred_class_index=ia_index,
                preferred_count=ia_count,
            )
        ]
        figure, axes = plt.subplots(4, 2, figsize=(12, 13), constrained_layout=True)
        for row_number, index in enumerate(indices):
            original_item = original[index]
            clean_item = clean[index]
            generated_item = generated[index]
            spectrum_axis = axes[row_number, 0]
            result_axis = axes[row_number, 1]
            original_spectrum = _masked_visit_median(original_item)
            clean_spectrum = _masked_visit_median(clean_item)
            spectrum_axis.plot(wavelength, original_spectrum, color="#5b6f8a", lw=0.8, label="original")
            spectrum_axis.plot(wavelength, clean_spectrum, color="#d95f02", lw=1.2, label="clean")
            spectrum_axis.axhline(0.0, color="0.75", lw=0.6)
            spectrum_axis.set_ylabel("flux / background scale")
            if row_number == 0:
                spectrum_axis.legend(frameon=False, ncol=2)

            original_result = _posterior(model, original_item, device)
            generated_result = _posterior(model, generated_item, device)
            result_axis.plot(model.redshift_grid.cpu(), original_result["p_z"], color="#5b6f8a", label="original")
            result_axis.plot(model.redshift_grid.cpu(), generated_result["p_z"], color="#1b9e77", label="new noise")
            result_axis.axvline(float(original_item["redshift"]), color="#d95f02", lw=1.2, label="true z")
            result_axis.set_ylabel("conditional redshift probability")
            if row_number == 0:
                result_axis.legend(frameon=False, ncol=3, fontsize=8)
            class_name = config["model"]["classes"][int(original_item["class_index"])]
            score = original_result["evidence_score"]
            grade = evidence_grade(score, grade_thresholds)
            result_axis.set_title(
                f"SNID {int(original_item['snid'])} · {class_name} · "
                f"z={float(original_item['redshift']):.3f} · "
                f"P(Ia)={original_result['p_ia']:.2f} · "
                f"{grade.lower()} · evidence score {score:.2f}",
                fontsize=9,
            )
        for axis in axes[-1, :]:
            axis.set_xlabel("observed wavelength (Å)" if axis is axes[-1, 0] else "redshift")
        figure.suptitle(
            f"Sundial {split} objects near z={target:.2f}", fontsize=14
        )
        path = figure_dir / f"{split}_examples_z{target:.2f}.png"
        figure.savefig(path, dpi=170)
        plt.close(figure)
        written.append(str(path))
    return {"device": str(device), "figures": written}


def _masked_visit_median(item: dict[str, torch.Tensor]) -> np.ndarray:
    flux = item["flux"].numpy()
    mask = item["wavelength_mask"].numpy() > 0
    values = np.where(mask, flux, np.nan)
    median = np.zeros(values.shape[1], dtype=np.float32)
    valid_columns = mask.any(axis=0)
    with np.errstate(all="ignore"):
        median[valid_columns] = np.nanmedian(values[:, valid_columns], axis=0)
    return median


def _posterior(
    model: Strider,
    item: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    batch = {name: value.to(device) for name, value in collate_objects([item]).items()}
    output = model(measurement_inputs(batch))
    joint = joint_probability(
        output["joint_logits"], model.redshift_cell_width, model.redshift_prior
    )[0]  # (C,Z)
    return {
        "p_z": joint.sum(dim=0).cpu().numpy(),
        "p_ia": float(joint[0].sum().cpu()),
        "evidence_score": float(
            torch.sigmoid(output["evidence_sufficiency_logit"])[0].cpu()
        ),
    }
