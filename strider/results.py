"""Stable public result schema for STRIDER."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from strider.model_info import model_id, sha256


REDSHIFT_QUALITY_LABELS = {
    "clean": "Reliable",
    "broad": "Uncertain",
    "ambiguous": "Two possible redshifts",
    "invalid": "Unavailable",
}


def redshift_quality_label(value: Any) -> str:
    quality = str(value)
    return REDSHIFT_QUALITY_LABELS.get(quality, quality)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def compact_result(
    output: dict[str, Any],
    *,
    metadata: dict[str, Any],
    checkpoint: str | Path,
) -> dict[str, Any]:
    names = list(output["class_names"])
    probabilities = np.asarray(output["p_class"], dtype=float)
    classification = output["classification"]
    redshift = output["redshift"]
    spectra = output["spectra"]

    return {
        "schema_version": "2.0",
        "object_id": str(metadata.get("object", "input")),
        "model": {
            "name": "STRIDER",
            "model_id": model_id(len(names)),
            "checkpoint": Path(checkpoint).name,
            "sha256": sha256(checkpoint) if checkpoint and Path(checkpoint).is_file() else None,
        },
        "input": {
            "n_spectra": int(spectra["n_epochs"]),
            "phase_days": spectra["phase_days"],
            "wavelength_range_aa": spectra["used_wave_range_aa"],
            "quality": spectra["quality"],
        },
        "classification": {
            "class": classification["class"],
            "confidence": float(classification["confidence"]),
            "p_Ia": float(classification["p_Ia"]),
            "p_Ia_raw": float(classification["p_Ia_raw"]),
            "gold_Ia_selected": bool(classification["gold_Ia_selected"]),
            "gold_Ia_threshold_raw": float(classification["gold_Ia_threshold_raw"]),
            "ia_selection_context": classification["ia_selection_context"],
            "ranking": classification["ranking"],
            "probability_type": classification["probabilities"],
            "top_classes": classification["top3"],
            "probabilities": {
                name: float(probabilities[index]) for index, name in enumerate(names)
            },
        },
        "redshift": {
            "z_STRIDER": float(redshift["z_STRIDER"]),
            "z_map": float(redshift["z_map"]),
            "z_p16": float(redshift["z_p16"]),
            "z_p84": float(redshift["z_p84"]),
            "z_p05": float(redshift["z_p05"]),
            "z_p95": float(redshift["z_p95"]),
            "z_p025": float(redshift["z_p025"]),
            "z_p975": float(redshift["z_p975"]),
            "quality": redshift.get("quality"),
            "interval_calibration": redshift["interval_calibration"],
            "spectra_only": redshift["spectra_only"],
            "with_controls": redshift["with_controls"],
        },
        "controls": output["controls"],
    }
