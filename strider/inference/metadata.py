"""Shared metadata for inference."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class InferenceMetadata:
    """Architecture-independent metadata needed by inference."""

    family: str
    class_names: list[str]
    z_grid: np.ndarray
    z_min: float
    z_max: float
    n_z_bins: int
    normalization: str
    patch_size: int
    n_channels: int
    peak_normalize_default: bool


def attach_inference_metadata(model, metadata: InferenceMetadata):
    """Attach common inference metadata to a model and return the model."""
    model.inference_metadata = metadata
    return model
