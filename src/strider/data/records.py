"""Names and small record types shared by preprocessing and training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourcePair:
    block: int
    model: str
    head_path: str
    spec_path: str


OBJECT_COLUMNS = [
    "object_index",
    "source_id",
    "source_product",
    "snid",
    "split",
    "block",
    "model",
    "gentype",
    "template_index",
    "redshift",
    "estimated_peak_mjd",
    "simulation_peak_mjd",
    "class_index",
    "class_name",
    "first_observation",
    "observation_count",
]

OBSERVATION_COLUMNS = [
    "observation_index",
    "object_index",
    "source_id",
    "source_product",
    "snid",
    "mjd",
    "exposure_seconds",
    "first_bin",
    "bin_count",
    "native_wavelength_min",
    "native_wavelength_max",
    "background_scale",
]
