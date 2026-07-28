"""What is inside a STRIDER checkpoint.

A checkpoint is one ``.pt`` file holding the weights plus everything needed
to rebuild and run the model on a fresh machine: class names, the wavelength
and redshift grids, the model config, the calibration maps and provenance.
Nothing else on disk is required.

Once a checkpoint ships its keys are fixed; only additive changes are allowed
without bumping ``CHECKPOINT_VERSION``. To run a model, use
:func:`strider.load_model` rather than this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

_SAFE_GLOBALS_REGISTERED = False


def _register_safe_numpy_globals() -> None:
    """Allowlist numpy's array/dtype reconstruction so ``weights_only=True`` can read the
    numpy content of a STRIDER bank without permitting arbitrary code execution."""
    global _SAFE_GLOBALS_REGISTERED
    if _SAFE_GLOBALS_REGISTERED:
        return
    multiarray = getattr(np, "_core", np.core).multiarray
    safe: list[Any] = [multiarray._reconstruct, multiarray.scalar, np.ndarray, np.dtype]
    safe += [t for t in vars(getattr(np, "dtypes", type("", (), {}))).values()
             if isinstance(t, type)]
    torch.serialization.add_safe_globals(safe)
    _SAFE_GLOBALS_REGISTERED = True


CHECKPOINT_VERSION = 2

FORMAT_MARKER = "strider_portable"   # value is baked into shipped .pt files


@dataclass(frozen=True)
class CheckpointMetadata:
    """Self-describing metadata for a STRIDER checkpoint.

    A checkpoint without this dataclass attached is unusable on a new machine.
    The schema records (a) what the model is, (b) how to rebuild it,
    (c) what its grids are, and (d) how to interpret its outputs.

    The ``model_config`` and ``detector_config`` blobs are the deserialized
    training configs; together with ``state_dict`` they let
    ``_build_underlying_model`` re-instantiate the trained model without
    any additional files on disk.
    """

    # model identity
    architecture: str                       # "single_scale" or "multi_scale"
    family: str                             # compatibility tag
    class_names: list[str]                  # e.g. ["Ia", "91bg", ..., "exotic"]
    n_classes: int

    # input grids
    wavelength_grid: list[float]            # rest-frame Å, length n_wave
    n_wave: int
    z_grid: list[float]                     # marginal z bins
    n_z_bins: int
    z_min: float
    z_max: float
    max_epochs: int                         # phase-time dim

    # preprocessing
    phase_norm: float                       # phases divided by this at input
    z_norm: float                           # z values divided by this at input
    normalization: str                      # spectrum normalization scheme
    n_channels: int                         # 1 (flux only) or 2 (flux + diff)
    patch_size: int

    # rebuild blocks
    # These are full config dicts (serialized STRIDERConfig + detector cfg),
    # not the dataclass instances — keep CheckpointMetadata import-cheap.
    model_config: dict[str, Any] = field(default_factory=dict)
    detector_config: dict[str, Any] | None = None
    bank_metadata: dict[str, Any] | None = None

    # calibration
    calibration_fitted_on: str = ""
    class_calibration: dict[str, Any] | None = None
    redshift_calibration: dict[str, Any] | None = None
    selection: dict[str, Any] | None = None
    calibration_evaluated_on: str = ""

    # provenance
    training_commit: str = ""               # git commit of training code
    training_config_hash: str = ""          # hash of the training YAML
    preprocessing_version: str = ""         # data pipeline version tag
    n_train_objects: int = 0
    created_at: str = ""                    # ISO 8601 UTC
    paper_citation: str = ""                # e.g. "Dixon+ 2026"
    portable_checkpoint_version: int = CHECKPOINT_VERSION   # key name is baked in

    def __post_init__(self):
        if self.n_classes != len(self.class_names):
            raise ValueError(
                f"n_classes={self.n_classes} but class_names has "
                f"{len(self.class_names)} entries"
            )
        if self.n_wave != len(self.wavelength_grid):
            raise ValueError(
                f"n_wave={self.n_wave} but wavelength_grid has "
                f"{len(self.wavelength_grid)} entries"
            )
        if self.n_z_bins != len(self.z_grid):
            raise ValueError(
                f"n_z_bins={self.n_z_bins} but z_grid has "
                f"{len(self.z_grid)} entries"
            )
        if self.architecture not in ("single_scale", "multi_scale"):
            raise ValueError(
                f"architecture must be 'single_scale' or 'multi_scale', "
                f"got {self.architecture!r}"
            )
        from strider.model_info import _CHECKPOINT_FAMILY_TAG

        if self.family != _CHECKPOINT_FAMILY_TAG:
            raise ValueError("checkpoint is not compatible with public STRIDER")
        if self.detector_config is None:
            raise ValueError("checkpoint requires detector_config")
        self._validate_class_calibration()
        self._validate_redshift_calibration()

    def _validate_class_calibration(self) -> None:
        calibration = self.class_calibration
        if calibration is None:
            return
        if calibration.get("method") != "dirichlet":
            raise ValueError("unsupported class calibration method")
        weights = calibration.get("weights")
        bias = calibration.get("bias")
        if not isinstance(weights, list) or len(weights) != self.n_classes:
            raise ValueError("Dirichlet calibration weights must match n_classes")
        if any(not isinstance(row, list) or len(row) != self.n_classes for row in weights):
            raise ValueError("Dirichlet calibration weights must be square")
        if not isinstance(bias, list) or len(bias) != self.n_classes:
            raise ValueError("Dirichlet calibration bias must match n_classes")

    def _validate_redshift_calibration(self) -> None:
        calibration = self.redshift_calibration
        if calibration is None:
            return
        if calibration.get("method") != "pit_recalibration":
            raise ValueError("unsupported redshift calibration method")
        for key in ("pit_full_sorted", "pit_gold_sorted", "hpd_scores_sorted"):
            values = calibration.get(key)
            if not isinstance(values, list) or not values:
                raise ValueError(f"redshift calibration requires {key}")
            if values != sorted(values) or values[0] < 0 or values[-1] > 1:
                raise ValueError(f"{key} must be sorted values in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CheckpointMetadata":
        v = d.get("portable_checkpoint_version", 0)
        if v > CHECKPOINT_VERSION:
            raise ValueError(
                f"Checkpoint version {v} is newer than this code "
                f"supports (version {CHECKPOINT_VERSION}). Upgrade "
                f"the strider package."
            )
        return cls(**d)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load and validate a checkpoint file.

    Returns the raw payload dict ({format, state_dict, metadata, optional
    bank_state}). Most callers should use :func:`strider.load_model`, which
    builds the actual STRIDER object.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}")
    _register_safe_numpy_globals()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected a dict-style checkpoint, got {type(payload)}. "
            "This is not a STRIDER checkpoint."
        )
    if payload.get("format") != FORMAT_MARKER:
        raise ValueError(
            f"Checkpoint at {path} is missing the "
            f"'format: {FORMAT_MARKER}' marker. This looks like a "
            "non-STRIDER model file."
        )
    if "metadata" not in payload:
        raise ValueError(
            f"Portable checkpoint at {path} has no 'metadata' block. "
            "It cannot be loaded on this machine without it. Re-export the "
            "Re-export it from the training code."
        )
    if "state_dict" not in payload:
        raise ValueError(
            f"Portable checkpoint at {path} has no 'state_dict' block. "
            "Re-export it from the training code."
        )
    return payload
