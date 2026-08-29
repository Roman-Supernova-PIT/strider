"""Measured-spectrum input for current STRIDER model packages.

Legacy public checkpoints require rest-frame phase and continue to use
``strider.io``. Current model packages instead use observer-frame dates and
never require a truth-derived rest phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from strider.io import _column, _number, _read_rows


@dataclass(frozen=True)
class ObservedSeriesInput:
    wavelength: np.ndarray
    flux: np.ndarray
    flux_error: np.ndarray
    observer_time: np.ndarray
    metadata: dict[str, Any]


def load_observed_inputs(
    paths: Sequence[str | Path],
    *,
    times: Sequence[float] | None = None,
    wavelength_unit: str = "angstrom",
) -> ObservedSeriesInput:
    """Load one object from NPZ or text files on a shared wavelength grid."""
    if not paths:
        raise ValueError("At least one spectrum file is required")
    if times is not None and len(times) != len(paths):
        raise ValueError("--time needs one value per input file")
    loaded = [
        _load_observed_input(
            path,
            time=None if times is None else float(times[index]),
        )
        for index, path in enumerate(paths)
    ]
    loaded = [_convert_wavelength_unit(item, wavelength_unit) for item in loaded]
    if len(loaded) == 1:
        return loaded[0]
    explicit_object_ids = {
        str(item.metadata["object"])
        for item in loaded
        if bool(item.metadata.get("_object_id_supplied"))
    }
    if len(explicit_object_ids) > 1:
        raise ValueError(
            "Input files identify different objects: "
            + ", ".join(sorted(explicit_object_ids))
        )
    reference = loaded[0].wavelength
    for item in loaded[1:]:
        if item.wavelength.shape != reference.shape or not np.allclose(
            item.wavelength, reference, rtol=1.0e-6, atol=1.0e-4
        ):
            raise ValueError("Input files use different wavelength grids; resample them first")
    return ObservedSeriesInput(
        wavelength=reference,
        flux=np.concatenate([np.atleast_2d(item.flux) for item in loaded]),
        flux_error=np.concatenate(
            [np.atleast_2d(item.flux_error) for item in loaded]
        ),
        observer_time=np.concatenate(
            [np.atleast_1d(item.observer_time) for item in loaded]
        ),
        metadata={
            "object": (
                next(iter(explicit_object_ids))
                if explicit_object_ids
                else loaded[0].metadata["object"]
            ),
            "source": [str(Path(path)) for path in paths],
            "wavelength_unit": "angstrom",
        },
    )


def _load_observed_input(
    path: str | Path, *, time: float | None
) -> ObservedSeriesInput:
    path = Path(path)
    if path.suffix.lower() == ".npz":
        return _load_observed_npz(path, time=time)
    if path.suffix.lower() in {
        ".ascii",
        ".csv",
        ".dat",
        ".spec",
        ".tab",
        ".tsv",
        ".txt",
    }:
        return _load_observed_table(path, time=time)
    raise ValueError(
        f"Current model packages accept NPZ and text/CSV input; got {path.suffix!r}"
    )


def _load_observed_npz(path: Path, *, time: float | None) -> ObservedSeriesInput:
    with np.load(path, allow_pickle=False) as data:
        wave_key = "wavelength" if "wavelength" in data else "wave"
        error_key = next(
            (key for key in ("flux_error", "flux_err", "flamerr") if key in data),
            None,
        )
        time_key = next(
            (key for key in ("observer_time", "mjd", "time") if key in data),
            None,
        )
        missing = [key for key in (wave_key, "flux") if key not in data]
        if error_key is None:
            missing.append("flux_error")
        if missing:
            raise ValueError(f"{path} is missing: {', '.join(missing)}")
        if time_key is not None and time is not None:
            raise ValueError("Input already contains observer time; remove --time")
        flux = np.asarray(data["flux"], dtype=np.float32)
        visits = 1 if flux.ndim == 1 else flux.shape[0]
        if time_key is None:
            if time is None:
                raise ValueError("NPZ needs observer_time or mjd, or pass --time")
            if visits != 1:
                raise ValueError("A time-series NPZ needs one observer time per spectrum")
            observer_time = np.asarray([time], dtype=np.float64)
        else:
            observer_time = np.atleast_1d(
                np.asarray(data[time_key], dtype=np.float64)
            )
        if len(observer_time) != visits:
            raise ValueError("NPZ observer time count does not match its spectra")
        object_id_supplied = "object_id" in data
        object_id = str(data["object_id"].item()) if object_id_supplied else path.stem
        return ObservedSeriesInput(
            wavelength=np.asarray(data[wave_key], dtype=np.float32),
            flux=flux,
            flux_error=np.asarray(data[error_key], dtype=np.float32),
            observer_time=observer_time,
            metadata={
                "object": object_id,
                "source": str(path),
                "_object_id_supplied": object_id_supplied,
            },
        )


def _load_observed_table(path: Path, *, time: float | None) -> ObservedSeriesInput:
    fieldnames, rows = _read_rows(path)
    wave_col = _column(fieldnames, "wavelength", "wavelength_aa", "wave", "lambda")
    flux_col = _column(fieldnames, "flux", "flam", "f_lambda")
    error_col = _column(
        fieldnames,
        "flux_error",
        "flux_err",
        "fluxerr",
        "flamerr",
        "error",
        "err",
        "sigma",
    )
    time_col = _column(
        fieldnames,
        "observer_time",
        "mjd",
        "mjd_obs",
        "time",
    )
    epoch_col = _column(fieldnames, "epoch", "epoch_id", "visit", "visit_id")
    object_col = _column(fieldnames, "object_id", "object")
    if wave_col is None or flux_col is None or error_col is None:
        raise ValueError("Current model input needs wavelength, flux and flux_error columns")
    if time_col is not None and time is not None:
        raise ValueError("Input already contains observer time; remove --time")
    if time_col is None and time is None:
        raise ValueError("Input needs observer_time or mjd, or pass --time")

    groups: dict[str, dict[str, Any]] = {}
    object_ids: set[str] = set()
    for row_number, row in rows:
        time_value = _number(row, time_col, row_number) if time_col else float(time)
        key = row.get(epoch_col, "").strip() if epoch_col else f"time:{time_value:.12g}"
        if not key:
            key = f"time:{time_value:.12g}"
        group = groups.setdefault(
            key,
            {"time": time_value, "wavelength": [], "flux": [], "error": []},
        )
        if not np.isclose(group["time"], time_value, atol=1.0e-7):
            raise ValueError(f"Epoch {key!r} contains more than one observer time")
        group["wavelength"].append(_number(row, wave_col, row_number))
        group["flux"].append(_number(row, flux_col, row_number))
        group["error"].append(_number(row, error_col, row_number))
        if object_col and row.get(object_col, "").strip():
            object_ids.add(row[object_col].strip())
    if len(object_ids) > 1:
        raise ValueError("Input contains more than one object_id")

    waves: list[np.ndarray] = []
    fluxes: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    times: list[float] = []
    for group in groups.values():
        wave = np.asarray(group["wavelength"], dtype=np.float32)
        order = np.argsort(wave, kind="stable")
        waves.append(wave[order])
        fluxes.append(np.asarray(group["flux"], dtype=np.float32)[order])
        errors.append(np.asarray(group["error"], dtype=np.float32)[order])
        times.append(float(group["time"]))
    reference = waves[0]
    for wave in waves[1:]:
        if wave.shape != reference.shape or not np.allclose(
            wave, reference, rtol=1.0e-6, atol=1.0e-4
        ):
            raise ValueError("Epochs use different wavelength grids; resample them first")
    object_id = next(iter(object_ids), path.stem)
    return ObservedSeriesInput(
        wavelength=reference,
        flux=np.stack(fluxes),
        flux_error=np.stack(errors),
        observer_time=np.asarray(times, dtype=np.float64),
        metadata={
            "object": object_id,
            "source": str(path),
            "_object_id_supplied": bool(object_ids),
        },
    )


def _convert_wavelength_unit(
    data: ObservedSeriesInput, unit: str
) -> ObservedSeriesInput:
    requested = str(unit).lower()
    if requested not in {"auto", "angstrom", "micron"}:
        raise ValueError("wavelength_unit must be auto, angstrom, or micron")
    wavelength = np.asarray(data.wavelength, dtype=np.float32)
    if requested == "auto":
        finite = wavelength[np.isfinite(wavelength)]
        if finite.size == 0:
            raise ValueError("wavelength must contain finite values")
        requested = "micron" if float(np.nanmedian(np.abs(finite))) < 100.0 else "angstrom"
    if requested == "micron":
        wavelength = wavelength * np.float32(10_000.0)
    metadata = dict(data.metadata)
    metadata["wavelength_unit"] = "angstrom"
    return ObservedSeriesInput(
        wavelength=wavelength,
        flux=data.flux,
        flux_error=data.flux_error,
        observer_time=data.observer_time,
        metadata=metadata,
    )
