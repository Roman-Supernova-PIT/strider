"""Input loaders for STRIDER inference."""

from __future__ import annotations

import csv
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


TEXT_SUFFIXES = {".ascii", ".csv", ".dat", ".spec", ".tab", ".tsv", ".txt"}
FITS_SUFFIXES = {".fit", ".fits", ".fts"}


@dataclass(frozen=True)
class SpectrumInput:
    wavelength: np.ndarray
    flux: np.ndarray
    phase: np.ndarray
    flux_err: np.ndarray | None
    metadata: dict[str, Any]


def _scalar(value: np.ndarray) -> Any:
    item = np.asarray(value).reshape(-1)[0]
    return item.item() if hasattr(item, "item") else item


def load_npz(path: str | Path, *, phase: float | None = None) -> SpectrumInput:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        wave_key = "wavelength" if "wavelength" in data else "wave"
        missing = [key for key in (wave_key, "flux") if key not in data]
        if missing:
            raise ValueError(f"{path} is missing: {', '.join(missing)}")

        phase_key = next((key for key in ("phase", "phases") if key in data), None)
        if phase_key is not None and phase is not None:
            raise ValueError("Input already contains phase; remove --phase")
        if phase_key is None and phase is None:
            raise ValueError("NPZ needs phase data, or pass --phase for one spectrum")

        flux = np.asarray(data["flux"], dtype=np.float32)
        if phase_key is None:
            n_epochs = 1 if flux.ndim == 1 else flux.shape[0]
            if n_epochs != 1:
                raise ValueError("A time-series NPZ needs one phase value per spectrum")
            phases = np.asarray([phase], dtype=np.float32)
        else:
            phases = np.atleast_1d(np.asarray(data[phase_key], dtype=np.float32))

        metadata: dict[str, Any] = {"object": path.stem, "source": str(path)}
        for key in ("object_id", "true_class", "z_true"):
            if key in data:
                metadata[key] = _scalar(data[key])
        if "object_id" in metadata:
            metadata["object"] = str(metadata["object_id"])

        return SpectrumInput(
            wavelength=np.asarray(data[wave_key], dtype=np.float32),
            flux=flux,
            phase=phases,
            flux_err=(
                np.asarray(data["flux_err"], dtype=np.float32)
                if "flux_err" in data
                else None
            ),
            metadata=metadata,
        )


def _column(fieldnames: list[str], *names: str) -> str | None:
    normalized = {name.strip().lower(): name for name in fieldnames}
    for name in names:
        if name in normalized:
            return normalized[name]
    return None


def _number(row: dict[str, str], column: str, row_number: int) -> float:
    raw = row.get(column, "").strip()
    if not raw:
        raise ValueError(f"Row {row_number} has no value for {column!r}")
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Row {row_number} has invalid {column!r} value {raw!r}") from exc


def _tokens(line: str) -> list[str]:
    if "," in line:
        return next(csv.reader([line], skipinitialspace=True))
    if "\t" in line:
        return next(csv.reader([line], delimiter="\t", skipinitialspace=True))
    return line.split()


def _is_header(values: list[str]) -> bool:
    names = {value.strip().lower() for value in values}
    wave_names = {"wave", "wavelength", "wavelength_aa", "lambda"}
    flux_names = {"flux", "flam", "f_lambda"}
    return bool(names & wave_names) and bool(names & flux_names)


def _read_rows(path: Path) -> tuple[list[str], list[tuple[int, dict[str, str]]]]:
    data_rows: list[tuple[int, list[str]]] = []
    comment_header: list[str] | None = None
    with path.open() as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                candidate = _tokens(line.lstrip("#").strip())
                if _is_header(candidate):
                    comment_header = candidate
                continue
            data_rows.append((line_number, _tokens(line)))

    if not data_rows:
        raise ValueError(f"Input has no data rows: {path}")

    _, first = data_rows[0]
    if _is_header(first):
        fieldnames = first
        data_rows = data_rows[1:]
    elif comment_header is not None:
        fieldnames = comment_header
    else:
        defaults = {
            2: ["wavelength", "flux"],
            3: ["wavelength", "flux", "flux_err"],
            4: ["wavelength", "flux", "flux_err", "phase"],
        }
        if len(first) not in defaults:
            raise ValueError(
                "Headerless text needs wavelength, flux, optional flux_err, "
                "and optional phase columns"
            )
        fieldnames = defaults[len(first)]

    rows: list[tuple[int, dict[str, str]]] = []
    for line_number, values in data_rows:
        if len(values) != len(fieldnames):
            raise ValueError(
                f"Row {line_number} has {len(values)} values; expected {len(fieldnames)}"
            )
        rows.append((line_number, dict(zip(fieldnames, values, strict=True))))
    if not rows:
        raise ValueError(f"Input has a header but no data rows: {path}")
    return fieldnames, rows


def load_table(path: str | Path, *, phase: float | None = None) -> SpectrumInput:
    """Load a single spectrum or time series from a text table."""
    path = Path(path)
    fieldnames, rows = _read_rows(path)
    wave_col = _column(fieldnames, "wavelength", "wavelength_aa", "wave", "lambda")
    flux_col = _column(fieldnames, "flux", "flam", "f_lambda")
    phase_col = _column(fieldnames, "phase", "phase_days", "rest_phase", "rest_phase_days")
    error_col = _column(
        fieldnames,
        "flux_err",
        "fluxerr",
        "flux_error",
        "flux_uncertainty",
        "error",
        "err",
        "sigma",
    )
    epoch_col = _column(fieldnames, "epoch", "epoch_id")
    object_col = _column(fieldnames, "object_id", "object")
    if wave_col is None or flux_col is None:
        raise ValueError("Input needs wavelength and flux columns")
    if phase_col is not None and phase is not None:
        raise ValueError("Input already contains a phase column; remove --phase")
    if phase_col is None and phase is None:
        raise ValueError("Input needs a phase column, or pass --phase for one spectrum")

    groups: OrderedDict[str, dict[str, Any]] = OrderedDict()
    object_ids: set[str] = set()
    for row_number, row in rows:
        phase_value = _number(row, phase_col, row_number) if phase_col else float(phase)
        key = row.get(epoch_col, "").strip() if epoch_col else f"phase:{phase_value:.12g}"
        if not key:
            key = f"phase:{phase_value:.12g}"
        group = groups.setdefault(
            key,
            {"phase": phase_value, "wave": [], "flux": [], "error": []},
        )
        if not np.isclose(group["phase"], phase_value, atol=1e-6):
            raise ValueError(f"Epoch {key!r} contains more than one phase")
        group["wave"].append(_number(row, wave_col, row_number))
        group["flux"].append(_number(row, flux_col, row_number))
        if error_col:
            group["error"].append(_number(row, error_col, row_number))
        if object_col and row.get(object_col, "").strip():
            object_ids.add(row[object_col].strip())

    if not groups:
        raise ValueError(f"Input has no data rows: {path}")
    if len(object_ids) > 1:
        raise ValueError("Input contains more than one object_id")

    waves: list[np.ndarray] = []
    fluxes: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    phases: list[float] = []
    for group in groups.values():
        wave = np.asarray(group["wave"], dtype=np.float32)
        order = np.argsort(wave)
        waves.append(wave[order])
        fluxes.append(np.asarray(group["flux"], dtype=np.float32)[order])
        if error_col:
            errors.append(np.asarray(group["error"], dtype=np.float32)[order])
        phases.append(float(group["phase"]))

    reference = waves[0]
    for index, wave in enumerate(waves[1:], start=2):
        if wave.shape != reference.shape or not np.allclose(
            wave, reference, rtol=1e-6, atol=1e-4
        ):
            raise ValueError(
                f"Epoch {index} has a different wavelength grid; resample it first"
            )

    object_id = next(iter(object_ids), path.stem)
    return SpectrumInput(
        wavelength=reference,
        flux=np.stack(fluxes),
        phase=np.asarray(phases, dtype=np.float32),
        flux_err=np.stack(errors) if error_col else None,
        metadata={"object": object_id, "object_id": object_id, "source": str(path)},
    )


def _fits_column(names: Sequence[str], *aliases: str) -> str | None:
    normalized = {name.upper(): name for name in names}
    return next((normalized[name] for name in aliases if name in normalized), None)


def _one_phase(values: Any, path: Path) -> float:
    phases = np.asarray(values, dtype=np.float32).reshape(-1)
    phases = phases[np.isfinite(phases)]
    if phases.size == 0:
        raise ValueError(f"FITS phase data is empty: {path}")
    if not np.allclose(phases, phases[0], atol=1e-6):
        raise ValueError("A FITS file may contain only one phase; use separate epoch files")
    return float(phases[0])


def _resolve_phase(embedded: Any | None, explicit: float | None, message: str) -> float:
    if embedded is not None and explicit is not None:
        raise ValueError("Input already contains phase; remove --phase")
    value = embedded if embedded is not None else explicit
    if value is None:
        raise ValueError(message)
    return float(value)


def _single_spectrum(
    path: Path,
    wavelength: Any,
    flux: Any,
    phase: float,
    flux_err: Any | None = None,
) -> SpectrumInput:
    wave = np.asarray(wavelength, dtype=np.float32).reshape(-1)
    values = np.asarray(flux, dtype=np.float32).reshape(-1)
    if wave.size != values.size:
        raise ValueError(f"FITS wavelength and flux lengths differ: {path}")
    order = np.argsort(wave)
    error = None
    if flux_err is not None:
        error = np.asarray(flux_err, dtype=np.float32).reshape(-1)
        if error.size != wave.size:
            raise ValueError(f"FITS flux_err length differs from flux: {path}")
        error = error[order]
    return SpectrumInput(
        wavelength=wave[order],
        flux=values[order][None, :],
        phase=np.asarray([phase], dtype=np.float32),
        flux_err=None if error is None else error[None, :],
        metadata={"object": path.stem, "object_id": path.stem, "source": str(path)},
    )


def load_fits(path: str | Path, *, phase: float | None = None) -> SpectrumInput:
    """Load one spectrum from a FITS table or a linear-WCS image."""
    try:
        from astropy.io import fits
    except ImportError as exc:
        raise RuntimeError('FITS input needs: pip install ".[fits]"') from exc

    path = Path(path)
    with fits.open(path) as hdus:
        for hdu in hdus:
            names = list(getattr(getattr(hdu, "columns", None), "names", None) or [])
            if hdu.data is None or not names:
                continue
            wave_col = _fits_column(names, "WAVELENGTH", "WAVE", "LAMBDA", "WAV")
            flux_col = _fits_column(names, "FLUX", "FLAM", "FLUX_DENSITY", "DATA")
            if wave_col is None or flux_col is None:
                continue
            error_col = _fits_column(
                names, "FLUX_ERR", "FLUXERR", "FLUX_ERROR", "ERROR", "ERR", "SIGMA"
            )
            phase_col = _fits_column(names, "PHASE", "PHASE_DAYS", "REST_PHASE")
            embedded_phase = (
                _one_phase(hdu.data[phase_col], path)
                if phase_col
                else hdu.header.get("PHASE")
            )
            phase_value = _resolve_phase(
                embedded_phase, phase, "FITS input needs phase data, or pass --phase"
            )
            return _single_spectrum(
                path,
                hdu.data[wave_col],
                hdu.data[flux_col],
                phase_value,
                hdu.data[error_col] if error_col else None,
            )

        for hdu in hdus:
            if hdu.data is None or np.asarray(hdu.data).ndim != 1:
                continue
            if "LOG" in str(hdu.header.get("CTYPE1", "")).upper():
                raise ValueError("Logarithmic FITS WCS is not supported; provide wavelength values")
            start = hdu.header.get("CRVAL1")
            step = hdu.header.get("CDELT1", hdu.header.get("CD1_1"))
            pixel = float(hdu.header.get("CRPIX1", 1.0))
            if start is None or step is None:
                continue
            phase_value = _resolve_phase(
                hdu.header.get("PHASE"),
                phase,
                "FITS input needs a PHASE header, or pass --phase",
            )
            wavelength = float(start) + (np.arange(hdu.data.size) - (pixel - 1.0)) * float(step)
            return _single_spectrum(path, wavelength, hdu.data, phase_value)

    raise ValueError(f"Could not find wavelength and flux data in FITS file: {path}")


def load_input(path: str | Path, *, phase: float | None = None) -> SpectrumInput:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return load_npz(path, phase=phase)
    if suffix in FITS_SUFFIXES:
        return load_fits(path, phase=phase)
    if suffix in TEXT_SUFFIXES:
        return load_table(path, phase=phase)
    choices = ", ".join(sorted(TEXT_SUFFIXES | FITS_SUFFIXES | {".npz"}))
    raise ValueError(f"Unsupported input type {suffix!r}; use one of: {choices}")


def load_inputs(
    paths: Sequence[str | Path], *, phases: Sequence[float] | None = None
) -> SpectrumInput:
    """Load one input or combine several epoch files on a shared wavelength grid."""
    if not paths:
        raise ValueError("At least one spectrum file is required")
    if phases is not None and len(phases) != len(paths):
        raise ValueError("--phase needs one value per input file")

    loaded = [
        load_input(path, phase=None if phases is None else float(phases[index]))
        for index, path in enumerate(paths)
    ]
    if len(loaded) == 1:
        return loaded[0]

    reference = loaded[0].wavelength
    for item in loaded[1:]:
        if item.wavelength.shape != reference.shape or not np.allclose(
            item.wavelength, reference, rtol=1e-6, atol=1e-4
        ):
            raise ValueError("Input files use different wavelength grids; resample them first")

    have_errors = [item.flux_err is not None for item in loaded]
    if any(have_errors) and not all(have_errors):
        raise ValueError("Either every input file must include flux_err, or none may include it")

    return SpectrumInput(
        wavelength=reference,
        flux=np.concatenate([np.atleast_2d(item.flux) for item in loaded]),
        phase=np.concatenate([np.atleast_1d(item.phase) for item in loaded]),
        flux_err=(
            np.concatenate([np.atleast_2d(item.flux_err) for item in loaded])
            if all(have_errors)
            else None
        ),
        metadata={
            "object": loaded[0].metadata["object"],
            "object_id": loaded[0].metadata["object"],
            "source": [str(Path(path)) for path in paths],
        },
    )
