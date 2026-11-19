"""Saved transients: one transient's input as STRIDER read it, its run's record and its result, in a folder of its
own, so that anyone can classify it again with the same model and settings and see whether the result is the same
(`rerun`).

A saved transient's folder holds ``spectra.parquet`` (`strider.io.write_spectra`), ``record.json`` (the run's record,
`strider.runs.run_record`), ``result.json`` (`strider.Classification.to_dict`, or the reason it failed) and, if the
run had one, ``redshift_prior.npy``.
"""

import hashlib
import json
import re
from pathlib import Path

import numpy as np

__all__ = ["save_transient", "rerun", "saved_folder"]

TOLERANCE = 1e-6


def save_transient(folder, key, transient, result, record, redshift_prior=None) -> Path:
    """Save one transient in its own folder in ``folder`` (`saved_folder`), and return that folder.
    ``transient`` is its input, or the reason it could not be read (then the folder has no spectra); ``result`` its
    `strider.Classification`, or the reason it failed; and ``record`` its run's (`strider.runs.run_record`)."""
    from .io import Transient, write_spectra

    path = saved_folder(folder, key)
    path.mkdir(parents=True, exist_ok=False)
    if isinstance(transient, Transient):
        write_spectra({str(key): transient}, path / "spectra.parquet")
    (path / "record.json").write_text(json.dumps(record, indent=1) + "\n")
    found = {"object_id": str(key)} | ({"failure": result} if isinstance(result, str) else result.to_dict())
    (path / "result.json").write_text(json.dumps(found, indent=1) + "\n")
    if redshift_prior is not None:
        np.save(path / "redshift_prior.npy", np.asarray(redshift_prior, dtype=np.float64))
    return path


def rerun(path, model=None) -> tuple[list[str], list[str]]:
    """Classify a saved transient again with its run's model and settings, and compare with its saved result.

    Parameters
    ----------
    path : path-like
        A saved transient's folder (`save_transient`).
    model : strider.Model, optional
        The model to use; by default the saved run's, by name, with its scan.

    Returns
    -------
    differences : list of str
        How the result differs, empty if it is the same within `TOLERANCE`.
    environment : list of str
        How the versions and the machine differ from the run's, which can explain small differences.
    """
    from . import load_model, registry
    from .io import read_spectra
    from .runs import environment

    path = Path(path)
    run = json.loads((path / "record.json").read_text())
    expected = json.loads((path / "result.json").read_text())
    settings = run["settings"]
    model = model or load_model(registry.name_of(settings["model"]), scan=settings["scan"])
    here, then = environment(), run["sessions"][-1]["environment"]
    machine = [f"{name}: {then.get(name)} then, {here[name]} now" for name in here if then.get(name) != here[name]]
    if model.fingerprint != settings["fingerprint"] or model.scan != settings["scan"]:
        used = f"{model.name} ({model.fingerprint[:12]}, {model.scan} scan)"
        saved = f"{settings['model']} ({settings['fingerprint'][:12]}, {settings['scan']} scan)"
        return [f"the model is {used}; the saved run's is {saved}"], machine
    if not (path / "spectra.parquet").exists():
        return [f"the input could not be read: {expected.get('failure')}"], machine
    differences = []
    ((key, transient),) = read_spectra(path / "spectra.parquet").items()
    prior = path / "redshift_prior.npy"
    options = {
        "classes": settings["classes"],
        "class_prior": settings["class_prior"],
        "redshift_range": None if settings["redshift_range"] is None else tuple(settings["redshift_range"]),
        "redshift_prior": np.load(prior) if prior.exists() else None,
        "host_redshift_outlier": settings["host_redshift_outlier"],
        "error_scale": settings["error_scale"],
        "light_curve_errors": settings["light_curve_errors"],
        "max_spectra": settings.get("max_spectra"),
    }
    try:
        found = model.classify(transient, **options).to_dict()
    except Exception as error:
        found = {"failure": f"{type(error).__name__}: {error}"}
    if "failure" in expected or "failure" in found:
        if expected.get("failure") != found.get("failure"):
            differences.append(f"failure: {expected.get('failure')!r} then, {found.get('failure')!r} now")
        return differences, machine
    return differences + _compared(expected, {"object_id": expected["object_id"]} | found), machine


def saved_folder(folder, key) -> Path:
    """The folder of one saved transient: its id, any character other than letters, digits, '.', '-' and '_'
    replaced by '_', then a short code of the id itself, so that different ids get different folders."""
    key = str(key)
    safe = re.sub(r"[^\w.-]", "_", key)
    return Path(folder) / f"{safe}-{hashlib.sha256(key.encode()).hexdigest()[:8]}"


def _compared(then, now, name=""):
    """How a saved result differs from a new one, field by field: numbers within `TOLERANCE` (NaN equal to NaN),
    each list of numbers by its largest difference, and everything else exactly."""
    if isinstance(then, dict) and isinstance(now, dict):
        found = []
        for key in sorted(set(then) | set(now), key=str):
            label = f"{name}.{key}" if name else str(key)
            if key not in then or key not in now:
                found.append(f"{label}: {'missing then' if key not in then else 'missing now'}")
            else:
                found += _compared(then[key], now[key], label)
        return found
    if isinstance(then, list) and isinstance(now, list):
        if len(then) != len(now):
            return [f"{name}: {len(then)} entries then, {len(now)} now"]
        if all(_number(value) for value in then + now):
            a, b = (np.array([np.nan if v is None else v for v in values], dtype=float) for values in (then, now))
            same = np.isclose(a, b, rtol=0.0, atol=TOLERANCE, equal_nan=True)
            if not same.all():
                largest = float(np.nanmax(np.abs(a - b))) if np.isfinite(a - b).any() else float("nan")
                return [f"{name}: {(~same).sum()} of {len(a)} values differ, by up to {largest:.3g}"]
            return []
        return [line for n, (x, y) in enumerate(zip(then, now)) for line in _compared(x, y, f"{name}[{n}]")]
    if _number(then) and _number(now):
        a, b = (np.nan if value is None else value for value in (then, now))
        return [] if np.isclose(a, b, rtol=0.0, atol=TOLERANCE, equal_nan=True) else [f"{name}: {then} then, {now} now"]
    return [] if then == now else [f"{name}: {then!r} then, {now!r} now"]


def _number(value):
    return value is None or (isinstance(value, (int, float)) and not isinstance(value, bool))
