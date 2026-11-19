"""Runs of many transients written to a parquet file: the run's record, saving a part at a time, resuming a run that
stopped, and splitting a run into jobs and joining their results.

A run writes its results to ``<output>.parts``, one parquet file per write. Each part is written to a temporary file,
flushed to disk and renamed, so a part is either whole or absent. The folder is locked while a run uses it. The parts
are joined into ``<output>`` only when every object is done; a run that stops keeps them, and resuming carries on
from them. The run's record is kept in ``_record.json`` and in every parquet file's metadata. The
folder's other files start with ``_``, so ``pyarrow.parquet.read_table`` reads the folder as the results saved so
far.
"""

import errno
import hashlib
import json
import os
import platform
import shutil
import uuid
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

__all__ = ["run_record", "in_job", "checked_job", "read_record", "summary", "merge"]

RUN_FORMAT = "strider-run-1"
RECORD_KEY = b"strider_run"
# the columns that say how a row was run rather than what was found, left out when duplicates are compared
RUN_COLUMNS = ("settings_fingerprint", "seconds", "cpu_seconds", "peak_memory_mb")
SUMMARY_COLUMNS = (
    "status",
    "warning_codes",
    "light_curve_check",
    "seconds",
    "cpu_seconds",
    "peak_memory_mb",
)


def run_record(
    model,
    priors=None,
    source=None,
    job=None,
    workers=1,
    error_scale=1.0,
    light_curve_errors="check",
    max_spectra=None,
) -> dict:
    """The record of a run: the model, the settings that change results, the input, the job and each session's
    environment.

    ``priors`` are the user's priors as `strider.Model.classify_many` takes them, and ``source`` describes the input:
    its ``path`` and ``sha256``, the ``only`` objects chosen from it, and the ``columns``, ``wavelength_unit`` and
    ``flux_density`` it was read with. The ``settings_fingerprint`` covers the model, the STRIDER and schema versions
    and the settings, not the input, the job or the environment.
    """
    from . import __version__
    from .io import SCHEMA_VERSION

    priors, source = dict(priors or {}), dict(source or {})
    classes = priors.get("classes")
    redshift_prior = priors.get("redshift_prior")
    settings = {
        "strider_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "model": model.name,
        "fingerprint": model.fingerprint,
        "package_sha256": model.files_sha256,
        "scan": model.scan,
        "classes": [classes] if isinstance(classes, str) else None if classes is None else list(classes),
        "class_prior": priors.get("class_prior"),
        "redshift_range": None if priors.get("redshift_range") is None else list(priors["redshift_range"]),
        "redshift_prior_sha256": None
        if redshift_prior is None
        else hashlib.sha256(np.ascontiguousarray(redshift_prior, "<f8").tobytes()).hexdigest(),
        "host_redshift_outlier": priors.get("host_redshift_outlier"),
        "error_scale": error_scale,
        "light_curve_errors": light_curve_errors,
        "max_spectra": max_spectra,
        "columns": source.get("columns"),
        "wavelength_unit": source.get("wavelength_unit", "angstrom"),
        "flux_density": source.get("flux_density", "f_lambda"),
    }
    only = source.get("only")
    started = _now()
    return {
        "format": RUN_FORMAT,
        "run_id": uuid.uuid4().hex,
        "status": "incomplete",
        "settings": settings,
        "settings_fingerprint": hashlib.sha256(_canonical(settings).encode()).hexdigest(),
        "input": {
            "path": source.get("path"),
            "sha256": source.get("sha256"),
            "only": None if only is None else sorted(str(key) for key in only),
        },
        "job": None if job is None else list(checked_job(job)),
        "started": started,
        "sessions": [{"started": started, "environment": environment(workers)}],
    }


def environment(workers=1) -> dict:
    """The versions and machine a session used, and its workers and the threads each ran. They can change results
    within rounding, so they are recorded."""
    import scipy
    import torch

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "workers": workers,
        "threads": 1 if workers > 1 else torch.get_num_threads(),  # workers run on one thread each
        "platform": platform.platform(),
    }


def checked_job(job) -> tuple[int, int]:
    """A job as (k, n), job k of n, from a pair or the text ``k/n``, with n ≥ 1 and 0 ≤ k < n."""
    try:
        k, n = (int(part) for part in (job.split("/") if isinstance(job, str) else job))
    except (TypeError, ValueError):
        raise ValueError(f"a job is K/N, such as 0/4, not {job!r}") from None
    if not (n >= 1 and 0 <= k < n):
        raise ValueError(f"a job K/N needs N of at least 1 and K from 0 to N - 1, not {k}/{n}")
    return k, n


def in_job(key, job) -> bool:
    """Whether object ``key`` falls to ``job`` (k, n): the first 16 hex digits of the sha256 of its id as UTF-8
    text, modulo n, are k. It depends on the id alone, so the jobs share an input the same way on any machine and in
    any input order."""
    k, n = job
    return int(hashlib.sha256(str(key).encode()).hexdigest()[:16], 16) % n == k


def read_record(path) -> dict:
    """The record of the run that wrote a parquet file of results."""
    import pyarrow.parquet as pq

    metadata = pq.read_schema(path).metadata or {}
    if RECORD_KEY not in metadata:
        raise ValueError(f"{path} has no record of its run; it was not written by STRIDER 0.2 or later")
    return json.loads(metadata[RECORD_KEY])


def summary(tables) -> dict:
    """What tables of results hold, together: how many objects, by status, warning code and light-curve error check,
    and what they took where it was measured (seconds per object, CPU seconds and the most memory a process used;
    None if none was, as for transients scored together on a GPU). The tables need only `SUMMARY_COLUMNS`."""
    objects, status, codes, checks, seconds, cpu, memory = 0, {}, {}, {}, [], [], []

    def count(found, value):
        found[value] = found.get(value, 0) + 1

    for table in tables:
        columns = table.select(list(SUMMARY_COLUMNS)).to_pydict()
        objects += table.num_rows
        for value in columns["status"]:
            count(status, value)
        for found in columns["warning_codes"]:
            for code in found or []:
                count(codes, code)
        for value in columns["light_curve_check"]:
            if value:
                count(checks, value)
        seconds += [value for value in columns["seconds"] if value is not None]
        cpu += [value for value in columns["cpu_seconds"] if value is not None]
        memory += [value for value in columns["peak_memory_mb"] if value is not None]
    seconds = np.array(seconds, dtype=float)
    measured = len(seconds) > 0
    return {
        "objects": objects,
        "status": dict(sorted(status.items())),
        "warning_codes": dict(sorted(codes.items())),
        "light_curve_check": dict(sorted(checks.items())),
        "seconds_per_object": {
            "measured": len(seconds),
            "total": float(seconds.sum()) if measured else None,
            "median": float(np.median(seconds)) if measured else None,
            "p90": float(np.percentile(seconds, 90)) if measured else None,
            "max": float(seconds.max()) if measured else None,
        },
        "cpu_seconds": float(sum(cpu)) if cpu else None,
        "largest_memory_mb": max(memory, default=None),
    }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _with_record(table, record):
    return table.replace_schema_metadata((table.schema.metadata or {}) | {RECORD_KEY: _canonical(record).encode()})


def _durable_replace(temporary: Path, final: Path) -> None:
    """Flush ``temporary`` to disk, rename it to ``final``, and flush the folder, so a crash leaves one or the
    other, whole. A failure to flush is an error."""
    with open(temporary, "rb") as written:
        os.fsync(written.fileno())
    os.replace(temporary, final)
    _flush_folder(final.parent)


def _flush_folder(folder) -> None:
    """Flush a folder's entries to disk, where the platform and file system can."""
    if os.name == "nt":  # Windows cannot open a folder to flush it
        return
    descriptor = os.open(folder, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    except OSError as error:
        if error.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP):  # a file system that cannot
            raise
    finally:
        os.close(descriptor)


class Parts:
    """The parts folder of a run writing ``path``, a parquet file: locked while open, written a whole part at a time,
    and joined into ``path`` when the run is done.

    Parameters
    ----------
    path : path-like
        The parquet file the run writes.
    schema : pyarrow.Schema
        The results schema, for joining a run with no rows.
    record : dict
        The run's record (`run_record`).
    resume : bool
        Carry on from the parts of a run that stopped, if its record allows it; with none, start the run.
    """

    def __init__(self, path, schema, record, resume=False):
        self.path, self.schema = Path(path), schema
        self.folder = self.path.with_name(self.path.name + ".parts")
        if self.path.exists():
            done = " (its run is complete)" if resume else ""
            raise ValueError(f"{self.path} exists already{done}; write to another file, or remove it first")
        if self.folder.exists() and not resume:
            raise ValueError(
                f"{self.folder} is left from a run that stopped: resume it (resume=True, --resume), or remove it"
            )
        created, self._lock = not self.folder.exists(), None
        try:
            self.folder.mkdir(exist_ok=resume)
        except FileExistsError:
            raise ValueError(f"another run started writing {self.folder}; write to another output") from None
        try:
            self._lock = _locked(self.folder / "_lock")
            if self.path.exists():  # another run finished while this one waited to start
                raise ValueError(f"{self.path} exists already (its run is complete); write to another file")
            saved = self.folder / "_record.json"
            if resume and saved.exists():
                self.record = _resumable(json.loads(saved.read_text()), record, self.folder)
            else:
                self.record = record
            self._write_text(saved, _canonical(self.record))
            for left in self.folder.glob("_*.tmp"):  # parts a stopped run was writing
                left.unlink()
            self.saved, numbers = {}, [-1]
            for part in self._parts():
                numbers.append(int(part.stem.split("-")[1]))
                self.saved.update(_fingerprints(part))
            self.count = max(numbers) + 1
        except BaseException as error:
            if created and not isinstance(error, RunBusy) and not self._parts():  # a folder this run made, unused
                shutil.rmtree(self.folder, ignore_errors=True)
            self.close()
            raise

    def write(self, table) -> None:
        """Save one part, whole."""
        import pyarrow.parquet as pq

        final = self.folder / f"part-{self.count:06d}.parquet"
        temporary = final.with_name(f"_{final.name}.tmp")
        pq.write_table(_with_record(table, self.record), temporary)
        _durable_replace(temporary, final)
        self.count += 1

    def finish(self, seen=None) -> None:
        """Join the parts into the run's file, a part at a time, and remove them, holding the folder throughout.
        ``seen`` are the ids of this run's input; a part that holds another object means the input changed, and
        nothing is joined."""
        import pyarrow.parquet as pq

        try:
            expected = set(self.saved) if seen is None else {str(key) for key in seen}
            others = set(self.saved) - expected
            if others:
                raise ValueError(
                    f"{self.folder} holds results for {len(others)} objects that are not in this input, such as "
                    f"{sorted(others)[0]}: the input changed since the run stopped; write to a new output"
                )
            parts = self._parts()
            found = summary(pq.read_table(part, columns=list(SUMMARY_COLUMNS)) for part in parts)
            record = self.record | {"status": "complete", "finished": _now(), "summary": found}
            temporary = self.path.with_name(self.path.name + ".tmp")
            with pq.ParquetWriter(temporary, _with_record(self.schema.empty_table(), record).schema) as writer:
                for part in parts:
                    writer.write_table(pq.read_table(part))
            _durable_replace(temporary, self.path)
            shutil.rmtree(self.folder)  # still locked, so no other run can start in it meanwhile
        finally:
            self.close()

    def close(self) -> None:
        """Release the folder, keeping its parts."""
        if self._lock is not None and not self._lock.closed:
            self._lock.close()

    def _parts(self):
        return sorted(self.folder.glob("part-*.parquet"))

    def _write_text(self, path, text):
        temporary = path.with_name(f"{path.name}.tmp")
        temporary.write_text(text)
        _durable_replace(temporary, path)


class RunBusy(ValueError):
    """Another run, or merge, holds the folder."""


def _locked(path):
    """``path`` opened and locked for this process alone; another process that tries is refused at once. The lock
    goes when the file is closed or the process ends, however it ends. Where there are no file locks, as on Windows, a
    run is refused."""
    try:
        import fcntl
    except ImportError:
        raise ValueError(
            "a .parquet run needs file locks, which this platform lacks; write the results to a .csv or .json file"
        ) from None
    lock = open(path, "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RunBusy(f"another run is writing {path.parent}; wait for it, or write to another output") from None
    except OSError as error:
        lock.close()
        raise ValueError(f"cannot lock {path.parent} ({error}); its file system needs to support file locks") from None
    return lock


def _resumable(saved, current, folder):
    """The stopped run's record with this session added, if this run may carry it on: the same STRIDER version,
    model, settings, job, input file and selection. Another environment is allowed, with a warning, as it changes
    results only within rounding."""
    checks = (
        ("the STRIDER version", saved["settings"]["strider_version"], current["settings"]["strider_version"]),
        ("the model package", saved["settings"]["package_sha256"], current["settings"]["package_sha256"]),
        ("the brightness scan", saved["settings"]["scan"], current["settings"]["scan"]),
        ("the settings", saved["settings_fingerprint"], current["settings_fingerprint"]),
        ("the job", saved["job"], current["job"]),
        ("the objects chosen", saved["input"]["only"], current["input"]["only"]),
    )
    for label, there, here in checks:
        if there != here:
            raise ValueError(
                f"cannot resume {folder}: {label} changed since it stopped ({there} then, {here} now); write to a "
                "new output"
            )
    then, now = saved["input"]["sha256"], current["input"]["sha256"]
    if then != now:
        why = (
            "the input file changed since the run stopped"
            if None not in (then, now)
            else (
                "the stopped run recorded its input file, and this run gives none (give source=)"
                if now is None
                else "the stopped run did not record its input file"
            )
        )
        raise ValueError(f"cannot resume {folder}: {why}; write to a new output")
    then, now = saved["sessions"][-1]["environment"], current["sessions"][-1]["environment"]
    if then != now:
        warnings.warn(f"resuming {folder} in another environment: {then} then, {now} now")
    return saved | {"sessions": saved["sessions"] + current["sessions"]}


def _fingerprints(part):
    """The object ids in a part, each with its input fingerprint."""
    import pyarrow.parquet as pq

    table = pq.read_table(part, columns=["object_id", "input_fingerprint"])
    return dict(zip(table.column("object_id").to_pylist(), table.column("input_fingerprint").to_pylist()))


def merge(paths, output, partial=False) -> dict:
    """Join the results files of a run's jobs into one, a batch of rows at a time.

    Every file must come from the same model, settings, input file and selection of objects, and the jobs must be
    complete: each of 0 to N - 1 once, of a run whose input file is recorded, unless ``partial``. An object in two
    files must have the same results there (its time and memory aside); it is kept once, from the lowest job.

    Returns
    -------
    dict
        The merged file's record.
    """
    paths, output = [Path(path) for path in paths], Path(output)
    if output.exists():
        raise ValueError(f"{output} exists already; merge into another file, or remove it first")
    folder = output.with_name(output.name + ".parts")  # held while merging, as a run holds it
    try:
        folder.mkdir()
    except FileExistsError:
        raise RunBusy(f"{folder} exists: another run or merge is writing {output}") from None
    lock = None
    try:
        lock = _locked(folder / "_lock")
        if output.exists():
            raise ValueError(f"{output} exists already; merge into another file, or remove it first")
        record = _merged(paths, output, partial)
        shutil.rmtree(folder)  # still locked
    except BaseException as error:
        if not isinstance(error, RunBusy):  # a busy folder is another writer's now: leave it
            shutil.rmtree(folder, ignore_errors=True)
        raise
    finally:
        if lock is not None:
            lock.close()
    return record


def _merged(paths, output, partial):
    """`merge`'s work, its output already held."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    records = [read_record(path) for path in paths]
    first = records[0]
    for path, record in zip(paths, records):
        if record["status"] != "complete":
            raise ValueError(f"{path} is from a run that did not finish")
        if record["settings_fingerprint"] != first["settings_fingerprint"]:
            raise ValueError(f"{path} was run with other settings or another model than {paths[0]}")
        if (record["input"]["sha256"], record["input"]["only"]) != (first["input"]["sha256"], first["input"]["only"]):
            raise ValueError(f"{path} was run on another input file, or other objects of it, than {paths[0]}")
    jobs = sorted(tuple(record["job"] or (0, 1)) for record in records)
    counts = {n for _, n in jobs}
    each_once = len(counts) == 1 and [k for k, _ in jobs] == list(range(counts.pop()))
    complete = each_once and first["input"]["sha256"] is not None
    if not (complete or partial):
        why = "are not each of one run's once" if not each_once else "come from a run whose input file is not recorded"
        raise ValueError(f"the jobs given, {', '.join(f'{k}/{n}' for k, n in jobs)}, {why}; or give partial")
    ordered = [path for path, _ in sorted(zip(paths, records), key=lambda pair: pair[1]["job"] or [0])]
    kept, chosen, summaries = {}, [], []
    for path in ordered:  # which rows to keep, and what they hold
        source, masks = pq.ParquetFile(path), []
        science = [name for name in source.schema_arrow.names if name not in RUN_COLUMNS]
        for batch in source.iter_batches(batch_size=1024):
            rows, keep = batch.select(science).to_pylist(), []
            for row in rows:
                key = row["object_id"]
                found = hashlib.sha256(_canonical(row).encode()).hexdigest()  # NaN equals NaN here
                if key in kept and kept[key] != found:
                    raise ValueError(f"object {key} has different results in two of the files")
                keep.append(key not in kept)
                kept.setdefault(key, found)
            masks.append(np.array(keep, dtype=bool))
            summaries.append(pa.Table.from_batches([batch]).select(list(SUMMARY_COLUMNS)).filter(masks[-1]))
        chosen.append(masks)
    record = first | {
        "status": "complete" if complete else "partial",
        "job": None,
        "jobs": [list(job) for job in jobs],
        "sessions": [session for record in records for session in record["sessions"]],
        "started": min(record["started"] for record in records),
        "finished": max(record["finished"] for record in records),
        "summary": summary(summaries),
    }
    schema = _with_record(pq.read_schema(ordered[0]).empty_table(), record).schema
    temporary = output.with_name(output.name + ".tmp")
    with pq.ParquetWriter(temporary, schema) as writer:
        for path, masks in zip(ordered, chosen):
            for batch, mask in zip(pq.ParquetFile(path).iter_batches(batch_size=1024), masks):
                writer.write_table(pa.Table.from_batches([batch]).filter(mask))
    _durable_replace(temporary, output)
    return record
