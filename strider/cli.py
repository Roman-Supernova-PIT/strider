"""The ``strider`` command: ``classify``, ``merge``, ``rerun``, ``info``, ``models`` and ``check``."""

import argparse
import contextlib
import hashlib
import itertools
import json
import logging
import os
import platform
import signal
import sys
import time
from importlib.resources import files
from pathlib import Path

import numpy as np

from . import __version__, registry
from .runs import checked_job
from .io import (
    EXAMPLE,
    FLUX_DENSITIES,
    WAVELENGTH_UNITS,
    RedshiftCurvesFile,
    ResultsFile,
    iter_spectra,
    read_spectra,
    read_transient,
)

__all__ = ["main", "check_example", "example_results"]

EXPECTED = files("strider") / "data" / "example_expected.json"
TOLERANCE = 1e-4


def classify(args):
    source = Path(args.input)
    if not source.is_file():
        raise FileNotFoundError(f"no file of spectra at {source}")
    output = None if args.output is None else Path(args.output)
    if output is not None and output.suffix.lower() not in ResultsFile.FORMATS:
        raise ValueError(f"cannot write results to {output}; give -o a file ending in {', '.join(ResultsFile.FORMATS)}")
    curves = RedshiftCurvesFile.FORMATS
    if args.redshift_curves is not None and Path(args.redshift_curves).suffix.lower() not in curves:
        raise ValueError(
            f"cannot write redshift curves to {args.redshift_curves}; give a file ending in {', '.join(curves)}"
        )
    for path, option in (
        (output, "-o"),
        (args.evidence_maps, "--evidence-maps"),
        (args.redshift_curves, "--redshift-curves"),
    ):
        if path is not None:
            writable(Path(path), source, option)
    chosen = [Path(p).resolve() for p in (output, args.evidence_maps, args.redshift_curves) if p is not None]
    if len(set(chosen)) < len(chosen):
        raise ValueError("give -o, --evidence-maps and --redshift-curves different files")
    if args.evidence_maps is not None:
        needs("h5py", "writing evidence maps needs h5py: pip install h5py")
    if args.redshift_curves is not None:
        needs("qp", "writing redshift curves needs qp: pip install qp-prob")
    if source.suffix.lower() == ".parquet":
        found = iter_spectra(source, args.columns, args.wavelength_unit, args.flux_density)
        first = next(found, None)  # the table's columns are checked before the model is loaded
        if first is None:
            raise ValueError(f"{source} has no visits")
        objects = itertools.chain([first], found)
    elif args.columns or args.wavelength_unit != "angstrom" or args.flux_density != "f_lambda":
        raise ValueError("--columns, --wavelength-unit and --flux-density are for a parquet table of spectra")
    else:
        objects = {source.stem: read_transient(source)}
    job = None if args.job is None else checked_job(args.job)
    if output is None and (job is not None or args.resume):
        raise ValueError("--job and --resume need a results file: give -o")
    if args.save_transient is not None and args.only is None:
        raise ValueError("--save-transient saves a folder for each transient: name them with --only")
    if args.save_transient is not None:
        from .saved import saved_folder

        folders = [saved_folder(args.save_transient, key) for key in args.only]
        taken = [str(path) for path in folders if path.exists()]
        if taken:
            raise ValueError(f"{', '.join(taken)} exist already; give --save-transient another folder")
        if len(set(folders)) < len(folders):
            raise ValueError("two of the transients named with --only would share a folder; save them separately")
    if args.only is not None:
        wanted = set(args.only)
        objects = {
            key: found
            for key, found in (objects.items() if isinstance(objects, dict) else objects)
            if str(key) in wanted
        }
        missing = wanted - {str(key) for key in objects}
        if missing:
            raise ValueError(f"{source} has no object {', '.join(sorted(missing))}")
    priors = {
        "classes": args.classes,
        "class_prior": args.class_prior,
        "host_redshift_outlier": args.host_redshift_outlier,
    }
    if args.zmin is not None or args.zmax is not None:
        priors["redshift_range"] = (
            -np.inf if args.zmin is None else args.zmin,
            np.inf if args.zmax is None else args.zmax,
        )
    described = {"path": str(source.resolve()), "sha256": file_sha256(source), "only": args.only}
    if source.suffix.lower() == ".parquet":
        described |= {
            "columns": args.columns,
            "wavelength_unit": args.wavelength_unit,
            "flux_density": args.flux_density,
        }
    from . import terminal

    model = load(args.model, args.device, "full" if args.full_scan else "fast")
    live = terminal.interactive()  # a line that updates in place for a person; log lines for a job's log
    if not live:
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO, format="strider: %(message)s", stream=sys.stderr
        )
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # a scheduler's stop first writes the transients done
    name = next(iter(objects)) if isinstance(objects, dict) and len(objects) == 1 else None
    total = planned(source, args, job, output) if live else None
    if live:  # which STRIDER and model, once, above the run
        scan = "full scan" if args.full_scan else "fast scan"
        about = terminal.GAP.join([f"STRIDER {__version__}", model.name, scan])
        print(terminal.Style(sys.stderr)(about, terminal.DIM), file=sys.stderr)
    started, began = time.monotonic(), time.time()
    display = terminal.LiveLine(args.verbose, name, args.workers, total) if live else contextlib.nullcontext()
    with display:
        results = model.classify_many(
            objects,
            output,
            plots=args.plot,
            evidence_maps=args.evidence_maps,
            redshift_curves=args.redshift_curves,
            workers=args.workers,
            batch=args.batch,
            error_scale=args.error_scale,
            light_curve_errors=args.light_curve_errors,
            max_spectra=args.max_spectra,
            chunk=args.save_every,
            keep=output is None or args.save_transient is not None,  # with a results file, keep only the failures
            job=job,
            resume=args.resume,
            source=described,
            **priors,
        )
    seconds = time.monotonic() - started
    style = terminal.Style(sys.stdout)
    cross = style(terminal.CROSS, terminal.RED)
    failed = [key for key, result in results.items() if isinstance(result, str)]
    if output is None:
        total, failures = len(results), len(failed)
        if live:
            print(terminal.finished(style, total - failures, failures, seconds) + "\n")
        print(
            "\n\n".join(
                f"{cross} {key}: not classified. {result}"
                if key in failed
                else terminal.report(result, key, style, saved_figure(args.plot, key, began))
                for key, result in results.items()
            )
        )
    else:
        total, failures = written(output)  # counted from the file, so a resumed run counts what it saved before
        if live:  # this session's count and time, and the file's total when it holds more
            done = display.done - display.failed
            print(terminal.finished(style, done, display.failed, seconds, output, total))
        else:
            print(f"Classified {total - failures} of {total} objects; results in {output}")
        for key in failed:
            print(f"{cross} {key}: {results[key]}")
    if args.save_transient is not None:
        from .saved import save_transient
        from .runs import run_record

        record = run_record(
            model,
            priors,
            described,
            job,
            args.workers,
            error_scale=args.error_scale,
            light_curve_errors=args.light_curve_errors,
            max_spectra=args.max_spectra,
        )
        for key, result in results.items():
            save_transient(args.save_transient, key, objects[key], result, record)
        print(f"Saved in {args.save_transient}; strider rerun {args.save_transient} classifies them again and compares")
    if total and failures == total:
        sys.exit("strider: no object could be classified; the reasons are above")
    if args.plot is not None and output is not None:  # else each report gives its figure
        print("Figures saved in " + ("this folder" if Path(args.plot) == Path(".") else f"{Path(args.plot)}/"))
    if args.evidence_maps is not None:
        print(f"Evidence maps in {args.evidence_maps}")
    if args.redshift_curves is not None and Path(args.redshift_curves).exists():
        print(f"Redshift curves in {args.redshift_curves}")


def planned(source, args, job, output):
    """How many transients a run will classify, counted from the table's ids alone, a batch at a time, so it is quick
    and small for any size; None if they cannot be counted."""
    if source.suffix.lower() != ".parquet":
        return len(args.only) if args.only else 1
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from .io import _ids
    from .runs import in_job

    try:
        column = (args.columns or {}).get("object_id", "object_id")
        keys = set()
        for batch in pq.ParquetFile(source).iter_batches(batch_size=65536, columns=[column]):
            keys.update(_ids(pc.unique(batch.column(0))))  # each batch's distinct ids, not every row
        keys.discard(None)
        if args.only is not None:
            keys &= set(args.only)
        if job is not None:
            keys = {key for key in keys if in_job(key, job)}
        parts = Path(f"{output}.parts")
        if args.resume and any(parts.glob("part-*.parquet")):
            keys -= set(_ids(pq.read_table(parts, columns=["object_id"]).column(0)))
    except (OSError, KeyError, ValueError):
        return None
    return len(keys)


def saved_figure(folder, key, began):
    """The file of a transient's figure if this run saved it (a figure that fails to save leaves a warning), else
    None."""
    from .model import figure_name

    path = None if folder is None else Path(folder) / figure_name(key)
    return path if path is not None and path.is_file() and path.stat().st_mtime >= began - 1.0 else None


def written(path):
    """How many objects a results file holds, and how many of them failed."""
    if path.suffix.lower() == ".json":
        statuses = [found["status"] for found in json.loads(path.read_text()).values()]
    elif path.suffix.lower() == ".csv":
        import pyarrow.csv

        include = pyarrow.csv.ConvertOptions(include_columns=["status"])
        statuses = pyarrow.csv.read_csv(path, convert_options=include).column("status").to_pylist()
    else:
        import pyarrow.parquet as pq

        statuses = pq.read_table(path, columns=["status"]).column("status").to_pylist()
    return len(statuses), statuses.count("failed")


def file_sha256(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def writable(path, source, option):
    """Refuse an output file that would overwrite the input or cannot be written."""
    if path.resolve() == source.resolve():
        raise ValueError(f"{option} would overwrite the input {source}; choose another file")
    if not path.parent.is_dir() or not os.access(path.parent, os.W_OK):
        raise ValueError(f"cannot write {path}: {path.parent} does not exist or is not writable")


def needs(module, message):
    try:
        __import__(module)
    except ImportError:
        raise ImportError(message) from None


def class_prior(text):
    """``Ia=0.6,IIP=0.4`` as a mapping of class to value."""
    try:
        return {name.strip(): float(value) for name, value in (pair.split("=") for pair in text.split(","))}
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a list of CLASS=VALUE, such as Ia=0.6,IIP=0.4") from None


def column_names(text):
    """``object_id=SNID,error=flux_error`` as a mapping of STRIDER's names to the file's."""
    try:
        return {name.strip(): column.strip() for name, column in (pair.split("=") for pair in text.split(","))}
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a list of NAME=COLUMN, such as object_id=SNID,error=flux_error"
        ) from None


def error_scale(text):
    """``auto`` or a number."""
    if text == "auto":
        return text
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError('error scale must be "auto" or a number') from None


PACKAGES = ("numpy", "torch", "scipy", "pyarrow", "pooch", "matplotlib", "cmasher", "astropy", "h5py", "qp-prob")


def info(args):
    found = described_run(Path(args.results)) if args.results else described_install()
    print(json.dumps(found, indent=1) if args.json else plain(found))


def described_install():
    """This install: STRIDER and its packages' versions (read without importing them), the machine, and the models."""
    from importlib.metadata import PackageNotFoundError, version

    def installed(name):
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    try:
        memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (AttributeError, OSError, ValueError):  # a platform without sysconf
        memory = None
    return {
        "strider": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processors": os.cpu_count(),
        "memory_gib": None if memory is None else round(memory, 1),
        "packages": {name: installed(name) for name in PACKAGES},
        "model_cache": str(registry.model_dir()),
        "models": {
            name: {"default": name == registry.DEFAULT_MODEL, "cached": registry.model_path(name).exists()}
            for name in registry.MODELS
        },
    }


def described_run(path):
    """The record of the run that wrote a results file, or of a run under way or stopped, from its parts folder,
    with how many objects it has saved."""
    from .runs import read_record

    if path.is_dir():
        import pyarrow.parquet as pq

        record = json.loads((path / "_record.json").read_text())
        return record | {
            "saved": pq.read_table(path, columns=["object_id"]).num_rows if any(path.glob("part-*")) else 0
        }
    return read_record(path)


def plain(found):
    """An install's or a run's description as a few lines of text."""
    if "packages" in found:
        packages = ", ".join(f"{name} {found['packages'][name] or 'not installed'}" for name in PACKAGES)
        models = ", ".join(
            f"{name}{' (default)' if model['default'] else ''} {'cached' if model['cached'] else 'not downloaded'}"
            for name, model in found["models"].items()
        )
        return "\n".join(
            [
                f"STRIDER {found['strider']}, Python {found['python']}, {found['platform']}",
                f"{found['processors']} processors, {found['memory_gib']} GiB of memory",
                f"Packages: {packages}",
                f"Models in {found['model_cache']}: {models or 'none released'}",
            ]
        )
    settings = found["settings"]
    lines = [
        f"{found['status'].capitalize()} run of {settings['model']} ({settings['fingerprint'][:12]}, "
        f"{settings['scan']} scan) by STRIDER {settings['strider_version']}, started {found['started']}"
        + (f", finished {found['finished']}" if found.get("finished") else ""),
        f"Input: {found['input']['path'] or 'given in Python'}"
        + (f" (sha256 {found['input']['sha256'][:12]})" if found["input"]["sha256"] else "")
        + (f", job {found['job'][0]}/{found['job'][1]}" if found.get("job") else ""),
    ]
    if "saved" in found:
        lines.append(f"Saved so far: {found['saved']} objects")
    if found.get("summary"):
        done, seconds = found["summary"], found["summary"]["seconds_per_object"]
        statuses = ", ".join(f"{name} {count}" for name, count in done["status"].items())
        lines.append(f"Objects: {done['objects']} ({statuses})")
        if seconds["median"] is not None:
            lines.append(
                f"Seconds per object: median {seconds['median']:.0f}, 90% {seconds['p90']:.0f}, longest "
                f"{seconds['max']:.0f}; {done['cpu_seconds'] / 3600:.1f} CPU hours; most memory a process "
                f"used {done['largest_memory_mb']:.0f} MB"
            )
        if done["warning_codes"]:
            lines.append("Warnings: " + ", ".join(f"{code} {n}" for code, n in done["warning_codes"].items()))
    return "\n".join(lines)


def rerun(args):
    from .saved import rerun as again

    folders = []
    for given in map(Path, args.folders):
        folders += (
            [given] if (given / "record.json").exists() else sorted(p.parent for p in given.glob("*/record.json"))
        )
    if not folders:
        raise ValueError(f"no saved transient in {', '.join(args.folders)}; its folder holds record.json")
    models, changed = {}, 0
    for folder in folders:
        scan = json.loads((folder / "record.json").read_text())["settings"]["scan"]
        if args.model is not None and scan not in models:
            models[scan] = load(args.model, scan=scan)
        differences, machine = again(folder, models.get(scan))
        changed += bool(differences)
        name = json.loads((folder / "result.json").read_text())["object_id"]
        print(f"{name}: {'differs' if differences else 'the same'}")
        for line in differences + (machine if differences else []):
            print(f"  {line}")
    if changed:
        sys.exit(f"strider: {changed} of {len(folders)} saved transients differ")


def merge(args):
    from .runs import merge as join

    record = join(args.inputs, args.output, partial=args.partial)
    print(f"Joined {len(args.inputs)} files into {args.output} ({record['status']})")


def models(args):
    if args.card is not None:
        from .package import read_card

        print(read_card(args.card if Path(args.card).exists() else registry.fetch(args.card)))
        return
    print(f"Model cache: {registry.model_dir()}")
    if not registry.MODELS:
        print(f"No models released yet. The bundled test model, {registry.TEST_MODEL}, gives placeholder results.")
    for name in registry.MODELS:
        default = " (default)" if name == registry.DEFAULT_MODEL else ""
        status = "cached" if registry.model_path(name).exists() else "not downloaded"
        print(f"{name}{default}  {status}")


def example_results(results):
    """The numbers `check` compares, for each object of the bundled example, from its results."""
    return {
        key: {
            "class_probabilities": list(result.class_probabilities.values()),
            "equal_prior_probabilities": list(result.equal_prior_probabilities.values()),
            "redshift": [result.redshift, *result.redshift_68, *result.redshift_95],
            "redshift_peaks": [value for peak in result.redshift_peaks for value in peak],
            "class_redshifts": [None if np.isnan(z) else z for z in result.class_redshifts.values()],
            "signal_to_noise": result.signal_to_noise,
            "source_probability": result.source_probability,
            "flags": result.flags,
            "dropped_visits": sorted(result.dropped_visits),
        }
        for key, result in results.items()
    }


def check_example(model):
    """Classify the bundled example with ``model`` and compare it with the results stored for this model.

    Returns
    -------
    problems : list of str or None
        The differences found, empty if the results match; None if no results are stored for this model.
    seconds : float
        Seconds per object.
    """
    objects = read_spectra(EXAMPLE)
    model.classify_many(dict(list(objects.items())[:1]))  # a warm-up, so that the timing is of the steady state
    start = time.perf_counter()
    results = model.classify_many(objects)
    seconds = (time.perf_counter() - start) / len(results)
    entry = json.loads(EXPECTED.read_text()).get(model.fingerprint)
    if entry is None:
        return None, seconds
    if not {"fast", "full"} & set(entry):  # stored by STRIDER 0.1, which had only the full scan
        entry = {"full": {key: value for key, value in entry.items() if key != "model"}}
    stored = entry.get(model.scan)
    if stored is None:
        return [f"no stored results for the {model.scan} scan of this model"], seconds
    problems = [f"{key}: {result}" for key, result in results.items() if isinstance(result, str)]
    found = example_results({key: result for key, result in results.items() if not isinstance(result, str)})
    for key, values in found.items():
        for name, value in values.items():
            expected = stored[key][name]
            if name in ("flags", "dropped_visits"):
                same = value == expected
            else:
                value, expected = np.asarray(value, dtype=float), np.asarray(expected, dtype=float)
                same = value.shape == expected.shape and np.allclose(
                    value, expected, rtol=0.0, atol=TOLERANCE, equal_nan=True
                )
            if not same:
                problems.append(f"{key}: {name} {value}, stored {expected}")
    return problems, seconds


def check(args):
    import torch

    print(
        f"strider {__version__}, Python {platform.python_version()}, NumPy {np.__version__}, PyTorch "
        f"{torch.__version__}, {torch.get_num_threads()} threads, {platform.platform()}"
    )
    model = load(args.model, scan="full" if args.full_scan else "fast")
    problems, seconds = check_example(model)
    print(
        f"{model.name} ({model.fingerprint[:12]}, {model.scan} scan): {seconds:.2f} s per object on the bundled example"
    )
    if problems is None:
        print("This model has no stored results for the example, so only the timing is checked.")
        return
    for problem in problems:
        print(problem)
    print(f"The example {'does not match' if problems else 'matches'} its stored results.")
    if problems:
        sys.exit(1)


def load(name, device="cpu", scan="fast"):
    from .model import load_model

    if name is None and not registry.MODELS:
        raise ValueError(
            f"no model is released yet; give --model {registry.TEST_MODEL} for the bundled test model, "
            "whose results are placeholders, or --model path/to/package.tar.gz"
        )
    return load_model(name, device, scan=scan)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="strider", description="Identify transients from their Roman prism spectra.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    model_help = (
        f"a released model's name, {registry.TEST_MODEL} for the bundled test model, or the path to a model "
        "package; if left out, the default model"
    )

    sub = commands.add_parser(
        "classify",
        help="classify one transient, or many",
        description="Classify the transients in a file. Print each one's result, or write the results with -o.",
    )
    sub.add_argument(
        "input",
        help="a CSV or ECSV file of one transient, one row per pixel; or a parquet table of many, one row per visit",
    )
    sub.add_argument("-o", "--output", help="the file of results to write: .parquet, .csv or .json")
    sub.add_argument(
        "--columns",
        type=column_names,
        metavar="NAME=COLUMN,...",
        help="the table's own names for STRIDER's columns, such as object_id=SNID,error=flux_error",
    )
    sub.add_argument(
        "--wavelength-unit", default="angstrom", choices=list(WAVELENGTH_UNITS), help="the table's wavelength unit"
    )
    sub.add_argument(
        "--flux-density",
        default="f_lambda",
        choices=list(FLUX_DENSITIES),
        help="flux per unit wavelength (f_lambda, the default) or per unit frequency (f_nu), in any units",
    )
    sub.add_argument("--model", help=model_help)
    sub.add_argument(
        "--classes",
        type=lambda text: [name.strip() for name in text.split(",")],
        metavar="CLASS,...",
        help="the only classes allowed, such as Ia,Iax,91bg",
    )
    sub.add_argument(
        "--class-prior",
        type=class_prior,
        metavar="CLASS=VALUE,...",
        help="how common each class allowed is, in place of the survey prior, such as Ia=0.6,IIP=0.4 "
        "with --classes Ia,IIP",
    )
    sub.add_argument(
        "--host-redshift-outlier",
        type=float,
        metavar="SHARE",
        help="the chance that the host redshift is wrong (default 0.05; about 0.03 for a spectroscopic redshift)",
    )
    sub.add_argument("--zmin", type=float, help="the lowest redshift allowed")
    sub.add_argument("--zmax", type=float, help="the highest redshift allowed")
    sub.add_argument(
        "--error-scale",
        type=error_scale,
        default=1.0,
        metavar="FACTOR",
        help="multiply every error by FACTOR, or by a factor measured from the spectra with 'auto' (default 1)",
    )
    sub.add_argument(
        "--light-curve-errors",
        default="check",
        choices=("check", "correct"),
        help="check (the default) whether the light curve's errors are too small for its scatter, or also correct "
        "them; correct cannot go with --error-scale",
    )
    sub.add_argument(
        "--plot",
        nargs="?",
        const=".",
        metavar="FOLDER",
        help="save each transient's figure as <name>.png, in FOLDER or here",
    )
    sub.add_argument(
        "--evidence-maps",
        metavar="FILE",
        help="write every class at every redshift, before any prior, to an HDF5 file; needs h5py",
    )
    sub.add_argument(
        "--redshift-curves",
        metavar="FILE",
        help="write the redshift posteriors in qp's format, as LSST and DESC redshift tools read them; needs qp",
    )
    sub.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="classify N transients at once, each on one CPU core; the fastest way through many on a CPU",
    )
    sub.add_argument(
        "--device", default="cpu", choices=("cpu", "cuda"), help="run on the CPU (the default) or a GPU (cuda)"
    )
    sub.add_argument(
        "--full-scan",
        action="store_true",
        help="start the brightness comparison's peak-date search 4 times finer, as STRIDER 0.1 did; slower",
    )
    sub.add_argument(
        "--batch",
        type=int,
        metavar="N",
        help="score N transients together; the fastest way through many on a GPU (default 8 there, 1 on a CPU)",
    )
    sub.add_argument(
        "--job",
        metavar="K/N",
        help="be job K of N (K from 0 to N - 1), so N jobs share one input; join their results with strider merge",
    )
    sub.add_argument(
        "--max-spectra",
        type=int,
        metavar="N",
        help="use only the first N usable spectra of each transient, in time order, to bound the time a transient "
        "with many takes; by default all",
    )
    sub.add_argument(
        "--save-every",
        type=int,
        default=100,
        metavar="N",
        help="save the results every N transients (default 100); with --workers, also at least every 5 minutes",
    )
    sub.add_argument(
        "--verbose", action="store_true", help="say on stderr how each transient went, with its time and memory"
    )
    sub.add_argument(
        "--only",
        type=lambda text: [name.strip() for name in text.split(",")],
        metavar="ID,...",
        help="classify only these transients",
    )
    sub.add_argument(
        "--save-transient",
        metavar="FOLDER",
        help="with --only: save each transient's input, settings and result in a folder of its own, which strider "
        "rerun classifies again",
    )
    sub.add_argument(
        "--resume",
        action="store_true",
        help="carry on a .parquet run that stopped, from the results it saved in <output>.parts; with none, start one",
    )
    sub.set_defaults(func=classify)

    sub = commands.add_parser(
        "merge",
        help="join the results of a run's jobs",
        description="Join the .parquet results of a run's jobs (strider classify --job) into one file. They must "
        "come from one input file, model and settings, and hold each job once.",
    )
    sub.add_argument("inputs", nargs="+", help="the jobs' results files")
    sub.add_argument("-o", "--output", required=True, help="the .parquet file to write")
    sub.add_argument("--partial", action="store_true", help="join them even if some jobs are missing")
    sub.set_defaults(func=merge)

    sub = commands.add_parser(
        "rerun",
        help="classify saved transients again and compare",
        description="Classify saved transients (strider classify --only ... --save-transient) again, with their "
        "run's model and settings, and say for each whether its result is the same.",
    )
    sub.add_argument("folders", nargs="+", help="the saved transients' folders, or folders of them")
    sub.add_argument("--model", help=model_help)
    sub.set_defaults(func=rerun)

    sub = commands.add_parser(
        "info",
        help="describe this install, or the run that wrote a results file",
        description="Describe this install: STRIDER's version, its packages, the machine and the models. Or, given a "
        "results file, or a stopped run's .parts folder, describe that run: its model, settings, input, times and "
        "what it holds.",
    )
    sub.add_argument("results", nargs="?", help="a .parquet results file, or the .parts folder of a run")
    sub.add_argument("--json", action="store_true", help="print it as JSON")
    sub.set_defaults(func=info)

    sub = commands.add_parser("models", help="list the released models", description="List the released models.")
    sub.add_argument("--card", metavar="MODEL", help="print the card of a model, given by name or package path")
    sub.set_defaults(func=models)

    sub = commands.add_parser(
        "check",
        aliases=["check-model"],
        help="check the install and a model on the bundled example",
        description="Check the install and a model. This prints the versions, times the bundled example, and "
        "compares its results with the stored ones.",
    )
    sub.add_argument("--model", help=model_help)
    sub.add_argument("--full-scan", action="store_true", help="check the full peak-date scan instead of the fast one")
    sub.set_defaults(func=check)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        resumable = args.command == "classify" and str(args.output or "").lower().endswith(".parquet")
        print("strider: stopped" + ("; carry on with --resume" if resumable else ""), file=sys.stderr)
        sys.exit(130)
    except (FileNotFoundError, ImportError, ValueError) as error:
        sys.exit(f"strider: {error}")
    except OSError as error:
        sys.exit(
            f"strider: {error}. If a model was being downloaded, check the network connection. Also check that "
            f"{registry.model_dir()} is writable, or set STRIDER_MODEL_DIR to a directory that is"
        )


if __name__ == "__main__":
    main()
