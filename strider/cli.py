"""The ``strider`` command: ``classify``, ``models`` and ``check``."""

import argparse
import json
import os
import platform
import signal
import sys
import time
from importlib.resources import files
from pathlib import Path

import numpy as np

from . import __version__, registry
from .io import EXAMPLE, RedshiftCurvesFile, ResultsFile, read_spectra, read_transient

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
    objects = read_spectra(source) if source.suffix.lower() == ".parquet" else {source.stem: read_transient(source)}
    if not objects:
        raise ValueError(f"{source} has no visits")
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
    model = load(args.model, args.device)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # a scheduler's stop first writes the transients done
    results = model.classify_many(
        objects,
        output,
        plots=args.plot,
        evidence_maps=args.evidence_maps,
        redshift_curves=args.redshift_curves,
        workers=args.workers,
        batch=args.batch,
        error_scale=args.error_scale,
        keep=output is None,  # with a results file, keep only the failures in memory
        **priors,
    )
    failed = [key for key, result in results.items() if isinstance(result, str)]
    if output is None:
        print(
            "\n\n".join(
                f"{key}: not classified. {result}" if key in failed else result.card(key)
                for key, result in results.items()
            )
        )
    else:
        print(f"Classified {len(objects) - len(failed)} of {len(objects)} objects; results in {output}")
        for key in failed:
            print(f"{key}: {results[key]}")
    if len(failed) == len(objects):
        sys.exit("strider: no object could be classified; the reasons are above")
    if args.plot is not None:
        print(f"Figures in {args.plot}")
    if args.evidence_maps is not None:
        print(f"Evidence maps in {args.evidence_maps}")
    if args.redshift_curves is not None and Path(args.redshift_curves).exists():
        print(f"Redshift curves in {args.redshift_curves}")


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


def error_scale(text):
    """``auto`` or a number."""
    if text == "auto":
        return text
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError('error scale must be "auto" or a number') from None


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
    stored = json.loads(EXPECTED.read_text()).get(model.fingerprint)
    if stored is None:
        return None, seconds
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
    model = load(args.model)
    problems, seconds = check_example(model)
    print(f"{model.name} ({model.fingerprint[:12]}): {seconds:.2f} s per object on the bundled example")
    if problems is None:
        print("This model has no stored results for the example, so only the timing is checked.")
        return
    for problem in problems:
        print(problem)
    print(f"The example {'does not match' if problems else 'matches'} its stored results.")
    if problems:
        sys.exit(1)


def load(name, device="cpu"):
    from .model import load_model

    if name is None and not registry.MODELS:
        raise ValueError(
            f"no model is released yet; give --model {registry.TEST_MODEL} for the bundled test model, "
            "whose results are placeholders, or --model path/to/package.tar.gz"
        )
    return load_model(name, device)


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
        "--plot",
        nargs="?",
        const=".",
        metavar="FOLDER",
        help="save each transient's figure as <name>.png, in FOLDER or here; needs matplotlib",
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
        "--batch",
        type=int,
        metavar="N",
        help="score N transients together; the fastest way through many on a GPU (default 8 there, 1 on a CPU)",
    )
    sub.set_defaults(func=classify)

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
    sub.set_defaults(func=check)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, ImportError, ValueError) as error:
        sys.exit(f"strider: {error}")
    except OSError as error:
        sys.exit(
            f"strider: {error}. If a model was being downloaded, check the network connection. Also check that "
            f"{registry.model_dir()} is writable, or set STRIDER_MODEL_DIR to a directory that is"
        )


if __name__ == "__main__":
    main()
