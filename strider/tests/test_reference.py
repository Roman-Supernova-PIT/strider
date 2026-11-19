"""Four cases of the internal network's 15-class release fixtures (random weights: the numbers mean nothing), with
the model's brightness tables at two of their template redshifts, z = 0 and 3, so that the tests run fast.

The coadd term and the source detector's logit must equal the internal network's. The other outputs, which the
brightness term's null copies change, are compared with those this STRIDER recorded, with its seed rule
(`strider.preprocess`), for each peak-date scan: the full scan's under their plain names, the fast scan's (the
default) under ``fast__``.

``python -m strider.tests.test_reference`` records them again; ``... fast`` records only the fast scan's.
"""

import platform
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import strider  # noqa: E402

DATA = Path(__file__).parent / "data"
PACKAGE = DATA / "strider-15class-fixture-1.0.tar.gz"
CASES = DATA / "reference_cases.npz"
NAMES = ("selection_02", "dropped_visit", "host_redshift", "bad_bins")
INTERNAL = ("coadd_term", "evidence_sufficiency_logit")
RECORDED = (
    "log_evidence",
    "class_probabilities",
    "equal_prior_probabilities",
    "redshift_posterior",
    "redshift",
    "redshift_probability",
    "redshift_peaks",
    "class_redshifts",
    "redshift_68",
    "redshift_95",
    "source_probability",
    "untrained_share",
    "flags",
    "visits_used",
    "signal_to_noise",
)
# the best match and how clearly it wins: the outputs a change of peak-date scan moves most; the gaps of these
# cases lie at least 0.02 from a level, far beyond any rounding, so the labels are compared on every machine
MATCHES = ("best_match_class", "best_match_redshift", "next_best_class", "match_gap", "match_strength", "warnings")
THREADS = 2  # the cases were made on 2 threads; more threads split the float32 sums differently


def tolerance(name):
    """1e-6 where the cases were made: torch 2.11 on arm64. Other torch versions and other processors (x86 on the
    test servers) sum float32 in another order: up to 1e-3 of the coadd term and the evidence, 1e-4 near zero, and a
    few 1e-6 of a probability or redshift."""
    if torch.__version__.startswith("2.11.") and platform.machine() == "arm64":
        return {"rtol": 2.4e-7, "atol": 1e-6}  # and two float32 rounding steps of the larger values
    logits = name in ("coadd_term", "log_evidence")
    return {"rtol": 2e-3 if logits else 1e-5, "atol": 1e-4 if logits else 5e-6}


def arguments(arrays):
    """The inputs of `strider.Model.classify` from a case's arrays."""
    cut = arrays["input_visit_offsets"]

    def split(name):
        return [arrays[name][start:stop] for start, stop in zip(cut[:-1], cut[1:])]

    host = {}
    if "input_host_redshift" in arrays:
        host = {
            "host_redshift": float(arrays["input_host_redshift"]),
            "host_redshift_error": float(arrays["input_host_redshift_error"]),
        }
    return [
        split("input_wavelength"),
        split("input_flux"),
        split("input_flux_error"),
        arrays["input_mjd"],
        split("input_mask"),
    ], host


def outputs(model, arrays):
    """The outputs compared, by name: numbers, then the match's text and numbers."""
    data, host = arguments(arrays)
    result = model.classify(*data, **host)
    matched = {
        "best_match_class": result.best_match[0],
        "best_match_redshift": result.best_match[1],
        "next_best_class": result.next_best_class,
        "match_gap": result.match_gap,
        "match_strength": result.match_strength,
        "warnings": "; ".join(result.warnings),
    }
    network = model._evaluate(model._prepare(*data))
    found = {name: getattr(result, name) for name in RECORDED}
    found |= {"coadd_term": network["coadd_term"], "evidence_sufficiency_logit": network["source_logit"]}
    found["class_probabilities"] = list(found["class_probabilities"].values())
    found["equal_prior_probabilities"] = list(found["equal_prior_probabilities"].values())
    found["flags"] = [found["flags"][name] for name in sorted(found["flags"])]
    found["class_redshifts"] = list(found["class_redshifts"].values())
    found["redshift_peaks"] = np.ravel(found["redshift_peaks"])
    return {name: np.asarray(value, dtype=np.float64) for name, value in found.items()} | {
        name: np.asarray(value) for name, value in matched.items()
    }


@pytest.fixture(scope="module", params=["full", "fast"])
def model(request):
    threads = torch.get_num_threads()
    torch.set_num_threads(THREADS)
    yield strider.load_model(PACKAGE, scan=request.param)
    torch.set_num_threads(threads)


def recorded(stored, case, scan):
    """A case's inputs and internal outputs, and the outputs recorded for ``scan``."""
    arrays = {name.split("__", 1)[1]: stored[name] for name in stored.files if name.startswith(f"{case}__")}
    fast = {name[len("fast__") :]: value for name, value in arrays.items() if name.startswith("fast__")}
    arrays = {name: value for name, value in arrays.items() if not name.startswith("fast__")}
    return arrays | fast if scan == "fast" else arrays


@pytest.mark.parametrize("case", NAMES)
def test_a_reference_case(model, case):
    arrays = recorded(np.load(CASES), case, model.scan)
    found = outputs(model, arrays)
    for name in INTERNAL + RECORDED + ("best_match_redshift", "match_gap"):
        np.testing.assert_allclose(
            found[name], arrays[name], err_msg=f"{case} ({model.scan}): {name}", **tolerance(name)
        )
    for name in ("best_match_class", "next_best_class", "match_strength", "warnings"):
        assert str(found[name]) == str(arrays[name]), f"{case} ({model.scan}): {name}"


def record_matches():
    """Add the match outputs of both scans to the stored cases, leaving everything else as it was."""
    torch.set_num_threads(THREADS)
    cases = np.load(CASES)
    stored = {name: cases[name] for name in cases.files}
    for scan in ("full", "fast"):
        model = strider.load_model(PACKAGE, scan=scan)
        prefix = "fast__" if scan == "fast" else ""
        for case in NAMES:
            found = outputs(model, recorded(cases, case, scan))
            stored |= {f"{case}__{prefix}{name}": found[name] for name in MATCHES}
    np.savez_compressed(CASES, **stored)


def record(scans=("full", "fast")):
    """Record this STRIDER's outputs for the stored cases again for ``scans``, keeping their inputs, the internal
    outputs and the other scan's."""
    torch.set_num_threads(THREADS)
    cases = np.load(CASES)
    stored = {name: cases[name] for name in cases.files}
    for scan in scans:
        model = strider.load_model(PACKAGE, scan=scan)
        prefix = "fast__" if scan == "fast" else ""
        for case in NAMES:
            arrays = recorded(cases, case, "full")
            found = outputs(model, arrays)
            stored |= {f"{case}__{prefix}{name}": found[name] for name in RECORDED + MATCHES}
            difference = max(float(np.abs(found[name] - arrays[name]).max()) for name in INTERNAL)
            print(case, scan, "internal outputs, largest difference:", difference)
    np.savez_compressed(CASES, **stored)


if __name__ == "__main__":
    import sys

    record(tuple(sys.argv[1:]) or ("full", "fast"))
