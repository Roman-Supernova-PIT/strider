"""Four cases of the internal network's 15-class release fixtures (random weights: the numbers mean nothing), with
the model's brightness tables at two of their template redshifts, z = 0 and 3, so that the tests run fast.

The coadd term and the source detector's logit must equal the internal network's. The other outputs, which the
brightness term's null copies change, are compared with those this STRIDER recorded, with its seed rule
(`strider.preprocess`).

``python -m strider.tests.test_reference`` records them again.
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
    """The outputs compared, by name."""
    data, host = arguments(arrays)
    result = model.classify(*data, **host)
    network = model._evaluate(model._prepare(*data))
    found = {name: getattr(result, name) for name in RECORDED}
    found |= {"coadd_term": network["coadd_term"], "evidence_sufficiency_logit": network["source_logit"]}
    found["class_probabilities"] = list(found["class_probabilities"].values())
    found["equal_prior_probabilities"] = list(found["equal_prior_probabilities"].values())
    found["flags"] = [found["flags"][name] for name in sorted(found["flags"])]
    found["class_redshifts"] = list(found["class_redshifts"].values())
    found["redshift_peaks"] = np.ravel(found["redshift_peaks"])
    return {name: np.asarray(value, dtype=np.float64) for name, value in found.items()}


@pytest.fixture(scope="module")
def model():
    threads = torch.get_num_threads()
    torch.set_num_threads(THREADS)
    yield strider.load_model(PACKAGE)
    torch.set_num_threads(threads)


@pytest.mark.parametrize("case", NAMES)
def test_a_reference_case(model, case):
    stored = np.load(CASES)
    arrays = {name.split("__", 1)[1]: stored[name] for name in stored.files if name.startswith(f"{case}__")}
    found = outputs(model, arrays)
    for name in INTERNAL + RECORDED:
        np.testing.assert_allclose(found[name], arrays[name], err_msg=f"{case}: {name}", **tolerance(name))


def record():
    """Record this STRIDER's outputs for the stored cases again, keeping their inputs and the internal outputs."""
    torch.set_num_threads(THREADS)
    model = strider.load_model(PACKAGE)
    cases = np.load(CASES)
    stored = {}
    for case in NAMES:
        arrays = {name.split("__", 1)[1]: cases[name] for name in cases.files if name.startswith(f"{case}__")}
        arrays = {name: value for name, value in arrays.items() if name.startswith("input_") or name in INTERNAL}
        found = outputs(model, arrays)
        arrays |= {name: found[name] for name in RECORDED}
        stored |= {f"{case}__{name}": value for name, value in arrays.items()}
        difference = max(float(np.abs(found[name] - arrays[name]).max()) for name in INTERNAL)
        print(case, "internal outputs, largest difference:", difference)
    np.savez_compressed(CASES, **stored)


if __name__ == "__main__":
    record()
