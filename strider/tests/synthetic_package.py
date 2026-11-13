"""A tiny synthetic model package for the tests: the released design at toy size, with seeded random weights.

Its results mean nothing. ``python -m strider.tests.synthetic_package`` writes ``ARCHIVE``, and the results of the
bundled example for it, which ``strider check`` compares, in ``strider/data/example_expected.json``.
"""

import gzip
import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import torch

from strider.bands import EDGES
from strider.package import FORMAT, FORMAT_VERSION, fingerprint, phase_bin_weights
from strider.posterior import redshift_grid

NAME, VERSION = "strider-synthetic-test", "1.0"
ARCHIVE = Path(__file__).parent / "data" / f"{NAME}-{VERSION}.tar.gz"
CLASSES = ["Ia", "91bg", "Iax", "IIP", "IIL", "IIb", "IIn", "Ib", "Ic", "Ic-BL", "SLSN", "TDE", "ILOT", "KN", "PISN"]
NODES = np.arange(-20.0, 81.0, 25.0)
PIXELS, REST_PIXELS = 128, 48
F_SHA256 = hashlib.sha256(b"synthetic spectral model").hexdigest()
CONFIG = {
    "data": {"class_scheme": "hourglass_15"},
    "observation": {"wavelength_min": 7500.0, "wavelength_max": 18175.0, "wavelength_bins": PIXELS},
    "model": {
        "classes": CLASSES,
        "dropout": 0.1,
        "evidence_use_visit_count_and_span": False,
        "evidence_visit_count_reference": 32,
        "redshift_bins": 12,
        "redshift_max": 3.0,
        "redshift_min": 0.0,
        "redshift_prior": "flat_z",
        "redshift_spacing": "log1p",
    },
    "fusion": {"spectrum_weight": 0.9, "brightness_weight": 0.5},
    "match_strength": {"high": 5, "medium": 2, "second_redshift": 3},
    "reference": {
        "attention_heads": 2,
        "continuum_width_km_s": 12000.0,
        "edge_taper_fraction": 0.05,
        "edge_trim_fraction": 0.0,
        "encoder_feedforward_multiplier": 2,
        "fine_class_temperature": 0.1,
        "minimum_encoder_support": 0.5,
        "minimum_profile_support": 5,
        "minimum_relative_spectral_precision": 1.1920929e-07,
        "minimum_rest_fraction": 0.15,
        "minimum_shared_fraction": 0.75,
        "phase_edges_days": [-20.0, 10.0, 80.0],
        "token_dim": 8,
        "token_pool_size": 4,
    },
}
PREPROCESSING = {
    "flux_error_input": True,
    "maximum_visits": None,
    "wavelength_frame": "observer",
    "wavelength_grid": {"bins": PIXELS, "maximum": 18175.0, "minimum": 7500.0, "spacing": "log"},
    "wavelength_support": "visits",
    "wavelength_unit": "angstrom",
}


def toy_brightness():
    """The brightness term's tables: light curves of a different width for each class, at two template redshifts
    (few, so that the tests run fast), with loadings for SN Ia only; the band edges; and a flat noise profile."""
    nodes, phase = np.log1p([0.0, 3.0]), np.arange(-10.0, 31.0)
    width = 6.0 + 2.0 * np.arange(len(CLASSES))[:, None, None, None]
    colour = np.array([1.0, 0.8, 0.6, 0.4])[:, None, None]
    mean = colour * (1.0 + 0.2 * np.arange(len(nodes)))[:, None] * np.exp(-0.5 * (phase / width) ** 2)
    loadings = np.zeros((*mean.shape, 2))
    loadings[0] = 0.02 * mean[0, ..., None] * np.stack([np.ones_like(phase), phase / 30.0], -1)
    wavelength = np.geomspace(7400.0, 18300.0, 60)
    return {
        "classes": np.array(CLASSES),
        "z_nodes": nodes,
        "mean": mean,
        "psi2": (0.05 * mean) ** 2 + 1e-6,
        "loadings": loadings,
        "window": np.array([-10.0, 30.0]),
        "tau": np.array([2.5, 58.0]),
        "discrepancy": np.tile([0.71, 32.0, 0.6, 2.83, 0.2], (len(CLASSES), 1)),
        "band_edges": EDGES,
        "profile_wavelength": wavelength,
        "profile_sigma": np.ones_like(wavelength),
    }


def spectral_model(state_dict=None):
    """The contents of ``spectral_model.pt``: KN has too few training spectra after +30 d."""
    coverage = np.zeros((len(CLASSES), len(NODES), REST_PIXELS), dtype=np.int64)
    coverage[..., 4:44] = 10
    coverage[CLASSES.index("KN"), NODES > 30.0] = 0
    return {
        "format": "strider-spectral-model-1",
        "class_names": CLASSES,
        "settings": {
            "basis_functions": 16,
            "class_embedding_dim": 4,
            "phase_frequencies": 2,
            "hidden_dim": 8,
            "layers": 1,
        },
        "state_dict": state_dict,
        "rest_wavelength": torch.from_numpy(np.geomspace(2500.0, 10000.0, REST_PIXELS).astype(np.float32)),
        "phase_range_days": [-20.0, 80.0],
        "phase_nodes_days": torch.from_numpy(NODES.astype(np.float32)),
        "coverage_counts": torch.from_numpy(coverage),
        "source_sha256": F_SHA256,
    }


def toy_network(seed=0):
    """The synthetic package's network, with seeded random weights and masks agreeing with the coverage."""
    from strider.network import Network

    torch.manual_seed(seed)
    network = Network(CONFIG, spectral_model())
    coverage, settings = spectral_model()["coverage_counts"].numpy(), CONFIG["reference"]
    weight = phase_bin_weights(NODES, settings["phase_edges_days"], coverage, settings["minimum_profile_support"])
    used, enough = weight > 0.0, coverage >= settings["minimum_profile_support"]
    mask = used.any(-1)[..., None] & np.all(~used[..., None] | enough[:, None], axis=2)
    with torch.no_grad():
        network.spectral_model.coefficients.bias.fill_(1.0)  # a flat, positive spectrum to start from
        for name, value in (
            ("continuum_removed_intercept", 0.4),
            ("log_coadd_scale", np.log(2.0)),
            ("log_prototype_temperature", np.log(0.08)),
        ):
            getattr(network, name).fill_(float(value))
        network.phase_nodes_days.copy_(torch.from_numpy(NODES))
        network.window_weight.copy_(torch.from_numpy(weight))
        network.coadd_profile_mask.copy_(torch.from_numpy(mask))
        network.coadd_profile_supported.copy_(torch.from_numpy(mask.any(-1)))
    return network


def priors():
    """A training prior flat in z, without KN above z = 1, and a survey prior with more objects at low z."""
    shares = dict(zip(CLASSES, [30.0, 1.0, 1.0, 20.0, 4.0, 8.0, 5.0, 6.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    edges = [0.0, 1.0, 3.0]
    training = {
        "class_fraction": dict.fromkeys(CLASSES, 1.0),
        "redshift_edges": edges,
        "redshift_density": {name: [0.0, 1.0] if name == "KN" else [1.0, 1.0] for name in CLASSES},
    }
    survey = {"class_fraction": shares, "redshift_edges": edges, "redshift_density": dict.fromkeys(CLASSES, [2.0, 1.0])}
    return {
        "fine_to_output": {name: name for name in CLASSES},
        "training_population": training | {"fine": training},
        "survey": survey | {"fine": survey, "description": "a synthetic survey: more objects at low redshift"},
    }


def package_files():
    """Every file of the package, by name."""
    network = toy_network()
    names = network.weight_names()
    weights = {names[key]: value for key, value in network.state_dict().items()}
    files = {
        "weights.pt": _torch_bytes(weights),
        "spectral_model.pt": _torch_bytes(spectral_model(network.spectral_model.state_dict())),
        "brightness.npz": _npz_bytes(toy_brightness()),
        "preprocessing.json": _json_bytes(PREPROCESSING),
        "priors.json": _json_bytes(priors()),
    }
    identity = fingerprint(CONFIG, {name: hashlib.sha256(data).hexdigest() for name, data in files.items()})
    grid = redshift_grid(CONFIG["model"])
    info = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "name": NAME,
        "version": VERSION,
        "fingerprint": identity,
        "classes": CLASSES,
        "fine_classes": CLASSES,
        "redshift_grid": {
            "minimum": 0.0,
            "maximum": 3.0,
            "bins": len(grid),
            "spacing": "log1p",
            "prior_measure": "flat_z",
        },
        "calibration_status": "fitted",
        "metrics_status": "not_included",
        "created": "2026-10-31",
        "provenance": {},
    }
    calibration = {
        "status": "fitted",
        "fingerprint": identity,
        "classes": CLASSES,
        "source_probability": {
            "method": "balanced-affine-logit",
            "slope": 1.2,
            "intercept": 0.3,
            "weak_source_below": 0.6,
            "blank_false_positive_rate": 0.05,
        },
    }
    texts = {
        "package.json": info,
        "model_config.json": CONFIG,
        "calibration.json": calibration,
    }
    files |= {name: _json_bytes(value) for name, value in texts.items()}
    files["redshift_grid.npy"] = _npy_bytes(grid)
    files["wavelength_grid.npy"] = _npy_bytes(np.geomspace(7500.0, 18175.0, PIXELS))
    files["MODEL_CARD.md"] = f"# {NAME} (version {VERSION})\n\nA synthetic model for the tests.\n".encode()
    files["SHA256SUMS"] = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in sorted(files.items())
    ).encode()
    return files


def write_archive(path=ARCHIVE):
    """Write the package as ``<name>-<version>.tar.gz`` with fixed dates and owners, so that it is reproducible."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, data in sorted(package_files().items()):
            member = tarfile.TarInfo(f"{NAME}-{VERSION}/{name}")
            member.size, member.mode = len(data), 0o644
            archive.addfile(member, io.BytesIO(data))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(gzip.compress(buffer.getvalue(), mtime=0))
    return Path(path)


def write_directory(directory, files, **changes):
    """The package with its JSON files changed, written to a directory; the fingerprint and checksums follow."""
    files = dict(files)
    for name, change in changes.items():
        files[name] = json.dumps(change(json.loads(files[name]))).encode()
    digest = {name: hashlib.sha256(files[name]).hexdigest() for name in files}
    identity = fingerprint(json.loads(files["model_config.json"]), digest)
    for name in ("package.json", "calibration.json"):
        files[name] = json.dumps(json.loads(files[name]) | {"fingerprint": identity}).encode()
    files["SHA256SUMS"] = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in files.items() if name != "SHA256SUMS"
    ).encode()
    directory.mkdir()
    for name, data in files.items():
        (directory / name).write_bytes(data)
    return directory


def _json_bytes(value):
    return json.dumps(value, indent=1, sort_keys=True).encode() + b"\n"


def _torch_bytes(value):
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _npy_bytes(array):
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.asanyarray(array), allow_pickle=False)
    return buffer.getvalue()


def _npz_bytes(arrays):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, array in arrays.items():
            archive.writestr(zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)), _npy_bytes(array))
    return buffer.getvalue()


def write_expected(path=ARCHIVE):
    """Store the results of the bundled example for the package at ``path``, by its fingerprint."""
    import strider
    from strider.cli import EXAMPLE, example_results
    from strider.io import read_spectra

    model = strider.load_model(path)
    expected = Path(__file__).parents[1] / "data" / "example_expected.json"
    stored = json.loads(expected.read_text()) if expected.exists() else {}
    stored = {key: value for key, value in stored.items() if value["model"] != model.name}  # an earlier build's
    stored[model.fingerprint] = {"model": model.name} | example_results(model.classify_many(read_spectra(EXAMPLE)))
    expected.write_text(json.dumps(stored, indent=1) + "\n")
    return expected


if __name__ == "__main__":
    print(write_archive())
    print(write_expected())
