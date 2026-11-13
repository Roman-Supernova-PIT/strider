"""Reading a STRIDER model package, a directory or its ``.tar.gz``, and checking it.

A package is refused unless every check passes: each file is present and matches its checksum in ``SHA256SUMS``;
the format is known; the fingerprint matches the declared config, both weight files, the brightness tables, the
preprocessing and the priors; every declared key is one this STRIDER implements, at a value it implements; the fusion
weights are positive, and the match-strength levels too, high above medium; the preprocessing, grids and classes
agree with the config; the spectral model in ``weights.pt`` equals ``spectral_model.pt``'s; the masks of the reference
spectra agree with the training coverage; the brightness tables are for this model's classes and bands; and a fitted
calibration names this model.
"""

import hashlib
import io
import json
import numbers
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bands import EDGES
from .posterior import redshift_grid

__all__ = ["Package", "read_package", "read_card", "fingerprint", "phase_bin_weights", "FORMAT", "FORMAT_VERSION"]

FORMAT, FORMAT_VERSION = "strider-model-package", 1
REQUIRED = (
    "package.json",
    "model_config.json",
    "weights.pt",
    "spectral_model.pt",
    "brightness.npz",
    "redshift_grid.npy",
    "wavelength_grid.npy",
    "preprocessing.json",
    "calibration.json",
    "priors.json",
    "MODEL_CARD.md",
    "SHA256SUMS",
)
# with model_config.json, the fingerprint covers the network, the brightness term, how inputs are made and the priors
FINGERPRINT_FILES = ("weights.pt", "spectral_model.pt", "brightness.npz", "preprocessing.json", "priors.json")
# brightness.npz's float64 tables, stored beside its classes, band edges and noise profile
BRIGHTNESS_TABLES = ("z_nodes", "mean", "psi2", "loadings", "window", "tau", "discrepancy")
SPECTRAL_MODEL_PREFIX = "roman_reference.spectral_model."
UPGRADE = "Upgrade STRIDER (pip install -U roman-snpit-strider), or use a model package made for this version"
DAMAGED = "the package is damaged. Download it again"
ANY = None
SOURCE_METHOD = "balanced-affine-logit"  # the source probability's calibration (`strider.posterior.source_probability`)
# the keys a package declares and the values this STRIDER implements; any other key is refused
IMPLEMENTED = {
    "data.class_scheme": ("hourglass_15",),
    "observation.wavelength_min": ANY,
    "observation.wavelength_max": ANY,
    "observation.wavelength_bins": ANY,
    "model.classes": ANY,
    "model.dropout": ANY,
    "model.redshift_min": ANY,
    "model.redshift_max": ANY,
    "model.redshift_bins": ANY,
    "model.redshift_prior": ("flat_z", "flat_log1p"),
    "model.redshift_spacing": ("log1p", "linear"),
    "model.evidence_use_visit_count_and_span": (True, False),
    "model.evidence_visit_count_reference": ANY,
    "reference.attention_heads": ANY,
    "reference.continuum_width_km_s": ANY,
    "reference.edge_taper_fraction": ANY,
    "reference.edge_trim_fraction": ANY,
    "reference.encoder_feedforward_multiplier": ANY,
    "reference.fine_class_temperature": ANY,
    "reference.minimum_encoder_support": ANY,
    "reference.minimum_profile_support": ANY,
    "reference.minimum_relative_spectral_precision": ANY,
    "reference.minimum_rest_fraction": ANY,
    "reference.minimum_shared_fraction": ANY,
    "reference.phase_edges_days": ANY,
    "reference.token_dim": ANY,
    "reference.token_pool_size": ANY,
    "fusion.spectrum_weight": ANY,
    "fusion.brightness_weight": ANY,
    "match_strength.high": ANY,
    "match_strength.medium": ANY,
    "match_strength.second_redshift": ANY,
}


@dataclass
class Package:
    """A checked model package.

    Parameters
    ----------
    path : pathlib.Path
        Where it was read from.
    info, config, preprocessing, calibration, priors : dict
        The contents of ``package.json``, ``model_config.json``, ``preprocessing.json``, ``calibration.json`` and
        ``priors.json``.
    card : str
        The model card.
    redshift_grid, wavelength_grid : numpy.ndarray
        The candidate redshifts and the observer-frame wavelength grid in Å.
    weights, spectral_model : dict
        The contents of ``weights.pt`` and ``spectral_model.pt``.
    brightness : dict of str to numpy.ndarray
        The contents of ``brightness.npz``: the brightness term's tables (`strider.brightness.BrightnessTerm`), its
        ``classes``, ``band_edges`` and the noise profile, ``profile_wavelength`` and ``profile_sigma``.
    """

    path: Path
    info: dict
    config: dict
    preprocessing: dict
    calibration: dict
    priors: dict
    card: str
    redshift_grid: np.ndarray
    wavelength_grid: np.ndarray
    weights: dict
    spectral_model: dict
    brightness: dict
    files_sha256: str = ""


def fingerprint(config, digests):
    """The model's identity: the sha256 of the canonical JSON of its declared config and of the sha256 of each of
    `FINGERPRINT_FILES`, given by name in ``digests``."""
    identity = {"model_config": config} | {name: digests[name] for name in FINGERPRINT_FILES}
    text = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_package(path):
    """Read and check a model package.

    Parameters
    ----------
    path : str or path-like
        A package directory or its ``.tar.gz``.

    Returns
    -------
    Package

    Raises
    ------
    FileNotFoundError
        If there is no package at ``path``.
    ValueError
        If any check fails, saying which file is wrong and what to do.
    """
    path = Path(path)
    files = _read_files(path)
    try:
        return _check(path, files)
    except KeyError as error:
        raise ValueError(f"model package {path}: the entry {error} is missing; {DAMAGED}") from None
    except ValueError as error:
        raise ValueError(f"model package {path}: {error}") from None


def read_card(path):
    """The model card of the package at ``path``, a directory or its ``.tar.gz``, unchecked."""
    files = _read_files(Path(path))
    if "MODEL_CARD.md" not in files:
        raise ValueError(f"model package {path} has no model card; {DAMAGED}")
    return files["MODEL_CARD.md"].decode("utf-8")


def _check(path, files):
    import torch

    digests = _check_files(files)
    info, config = json.loads(files["package.json"]), json.loads(files["model_config.json"])
    if (info.get("format"), info.get("format_version")) != (FORMAT, FORMAT_VERSION):
        raise ValueError(
            f"the package has format {info.get('format')!r} {info.get('format_version')!r}, and this "
            f"STRIDER reads {FORMAT!r} {FORMAT_VERSION}. " + UPGRADE
        )
    if fingerprint(config, digests) != info["fingerprint"]:
        raise ValueError("its files do not match its fingerprint; " + DAMAGED)
    _check_implemented(config)
    weights = list(config["fusion"].values())
    if not all(_number(weight) and weight > 0.0 for weight in weights):
        raise ValueError(f"model_config.json has the fusion weights {weights}, which must be positive; " + DAMAGED)
    levels = config["match_strength"]
    if not (all(_number(level) and level > 0 for level in levels.values()) and levels["high"] > levels["medium"]):
        raise ValueError(
            f"model_config.json has the match-strength levels {levels}, which must be positive, high above medium; "
            + DAMAGED
        )
    preprocessing = json.loads(files["preprocessing.json"])
    _check_preprocessing(preprocessing, config)

    grids = {key: np.load(io.BytesIO(files[f"{key}.npy"])) for key in ("redshift_grid", "wavelength_grid")}
    grid = preprocessing["wavelength_grid"]
    described = np.geomspace(grid["minimum"], grid["maximum"], grid["bins"])
    packaged = grids["wavelength_grid"]
    if packaged.shape != described.shape or not np.allclose(packaged, described, rtol=1e-12, atol=0.0):
        raise ValueError("wavelength_grid.npy does not match preprocessing.json; " + DAMAGED)
    if not np.array_equal(grids["redshift_grid"], redshift_grid(config["model"])):
        raise ValueError("redshift_grid.npy does not match model_config.json; " + DAMAGED)
    if info["redshift_grid"]["prior_measure"] != config["model"]["redshift_prior"]:
        raise ValueError("package.json and model_config.json give different redshift prior measures; " + DAMAGED)
    classes = config["model"]["classes"]
    if info["classes"] != classes:
        raise ValueError(f"package.json has the classes {info['classes']} and model_config.json {classes}; " + DAMAGED)

    calibration = json.loads(files["calibration.json"])
    _check_calibration(calibration, info, classes)
    weights, spectral_model = (
        torch.load(io.BytesIO(files[key]), map_location="cpu", weights_only=True)
        for key in ("weights.pt", "spectral_model.pt")
    )
    prefix = SPECTRAL_MODEL_PREFIX
    trained = {key.removeprefix(prefix): value for key, value in weights.items() if key.startswith(prefix)}
    packaged = spectral_model["state_dict"]
    if trained.keys() != packaged.keys() or not all(torch.equal(trained[key], packaged[key]) for key in trained):
        raise ValueError("weights.pt and spectral_model.pt hold different spectral models; " + DAMAGED)
    _check_phase_bins(weights, spectral_model, config["reference"])
    _check_masks(weights, spectral_model["coverage_counts"].numpy(), config["reference"]["minimum_profile_support"])
    brightness = dict(np.load(io.BytesIO(files["brightness.npz"]), allow_pickle=False))
    _check_brightness(brightness, classes, grids["redshift_grid"])
    priors = json.loads(files["priors.json"])
    if "fine_to_output" not in priors:
        raise ValueError("priors.json has no subtypes, which this STRIDER needs. " + UPGRADE)
    return Package(
        path=path,
        info=info,
        config=config,
        preprocessing=preprocessing,
        calibration=calibration,
        priors=priors,
        card=files["MODEL_CARD.md"].decode("utf-8"),
        weights=weights,
        spectral_model=spectral_model,
        brightness=brightness,
        files_sha256=hashlib.sha256(files["SHA256SUMS"]).hexdigest(),
        **grids,
    )


def _read_files(path):
    """The package's files by name, from a directory or a ``.tar.gz``, leaving out hidden files (``.DS_Store``)."""
    if path.is_dir():
        items = (item for item in path.iterdir() if item.is_file())
        return {item.name: item.read_bytes() for item in items if not item.name.startswith(".")}
    if not path.is_file():
        raise FileNotFoundError(f"no model package at {path}")
    with tarfile.open(path, "r:gz") as archive:
        members = [member for member in archive.getmembers() if not Path(member.name).name.startswith(".")]
        parts = [Path(member.name).parts for member in members]
        if len({part[0] for part in parts}) != 1 or any(len(part) > 2 for part in parts):
            raise ValueError(f"model package {path}: the archive should hold a single directory; {DAMAGED}")
        return {Path(m.name).name: archive.extractfile(m).read() for m in members if m.isfile()}


def _check_files(files):
    """Check every file against SHA256SUMS, which must list each of them; return the listed sha256 by name."""
    missing = [name for name in REQUIRED if name not in files]
    if missing:
        raise ValueError(f"the package lacks {', '.join(missing)}; {DAMAGED}")
    listed = {}
    for line in files["SHA256SUMS"].decode("utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            listed[name] = digest
    unlisted = sorted(set(files) - set(listed) - {"SHA256SUMS"})
    if unlisted:
        raise ValueError(f"SHA256SUMS does not list {', '.join(unlisted)}; {DAMAGED}")
    for name, digest in listed.items():
        if name not in files or hashlib.sha256(files[name]).hexdigest() != digest:
            raise ValueError(f"{name} is missing or does not match its checksum; {DAMAGED}")
    return listed


def _check_implemented(config):
    declared = {f"{section}.{name}": value for section, values in config.items() for name, value in values.items()}
    unknown = sorted(declared.keys() - IMPLEMENTED.keys())
    if unknown:
        raise ValueError(
            f"model_config.json uses {', '.join(unknown)}, which this STRIDER does not implement. " + UPGRADE
        )
    for key, allowed in IMPLEMENTED.items():
        if key not in declared:
            raise ValueError(f"model_config.json lacks {key}; " + DAMAGED)
        if allowed is not ANY and declared[key] not in allowed:
            raise ValueError(
                f"model_config.json has {key} = {declared[key]!r}; this STRIDER implements "
                f"{' or '.join(map(repr, allowed))}. " + UPGRADE
            )


def _check_preprocessing(preprocessing, config):
    grid, observation = preprocessing["wavelength_grid"], config["observation"]
    if (grid["minimum"], grid["maximum"], grid["bins"], grid["spacing"]) != (
        observation["wavelength_min"],
        observation["wavelength_max"],
        observation["wavelength_bins"],
        "log",
    ):
        raise ValueError("preprocessing.json and model_config.json give different wavelength grids; " + DAMAGED)
    if preprocessing["wavelength_frame"] != "observer" or preprocessing["wavelength_unit"] != "angstrom":
        raise ValueError("preprocessing.json expects wavelengths other than observer-frame Å. " + UPGRADE)
    if preprocessing["wavelength_support"] not in ("complete", "visits") or not preprocessing["flux_error_input"]:
        raise ValueError(
            "preprocessing.json does not require full wavelength coverage and flux errors, which this "
            "STRIDER needs. " + UPGRADE
        )


def phase_bin_weights(nodes, edges, coverage, minimum):
    """(classes, phase bins, nodes) float32: the weights on the spectral model's nodes that average it over each
    phase bin, interpolated linearly between the nodes (trapezoids on the nodes are exact), using only the nodes
    with ``minimum`` training spectra at some pixel of ``coverage`` (classes, nodes, pixels), renormalised."""
    nodes, edges = np.asarray(nodes, np.float64), np.asarray(edges, np.float32).astype(np.float64)
    weights = np.zeros((len(edges) - 1, len(nodes)))
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        points = np.unique(np.concatenate([[low, high], nodes[(nodes > low) & (nodes < high)]]))
        hats = np.stack([np.interp(points, nodes, row) for row in np.eye(len(nodes))], axis=-1)
        weights[index] = (np.diff(points)[:, None] * (hats[1:] + hats[:-1])).sum(0) / (2.0 * (high - low))
    weights = weights[None] * (coverage.max(-1) >= minimum)[:, None, :]
    total = weights.sum(-1, keepdims=True)
    return np.divide(weights, total, out=np.zeros_like(weights), where=total > 0.0).astype(np.float32)


def _check_phase_bins(weights, spectral_model, settings):
    """The phase nodes and phase-bin weights in weights.pt must be those built from spectral_model.pt."""
    nodes = spectral_model["phase_nodes_days"].numpy().astype(np.float32)
    built = phase_bin_weights(
        nodes,
        settings["phase_edges_days"],
        spectral_model["coverage_counts"].numpy(),
        settings["minimum_profile_support"],
    )
    packaged = [weights[f"roman_reference.{name}"].numpy() for name in ("phase_nodes_days", "window_weight")]
    if not all(
        found.dtype == np.float32 and np.array_equal(found, expected)
        for found, expected in zip(packaged, (nodes, built))
    ):
        raise ValueError(
            "weights.pt's phase nodes and phase-bin weights are not those built from spectral_model.pt "
            "and model_config.json; " + DAMAGED
        )


def _check_masks(weights, coverage, minimum):
    """The masks of the reference spectra in weights.pt are authoritative, and must agree with the training coverage
    (class, node, rest pixel): a reference spectrum averages the nodes where its phase-bin weight is positive, and its
    mask holds the pixels where each of them has ``minimum`` training spectra; it is supported where its mask holds
    any pixel."""
    used = weights["roman_reference.window_weight"].numpy() > 0.0
    enough = coverage >= minimum
    expected = used.any(-1)[..., None] & np.all(~used[..., None] | enough[:, None], axis=2)
    mask = weights["roman_reference.coadd_profile_mask"].numpy()
    supported = weights["roman_reference.coadd_profile_supported"].numpy()
    if not (np.array_equal(mask, expected) and np.array_equal(supported, mask.any(-1))):
        raise ValueError(
            "the reference-spectrum masks in weights.pt do not agree with the training coverage in "
            "spectral_model.pt; " + DAMAGED
        )


def _check_brightness(brightness, classes, grid):
    """The brightness term's tables: finite float64, for the model's classes and the bands `strider.bands` makes, of
    consistent shapes and within their bounds, with template redshifts that cover the redshift grid."""
    from .brightness import STEP

    def refuse(problem):
        raise ValueError(f"brightness.npz {problem}; " + DAMAGED)

    names = [str(name) for name in brightness["classes"]]
    if names != classes:
        refuse(f"has the classes {names}, not the model's")
    if not np.array_equal(brightness["band_edges"], EDGES):
        raise ValueError(f"brightness.npz has the band edges {brightness['band_edges']}, not {EDGES}. " + UPGRADE)
    tables = {name: brightness[name] for name in (*BRIGHTNESS_TABLES, "profile_wavelength", "profile_sigma")}
    if not all(table.dtype == np.float64 and np.isfinite(table).all() for table in tables.values()):
        refuse("holds tables that are not finite float64")
    nodes, window, error = tables["z_nodes"], tables["window"], tables["discrepancy"]
    if nodes.ndim != 1 or len(nodes) < 2 or np.any(np.diff(nodes) <= 0):
        refuse("has template redshifts that do not increase")
    if nodes[0] > np.log1p(float(grid[0])) or nodes[-1] < np.log1p(float(grid[-1])):
        refuse("has template redshifts that do not cover the redshift grid")
    phases = (window[1] - window[0]) / STEP + 1 if window.shape == (2,) else 0.0
    if phases < 2 or phases != round(phases):
        refuse(f"has the phase window {window}, which must increase in steps of {STEP:g} day")
    shape = (len(classes), len(EDGES) - 1, len(nodes), round(phases))
    if not tables["mean"].shape == tables["psi2"].shape == tables["loadings"].shape[:-1] == shape:
        refuse(f"has mean, psi2 and loadings not of the shape {shape} (and modes)")
    if np.any(tables["psi2"] < 0) or tables["tau"].shape != (2,) or np.any(tables["tau"] <= 0):
        refuse("has a negative psi2, or tau not two positive scales")
    if error.shape != (len(classes), 5):
        refuse(f"has a discrepancy of shape {error.shape}, not {(len(classes), 5)}")
    kappa, ell, rho, kappa_wide, wide = error.T
    if np.any((kappa <= 0) | (ell <= 0) | (kappa_wide <= 0) | (rho < 0) | (rho > 1) | (wide < 0) | (wide > 1)):
        refuse("has a discrepancy outside its bounds: kappa, ell and kappa_wide above 0, rho and wide from 0 to 1")
    wavelength, sigma = tables["profile_wavelength"], tables["profile_sigma"]
    if (
        wavelength.ndim != 1
        or wavelength.shape != sigma.shape
        or np.any(np.diff(wavelength) <= 0)
        or np.any(sigma <= 0)
    ):
        refuse("has a noise profile without increasing wavelengths and positive errors")


def _check_calibration(calibration, info, classes):
    """The source probability's calibration: fitted for this model, or not fitted."""
    if calibration["status"] not in ("fitted", "not_fitted"):
        raise ValueError(f"calibration.json has status {calibration['status']!r}. " + UPGRADE)
    if calibration["status"] != info["calibration_status"]:
        raise ValueError("package.json and calibration.json give different calibration statuses; " + DAMAGED)
    if calibration["status"] == "not_fitted":
        return
    if calibration["fingerprint"] != info["fingerprint"] or calibration["classes"] != classes:
        raise ValueError("calibration.json was fitted for another model; " + DAMAGED)
    fit = calibration["source_probability"]
    if fit["method"] != SOURCE_METHOD:
        raise ValueError(
            f"calibration.json has a {fit['method']!r} source probability; this STRIDER reads "
            f"{SOURCE_METHOD!r}. " + UPGRADE
        )
    if not (
        all(_number(fit[name]) for name in ("slope", "intercept", "weak_source_below"))
        and fit["slope"] > 0
        and 0 <= fit["weak_source_below"] <= 1
    ):
        raise ValueError(
            "calibration.json's source probability needs a positive slope, a finite intercept and a cut-off from 0 to "
            "1; " + DAMAGED
        )


def _number(value):
    """Whether ``value`` is a finite number, not a boolean."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and np.isfinite(value)
