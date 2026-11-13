"""Reading the synthetic package, and refusing it when any file is wrong."""

import io

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from strider.package import _read_files, read_package  # noqa: E402
from strider.tests import synthetic_package  # noqa: E402
from strider.tests.synthetic_package import ARCHIVE, CLASSES, write_directory as write  # noqa: E402


@pytest.fixture(scope="module")
def files():
    return synthetic_package.package_files()


def test_the_archive_and_its_directory_are_read(files, tmp_path):
    archived = read_package(ARCHIVE)
    unpacked = read_package(write(tmp_path / "package", _read_files(ARCHIVE)))  # its own files, on any machine
    assert archived.info["classes"] == unpacked.config["model"]["classes"] == CLASSES
    assert archived.info["fingerprint"] == unpacked.info["fingerprint"]
    assert unpacked.brightness["mean"].shape[:2] == (len(CLASSES), 4)
    assert unpacked.config["fusion"] == {"spectrum_weight": 0.9, "brightness_weight": 0.5}


def section(name, key, value):
    return lambda content: content | {name: content[name] | {key: value}}


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"model_config.json": section("data", "class_scheme", "normal_ia_binary")}, "implements 'hourglass_15'"),
        ({"model_config.json": section("reference", "sequence_visits", 8)}, "does not implement"),
        ({"package.json": lambda info: info | {"format_version": 3}}, "this STRIDER reads 'strider-model-package' 1"),
        ({"model_config.json": section("fusion", "brightness_weight", 0.0)}, "fusion weights"),
        ({"model_config.json": section("fusion", "brightness_weight", True)}, "fusion weights"),
        ({"model_config.json": section("match_strength", "low", 1)}, "does not implement"),
        ({"model_config.json": section("match_strength", "medium", 6)}, "match-strength levels"),
        ({"calibration.json": lambda calibration: calibration | {"classes": CLASSES[::-1]}}, "another model"),
        ({"calibration.json": section("source_probability", "method", "isotonic")}, "reads 'balanced-affine-logit'"),
        ({"calibration.json": section("source_probability", "slope", -1.0)}, "positive slope"),
        ({"calibration.json": section("source_probability", "weak_source_below", 2.0)}, "cut-off from 0 to 1"),
        ({"priors.json": lambda priors: {k: v for k, v in priors.items() if k != "fine_to_output"}}, "no subtypes"),
    ],
)
def test_a_package_with_a_wrong_file_is_refused(files, tmp_path, changes, message):
    path = write(tmp_path / "package", files, **changes)
    with pytest.raises(ValueError, match=f"model package {path}: .*{message}"):
        read_package(path)


def test_whole_fusion_weights_are_read(files, tmp_path):
    path = write(tmp_path / "package", files, **{"model_config.json": section("fusion", "spectrum_weight", 1)})
    assert read_package(path).config["fusion"]["spectrum_weight"] == 1


def test_a_damaged_package_is_refused(files, tmp_path):
    path = write(tmp_path / "package", files)
    (path / "MODEL_CARD.md").write_text("changed")
    with pytest.raises(ValueError, match="MODEL_CARD.md is missing or does not match its checksum"):
        read_package(path)
    (path / "weights.pt").unlink()
    with pytest.raises(ValueError, match="the package lacks weights.pt"):
        read_package(path)
    with pytest.raises(FileNotFoundError, match="no model package at"):
        read_package(tmp_path / "nothing.tar.gz")


@pytest.mark.parametrize(
    "change, message",
    [
        ({"band_edges": np.array([7500.0, 9000.0, 11000.0, 14000.0, 18000.0])}, "band edges"),
        ({"classes": np.array(CLASSES[::-1])}, "not the model's"),
        ({"tau": np.array([2, 58])}, "not finite float64"),
        ({"tau": np.array([np.nan, 58.0])}, "not finite float64"),
        ({"z_nodes": np.log1p([0.0, 2.0])}, "do not cover the redshift grid"),
        ({"z_nodes": np.log1p([3.0, 0.0])}, "do not increase"),
        ({"window": np.array([-10.0, 30.5])}, "phase window"),
        ({"mean": np.ones((15, 4, 2, 40))}, "not of the shape"),
        ({"psi2": np.full((15, 4, 2, 41), -1.0)}, "negative psi2"),
        ({"discrepancy": np.tile([0.71, 32.0, 1.5, 2.83, 0.2], (15, 1))}, "outside its bounds"),
        ({"profile_sigma": np.zeros(60)}, "noise profile"),
    ],
)
def test_brightness_tables_out_of_their_bounds_are_refused(files, tmp_path, change, message):
    files = files | {"brightness.npz": synthetic_package._npz_bytes(synthetic_package.toy_brightness() | change)}
    with pytest.raises(ValueError, match=message):
        read_package(write(tmp_path / "package", files))


@pytest.mark.parametrize("name", ["phase_nodes_days", "window_weight"])
def test_phase_nodes_and_bin_weights_must_be_built_from_the_spectral_model(files, tmp_path, name):
    weights = torch.load(io.BytesIO(files["weights.pt"]), weights_only=True)
    weights[f"roman_reference.{name}"] = weights[f"roman_reference.{name}"] * 1.01
    files = files | {"weights.pt": synthetic_package._torch_bytes(weights)}
    with pytest.raises(ValueError, match="weights.pt's phase nodes and phase-bin weights are not those built"):
        read_package(write(tmp_path / "package", files))


def test_the_packaged_wavelength_grid_may_differ_from_its_description_by_rounding_only(files, tmp_path):
    grid = np.load(io.BytesIO(files["wavelength_grid.npy"]))
    nudged = files | {"wavelength_grid.npy": synthetic_package._npy_bytes(grid * (1.0 + 1e-15))}
    assert np.array_equal(read_package(write(tmp_path / "close", nudged)).wavelength_grid, grid * (1.0 + 1e-15))
    shifted = files | {"wavelength_grid.npy": synthetic_package._npy_bytes(grid * (1.0 + 1e-9))}
    with pytest.raises(ValueError, match="wavelength_grid.npy does not match preprocessing.json"):
        read_package(write(tmp_path / "far", shifted))
