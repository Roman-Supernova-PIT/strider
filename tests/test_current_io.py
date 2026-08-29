from pathlib import Path

import numpy as np
import pytest

from strider.current_io import load_observed_inputs


def test_current_csv_uses_observer_time_and_reported_error(tmp_path: Path) -> None:
    path = tmp_path / "series.csv"
    path.write_text(
        "object_id,epoch,mjd,wavelength,flux,flux_error\n"
        "roman-1,1,62000,7500,1.0,0.5\n"
        "roman-1,1,62000,8000,2.0,0.5\n"
        "roman-1,2,62012,7500,1.5,0.6\n"
        "roman-1,2,62012,8000,2.5,0.6\n"
    )

    data = load_observed_inputs([path])

    assert data.metadata["object"] == "roman-1"
    assert data.observer_time.tolist() == [62000.0, 62012.0]
    assert data.flux.shape == (2, 2)
    np.testing.assert_allclose(data.flux_error, [[0.5, 0.5], [0.6, 0.6]])


def test_current_npz_accepts_explicit_time_for_one_spectrum(tmp_path: Path) -> None:
    path = tmp_path / "spectrum.npz"
    np.savez(
        path,
        wavelength=np.asarray([7500.0, 8000.0]),
        flux=np.asarray([1.0, 2.0]),
        flux_error=np.asarray([0.5, 0.5]),
    )

    data = load_observed_inputs([path], times=[62000.0])

    assert data.observer_time.tolist() == [62000.0]
    assert data.flux.shape == (2,)


def test_current_input_rejects_phase_without_observer_time(tmp_path: Path) -> None:
    path = tmp_path / "legacy.csv"
    path.write_text(
        "wavelength,flux,flux_error,phase\n"
        "7500,1.0,0.5,-5\n"
        "8000,2.0,0.5,-5\n"
    )

    with pytest.raises(ValueError, match="observer_time or mjd"):
        load_observed_inputs([path])


def test_current_input_converts_microns_to_angstrom(tmp_path: Path) -> None:
    path = tmp_path / "spectrum.csv"
    path.write_text(
        "mjd,wavelength,flux,flux_error\n"
        "62000,0.75,1.0,0.5\n"
        "62000,1.80,2.0,0.5\n"
    )

    data = load_observed_inputs([path], wavelength_unit="micron")

    np.testing.assert_allclose(data.wavelength, [7500.0, 18000.0])


def test_current_input_rejects_different_explicit_objects(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text(
        "object_id,mjd,wavelength,flux,flux_error\n"
        "roman-1,62000,7500,1.0,0.5\n"
        "roman-1,62000,8000,2.0,0.5\n"
    )
    second.write_text(
        "object_id,mjd,wavelength,flux,flux_error\n"
        "roman-2,62012,7500,1.5,0.6\n"
        "roman-2,62012,8000,2.5,0.6\n"
    )

    with pytest.raises(ValueError, match="different objects"):
        load_observed_inputs([first, second])
