from __future__ import annotations

import numpy as np
import pytest

from strider.io import load_input, load_inputs, load_npz, load_table


def test_single_spectrum_csv_uses_explicit_phase(tmp_path):
    path = tmp_path / "single.csv"
    path.write_text("wavelength,flux\n7500,1\n7600,2\n")
    data = load_table(path, phase=3.0)
    assert data.flux.shape == (1, 2)
    assert data.phase.tolist() == [3.0]


def test_time_series_csv_groups_phases(tmp_path):
    path = tmp_path / "series.csv"
    path.write_text(
        "wavelength,flux,phase,flux_err,object_id\n"
        "7600,2,-7,0.2,SN1\n"
        "7500,1,-7,0.1,SN1\n"
        "7600,4,0,0.4,SN1\n"
        "7500,3,0,0.3,SN1\n"
    )
    data = load_table(path)
    assert data.metadata["object"] == "SN1"
    assert data.phase.tolist() == [-7.0, 0.0]
    np.testing.assert_allclose(data.wavelength, [7500, 7600])
    np.testing.assert_allclose(data.flux, [[1, 2], [3, 4]])
    np.testing.assert_allclose(data.flux_err, [[0.1, 0.2], [0.3, 0.4]])


def test_time_series_csv_rejects_different_grids(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        "wavelength,flux,phase\n"
        "7500,1,-7\n7600,2,-7\n"
        "7500,3,0\n7700,4,0\n"
    )
    with pytest.raises(ValueError, match="different wavelength grid"):
        load_table(path)


def test_headerless_text_accepts_optional_flux_error(tmp_path):
    path = tmp_path / "spectrum.txt"
    path.write_text("7500 1.0 0.1\n7600 2.0 0.2\n")
    data = load_input(path, phase=-4.0)
    assert data.phase.tolist() == [-4.0]
    np.testing.assert_allclose(data.flux, [[1.0, 2.0]])
    np.testing.assert_allclose(data.flux_err, [[0.1, 0.2]])


def test_headered_text_can_hold_a_time_series(tmp_path):
    path = tmp_path / "series.dat"
    path.write_text(
        "wavelength flux fluxerr phase\n"
        "7500 1.0 0.1 -7\n7600 2.0 0.2 -7\n"
        "7500 3.0 0.3 0\n7600 4.0 0.4 0\n"
    )
    data = load_input(path)
    assert data.phase.tolist() == [-7.0, 0.0]
    np.testing.assert_allclose(data.flux, [[1.0, 2.0], [3.0, 4.0]])


def test_commented_header_is_accepted(tmp_path):
    path = tmp_path / "spectrum.spec"
    path.write_text("# wavelength flux\n7500 1.0\n7600 2.0\n")
    data = load_input(path, phase=1.5)
    assert data.phase.tolist() == [1.5]
    assert data.flux_err is None


def test_single_spectrum_npz_can_take_phase_from_cli(tmp_path):
    path = tmp_path / "spectrum.npz"
    np.savez(path, wavelength=[7500, 7600], flux=[1.0, 2.0])
    data = load_npz(path, phase=2.0)
    assert data.phase.tolist() == [2.0]


def test_phase_is_never_silently_overridden(tmp_path):
    path = tmp_path / "spectrum.tsv"
    path.write_text("wavelength\tflux\tphase\n7500\t1\t0\n7600\t2\t0\n")
    with pytest.raises(ValueError, match="already contains a phase"):
        load_input(path, phase=3.0)


def test_separate_epoch_files_use_explicit_phases(tmp_path):
    first = tmp_path / "epoch1.txt"
    second = tmp_path / "epoch2.txt"
    first.write_text("7500 1.0 0.1\n7600 2.0 0.2\n")
    second.write_text("7500 3.0 0.3\n7600 4.0 0.4\n")
    data = load_inputs([first, second], phases=[-7.0, 0.0])
    assert data.phase.tolist() == [-7.0, 0.0]
    np.testing.assert_allclose(data.flux, [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(data.flux_err, [[0.1, 0.2], [0.3, 0.4]])


def test_separate_epoch_files_need_matching_phase_count(tmp_path):
    paths = [tmp_path / "epoch1.txt", tmp_path / "epoch2.txt"]
    with pytest.raises(ValueError, match="one value per input"):
        load_inputs(paths, phases=[0.0])


def test_fits_table_input(tmp_path):
    fits = pytest.importorskip("astropy.io.fits")
    path = tmp_path / "spectrum.fits"
    columns = [
        fits.Column(name="WAVELENGTH", format="E", array=[7600.0, 7500.0]),
        fits.Column(name="FLUX", format="E", array=[2.0, 1.0]),
        fits.Column(name="FLUX_ERR", format="E", array=[0.2, 0.1]),
    ]
    fits.BinTableHDU.from_columns(columns).writeto(path)
    data = load_input(path, phase=3.0)
    assert data.phase.tolist() == [3.0]
    np.testing.assert_allclose(data.wavelength, [7500.0, 7600.0])
    np.testing.assert_allclose(data.flux, [[1.0, 2.0]])
    np.testing.assert_allclose(data.flux_err, [[0.1, 0.2]])
