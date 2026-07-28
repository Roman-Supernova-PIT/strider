from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from strider import load_model
from strider.io import load_inputs
from strider.model_info import verify_model
from strider.model import _quantile


@pytest.mark.model
def test_production_model():
    model_path = Path("models/strider.pt")
    verify_model(model_path)
    paths = sorted(Path("examples/SN20088677_ou").glob("spectrum_*.csv"))
    data = load_inputs(paths)
    model = load_model(model_path)
    output = model.classify(
        data.wavelength,
        data.flux,
        data.phase,
    )
    assert output["strider_class"] == "Ia"
    assert float(output["p_Ia"]) > 0.98
    assert np.isfinite(output["z_STRIDER"])
    assert len(output["class_names"]) == 15
    assert model.metadata.class_calibration["method"] == "dirichlet"
    assert model.metadata.redshift_calibration["method"] == "pit_recalibration"
    assert model.metadata.family == "strider"
    assert output["strider_class_index"] == int(
        np.argmax(output["p_class_uncal_with_controls"])
    )
    assert np.isclose(np.sum(output["p_class_cal_with_controls"]), 1.0)
    assert output["redshift"]["interval_calibration"]["applied"] is True
    assert output["spectra"]["quality"]["model_wavelength_coverage_fraction"] > 0.99
    assert output["spectra"]["quality"]["flux_error_supplied"] is False
    assert len(output["spectra"]["quality"]["valid_flux_fraction_per_epoch"]) == 5

    raw = np.asarray(output["p_class_uncal_with_controls"], dtype=float)
    class_calibration = model.metadata.class_calibration
    log_raw = np.log(np.clip(raw, 1e-300, None))
    log_raw -= np.logaddexp.reduce(log_raw)
    logits = (
        np.asarray(class_calibration["weights"], dtype=float)
        @ log_raw
        + np.asarray(class_calibration["bias"], dtype=float)
    )
    expected = np.exp(logits - logits.max())
    expected /= expected.sum()
    np.testing.assert_allclose(output["p_class_cal_with_controls"], expected)

    redshift_calibration = model.metadata.redshift_calibration
    population = output["redshift"]["interval_calibration"]["population"]
    pit_key = "pit_gold_sorted" if population == "gold_Ia" else "pit_full_sorted"
    raw_level = np.quantile(redshift_calibration[pit_key], 0.16)
    expected_p16 = _quantile(
        np.asarray(model.metadata.z_grid),
        np.asarray(output["z_marginal_uncal_with_controls"]),
        raw_level,
    )
    assert output["z_p16"] == pytest.approx(expected_p16)


@pytest.mark.model
def test_redshift_prior_disables_interval_recalibration():
    model = load_model(Path("models/strider.pt"))
    paths = sorted(Path("examples/SN20088677_ou").glob("spectrum_*.csv"))
    data = load_inputs(paths)
    output = model.classify(
        data.wavelength,
        data.flux,
        data.phase,
        z_prior=(0.12, 0.05),
    )
    calibration = output["redshift"]["interval_calibration"]
    assert calibration["applied"] is False
    assert "not validated" in calibration["reason"]
    assert output["redshift"]["hpd_conformal"]["applied"] is False
    assert output["redshift"]["spectra_only"]["z_S"] != pytest.approx(
        output["redshift"]["with_controls"]["z_S"]
    )


@pytest.mark.model
def test_hourglass2_examples():
    model = load_model(Path("models/strider.pt"))
    examples = (
        ("hourglass2_Ia_2791003", "Ia", 0.312832),
        ("hourglass2_IIP_452808", "IIP", 0.808779),
        ("hourglass2_Ic_1209532", "Ic", 0.868175),
    )

    for filename, expected_class, true_redshift in examples:
        paths = sorted((Path("examples") / filename).glob("spectrum_*.csv"))
        data = load_inputs(paths)
        assert np.all(np.diff(data.phase) > 0)
        output = model.classify(data.wavelength, data.flux, data.phase)
        assert output["strider_class"] == expected_class
        assert abs(output["z_STRIDER"] - true_redshift) / (1 + true_redshift) < 0.01


def test_example_folder_is_chronological():
    paths = sorted(Path("examples/SN20088677_ou").glob("spectrum_*.csv"))
    data = load_inputs(paths)
    assert data.phase.tolist() == [-15.0, -7.0, 0.0, 7.0, 15.0]
