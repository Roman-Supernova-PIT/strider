import numpy as np

from strider.bands import EDGES, band_fluxes, full_weights

WAVELENGTH = np.linspace(7400.0, 18300.0, 400)
PROFILE = (WAVELENGTH, np.full(len(WAVELENGTH), 0.1))
FULL = full_weights(PROFILE)


def test_a_flat_spectrum_gives_its_flux_density_in_every_band():
    value, sigma = band_fluxes(WAVELENGTH, np.full(400, 2.0), np.full(400, 0.1), PROFILE, FULL)
    np.testing.assert_allclose(value, 2.0)
    assert np.all(sigma > 0)


def test_a_gap_gives_the_same_bands_as_masking_its_pixels():
    rng = np.random.default_rng(1)
    flux, error = rng.normal(1.0, 0.1, 400), np.full(400, 0.1)
    gap = (WAVELENGTH > 10000.0) & (WAVELENGTH < 10100.0)
    masked = band_fluxes(WAVELENGTH, flux, np.where(gap, np.nan, error), PROFILE, FULL)
    removed = band_fluxes(WAVELENGTH[~gap], flux[~gap], error[~gap], PROFILE, FULL)
    np.testing.assert_allclose(removed, masked)


def test_a_band_with_under_90_percent_of_its_weight_is_left_out():
    error = np.full(400, 0.1)
    error[(WAVELENGTH >= EDGES[0]) & (WAVELENGTH < 9000.0)] = np.nan
    value, sigma = band_fluxes(WAVELENGTH, np.ones(400), error, PROFILE, FULL)
    assert np.isnan(value[0]) and np.isnan(sigma[0])
    assert np.all(np.isfinite(value[1:]))
