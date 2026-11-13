import numpy as np
import pytest

from strider.preprocess import measured_error_scale, prepare, wavelength_grid

SETTINGS = {"wavelength_grid": {"minimum": 7500.0, "maximum": 18175.0, "bins": 64, "spacing": "log"}}


def visit(low=7400.0, high=18300.0, pixels=120, seed=0):
    rng = np.random.default_rng(seed)
    wavelength = np.geomspace(low, high, pixels)  # log-uniform, as the prism's pixels widen to the red
    return wavelength, 1.0 + 0.1 * rng.normal(size=pixels), np.full(pixels, 0.1) + 0.01 * rng.random(pixels)


def test_errors_are_required():
    wavelength, flux, _ = visit()
    with pytest.raises(ValueError, match="errors are required"):
        prepare([wavelength], [flux], None, [0.0], settings=SETTINGS)


def test_a_visit_on_the_model_grid_keeps_its_values():
    grid = wavelength_grid(SETTINGS)
    flux, error = np.linspace(1.0, 2.0, len(grid)), np.full(len(grid), 0.5)
    prepared = prepare([grid], [flux], [error], [61000.0], settings=SETTINGS)
    np.testing.assert_allclose(prepared.flux[0], flux / 0.5, rtol=1e-6)
    np.testing.assert_allclose(prepared.log_error[0], 0.0, atol=1e-5)  # the wavelengths are float32
    assert prepared.measured.all() and prepared.scale[0] == 0.5 and prepared.days[0] == 0.0


def test_a_visit_missing_much_of_the_range_is_dropped_and_reported():
    full, short = visit(), visit(low=8200.0)
    prepared = prepare(*zip(full, short), [0.0, 3.0], settings=SETTINGS)
    assert list(prepared.visits) == [0]
    assert prepared.dropped[1] == (
        "covers 8172-18362 Å, and STRIDER needs a visit to reach from 7634 Å or below to 17856 Å or above"
    )
    with pytest.raises(ValueError, match="no usable visit"):
        prepare(*zip(short), [0.0], settings=SETTINGS)
    wavelength, flux, error = full
    with pytest.raises(ValueError, match="17856 Å or above; its wavelengths look like µm"):
        prepare([wavelength / 1e4], [flux], [error], [0.0], settings=SETTINGS)


def test_a_visit_a_little_short_is_kept_with_the_grid_it_misses_unmeasured():
    """Up to COVERAGE_SHORTFALL of the grid's ln λ span may be missing at each end; the grid it misses is masked and
    the rest is as for the same pixels on a visit that covers the grid."""
    grid = wavelength_grid(SETTINGS)
    extent = np.log(grid[-1] / grid[0])
    full = visit(low=7000.0, high=19000.0, pixels=400)
    wavelength, flux, error = full
    for low, high in [(7531.0, 18150.0), (grid[0] * np.exp(0.019 * extent), grid[-1] * np.exp(-0.019 * extent))]:
        keep = (wavelength > low) & (wavelength < high)
        short = prepare([wavelength[keep]], [flux[keep]], [error[keep]], [0.0], settings=SETTINGS)
        whole = prepare([wavelength], [flux], [error], [0.0], settings=SETTINGS)
        inside = (grid > wavelength[keep][1]) & (grid < wavelength[keep][-2])
        assert not short.measured[0][grid < wavelength[keep][0]].any()
        assert not short.measured[0][grid > wavelength[keep][-1]].any()
        assert short.measured[0][inside].all()
        np.testing.assert_allclose(
            short.flux[0][inside] * short.scale[0], whole.flux[0][inside] * whole.scale[0], rtol=1e-6
        )
    keep = wavelength > grid[0] * np.exp(0.025 * extent)
    with pytest.raises(ValueError, match="needs a visit to reach from 7634 Å or below"):
        prepare([wavelength[keep]], [flux[keep]], [error[keep]], [0.0], settings=SETTINGS)


def test_a_short_visit_is_the_same_visit_with_the_pixels_it_lacks_masked():
    """At an end a visit falls short of, nothing beyond its last pixel is measured: as if those pixels were masked."""
    grid = wavelength_grid(SETTINGS)
    extent = np.log(grid[-1] / grid[0])
    wavelength, flux, error = visit(pixels=400)
    for keep in (wavelength < grid[-1] * np.exp(-0.019 * extent), wavelength > grid[0] * np.exp(0.019 * extent)):
        short = prepare([wavelength[keep]], [flux[keep]], [error[keep]], [0.0], settings=SETTINGS)
        masked = prepare([wavelength], [flux], [error], [0.0], [~keep], settings=SETTINGS)
        for name in ("flux", "log_error", "measured", "scale"):
            np.testing.assert_array_equal(getattr(short, name), getattr(masked, name))


def test_a_visit_padded_with_zeros_is_the_visit():
    """Fixed-length arrays padded with zero wavelength, flux and error: the padding is ignored."""
    wavelength, flux, error = visit()
    padded = prepare(
        [np.r_[np.zeros(5), wavelength]],
        [np.r_[np.zeros(5), flux]],
        [np.r_[np.zeros(5), error]],
        [0.0],
        settings=SETTINGS,
    )
    plain = prepare([wavelength], [flux], [error], [0.0], settings=SETTINGS)
    for name in ("flux", "log_error", "measured", "scale"):
        np.testing.assert_array_equal(getattr(padded, name), getattr(plain, name))


def test_pixels_flagged_bad_are_masked_and_the_visit_kept():
    """Every grid point whose interpolation touches a bad pixel is masked; the rest is unchanged."""
    wavelength, flux, _ = visit()
    error = np.full(len(flux), 0.1)
    bad_flux, bad_error = flux.copy(), error.copy()
    bad_flux[20], bad_error[30] = np.nan, 0.0
    mask = np.zeros(len(flux), bool)
    mask[[0, 60, 61, 62]] = True  # one below the range, and three in a row
    found = prepare([wavelength], [bad_flux], [bad_error], [0.0], [mask], settings=SETTINGS)
    clean = prepare([wavelength], [flux], [error], [0.0], settings=SETTINGS)
    grid = wavelength_grid(SETTINGS)
    upper = np.clip(np.searchsorted(wavelength.astype(np.float32), grid), 1, len(wavelength) - 1)
    touches = np.isin(upper - 1, [0, 20, 30, 60, 61, 62]) | np.isin(upper, [0, 20, 30, 60, 61, 62])
    assert not found.dropped and touches.any()
    np.testing.assert_array_equal(found.measured[0], ~touches)
    np.testing.assert_array_equal(found.flux[0][~touches], clean.flux[0][~touches])
    np.testing.assert_array_equal(found.flux[0][touches], 0.0)


def test_visits_are_put_in_time_order():
    first, second = visit(seed=1), visit(seed=2)
    prepared = prepare(*zip(second, first), [60010.0, 60000.0], settings=SETTINGS)
    assert list(prepared.visits) == [1, 0]
    np.testing.assert_array_equal(prepared.days, [0.0, 10.0])
    alone = prepare(*zip(first), [60000.0], settings=SETTINGS)
    np.testing.assert_array_equal(prepared.flux[0], alone.flux[0])


def test_visits_must_agree():
    wavelength, flux, error = visit()
    with pytest.raises(ValueError, match="same number of visits"):
        prepare([wavelength], [flux], [error], [0.0, 1.0], settings=SETTINGS)
    with pytest.raises(ValueError, match="must have the same length"):
        prepare([wavelength], [flux[:-1]], [error], [0.0], settings=SETTINGS)


def test_one_grid_for_every_visit_or_a_list_per_visit_and_nothing_else():
    wavelength, flux, error = visit()
    shared = prepare(wavelength, [flux, flux + 1.0], [error, error], [0.0, 1.0], settings=SETTINGS)
    separate = prepare([wavelength] * 2, [flux, flux + 1.0], [error, error], [0.0, 1.0], settings=SETTINGS)
    np.testing.assert_array_equal(shared.flux, separate.flux)
    with pytest.raises(ValueError, match="flux must be a list with one 1-d array per visit"):
        prepare(wavelength, flux, error, [0.0], settings=SETTINGS)
    with pytest.raises(ValueError, match="same number of visits"):
        prepare([wavelength], [flux, flux], [error, error], [0.0, 1.0], settings=SETTINGS)


def test_the_band_fluxes_and_the_seed_come_from_the_data():
    (wavelength, flux, error), other = visit(), visit(seed=1)
    profile = (wavelength, np.full(len(wavelength), 0.1))
    prepared = prepare(
        [wavelength] * 2, [flux, other[1]], [error, other[2]], [5.0, 1.0], settings=SETTINGS, profile=profile
    )
    assert prepared.mjd.tolist() == [1.0, 5.0] and prepared.band_flux.shape == prepared.band_error.shape == (2, 4)
    np.testing.assert_allclose(prepared.band_flux, 1.0, atol=0.05)
    assert np.isnan(prepare([wavelength], [flux], [error], [0.0], settings=SETTINGS).band_flux).all()

    def seed(unit=1.0, spread=1.0, dates=(0.0, 1.0)):
        return prepare(
            [wavelength] * 2,
            [unit * flux, unit * other[1]],
            [unit * error, unit * spread * other[2]],
            dates,
            settings=SETTINGS,
        ).seed

    assert seed(unit=1e3) == seed() == seed(dates=(10.0, 11.0))
    assert len({seed(), seed(spread=1.01), seed(dates=(0.0, 2.0))}) == 3


def test_edge_cases_of_the_input():
    import warnings

    pandas = pytest.importorskip("pandas")
    wavelength, flux, error = visit()
    with pytest.raises(ValueError, match="no visit given"):
        prepare([], [], [], [], settings=SETTINGS)
    series = prepare(
        pandas.Series(wavelength, index=range(7, 127)),
        [flux],
        [error],
        pandas.Series([1.0], index=[3]),
        settings=SETTINGS,
    )
    np.testing.assert_array_equal(series.flux, prepare([wavelength], [flux], [error], [1.0], settings=SETTINGS).flux)

    inside = (wavelength > 7450.0) & (wavelength < 18250.0)
    found = prepare([wavelength] * 2, [flux] * 2, [error] * 2, [0.0, 1.0], [inside, None], settings=SETTINGS)
    assert found.dropped == {0: "has no two neighbouring usable pixels in the model's wavelength range"}
    assert list(found.visits) == [1]

    sentinel = error.copy()
    sentinel[30:40] = -1e300
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not prepare([wavelength], [flux], [sentinel], [0.0], settings=SETTINGS).dropped


def test_the_signal_to_noise_is_the_median_of_the_visits_combined_on_their_own_pixels():
    wavelength, _, error = visit()
    flux = 3.0 * error
    flux[:3] = flux[-3:] = 1e3  # the ends, outside the range less 5% of it, do not count
    prepared = prepare([wavelength] * 2, [flux] * 2, [error] * 2, [0.0, 5.0], settings=SETTINGS)
    assert prepared.signal_to_noise == pytest.approx(3.0 * np.sqrt(2.0), rel=1e-6)
    masked = prepare(
        [wavelength] * 2, [flux] * 2, [error] * 2, [0.0, 5.0], [None, np.arange(120) < 60], settings=SETTINGS
    )
    assert list(masked.visits) == [0, 1] and 3.0 < masked.signal_to_noise < 3.0 * np.sqrt(2.0)


@pytest.mark.parametrize("seed", range(5))
def test_on_one_shared_grid_the_signal_to_noise_is_the_direct_coadd(seed):
    """Visits on one grid, with masked pixels: the median over the trimmed range of the inverse-variance coadd's
    S/N per pixel, the pixels whose combined error is over 3 times the median left out."""
    rng = np.random.default_rng(seed)
    wavelength, _, _ = visit()
    flux = [1.0 + 0.3 * rng.normal(size=120) for _ in range(3)]
    error = [0.1 + 0.2 * rng.random(120) for _ in range(3)]
    masks = [rng.random(120) < 0.2 for _ in range(3)]
    prepared = prepare([wavelength] * 3, flux, error, [0.0, 4.0, 9.0], masks, settings=SETTINGS)
    precision = sum(np.where(m, 0.0, 1.0 / e**2) for e, m in zip(error, masks))
    weighted = sum(np.where(m, 0.0, f / e**2) for f, e, m in zip(flux, error, masks))
    grid = wavelength_grid(SETTINGS)
    low, high = np.log(grid[0]), np.log(grid[-1])
    inside = (
        (wavelength >= np.exp(low + 0.05 * (high - low)))
        & (wavelength <= np.exp(high - 0.05 * (high - low)))
        & (precision > 0)
    )
    combined = 1.0 / np.sqrt(precision[inside])
    kept = combined <= 3.0 * np.median(combined)
    expected = np.median(weighted[inside][kept] / np.sqrt(precision[inside][kept]))
    assert prepared.signal_to_noise == pytest.approx(expected, rel=1e-5)


def test_the_signal_to_noise_combines_visits_whose_pixels_are_shifted_by_less_than_half_a_pixel():
    wavelength, _, error = visit()
    flux = 3.0 * error
    shifted = wavelength + 0.2 * np.gradient(wavelength)  # each visit's own wavelength solution
    same = prepare([wavelength] * 2, [flux] * 2, [error] * 2, [0.0, 5.0], settings=SETTINGS)
    moved = prepare([wavelength, shifted], [flux] * 2, [error] * 2, [0.0, 5.0], settings=SETTINGS)
    assert moved.signal_to_noise == pytest.approx(same.signal_to_noise, rel=1e-6)


def noisy(scale, lines=0.0, visits=20, pixels=1000, seed=3):
    """Visits with errors of 0.1 and white noise ``scale`` times larger, on a faint continuum (S/N 0.1), with sharp
    lines of peak ``lines`` every 10 pixels over the first 40% of the spectrum; pixels shuffled."""
    rng = np.random.default_rng(seed)
    wavelength, index = np.geomspace(7400.0, 18300.0, pixels), np.arange(pixels)
    shape = 0.01 + lines * sum(
        np.exp(-0.5 * ((index - centre) / 1.5) ** 2) for centre in range(5, int(0.4 * pixels), 10)
    )
    order = rng.permutation(pixels)  # the scale is measured in wavelength order, whatever order the pixels come in
    flux = [(shape + 0.1 * scale * rng.normal(size=pixels))[order] for _ in range(visits)]
    return [wavelength[order]] * visits, flux, [np.full(pixels, 0.1)] * visits


@pytest.mark.parametrize("scale", [1.25, 1.5, 2.0])
def test_the_measured_error_scale_recovers_white_noise_larger_than_the_errors(scale):
    assert measured_error_scale(*noisy(scale)) == pytest.approx(scale, rel=0.03)


def test_the_measured_error_scale_never_shrinks_the_errors_or_reads_bright_lines_as_noise():
    assert 1.0 <= measured_error_scale(*noisy(1.0)) < 1.03
    assert measured_error_scale(*noisy(0.5)) == 1.0
    # sharp lines at a peak S/N of 100 over 40% of the spectrum would read about 2 without the safeguard
    assert measured_error_scale(*noisy(1.0, lines=10.0)) == pytest.approx(1.0, abs=0.03)
    assert measured_error_scale(*noisy(1.25, lines=10.0)) == pytest.approx(1.25, rel=0.03)


def test_the_measured_error_scale_leaves_out_masked_and_unusable_pixels():
    rng = np.random.default_rng(3)
    wavelength = np.geomspace(7400.0, 18300.0, 1000)
    flux = 0.01 + 0.125 * rng.normal(size=1000)  # errors 0.1, noise 1.25 times larger
    error = np.full(1000, 0.1)
    bad = np.zeros(1000, bool)
    bad[200:600] = True
    wild = np.where(bad, 1e3 * rng.normal(size=1000), flux)
    expected = measured_error_scale([wavelength], [flux], [error], [bad])
    assert expected > 1.1
    assert measured_error_scale([wavelength], [wild], [error], [bad]) == expected
    assert measured_error_scale([wavelength], [wild], [np.where(bad, 0.0, error)]) == expected
    assert measured_error_scale([wavelength], [wild], [np.where(bad, -0.1, error)]) == expected


def test_the_measured_error_scale_needs_twenty_ratios():
    assert measured_error_scale(*noisy(3.0, visits=1, pixels=23)) == 1.0  # 19 ratios
    assert measured_error_scale(*noisy(3.0, visits=1, pixels=24)) > 2.0  # 20 ratios


def test_visits_on_one_date_with_the_same_flux_are_ordered_by_their_errors():
    wavelength, flux, error = visit()
    one, other = ([error, 1.5 * error], [1.5 * error, error])
    first, second = (
        prepare([wavelength] * 2, [flux, flux], errors, [0.0, 0.0], settings=SETTINGS) for errors in (one, other)
    )
    assert first.visits.tolist() == second.visits.tolist()[::-1] and first.seed == second.seed
    np.testing.assert_array_equal(first.log_error, second.log_error)


def test_the_order_of_visits_on_one_date_does_not_depend_on_the_error_scale():
    wavelength, flux, error = (parts[:3] for parts in noisy(1.0, visits=3, pixels=400))
    dates = [61000.0, 61000.0, 61005.0]
    base = prepare(wavelength, flux, error, dates, settings=SETTINGS)
    for scale in np.arange(1.01, 1.51, 0.01):
        scaled = prepare(
            wavelength, flux, [part * scale for part in error], dates, settings=SETTINGS, error_scale=scale
        )
        assert scaled.visits.tolist() == base.visits.tolist()
