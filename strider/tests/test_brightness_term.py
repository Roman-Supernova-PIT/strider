import numpy as np
import pytest

from strider import brightness
from strider.brightness import BrightnessTerm

NODES = np.linspace(0.0, np.log(2.0), 4)
PHASES = np.arange(-20.0, 51.0)
COLOUR = np.array([1.0, 0.8, 0.6, 0.4])
WIDTHS = (8.0, 25.0)  # a fast and a slow class
GRID = np.linspace(0.0, 1.0, 9)
DAYS = 60000.0 + np.repeat([-12.0, -4.0, 3.0, 10.0, 24.0], 4)
BAND = np.tile(np.arange(4), 5)


def curve(width, band, node, phase):
    return COLOUR[band] * (1.0 + 0.2 * node) * np.exp(-0.5 * (phase / width) ** 2)


def toy_templates():
    mean = np.stack([curve(w, *np.ix_(np.arange(4), np.arange(4.0), PHASES)) for w in WIDTHS])
    loadings = np.zeros((*mean.shape, 2))
    loadings[0] = 0.02 * mean[0, ..., None] * np.stack([np.ones_like(PHASES), PHASES / 50.0], -1)
    return {
        "classes": np.array(["fast", "slow"]),
        "z_nodes": NODES,
        "mean": mean,
        "psi2": (0.05 * mean) ** 2 + 1e-6,
        "loadings": loadings,
        "window": np.array([-20.0, 50.0]),
        "tau": np.array([2.5, 58.0]),
        "discrepancy": np.array([[1.41, 16.0, 0.3, 5.66, 0.05], [0.71, 32.0, 0.6, 2.83, 0.2]]),
    }


TERM = BrightnessTerm(toy_templates(), GRID)
NODE = np.interp(np.log1p(0.5), NODES, np.arange(4))
FLUX = 30.0 * curve(WIDTHS[0], BAND, NODE, (DAYS - 60000.0) / 1.5) + np.random.default_rng(3).normal(size=20)
ERROR = np.ones(20)


@pytest.fixture(scope="module")
def result():
    return TERM.score(DAYS, BAND, FLUX, ERROR, seed=7)


def test_the_class_the_light_curve_came_from_scores_highest(result):
    assert np.all(np.isfinite(result.log_ratio))
    assert result.log_ratio[0].max() > result.log_ratio[1].max()
    best = np.argmax(result.peak_log_weight[0, :, 0])
    assert abs(result.peak_date[0, best, 0] - 60000.0) < 5.0


def test_a_flux_unit_and_a_date_zero_point_change_nothing(result):
    other = TERM.score(DAYS - 60000.0, BAND, 1e3 * FLUX, 1e3 * ERROR, seed=7)
    np.testing.assert_allclose(other.log_ratio, result.log_ratio, rtol=0, atol=1e-8)
    np.testing.assert_allclose(other.peak_date + 60000.0, result.peak_date, rtol=0, atol=1e-6)


def test_the_null_copies_follow_the_seed(result):
    again = TERM.score(DAYS, BAND, FLUX, ERROR, seed=7)
    np.testing.assert_array_equal(again.log_ratio, result.log_ratio)
    assert not np.allclose(TERM.score(DAYS, BAND, FLUX, ERROR, seed=8).log_ratio, result.log_ratio)


def test_the_batch_size_changes_nothing(result, monkeypatch):
    monkeypatch.setattr(brightness, "BUDGET", 1.0)
    np.testing.assert_allclose(TERM.score(DAYS, BAND, FLUX, ERROR, seed=7).log_ratio, result.log_ratio, atol=1e-12)


def test_cells_not_scored_are_minus_infinity_and_leave_the_rest_alone(result):
    scored = np.ones((2, len(GRID)), bool)
    scored[1] = False
    scored[0, -3:] = False
    out = BrightnessTerm(toy_templates(), GRID, scored).score(DAYS, BAND, FLUX, ERROR, seed=7)
    assert np.all(np.isneginf(out.log_ratio[~scored]))
    np.testing.assert_allclose(out.log_ratio[scored], result.log_ratio[scored], rtol=0, atol=1e-12)
    assert np.all(np.isnan(out.peak_date[1]))


def test_one_visit_or_visits_on_one_date_add_nothing(result):
    out = TERM.score(DAYS[:4], BAND[:4], FLUX[:4], ERROR[:4], seed=7)
    np.testing.assert_array_equal(out.log_ratio, 0.0)
    same = np.r_[DAYS[:4], DAYS[:4]], np.r_[BAND[:4], BAND[:4]], np.r_[FLUX[:4], FLUX[4:8]], np.r_[ERROR[:4], ERROR[:4]]
    np.testing.assert_array_equal(TERM.score(*same, seed=7).log_ratio, 0.0)
    assert TERM.curve(*same, 0, 0.5, result.peak_date[0, :, 0], DATES) is None


def test_a_band_outside_the_four_is_refused():
    with pytest.raises(ValueError, match="band"):
        TERM.score(DAYS, np.where(BAND == 3, 4, BAND), FLUX, ERROR, seed=7)


DATES = np.linspace(59960.0, 60060.0, 201)


def test_the_fitted_curves_follow_the_light_curve(result):
    dates = result.peak_date[0, :, 0]
    fit = TERM.curve(DAYS, BAND, FLUX + 20.0 * BAND, ERROR, 0, 0.5, dates, DATES)
    assert abs(fit.peak_date - 60000.0) < 3.0
    np.testing.assert_allclose(fit.level, 20.0 * np.arange(4), atol=3.0)
    peaks = fit.mean.max(0)
    np.testing.assert_allclose(peaks / peaks[0], COLOUR, rtol=0.05)
    np.testing.assert_allclose(peaks, FLUX[DAYS == 60003.0], rtol=0.15)
    assert np.all(np.abs(DATES[fit.mean.argmax(0)] - fit.peak_date) < 3.0)
    assert np.all(fit.sd > 0) and np.all(fit.sd.max(0) < 0.2 * peaks)


def test_a_flux_unit_and_a_date_zero_point_change_the_curves_alike(result):
    dates = result.peak_date[0, :, 0]
    fit = TERM.curve(DAYS, BAND, FLUX, ERROR, 0, 0.5, dates, DATES)
    other = TERM.curve(DAYS - 6e4, BAND, 1e-17 * FLUX, 1e-17 * ERROR, 0, 0.5, dates - 6e4, DATES - 6e4)
    np.testing.assert_allclose(1e17 * other.mean, fit.mean, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(1e17 * other.sd, fit.sd, rtol=1e-8, atol=1e-8)
    assert other.peak_date + 6e4 == pytest.approx(fit.peak_date, abs=1e-6)


def test_a_peak_date_that_leaves_a_visit_outside_the_templates_is_moved():
    late = np.full(len(NODES), 60060.0)  # the first visit 48 rest days before it, beyond the window's -20
    fit = TERM.curve(DAYS, BAND, FLUX, ERROR, 0, 0.5, late, DATES)
    z1 = np.exp(NODES[np.argmin(np.abs(NODES - np.log1p(0.5)))])
    phase = (DAYS - fit.peak_date) / z1
    assert fit.peak_date < 60060.0 and phase.min() >= -20.0 and phase.max() <= 50.0


def test_one_visit_has_no_fitted_curve(result):
    assert TERM.curve(DAYS[:4], BAND[:4], FLUX[:4], ERROR[:4], 0, 0.5, result.peak_date[0, :, 0], DATES) is None
