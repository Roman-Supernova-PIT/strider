import math

import numpy as np
import pytest

from strider import posterior

GRID = np.linspace(0.0, 3.0, 301)
WIDTHS = posterior.cell_widths(GRID)
CLASSES = ["a", "b"]


def gaussian(mean, sigma, mass=1.0):
    """``mass`` times a Gaussian density of height 1, as probabilities per cell."""
    return mass * np.exp(-0.5 * ((GRID - mean) / sigma) ** 2) * WIDTHS


def priors(survey_top=1.5):
    """Class b pools the subtypes b1 and b2, and b2 has no training mass."""

    def part(fine, top):
        edges, density = ([0.0, 3.0], [1.0]) if top == 3.0 else ([0.0, top, 3.0], [1.0, 0.0])
        return {"class_fraction": fine, "redshift_edges": edges, "redshift_density": {name: density for name in fine}}

    return {
        "fine_to_output": {"a": "a", "b1": "b", "b2": "b"},
        "training_population": {"fine": part({"a": 0.5, "b1": 0.5}, 3.0)},
        "survey": {"class_fraction": {"a": 0.6, "b": 0.4}, "fine": part({"a": 0.6, "b1": 0.3, "b2": 0.1}, survey_top)},
    }


def test_cell_widths_cover_the_grid():
    assert np.isclose(WIDTHS.sum(), 3.0)
    assert np.isclose(WIDTHS[0], 0.5 * WIDTHS[1])
    grid = posterior.redshift_grid(
        {"redshift_min": 0.0, "redshift_max": 3.0, "redshift_bins": 5, "redshift_spacing": "log1p"}
    )
    np.testing.assert_allclose(np.log1p(grid), np.linspace(0.0, np.log(4.0), 5), rtol=1e-6)


def test_the_posterior_is_normalised_and_a_change_of_prior_reverses():
    rng = np.random.default_rng(0)
    log_posterior = posterior.log_posterior(rng.normal(size=(2, len(GRID))), WIDTHS, "flat_z")
    assert np.isclose(np.exp(log_posterior).sum(), 1.0)
    ratio = rng.normal(size=(2, len(GRID)))
    changed = posterior.change_prior(log_posterior, ratio)
    np.testing.assert_allclose(posterior.change_prior(changed, -ratio), log_posterior, atol=1e-12)


def test_the_survey_prior_counts_only_subtypes_with_training_mass():
    """With equal evidence everywhere, the posterior is the prior: the survey's mass that the model trained on."""
    prior = posterior.prior_change(priors(), CLASSES, GRID, WIDTHS)
    flat = posterior.log_posterior(np.zeros((2, len(GRID))), WIDTHS, "flat_z")
    found = posterior.summarise(flat, 0.0, GRID, WIDTHS, "flat_z", prior)
    np.testing.assert_allclose(found["class_probabilities"], [2 / 3, 1 / 3], rtol=1e-6)
    # with every class equally common, a keeps all its mass and b the three quarters in b1
    np.testing.assert_allclose(found["equal_prior_probabilities"], [1 / 1.75, 0.75 / 1.75], rtol=1e-6)
    assert found["redshift_posterior"][GRID > 1.5].sum() < 1e-12
    assert posterior.quantile(GRID, found["redshift_posterior"], 0.5) == pytest.approx(0.75, abs=0.01)
    # flat over 0 < z < 1.5: one peak, holding the 0.2 of the 1.5 that a window of ±0.1 covers
    assert found["redshift"] < 1.5 and found["redshift_probability"] == pytest.approx(0.2 / 1.5, abs=0.01)
    # a quarter of b's survey mass is b2, which the model never saw: 1/12 of the posterior
    np.testing.assert_allclose(prior["untrained"][1, GRID < 1.5], 0.25)
    assert found["untrained_share"] == pytest.approx(1 / 12, rel=1e-6)
    assert found["flags"] == {
        "several_redshift_peaks": False,
        "weak_source": None,
        "untrained_subtypes": True,
        "few_classes_trained": False,
    }
    assert found["source_probability"] is None


def test_a_class_untrained_at_a_redshift_is_not_an_untrained_subtype():
    """With one subtype per class, the untrained subtypes' share, of subtypes missing from classes trained at the
    redshift, stays 0 where a class had no training."""
    declared = {
        "fine_to_output": {"a": "a", "b": "b"},
        "training_population": {
            "fine": {
                "class_fraction": {"a": 0.5, "b": 0.5},
                "redshift_edges": [0.0, 1.5, 3.0],
                "redshift_density": {"a": [1.0, 1.0], "b": [1.0, 0.0]},
            }
        },
        "survey": {
            "class_fraction": {"a": 0.6, "b": 0.4},
            "fine": {
                "class_fraction": {"a": 0.6, "b": 0.4},
                "redshift_edges": [0.0, 3.0],
                "redshift_density": {"a": [1.0], "b": [1.0]},
            },
        },
    }
    prior = posterior.prior_change(declared, CLASSES, GRID, WIDTHS)
    logits = np.zeros((2, len(GRID)))
    logits[1, GRID > 1.5] = 5.0  # the network puts b where it had no b to train on
    found = posterior.summarise(logits, 0.0, GRID, WIDTHS, "flat_z", prior)
    assert found["untrained_share"] == 0.0 and not found["flags"]["untrained_subtypes"]


def test_a_survey_prior_without_some_class_is_refused():
    declared = priors()
    declared["survey"]["class_fraction"]["b"] = 0.0
    with pytest.raises(ValueError, match="priors.json gives b no share of the survey prior"):
        posterior.prior_change(declared, CLASSES, GRID, WIDTHS)


def test_median_and_intervals_of_a_gaussian():
    probability = gaussian(1.2, 0.1)
    assert posterior.quantile(GRID, probability, 0.5) == pytest.approx(1.2, abs=1e-6)
    lower, upper = posterior.central_interval(GRID, probability)
    assert lower == pytest.approx(1.2 - 0.0994, abs=1e-3) and upper == pytest.approx(1.2 + 0.0994, abs=1e-3)
    lower, upper = posterior.central_interval(GRID, probability, 0.95)
    assert lower == pytest.approx(1.2 - 0.196, abs=1e-3) and upper == pytest.approx(1.2 + 0.196, abs=1e-3)


def within(sigma, height=1.0):
    """The probability of a Gaussian of `gaussian`'s height and width within 0.1 of its mean."""
    return height * sigma * math.sqrt(2.0 * math.pi) * math.erf(0.1 / (sigma * math.sqrt(2.0)))


def area(sigma, height=1.0):
    return height * sigma * math.sqrt(2.0 * math.pi)


@pytest.mark.parametrize(
    "probability, expected",
    [
        (gaussian(1.0, 0.1), [(1.0, math.erf(1.0 / math.sqrt(2.0)))]),
        (
            gaussian(0.6, 0.1) + gaussian(2.0, 0.1, 0.5),
            [(0.6, within(0.1) / area(0.1) / 1.5), (2.0, 0.5 * within(0.1) / area(0.1) / 1.5)],
        ),
        (gaussian(0.6, 0.1) + gaussian(2.0, 0.1, 0.01), [(0.6, within(0.1) / area(0.1) / 1.01)]),
        # a narrow spike holds more within 0.1 than a broad peak with more probability in all, so it is the main one
        (
            gaussian(0.5, 0.02, 3.0) + gaussian(1.8, 0.3, 0.7),
            [
                (0.5, within(0.02, 3.0) / (area(0.02, 3.0) + area(0.3, 0.7))),
                (1.8, within(0.3, 0.7) / (area(0.02, 3.0) + area(0.3, 0.7))),
            ],
        ),
        # two equal peaks, each holding half
        (
            gaussian(0.5, 0.05) + gaussian(2.0, 0.05),
            [(0.5, 0.5 * math.erf(2.0 / math.sqrt(2.0))), (2.0, 0.5 * math.erf(2.0 / math.sqrt(2.0)))],
        ),
        # a peak at the grid's end: half a Gaussian, nearly all of it within 0.1
        (gaussian(0.0, 0.05), [(0.0, math.erf(2.0 / math.sqrt(2.0)))]),
    ],
)
def test_redshift_peaks_most_probable_first(probability, expected):
    found = posterior.redshift_peaks(GRID, probability, WIDTHS)
    assert len(found) == len(expected)
    for (redshift, near), (want_redshift, want_near) in zip(sorted(found), sorted(expected)):
        assert redshift == pytest.approx(want_redshift, abs=2e-3) and near == pytest.approx(want_near, abs=3e-3)
    assert found[0][1] == pytest.approx(max(near for _, near in expected), abs=3e-3)  # the main one first


@pytest.mark.parametrize("seed", range(20))
def test_the_main_peak_holds_the_most_and_the_others_at_least_the_minimum(seed):
    """On rough posteriors: the main peak is quoted at no less than 93% of the most any redshift holds within 0.1,
    and every other peak at no less than 10%, in falling order."""
    rng = np.random.default_rng(seed)
    probability = sum(
        gaussian(rng.uniform(0.0, 3.0), rng.uniform(0.01, 0.3), rng.uniform(0.2, 1.0))
        for _ in range(rng.integers(1, 5))
    ) * rng.uniform(0.7, 1.3, len(GRID))
    found = posterior.redshift_peaks(GRID, probability, WIDTHS)
    best = posterior.window_probability(GRID, probability / probability.sum()).max()
    assert found[0][1] >= 0.93 * best
    assert all(near >= 0.10 for _, near in found[1:])
    assert [near for _, near in found[1:]] == sorted((near for _, near in found[1:]), reverse=True)


@pytest.mark.parametrize("per", ["cell", "redshift"])
def test_a_flat_posterior_has_one_peak_and_a_low_probability(per):
    found = posterior.redshift_peaks(GRID, np.ones(len(GRID)) if per == "cell" else WIDTHS.copy(), WIDTHS)
    assert len(found) == 1 and found[0][1] < 0.1


def test_ripples_on_one_peak_are_one_peak():
    """As a host redshift's prior can leave them: one answer, not several within its own width."""
    ripples = gaussian(1.0, 0.05) * (1.0 + 0.6 * np.sin(2.0 * np.pi * GRID / 0.04))
    found = posterior.redshift_peaks(GRID, ripples, WIDTHS)
    assert len(found) == 1 and found[0][1] > 0.9


def test_two_broad_jagged_humps_are_two_peaks():
    jagged = (gaussian(0.45, 0.2) + gaussian(1.25, 0.2)) * np.random.default_rng(1).lognormal(0.0, 0.5, len(GRID))
    found = posterior.redshift_peaks(GRID, jagged, WIDTHS)
    assert len(found) == 2 and sorted(round(z, 1) for z, _ in found) == pytest.approx([0.4, 1.2], abs=0.15)


def test_a_peak_between_grid_points_is_found_between_them():
    """The parabola in ln(1 + z) is exact for a density that is Gaussian in ln(1 + z)."""
    off_grid = 1.0 + 0.4 * (GRID[1] - GRID[0])
    density = np.exp(-0.5 * ((np.log1p(GRID) - np.log1p(off_grid)) / 0.05) ** 2)
    assert posterior.redshift_peaks(GRID, density * WIDTHS, WIDTHS)[0][0] == pytest.approx(off_grid, abs=1e-9)


def test_each_class_redshift_is_its_main_peak():
    rows = np.stack([gaussian(0.4, 0.1), gaussian(2.2, 0.1), np.full(len(GRID), np.nan)])
    found = posterior.class_redshifts(GRID, rows / np.nansum(rows, axis=1, keepdims=True), WIDTHS)
    assert found[:2] == pytest.approx([0.4, 2.2], abs=2e-3) and np.isnan(found[2])


def test_the_fusion_weighs_the_spectrum_s_evidence_and_adds_the_brightness_term():
    """a (log q - log pi) + log pi + w L: with q = pi and L = 0 the posterior is the training prior, and none is left
    where pi is 0."""
    with np.errstate(divide="ignore"):
        training = np.log(np.stack([gaussian(1.0, 0.5), np.where(GRID < 2.0, WIDTHS, 0.0)]))
    training -= np.logaddexp.reduce(training[np.isfinite(training)])
    brightness = np.where(np.isfinite(training), 0.0, -np.inf)
    np.testing.assert_allclose(posterior.fuse(training, brightness, training, 0.8, 0.4), training, atol=1e-12)
    spectrum = posterior.log_posterior(np.random.default_rng(2).normal(size=training.shape), WIDTHS, "flat_z")
    found = posterior.fuse(spectrum, brightness + 1.0 * (GRID > 1.0), training, 0.8, 0.4)
    trained = np.isfinite(training)
    expected = np.where(
        trained, 0.8 * (spectrum - np.where(trained, training, 0.0)) + training + 0.4 * (GRID > 1.0), -np.inf
    )
    np.testing.assert_allclose(found, expected - np.logaddexp.reduce(expected[trained]), atol=1e-12)
    assert np.isneginf(found[1, GRID >= 2.0]).all()


LEVELS = {"high": 5, "medium": 2, "second_redshift": 3}


def match(other=4.0, second=6.0, light=0.0, at=2.0):
    """Class a's score peaks at 8 at z = 1 with a second peak of ``second`` at z = ``at``, and class b's is ``other``
    everywhere; the brightness term alone favours class b by ``light`` at z = 1, at the weights a = 1, w = 0.5."""
    training = np.log(np.tile(WIDTHS / (2 * WIDTHS.sum()), (2, 1)))
    a = 8.0 * np.exp(-0.5 * ((GRID - 1.0) / 0.1) ** 2) + second * np.exp(-0.5 * ((GRID - at) / 0.1) ** 2)
    brightness = np.stack([np.zeros_like(GRID), light * np.exp(-0.5 * ((GRID - 1.0) / 0.1) ** 2)])
    spectrum = training + np.stack([a, np.full_like(GRID, other)]) - 0.5 * brightness
    return posterior.match(spectrum, brightness, training, 1.0, 0.5, GRID, LEVELS)


def test_the_match_strength_follows_the_gap_to_the_next_class():
    found = match()
    assert (found["best_class"], found["next_best_class"]) == (0, 1) and found["best_redshift"] == GRID[100]
    assert found["match_gap"] == pytest.approx(4.0) and found["match_strength"] == "medium"
    assert match(other=2.0)["match_strength"] == "high" and match(other=7.5)["match_strength"] == "low"


def test_a_second_redshift_close_behind_and_a_disagreeing_brightness_term_are_warned_of():
    assert match()["warnings"] == ["second redshift near z 2.00"]
    assert match(at=3.0)["warnings"] == ["second redshift near z 3.00"]  # at the end of the grid
    assert match(second=4.0)["warnings"] == []  # 4 behind: far enough
    assert match(second=4.0, light=3.0)["warnings"] == ["spectra and brightness disagree"]
    assert match(second=4.0, light=0.5)["warnings"] == []  # the brightness term alone too weak to disagree


def test_the_survey_prior_and_the_source_probability():
    prior = posterior.prior_change(priors(survey_top=3.0), CLASSES, GRID, WIDTHS)
    log_joint = np.log(np.stack([np.full(len(GRID), math.e), np.ones(len(GRID))]) * WIDTHS / (math.e + 1.0) / 3.0)
    calibration = {"source_probability": {"slope": 2.0, "intercept": -1.0, "weak_source_below": 0.5}}
    found = posterior.summarise(log_joint, 0.3, GRID, WIDTHS, "flat_z", prior, calibration)
    expected = np.array([math.e, 1.0]) * np.array([0.6, 0.3]) / np.array([0.5, 0.5])
    np.testing.assert_allclose(found["class_probabilities"], expected / expected.sum(), rtol=1e-6)
    np.testing.assert_allclose(found["log_evidence"], log_joint - prior["training"], rtol=1e-12)
    assert found["source_probability"] == pytest.approx(1.0 / (1.0 + math.exp(-(2.0 * 0.3 - 1.0))))
    assert found["flags"]["weak_source"] is True
    assert (
        posterior.summarise(log_joint, 2.0, GRID, WIDTHS, "flat_z", prior, calibration)["flags"]["weak_source"] is False
    )


@pytest.mark.parametrize("measure", ["flat_z", "flat_log1p"])
def test_a_host_redshift_multiplies_the_evidence_by_its_probability_in_each_cell(measure):
    pytest.importorskip("torch")
    logits = np.random.default_rng(1).normal(size=(2, len(GRID)))
    log_posterior = posterior.log_posterior(logits, WIDTHS, measure)
    edges = np.concatenate([GRID[:1], 0.5 * (GRID[1:] + GRID[:-1]), GRID[-1:]])
    below = np.array([0.5 * math.erfc(-(edge - 1.2) / 0.05 / math.sqrt(2.0)) for edge in edges])
    for outlier in (0.0, 0.05):
        ratio = posterior.host_redshift_log_prior(GRID, WIDTHS, measure, 1.2, 0.05, outlier)
        found = np.exp(posterior.change_prior(log_posterior, ratio))
        # the Gaussian's mass in each cell, with a share spread over the cells by width (the paper's form)
        expected = np.exp(logits) * ((1.0 - outlier) * np.diff(below) + outlier * WIDTHS / WIDTHS.sum())
        np.testing.assert_allclose(found, expected / expected.sum(), rtol=1e-6, atol=1e-15)
    with pytest.raises(ValueError, match="positive"):
        posterior.host_redshift_log_prior(GRID, WIDTHS, measure, 1.2, 0.0)


def test_a_wrong_host_redshift_cannot_overrule_strong_evidence():
    """With the default share left for a wrong host, evidence of 20 nats at z 0.5 beats a host at 2.5 ± 0.01; with no
    share it does not."""
    pytest.importorskip("torch")
    logits = np.full((2, len(GRID)), -10.0)
    logits[0, np.argmin(np.abs(GRID - 0.5))] = 10.0
    prior = posterior.prior_change(priors(3.0), CLASSES, GRID, WIDTHS)
    trusted = posterior.summarise(logits, 0.0, GRID, WIDTHS, "flat_z", prior, host=(2.5, 0.01))
    certain = posterior.summarise(logits, 0.0, GRID, WIDTHS, "flat_z", prior, host=(2.5, 0.01), host_outlier=0.0)
    assert trusted["redshift"] == pytest.approx(0.5, abs=0.02) and certain["redshift"] > 2.4


def evidence(seed=2):
    """A posterior from random logits under the survey prior of `priors`, with each class over the whole grid."""
    return np.random.default_rng(seed).normal(size=(2, len(GRID))), posterior.prior_change(
        priors(3.0), CLASSES, GRID, WIDTHS
    )


def test_a_class_restriction_renormalises_over_the_classes_kept():
    logits, prior = evidence()
    found = posterior.summarise(logits, 0.0, GRID, WIDTHS, "flat_z", prior)
    kept = posterior.summarise(logits, 0.0, GRID, WIDTHS, "flat_z", prior, classes=np.array([False, True]))
    np.testing.assert_allclose(kept["class_probabilities"], [0.0, 1.0], atol=1e-15)
    np.testing.assert_allclose(kept["equal_prior_probabilities"], [0.0, 1.0], atol=1e-15)
    np.testing.assert_allclose(kept["class_redshift_posterior"], found["class_redshift_posterior"], rtol=1e-12)
    np.testing.assert_allclose(kept["redshift_posterior"], found["class_redshift_posterior"][1], rtol=1e-12)
    everything = posterior.summarise(logits, 0.0, GRID, WIDTHS, "flat_z", prior, classes=np.array([True, True]))
    assert np.array_equal(everything["class_probabilities"], found["class_probabilities"])
    assert np.array_equal(everything["redshift_posterior"], found["redshift_posterior"])


def test_a_class_prior_replaces_the_survey_mix_of_classes():
    logits, prior = evidence()
    found = posterior.summarise(logits, 0.0, GRID, WIDTHS, "flat_z", prior)
    equal = posterior.summarise(logits, 0.0, GRID, WIDTHS, "flat_z", prior, class_prior=np.array([3.0, 3.0]))
    np.testing.assert_allclose(equal["class_probabilities"], found["equal_prior_probabilities"], rtol=1e-12)
    survey = posterior.summarise(logits, 0.0, GRID, WIDTHS, "flat_z", prior, class_prior=np.array([0.6, 0.4]))
    np.testing.assert_allclose(survey["class_probabilities"], found["class_probabilities"], rtol=1e-12)
    odds = found["equal_prior_probabilities"] * [0.9, 0.1]
    tilted = posterior.summarise(logits, 0.0, GRID, WIDTHS, "flat_z", prior, class_prior=np.array([0.9, 0.1]))
    np.testing.assert_allclose(tilted["class_probabilities"], odds / odds.sum(), rtol=1e-12)


def test_a_redshift_range_keeps_the_share_of_each_cell_inside_it():
    logits, prior = evidence()
    found = posterior.summarise(logits, 0.0, GRID, WIDTHS, "flat_z", prior)["redshift_posterior"]
    low, high = 1.0025, 2.0  # a quarter of the cell at 1.0 and all of the cell at 2.0 lie inside
    ranged = posterior.summarise(
        logits, 0.0, GRID, WIDTHS, "flat_z", prior, redshift_prior=posterior.redshift_range_log_prior(GRID, low, high)
    )
    share = np.where((GRID > low) & (GRID < high), 1.0, 0.0)
    share[np.isclose(GRID, 1.0)], share[np.isclose(GRID, 2.0)] = 0.25, 0.5
    np.testing.assert_allclose(
        ranged["redshift_posterior"], found * share / (found * share).sum(), rtol=1e-9, atol=1e-15
    )
    assert low <= ranged["redshift_95"][0] < ranged["redshift_95"][1] <= high
    assert not np.any(posterior.redshift_range_log_prior(GRID, -np.inf, np.inf))
    with pytest.raises(ValueError, match="no probability is left"):
        posterior.summarise(
            logits,
            0.0,
            GRID,
            WIDTHS,
            "flat_z",
            prior,
            redshift_prior=posterior.redshift_range_log_prior(GRID, 5.0, 6.0),
        )


@pytest.mark.parametrize("measure", ["flat_z", "flat_log1p"])
def test_a_redshift_density_multiplies_the_posterior_as_a_host_redshift_does(measure):
    pytest.importorskip("torch")
    logits, prior = evidence()
    density = np.exp(-0.5 * ((GRID - 1.2) / 0.05) ** 2) / (0.05 * math.sqrt(2.0 * math.pi))
    by_density = posterior.redshift_density_log_prior(WIDTHS, measure, density)
    by_host = posterior.host_redshift_log_prior(GRID, WIDTHS, measure, 1.2, 0.05, outlier=0.0)
    near = np.abs(GRID - 1.2) < 0.1  # the density at a grid point against the mass in its cell: (w/σ)²/24 apart
    np.testing.assert_allclose(by_density[near], by_host[near], atol=5e-3)
    found = posterior.summarise(logits, 0.0, GRID, WIDTHS, measure, prior)
    flat = posterior.summarise(
        logits,
        0.0,
        GRID,
        WIDTHS,
        measure,
        prior,
        redshift_prior=posterior.redshift_density_log_prior(WIDTHS, measure, np.ones(len(GRID))),
    )
    expected = found["redshift_posterior"] * (WIDTHS if measure == "flat_log1p" else 1.0)
    np.testing.assert_allclose(flat["redshift_posterior"], expected / expected.sum(), rtol=1e-6)
