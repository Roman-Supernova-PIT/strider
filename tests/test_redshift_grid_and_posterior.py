"""Scientific contracts for the candidate-redshift probability measure."""

from __future__ import annotations

import numpy as np
import torch

from strider.data.dataset import log_wavelength_grid
from strider.evaluation.evaluate import (
    _candidate_route_support,
    _interpolated_peak,
    _posterior_basin_candidates,
    _joint_posterior_basin_candidates,
    _posterior_information,
    _posterior_peak_summary,
    _posterior_quantile,
)
from strider.model.posterior import joint_probability
from strider.model.redshift_scan import build_redshift_grid, redshift_cell_widths
from strider.training.losses import joint_targets


def test_log1p_grid_has_constant_spectral_shift() -> None:
    grid = build_redshift_grid(0.05, 3.0, 500, "log1p")
    shifts = np.diff(np.log1p(grid))
    assert np.allclose(shifts, shifts[0], rtol=1e-4)


def test_observed_wavelength_grid_stays_log_uniform_at_full_resolution() -> None:
    grid = log_wavelength_grid(7500.0, 20000.0, 1024)
    shifts = np.diff(np.log(grid))

    assert grid.dtype == np.float64
    assert np.allclose(shifts, shifts[0], rtol=1e-10, atol=1e-12)


def test_redshift_cell_widths_cover_the_declared_interval() -> None:
    grid = build_redshift_grid(0.05, 3.0, 500, "log1p")
    widths = redshift_cell_widths(grid)
    assert np.all(widths > 0.0)
    assert np.isclose(widths.sum(), grid[-1] - grid[0], rtol=1e-6)


def test_flat_z_prior_accounts_for_unequal_grid_cells() -> None:
    grid = build_redshift_grid(0.05, 3.0, 50, "log1p")
    widths = torch.from_numpy(redshift_cell_widths(grid))
    evidence = torch.zeros(2, 3, len(grid))
    probability = joint_probability(evidence, widths, prior="flat_z")
    redshift_mass = probability.sum(dim=1)
    density = redshift_mass / widths[None, :]
    assert torch.allclose(density, density[:, :1].expand_as(density), rtol=1e-5)


def test_peak_interpolation_can_return_a_value_between_grid_points() -> None:
    grid = build_redshift_grid(0.05, 3.0, 60, "log1p").astype(np.float64)
    centre = 1.234
    probability = np.exp(-0.5 * ((np.log1p(grid) - np.log1p(centre)) / 0.02) ** 2)
    estimate = _interpolated_peak(grid, probability)
    assert min(abs(grid - centre)) > abs(estimate - centre)
    assert abs(estimate - centre) < 1e-3


def test_density_mode_does_not_mistake_unequal_cell_mass_for_evidence() -> None:
    grid = build_redshift_grid(0.05, 3.0, 60, "log1p").astype(np.float64)
    widths = redshift_cell_widths(grid).astype(np.float64)
    flat_density = widths / widths
    assert _interpolated_peak(grid, flat_density) == grid[0]


def test_posterior_information_is_finite_with_exact_zero_mass() -> None:
    probability = np.array([0.0, 0.25, 0.75, 0.0], dtype=np.float32)
    cell_width = np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    entropy, information_gain = _posterior_information(
        probability, cell_width, "flat_z"
    )
    assert np.isfinite(entropy)
    assert np.isfinite(information_gain)


def test_peak_summary_separates_a_sharp_solution_from_a_high_redshift_tail() -> None:
    grid = build_redshift_grid(0.05, 3.0, 500, "log1p").astype(np.float64)
    widths = redshift_cell_widths(grid).astype(np.float64)
    coordinate = np.log1p(grid)
    primary = np.exp(-0.5 * ((coordinate - np.log1p(1.25)) / 0.012) ** 2)
    secondary = 0.22 * np.exp(
        -0.5 * ((coordinate - np.log1p(2.15)) / 0.08) ** 2
    )
    density = primary + secondary
    mass = density * widths
    mass /= mass.sum()

    summary = _posterior_peak_summary(grid, mass, widths)
    median = _posterior_quantile(grid, mass, 0.5)

    assert abs(float(summary["dominant_redshift"]) - 1.25) < 0.01
    assert median > float(summary["dominant_redshift"])
    assert abs(float(summary["secondary_redshift"]) - 2.15) < 0.03
    assert float(summary["secondary_mass"]) > 0.05
    assert int(summary["distinct_peak_count"]) >= 2


def test_basin_candidate_median_is_not_dragged_between_distinct_solutions() -> None:
    grid = build_redshift_grid(0.05, 3.0, 500, "log1p").astype(np.float64)
    widths = redshift_cell_widths(grid).astype(np.float64)
    coordinate = np.log1p(grid)
    primary = np.exp(-0.5 * ((coordinate - np.log1p(1.25)) / 0.012) ** 2)
    secondary = 0.22 * np.exp(
        -0.5 * ((coordinate - np.log1p(2.15)) / 0.08) ** 2
    )
    mass = (primary + secondary) * widths
    mass /= mass.sum()

    candidates = _posterior_basin_candidates(grid, mass, widths)
    global_median = _posterior_quantile(grid, mass, 0.5)

    assert len(candidates) == 2
    assert abs(float(candidates[0]["peak_redshift"]) - 1.25) < 0.01
    assert abs(float(candidates[0]["median_redshift"]) - 1.25) < 0.02
    assert global_median > float(candidates[0]["median_redshift"])
    assert abs(float(candidates[1]["peak_redshift"]) - 2.15) < 0.03
    assert float(candidates[1]["mass"]) > 0.05
    assert float(candidates[1]["height_ratio"]) < 1.0
    assert np.isfinite(
        float(candidates[0]["log_peak_to_strongest_competitor_saddle_ratio"])
    )
    assert (
        float(candidates[0]["primary_to_strongest_competitor_height_ratio"])
        > 1.0
    )
    assert not bool(candidates[0]["is_largest_mass_basin"])
    assert bool(candidates[1]["is_largest_mass_basin"])


def test_basin_candidates_limit_output_without_losing_primary() -> None:
    grid = build_redshift_grid(0.05, 3.0, 500, "log1p").astype(np.float64)
    widths = redshift_cell_widths(grid).astype(np.float64)
    coordinate = np.log1p(grid)
    density = sum(
        amplitude
        * np.exp(-0.5 * ((coordinate - np.log1p(redshift)) / 0.018) ** 2)
        for redshift, amplitude in ((0.5, 1.0), (1.1, 0.8), (1.8, 0.7), (2.5, 0.6))
    )

    candidates = _posterior_basin_candidates(
        grid,
        density * widths,
        widths,
        maximum_candidates=3,
    )

    assert len(candidates) == 3
    assert abs(float(candidates[0]["peak_redshift"]) - 0.5) < 0.02
    assert all(float(candidate["mass"]) > 0.05 for candidate in candidates)


def test_joint_candidates_preserve_classes_hidden_by_one_redshift_basin() -> None:
    grid = build_redshift_grid(0.05, 3.0, 500, "log1p").astype(np.float64)
    widths = redshift_cell_widths(grid).astype(np.float64)
    coordinate = np.log1p(grid)
    shared_mode = np.exp(
        -0.5 * ((coordinate - np.log1p(1.1)) / 0.025) ** 2
    ) * widths
    joint = np.stack((0.62 * shared_mode, 0.38 * shared_mode))
    joint /= joint.sum()

    marginal = _posterior_basin_candidates(grid, joint.sum(axis=0), widths)
    candidates = _joint_posterior_basin_candidates(
        grid,
        joint,
        widths,
        ["Ia", "other"],
    )

    assert len(marginal) == 1
    assert len(candidates) == 2
    assert [candidate["class_name"] for candidate in candidates] == ["Ia", "other"]
    assert all(abs(float(candidate["peak_redshift"]) - 1.1) < 0.02 for candidate in candidates)
    assert np.isclose(sum(float(candidate["mass"]) for candidate in candidates), 1.0)


def test_joint_candidates_rank_distinct_class_redshift_hypotheses() -> None:
    grid = build_redshift_grid(0.05, 3.0, 500, "log1p").astype(np.float64)
    widths = redshift_cell_widths(grid).astype(np.float64)
    coordinate = np.log1p(grid)
    ia = np.exp(-0.5 * ((coordinate - np.log1p(0.8)) / 0.02) ** 2) * widths
    other = 0.45 * np.exp(
        -0.5 * ((coordinate - np.log1p(1.7)) / 0.03) ** 2
    ) * widths
    joint = np.stack((ia, other))
    joint /= joint.sum()

    candidates = _joint_posterior_basin_candidates(
        grid,
        joint,
        widths,
        ["Ia", "other"],
    )

    assert len(candidates) == 2
    assert candidates[0]["class_name"] == "Ia"
    assert abs(float(candidates[0]["peak_redshift"]) - 0.8) < 0.02
    assert candidates[1]["class_name"] == "other"
    assert abs(float(candidates[1]["peak_redshift"]) - 1.7) < 0.03


def test_route_support_rewards_its_peak_and_downweights_flat_routes() -> None:
    candidates = [
        {"left_index": 0, "right_index": 2},
        {"left_index": 2, "right_index": 4},
    ]
    informative = np.asarray(
        [
            [0.0, 0.2, 0.0, 0.0],
            [0.0, 0.0, 1.5, 2.0],
        ]
    )
    flat = np.zeros((2, 4))

    informative_support = _candidate_route_support(
        informative,
        candidates,
        [0, 1],
    )
    flat_support = _candidate_route_support(flat, candidates, [0, 1])

    assert informative_support[1] > informative_support[0]
    assert informative_support[1] > 0.8
    assert flat_support == [0.0, 0.0]


def test_route_support_respects_unsupported_cells() -> None:
    candidates = [{"left_index": 0, "right_index": 2}]
    logits = np.asarray([[-1.0e4, -1.0e4], [0.0, 1.0]])

    assert _candidate_route_support(logits, candidates, [0]) == [0.0]


def test_peak_summary_does_not_invent_a_competing_mode_for_one_gaussian() -> None:
    grid = build_redshift_grid(0.05, 3.0, 500, "log1p").astype(np.float64)
    widths = redshift_cell_widths(grid).astype(np.float64)
    density = np.exp(-0.5 * ((np.log1p(grid) - np.log1p(0.8)) / 0.03) ** 2)
    mass = density * widths

    summary = _posterior_peak_summary(grid, mass, widths)

    assert int(summary["distinct_peak_count"]) == 1
    assert np.isnan(float(summary["secondary_redshift"]))
    assert float(summary["secondary_mass"]) == 0.0


def test_training_rejects_truth_redshift_outside_candidate_grid() -> None:
    grid = torch.tensor([0.1, 0.2, 0.3])
    try:
        joint_targets(torch.tensor([0]), torch.tensor([0.4]), grid)
    except ValueError as error:
        assert "outside the candidate grid" in str(error)
    else:
        raise AssertionError("out-of-grid training target was silently clipped")
