import torch

from strider.model.phase_consistency import CandidatePhaseConsistency


def test_uniform_phase_predictions_add_no_candidate_evidence() -> None:
    module = CandidatePhaseConsistency(initial_scale=1.0)
    logits = torch.zeros(2, 4, 3, 5, 9)
    observer_days = torch.tensor(
        [[0.0, 8.0, 17.0, 28.0], [0.0, 5.0, 11.0, 19.0]]
    )
    visit_mask = torch.ones(2, 4)
    redshift_grid = torch.linspace(0.0, 2.0, 5)
    phase_grid = torch.linspace(-20.0, 20.0, 9)

    scaled, raw = module(
        logits, observer_days, visit_mask, redshift_grid, phase_grid
    )

    assert torch.allclose(raw, torch.zeros_like(raw), atol=1.0e-6)
    assert torch.allclose(scaled, torch.zeros_like(scaled), atol=1.0e-6)


def test_phase_trajectory_prefers_its_candidate_redshift() -> None:
    module = CandidatePhaseConsistency(initial_scale=1.0)
    phase_grid = torch.linspace(-20.0, 20.0, 9)
    redshift_grid = torch.tensor([0.0, 1.0])
    observer_days = torch.tensor([[0.0, 10.0, 20.0]])
    visit_mask = torch.ones(1, 3)
    logits = torch.full((1, 3, 1, 2, 9), -8.0)
    true_phase_indices = (2, 3, 4)
    for visit, phase_index in enumerate(true_phase_indices):
        logits[0, visit, 0, :, phase_index] = 8.0

    _, raw = module(
        logits, observer_days, visit_mask, redshift_grid, phase_grid
    )

    assert raw[0, 0, 1] > raw[0, 0, 0] + 1.0


def test_one_visit_cannot_create_phase_consistency() -> None:
    module = CandidatePhaseConsistency(initial_scale=1.0)
    logits = torch.randn(1, 3, 2, 4, 7)
    observer_days = torch.tensor([[0.0, 10.0, 20.0]])
    visit_mask = torch.tensor([[1.0, 0.0, 0.0]])

    _, raw = module(
        logits,
        observer_days,
        visit_mask,
        torch.linspace(0.0, 2.0, 4),
        torch.linspace(-15.0, 15.0, 7),
    )

    assert torch.equal(raw, torch.zeros_like(raw))


def test_candidate_phase_consistency_uses_relative_dates() -> None:
    module = CandidatePhaseConsistency(initial_scale=0.5)
    phase_grid = torch.linspace(-20.0, 50.0, 15)
    redshift_grid = torch.tensor([0.5, 1.0, 1.5])
    phase_logits = torch.randn(2, 4, 2, 3, 15)
    observer_days = torch.tensor(
        [[0.0, 5.0, 15.0, 25.0], [3.0, 8.0, 18.0, 28.0]]
    )
    visit_mask = torch.ones(2, 4)

    original = module(
        phase_logits, observer_days, visit_mask, redshift_grid, phase_grid
    )
    shifted = module(
        phase_logits, observer_days + 120.0, visit_mask, redshift_grid, phase_grid
    )

    assert torch.allclose(original[0], shifted[0], atol=1e-5)
    assert torch.allclose(original[1], shifted[1], atol=1e-5)


def test_peak_informed_trajectory_prefers_the_correct_redshift() -> None:
    module = CandidatePhaseConsistency(
        initial_scale=1.0,
        use_peak_date=True,
        peak_uncertainty_days=0.0,
        peak_quadrature_points=1,
        minimum_coverage_fraction=1.0,
    )
    phase_grid = torch.linspace(-20.0, 20.0, 9)
    redshift_grid = torch.tensor([0.0, 1.0])
    observer_days = torch.tensor([[0.0, 10.0, 20.0]])
    visit_mask = torch.ones(1, 3)
    logits = torch.full((1, 3, 1, 2, 9), -8.0)
    # A peak at observer day 20 gives phases -10, -5 and 0 at z=1.
    for visit, phase_index in enumerate((2, 3, 4)):
        logits[0, visit, 0, :, phase_index] = 8.0

    _, raw = module(
        logits,
        observer_days,
        visit_mask,
        redshift_grid,
        phase_grid,
        peak_day_offset=torch.tensor([20.0]),
        peak_date_valid=torch.tensor([1.0]),
    )

    assert raw[0, 0, 1] > raw[0, 0, 0] + 1.0


def test_uniform_phase_predictions_add_no_peak_date_evidence() -> None:
    module = CandidatePhaseConsistency(
        initial_scale=1.0,
        use_peak_date=True,
        peak_uncertainty_days=10.0,
        peak_quadrature_points=5,
        peak_outlier_fraction=0.1,
    )
    logits = torch.zeros(2, 4, 3, 5, 9)
    observer_days = torch.tensor(
        [[0.0, 8.0, 17.0, 28.0], [0.0, 5.0, 11.0, 19.0]]
    )

    scaled, raw = module(
        logits,
        observer_days,
        torch.ones(2, 4),
        torch.linspace(0.0, 2.0, 5),
        torch.linspace(-20.0, 20.0, 9),
        peak_day_offset=torch.tensor([12.0, 15.0]),
        peak_date_valid=torch.ones(2),
    )

    assert torch.allclose(raw, torch.zeros_like(raw), atol=1.0e-6)
    assert torch.allclose(scaled, torch.zeros_like(scaled), atol=1.0e-6)


def test_invalid_peak_date_makes_the_route_abstain() -> None:
    module = CandidatePhaseConsistency(
        initial_scale=1.0,
        use_peak_date=True,
    )
    logits = torch.randn(2, 3, 2, 4, 7)
    scaled, raw = module(
        logits,
        torch.tensor([[0.0, 10.0, 20.0], [0.0, 5.0, 12.0]]),
        torch.ones(2, 3),
        torch.linspace(0.0, 2.0, 4),
        torch.linspace(-15.0, 15.0, 7),
        peak_day_offset=torch.zeros(2),
        peak_date_valid=torch.zeros(2),
    )

    assert torch.equal(raw, torch.zeros_like(raw))
    assert torch.equal(scaled, torch.zeros_like(scaled))


def test_peak_date_and_observer_date_share_one_frame() -> None:
    module = CandidatePhaseConsistency(
        initial_scale=0.5,
        use_peak_date=True,
        peak_uncertainty_days=5.0,
        peak_quadrature_points=3,
    )
    phase_logits = torch.randn(2, 4, 2, 3, 15)
    observer_days = torch.tensor(
        [[0.0, 5.0, 15.0, 25.0], [3.0, 8.0, 18.0, 28.0]]
    )
    peak = torch.tensor([11.0, 14.0])
    arguments = (
        torch.ones(2, 4),
        torch.tensor([0.5, 1.0, 1.5]),
        torch.linspace(-20.0, 50.0, 15),
    )
    original = module(
        phase_logits,
        observer_days,
        *arguments,
        peak_day_offset=peak,
        peak_date_valid=torch.ones(2),
    )
    shifted = module(
        phase_logits,
        observer_days + 120.0,
        *arguments,
        peak_day_offset=peak + 120.0,
        peak_date_valid=torch.ones(2),
    )

    assert torch.allclose(original[0], shifted[0], atol=1.0e-5)
    assert torch.allclose(original[1], shifted[1], atol=1.0e-5)
