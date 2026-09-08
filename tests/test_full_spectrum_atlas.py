from pathlib import Path

import numpy as np
import torch

from strider.atlas.full_spectrum import (
    CandidateRestFrameScan,
    FullSpectrumAtlas,
    PhaseIndexedAtlas,
    best_measured_visit,
    build_phase_indexed_atlas,
    combine_view_scores,
    measurement_faithful_coadd,
    phase_sequence_match,
    score_atlas_view,
)


def _profile(grid: np.ndarray, centers: tuple[float, ...]) -> np.ndarray:
    values = np.zeros_like(grid, dtype=np.float32)
    for center in centers:
        values += np.exp(-0.5 * ((np.log(grid) - np.log(center)) / 0.018) ** 2)
    return values.astype(np.float32)


def test_measurement_faithful_study_coadd_uses_reported_errors() -> None:
    flux = torch.tensor([[1.0, 3.0], [3.0, 5.0]])
    mask = torch.ones_like(flux)
    log_error = torch.log(torch.tensor([[1.0, 1.0], [2.0, 2.0]]))
    scale = torch.ones(2)

    coadd, error, support = measurement_faithful_coadd(
        flux,
        mask,
        log_error,
        scale,
        maximum_relative_error=3.0,
        edge_trim_fraction=0.0,
    )

    assert np.allclose(coadd, [1.4, 3.4])
    assert np.allclose(error, np.sqrt(0.8))
    assert support.tolist() == [True, True]


def test_best_measured_visit_uses_signed_median_signal_to_noise() -> None:
    flux = torch.tensor([[1.0, 1.0, 1.0], [4.0, 4.0, 4.0]])
    mask = torch.ones_like(flux)
    log_error = torch.log(torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))

    index, selected, error, support, signal_to_noise = best_measured_visit(
        flux,
        mask,
        log_error,
        edge_trim_fraction=0.0,
    )

    assert index == 1
    assert np.allclose(selected, 4.0)
    assert np.allclose(error, 2.0)
    assert support.all()
    assert signal_to_noise == 2.0


def test_candidate_scan_recovers_class_and_redshift() -> None:
    rest = np.geomspace(2500.0, 10000.0, 256).astype(np.float32)
    observed = np.geomspace(7500.0, 18175.0, 512).astype(np.float32)
    redshift = np.linspace(0.5, 2.5, 161).astype(np.float32)
    class_zero = _profile(rest, (4100.0, 5300.0, 6800.0))
    class_one = _profile(rest, (4550.0, 6100.0, 7900.0))
    truth = 1.2
    measured = np.interp(
        observed / (1.0 + truth),
        rest,
        class_zero,
        left=0.0,
        right=0.0,
    ).astype(np.float32)
    mask = measured != 0.0

    atlas = FullSpectrumAtlas(
        class_names=("zero", "one"),
        rest_wavelength=rest,
        whole_profiles=np.stack([class_zero, class_one])[:, None, :],
        whole_masks=np.ones((2, 1, len(rest)), dtype=bool),
        detail_profiles=np.stack([class_zero, class_one])[:, None, :],
        detail_masks=np.ones((2, 1, len(rest)), dtype=bool),
        support_counts=np.ones((2, 1), dtype=np.int64),
    )
    atlas.validate()
    aligned, aligned_mask = CandidateRestFrameScan(
        observed, rest, redshift
    ).align(measured, mask)
    score, support = score_atlas_view(
        aligned,
        aligned_mask,
        atlas.whole_profiles,
        atlas.whole_masks,
        atlas.support_counts,
        minimum_rest_fraction=0.1,
    )
    candidate, predicted_class = np.unravel_index(np.argmax(score), score.shape)

    assert support[candidate, predicted_class]
    assert predicted_class == 0
    assert abs(float(redshift[candidate]) - truth) < 0.03


def test_combined_view_requires_support_from_both_views() -> None:
    whole = np.asarray([[0.8, 0.5], [0.4, 0.3]], dtype=np.float32)
    detail = np.asarray([[0.6, 0.7], [0.2, 0.1]], dtype=np.float32)
    whole_support = np.ones_like(whole, dtype=bool)
    detail_support = np.asarray([[True, False], [True, True]])

    combined, support = combine_view_scores(
        whole,
        whole_support,
        detail,
        detail_support,
        detail_fraction=0.25,
    )

    assert np.isclose(combined[0, 0], 0.75)
    assert np.isneginf(combined[0, 1])
    assert support.tolist() == [[True, False], [True, True]]


def test_atlas_round_trip(tmp_path: Path) -> None:
    rest = np.geomspace(2500.0, 10000.0, 8).astype(np.float32)
    values = np.arange(16, dtype=np.float32).reshape(2, 1, 8)
    atlas = FullSpectrumAtlas(
        class_names=("Ia", "other"),
        rest_wavelength=rest,
        whole_profiles=values,
        whole_masks=np.ones_like(values, dtype=bool),
        detail_profiles=values / 2.0,
        detail_masks=np.ones_like(values, dtype=bool),
        support_counts=np.ones((2, 1), dtype=np.int64),
    )

    loaded = FullSpectrumAtlas.load(atlas.save(tmp_path / "atlas.npz"))

    assert loaded.class_names == atlas.class_names
    assert np.array_equal(loaded.rest_wavelength, rest)
    assert np.array_equal(loaded.whole_profiles, values)


def test_phase_indexed_atlas_build_and_round_trip(tmp_path: Path) -> None:
    rest = np.geomspace(2500.0, 10000.0, 12).astype(np.float32)
    rows = np.stack(
        [
            _profile(rest, (4000.0 + 200.0 * index, 6500.0))
            for index in range(8)
        ]
    )
    mask = np.ones_like(rows, dtype=bool)
    atlas = build_phase_indexed_atlas(
        rows,
        mask,
        rows,
        mask,
        class_index=np.asarray([0, 0, 0, 0, 1, 1, 1, 1]),
        phase_index=np.asarray([0, 0, 1, 1, 0, 0, 1, 1]),
        class_names=("Ia", "other"),
        rest_wavelength=rest,
        phase_edges_days=np.asarray([-20.0, 0.0, 20.0]),
        prototype_count=2,
    )

    loaded = PhaseIndexedAtlas.load(atlas.save(tmp_path / "phase_atlas.npz"))

    assert loaded.whole_profiles.shape == (2, 2, 2, 12)
    assert np.all(loaded.support_counts.sum(axis=2) == 2)
    assert np.array_equal(loaded.phase_edges_days, [-20.0, 0.0, 20.0])


def test_phase_averaging_rewards_a_consistent_visit_trajectory() -> None:
    score = np.zeros((3, 2, 1, 4), dtype=np.float32)
    support = np.ones_like(score, dtype=bool)
    # At the first redshift, each visit independently prefers the same phase;
    # elapsed time makes that sequence inconsistent. At the second redshift the
    # preferred phases follow a valid 0, 10, 20 day trajectory.
    score[:, 0, 0, 1] = 1.0
    score[0, 1, 0, 1] = 1.0
    score[1, 1, 0, 1] = 1.0
    score[2, 1, 0, 2] = 1.0
    phase_edges = np.asarray([-20.0, 0.0, 20.0, 40.0, 80.0])
    redshift = np.asarray([0.0, 2.0])
    observer_days = np.asarray([0.0, 30.0, 60.0])

    phase_independent, _ = phase_sequence_match(
        score,
        support,
        observer_days,
        redshift,
        phase_edges,
        mode="phase_independent",
    )
    averaged, _ = phase_sequence_match(
        score,
        support,
        observer_days,
        redshift,
        phase_edges,
        mode="phase_averaged",
        starting_phase_grid=np.asarray([-20.0, 0.0, 20.0]),
    )
    truth_upper_bound, _ = phase_sequence_match(
        score,
        support,
        observer_days,
        redshift,
        phase_edges,
        mode="truth_phase_upper_bound",
        truth_starting_phase=0.0,
    )

    assert np.isclose(phase_independent[0, 0], phase_independent[1, 0])
    assert averaged[1, 0] > averaged[0, 0]
    assert truth_upper_bound[1, 0] > truth_upper_bound[0, 0]
