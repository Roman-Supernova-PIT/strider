"""Numerical contracts for the measurement-faithful coadd route."""

from __future__ import annotations

import math

import torch

from strider.model.coadd import (
    cosine_edge_taper,
    cumulative_inverse_variance_coadd,
    final_equal_weight_coadd,
    final_inverse_variance_coadd,
    minimum_relative_precision_mask,
    relative_inverse_variance,
)
from strider.model.spectral_tokens import normalize_valid_bins


def _scaled_measurement(
    physical_flux: torch.Tensor,
    physical_error: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        physical_flux / scale[..., None],
        torch.log(physical_error / scale[..., None]),
    )


def test_unequal_error_coadd_matches_hand_calculation() -> None:
    physical_flux = torch.tensor([[[10.0, 20.0], [14.0, 8.0]]])
    physical_error = torch.tensor([[[1.0, 2.0], [2.0, 1.0]]])
    scale = torch.tensor([[2.0, 4.0]])
    flux, log_error = _scaled_measurement(physical_flux, physical_error, scale)

    coadd, error, mask = final_inverse_variance_coadd(
        flux,
        torch.ones_like(flux),
        torch.ones(1, 2),
        log_error,
        scale,
    )

    object_scale = math.sqrt(8.0)
    expected_flux = torch.tensor(
        [[(10.0 + 0.25 * 14.0) / 1.25, (0.25 * 20.0 + 8.0) / 1.25]]
    ) / object_scale
    expected_error = torch.full((1, 2), math.sqrt(1.0 / 1.25) / object_scale)
    assert torch.allclose(coadd, expected_flux, atol=1.0e-6)
    assert torch.allclose(error, expected_error, atol=1.0e-6)
    assert torch.equal(mask, torch.ones_like(mask))


def test_missing_and_padded_measurements_never_contribute() -> None:
    physical_flux = torch.tensor([[[2.0, 4.0], [1000.0, 8.0], [9000.0, 9000.0]]])
    physical_error = torch.ones_like(physical_flux)
    scale = torch.ones(1, 3)
    flux, log_error = _scaled_measurement(physical_flux, physical_error, scale)
    wavelength_mask = torch.tensor([[[1.0, 1.0], [0.0, 1.0], [1.0, 1.0]]])
    visit_mask = torch.tensor([[1.0, 1.0, 0.0]])

    coadd, error, mask = final_inverse_variance_coadd(
        flux,
        wavelength_mask,
        visit_mask,
        log_error,
        scale,
    )

    assert torch.allclose(coadd, torch.tensor([[2.0, 6.0]]))
    assert torch.allclose(error, torch.tensor([[1.0, math.sqrt(0.5)]]))
    assert torch.equal(mask, torch.ones_like(mask))


def test_nonfinite_padded_values_cannot_poison_valid_coadd() -> None:
    flux = torch.tensor([[[2.0, 4.0], [float("nan"), float("inf")]]])
    log_error = torch.tensor([[[0.0, 0.0], [float("nan"), float("nan")]]])
    scale = torch.tensor([[1.0, float("nan")]])

    coadd, error, mask = final_inverse_variance_coadd(
        flux,
        torch.ones_like(flux),
        torch.tensor([[1.0, 0.0]]),
        log_error,
        scale,
    )

    assert torch.allclose(coadd, torch.tensor([[2.0, 4.0]]))
    assert torch.allclose(error, torch.ones(1, 2))
    assert torch.equal(mask, torch.ones_like(mask))


def test_normalized_coadd_is_invariant_to_preprocessing_scales() -> None:
    physical_flux = torch.tensor(
        [[[1.0, 2.0, 5.0, 3.0], [2.0, 1.0, 4.0, 6.0]]]
    )
    physical_error = torch.tensor(
        [[[1.0, 2.0, 1.0, 3.0], [2.0, 1.0, 2.0, 1.0]]]
    )
    first_scale = torch.tensor([[1.0, 1.0]])
    second_scale = torch.tensor([[0.125, 32.0]])
    mask = torch.ones_like(physical_flux)
    visits = torch.ones(1, 2)

    first_flux, first_log_error = _scaled_measurement(
        physical_flux, physical_error, first_scale
    )
    second_flux, second_log_error = _scaled_measurement(
        physical_flux, physical_error, second_scale
    )
    first, _, first_mask = final_inverse_variance_coadd(
        first_flux, mask, visits, first_log_error, first_scale
    )
    second, _, second_mask = final_inverse_variance_coadd(
        second_flux, mask, visits, second_log_error, second_scale
    )

    assert torch.equal(first_mask, second_mask)
    assert torch.allclose(
        normalize_valid_bins(first, first_mask),
        normalize_valid_bins(second, second_mask),
        atol=1.0e-6,
    )


def test_relative_error_quality_rule_masks_only_bad_bins() -> None:
    physical_flux = torch.ones(1, 1, 3)
    physical_error = torch.tensor([[[1.0, 2.0, 100.0]]])
    scale = torch.ones(1, 1)
    flux, log_error = _scaled_measurement(physical_flux, physical_error, scale)

    coadd, error, mask = final_inverse_variance_coadd(
        flux,
        torch.ones_like(flux),
        torch.ones(1, 1),
        log_error,
        scale,
        maximum_relative_error=3.0,
    )

    assert torch.equal(mask, torch.tensor([[1.0, 1.0, 0.0]]))
    assert torch.allclose(coadd, torch.tensor([[1.0, 1.0, 0.0]]))
    assert torch.allclose(error, torch.tensor([[1.0, 2.0, 0.0]]))


def test_no_quality_limit_retains_high_error_measurements() -> None:
    physical_flux = torch.ones(1, 1, 3)
    physical_error = torch.tensor([[[1.0, 2.0, 100.0]]])
    scale = torch.ones(1, 1)
    flux, log_error = _scaled_measurement(physical_flux, physical_error, scale)

    coadd, error, mask = final_inverse_variance_coadd(
        flux,
        torch.ones_like(flux),
        torch.ones(1, 1),
        log_error,
        scale,
        maximum_relative_error=None,
    )

    assert torch.equal(mask, torch.ones_like(mask))
    assert torch.allclose(coadd, physical_flux[:, 0])
    assert torch.allclose(error, physical_error[:, 0])


def test_relative_inverse_variance_is_continuous_and_scale_free() -> None:
    error = torch.tensor([[1.0, 2.0, 8.0, 0.0]])
    mask = torch.ones_like(error)

    first = relative_inverse_variance(error, mask)
    second = relative_inverse_variance(17.0 * error, mask)

    assert torch.allclose(first, torch.tensor([[4.0, 1.0, 0.0625, 0.0]]))
    assert torch.allclose(second, first)


def test_relative_inverse_variance_preserves_extreme_error_dynamic_range() -> None:
    error = torch.tensor([[1.0, 1.0, 1.0, 1.0e8]])
    mask = torch.ones_like(error)

    weight = relative_inverse_variance(error, mask)

    assert weight[0, -1] > 0.0
    assert torch.allclose(weight[0, -1], torch.tensor(1.0e-16), rtol=1.0e-5)


def test_relative_inverse_variance_falls_back_without_errors() -> None:
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    weight = relative_inverse_variance(torch.zeros_like(mask), mask)

    assert torch.equal(weight, mask)


def test_minimum_relative_precision_mask_removes_only_numerically_unresolved_bins() -> None:
    reliability = torch.tensor([[1.0, 1.0e-4, 1.0e-16, 0.0]])
    mask = torch.ones_like(reliability)

    resolved = minimum_relative_precision_mask(
        reliability,
        mask,
        torch.finfo(torch.float32).eps,
    )

    assert torch.equal(
        resolved,
        torch.tensor([[True, True, False, False]]),
    )


def test_zero_minimum_relative_precision_preserves_measured_support() -> None:
    reliability = torch.tensor([[1.0, 0.0, float("nan")]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])

    resolved = minimum_relative_precision_mask(reliability, mask, 0.0)

    assert torch.equal(resolved, mask.bool())


def test_cosine_edge_taper_preserves_interior_and_is_symmetric() -> None:
    taper = cosine_edge_taper(21, 0.20)

    assert taper.shape == (21,)
    assert taper[0] == 0.0
    assert taper[-1] == 0.0
    assert torch.allclose(taper, taper.flip(0), atol=1.0e-7)
    assert torch.allclose(taper[4:17], torch.ones(13), atol=1.0e-7)


def test_zero_edge_taper_keeps_every_wavelength_bin() -> None:
    assert torch.equal(cosine_edge_taper(9, 0.0), torch.ones(9))


def test_cumulative_coadd_uses_only_each_observed_prefix() -> None:
    physical_flux = torch.tensor([[[2.0], [6.0], [1000.0]]])
    physical_error = torch.ones_like(physical_flux)
    scale = torch.ones(1, 3)
    flux, log_error = _scaled_measurement(physical_flux, physical_error, scale)

    coadd, error, mask = cumulative_inverse_variance_coadd(
        flux,
        torch.ones_like(flux),
        torch.tensor([[1.0, 1.0, 0.0]]),
        log_error,
        scale,
    )

    assert torch.allclose(coadd[:, :, 0], torch.tensor([[2.0, 4.0, 0.0]]))
    assert torch.allclose(
        error[:, :, 0], torch.tensor([[1.0, math.sqrt(0.5), 0.0]])
    )
    assert torch.equal(mask[:, :, 0], torch.tensor([[1.0, 1.0, 0.0]]))


def test_equal_weight_coadd_works_without_reported_errors() -> None:
    physical_flux = torch.tensor(
        [[[2.0, 4.0, 8.0], [6.0, 20.0, 10.0], [1000.0, 1000.0, 1000.0]]]
    )
    scale = torch.tensor([[2.0, 5.0, 10.0]])
    scaled_flux = physical_flux / scale[..., None]
    wavelength_mask = torch.tensor(
        [[[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]]
    )

    coadd, mask = final_equal_weight_coadd(
        scaled_flux,
        wavelength_mask,
        torch.tensor([[1.0, 1.0, 0.0]]),
        scale,
    )

    object_scale = math.sqrt(10.0)
    expected = torch.tensor([[(2.0 + 6.0) / 2.0, 4.0, (8.0 + 10.0) / 2.0]])
    assert torch.allclose(coadd, expected / object_scale, atol=1.0e-6)
    assert torch.equal(mask, torch.ones_like(mask))


def test_equal_weight_coadd_assumes_common_units_when_scale_is_absent() -> None:
    flux = torch.tensor([[[2.0, 4.0], [6.0, 8.0]]])

    coadd, mask = final_equal_weight_coadd(
        flux,
        torch.ones_like(flux),
        torch.ones(1, 2),
    )

    assert torch.allclose(coadd, torch.tensor([[4.0, 6.0]]))
    assert torch.equal(mask, torch.ones_like(mask))
