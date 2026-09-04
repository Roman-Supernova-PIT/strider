"""Measurement-faithful inverse-variance spectral coadds.

Dataset spectra are divided by a robust visit scale before entering the model.
That preprocessing is useful numerically, but spectra from different visits are
not in a common scale afterwards.  This module reverses the visit scaling,
combines flux using the uncertainty that belongs to each measurement, and only
then permits an object-wide normalization in the downstream spectral encoder.
"""

from __future__ import annotations

import math

import torch


def cosine_edge_taper(
    wavelength_bins: int,
    fraction_per_side: float,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a smooth log-grid edge weight without discarding valid bins.

    The weight rises from zero to one across ``fraction_per_side`` of each
    edge and is unity through the interior.  A zero fraction returns all ones.
    This is intended for spectral matching, not for changing measurement
    uncertainties or the inverse-variance coadd itself.
    """
    if wavelength_bins < 2:
        raise ValueError("A wavelength taper requires at least two bins")
    if not 0.0 <= fraction_per_side < 0.5:
        raise ValueError("Edge taper fraction must lie in [0, 0.5)")
    coordinate = torch.linspace(
        0.0,
        1.0,
        wavelength_bins,
        dtype=dtype,
        device=device,
    )
    if fraction_per_side == 0.0:
        return torch.ones_like(coordinate)
    distance = torch.minimum(coordinate, 1.0 - coordinate)
    phase = (distance / float(fraction_per_side)).clamp(0.0, 1.0)
    return 0.5 - 0.5 * torch.cos(math.pi * phase)


def relative_inverse_variance(
    error: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return continuous per-bin reliability from reported uncertainty.

    The absolute error scale is irrelevant for a scale-free spectral match, so
    precision is expressed relative to the median valid error of each spectrum.
    Every finite measured bin retains non-zero weight; a spectrum without a
    usable error vector falls back to uniform weight over its measured support.
    """
    if error.shape != mask.shape:
        raise ValueError("Error and wavelength mask must have the same shape")
    measured = mask.bool()
    valid = measured & torch.isfinite(error) & (error > 0.0)
    log_error = torch.where(
        valid,
        torch.log(error.clamp_min(torch.finfo(error.dtype).tiny)),
        torch.full_like(error, torch.nan),
    )
    median_log_error = torch.nanmedian(log_error, dim=-1).values
    has_error = valid.any(dim=-1) & torch.isfinite(median_log_error)
    relative_log_precision = -2.0 * (
        log_error - median_log_error[..., None]
    )
    # This bound is only a floating-point guard. Some SNANA template-support
    # edges carry errors eight or nine orders of magnitude above the interior,
    # so the guard must preserve a correspondingly tiny but non-zero weight.
    relative_precision = torch.exp(relative_log_precision.clamp(-60.0, 60.0))
    relative_precision = torch.where(
        valid, relative_precision, torch.zeros_like(relative_precision)
    )
    uniform = measured.to(error.dtype)
    return torch.where(has_error[..., None], relative_precision, uniform)


def minimum_relative_precision_mask(
    reliability: torch.Tensor,
    mask: torch.Tensor,
    minimum_relative_precision: float,
) -> torch.Tensor:
    """Exclude only measurements below a declared numerical precision floor.

    ``reliability`` is the scale-free relative inverse variance returned by
    :func:`relative_inverse_variance`. A zero floor preserves the original
    measured support exactly. A positive floor is intended for numerical
    limits such as the machine epsilon of the model dtype, not for selecting
    spectra by signal quality.
    """
    if reliability.shape != mask.shape:
        raise ValueError("Reliability and wavelength mask must have the same shape")
    if not 0.0 <= minimum_relative_precision < 1.0:
        raise ValueError("Minimum relative precision must lie in [0, 1)")
    measured = mask.bool()
    if minimum_relative_precision == 0.0:
        return measured
    return (
        measured
        & torch.isfinite(reliability)
        & (reliability >= float(minimum_relative_precision))
    )


def final_inverse_variance_coadd(
    flux: torch.Tensor,
    wavelength_mask: torch.Tensor,
    visit_mask: torch.Tensor,
    log_scaled_error: torch.Tensor,
    visit_flux_scale: torch.Tensor,
    *,
    maximum_relative_error: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the final coadded flux, propagated error, and quality mask.

    ``flux`` and ``log_scaled_error`` are the visit-scaled quantities emitted
    by :mod:`strider.data.dataset`. ``visit_flux_scale`` reverses that
    preprocessing.  A per-object geometric reference keeps the arithmetic in
    a safe numerical range; it is a common multiplicative factor and therefore
    cannot alter the inverse-variance weights.
    """
    _validate_inputs(
        flux,
        wavelength_mask,
        visit_mask,
        log_scaled_error,
        visit_flux_scale,
    )
    if maximum_relative_error is not None and maximum_relative_error <= 0.0:
        raise ValueError("maximum_relative_error must be positive")

    visit_valid = visit_mask.bool() & torch.isfinite(visit_flux_scale)
    visit_valid &= visit_flux_scale > 0.0
    safe_scale = visit_flux_scale.clamp_min(torch.finfo(flux.dtype).tiny)
    log_scale = torch.log(safe_scale)
    valid_count = visit_valid.sum(dim=1, keepdim=True).clamp_min(1)
    object_log_scale = torch.where(
        visit_valid,
        log_scale,
        torch.zeros_like(log_scale),
    ).sum(dim=1, keepdim=True) / valid_count
    log_scale_ratio = log_scale - object_log_scale
    scale_ratio = torch.exp(log_scale_ratio)

    common_flux = flux * scale_ratio[..., None]
    common_log_error = log_scaled_error + log_scale_ratio[..., None]
    measured = wavelength_mask.bool() & visit_valid[..., None]
    measured &= torch.isfinite(flux) & torch.isfinite(log_scaled_error)
    common_flux = torch.where(measured, common_flux, torch.zeros_like(common_flux))

    # Normalize precision independently at each wavelength before summing.
    # This preserves the exact weighted mean while avoiding overflow for FLAM
    # values and errors expressed in very small physical units.
    log_precision = -2.0 * common_log_error
    negative_infinity = torch.full_like(log_precision, -torch.inf)
    supported_log_precision = torch.where(
        measured, log_precision, negative_infinity
    )
    reference = supported_log_precision.max(dim=1).values
    has_measurement = torch.isfinite(reference)
    safe_reference = torch.where(
        has_measurement,
        reference,
        torch.zeros_like(reference),
    )
    relative_precision = torch.exp(
        supported_log_precision - safe_reference[:, None, :]
    )
    relative_precision = torch.where(
        measured, relative_precision, torch.zeros_like(relative_precision)
    )
    precision_sum = relative_precision.sum(dim=1)
    coadded_flux = (
        relative_precision * common_flux
    ).sum(dim=1) / precision_sum.clamp_min(torch.finfo(flux.dtype).tiny)
    coadded_error = torch.exp(-0.5 * safe_reference) / torch.sqrt(
        precision_sum.clamp_min(torch.finfo(flux.dtype).tiny)
    )
    coadded_mask = has_measurement & (precision_sum > 0.0)

    if maximum_relative_error is not None:
        masked_error = torch.where(
            coadded_mask,
            coadded_error,
            torch.full_like(coadded_error, torch.nan),
        )
        median_error = torch.nanmedian(masked_error, dim=-1).values
        usable_median = torch.isfinite(median_error) & (median_error > 0.0)
        quality = coadded_error <= (
            float(maximum_relative_error) * median_error[:, None]
        )
        coadded_mask &= usable_median[:, None] & quality

    coadded_flux = torch.where(
        coadded_mask, coadded_flux, torch.zeros_like(coadded_flux)
    )
    coadded_error = torch.where(
        coadded_mask, coadded_error, torch.zeros_like(coadded_error)
    )
    return (
        coadded_flux,
        coadded_error,
        coadded_mask.to(coadded_flux.dtype),
    )


def final_equal_weight_coadd(
    flux: torch.Tensor,
    wavelength_mask: torch.Tensor,
    visit_mask: torch.Tensor,
    visit_flux_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average measured visits when reported uncertainties are unavailable.

    If visit scale factors are supplied, they reverse STRIDER's numerical
    per-visit scaling before accumulation.  Otherwise the input spectra are
    assumed to already share one physical flux scale.  No uncertainty is
    estimated or implied by this fallback.
    """
    if flux.ndim != 3:
        raise ValueError("Coadd flux must have shape (objects, visits, wavelength)")
    if wavelength_mask.shape != flux.shape:
        raise ValueError("Coadd wavelength mask must match flux")
    if visit_mask.shape != flux.shape[:2]:
        raise ValueError("Coadd visit mask must match object and visit axes")
    if visit_flux_scale is None:
        visit_flux_scale = torch.ones_like(visit_mask, dtype=flux.dtype)
    if visit_flux_scale.shape != flux.shape[:2]:
        raise ValueError("Coadd visit scale must match object and visit axes")

    visit_valid = visit_mask.bool() & torch.isfinite(visit_flux_scale)
    visit_valid &= visit_flux_scale > 0.0
    safe_scale = visit_flux_scale.clamp_min(torch.finfo(flux.dtype).tiny)
    log_scale = torch.log(safe_scale)
    valid_count = visit_valid.sum(dim=1, keepdim=True).clamp_min(1)
    object_log_scale = torch.where(
        visit_valid,
        log_scale,
        torch.zeros_like(log_scale),
    ).sum(dim=1, keepdim=True) / valid_count
    scale_ratio = torch.exp(log_scale - object_log_scale)

    common_flux = flux * scale_ratio[..., None]
    measured = wavelength_mask.bool() & visit_valid[..., None]
    measured &= torch.isfinite(common_flux)
    weight = measured.to(common_flux.dtype)
    count = weight.sum(dim=1)
    coadded_mask = count > 0.0
    coadded_flux = (common_flux * weight).sum(dim=1)
    coadded_flux = coadded_flux / count.clamp_min(1.0)
    coadded_flux = torch.where(
        coadded_mask, coadded_flux, torch.zeros_like(coadded_flux)
    )
    return coadded_flux, coadded_mask.to(coadded_flux.dtype)


def cumulative_inverse_variance_coadd(
    flux: torch.Tensor,
    wavelength_mask: torch.Tensor,
    visit_mask: torch.Tensor,
    log_scaled_error: torch.Tensor,
    visit_flux_scale: torch.Tensor,
    *,
    maximum_relative_error: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the independently defined coadd after every visit prefix.

    Prefix ``j`` uses visits ``0..j`` only.  This deliberately separate helper
    is intended for accumulation diagnostics and animations; the production
    dense route calls :func:`final_inverse_variance_coadd` once on all retained
    visits and cannot silently substitute a future visit into an early prefix.
    """
    _validate_inputs(
        flux,
        wavelength_mask,
        visit_mask,
        log_scaled_error,
        visit_flux_scale,
    )
    prefix_flux: list[torch.Tensor] = []
    prefix_error: list[torch.Tensor] = []
    prefix_mask: list[torch.Tensor] = []
    for stop in range(1, flux.shape[1] + 1):
        coadd, error, mask = final_inverse_variance_coadd(
            flux[:, :stop],
            wavelength_mask[:, :stop],
            visit_mask[:, :stop],
            log_scaled_error[:, :stop],
            visit_flux_scale[:, :stop],
            maximum_relative_error=maximum_relative_error,
        )
        # Padded visit slots are not valid observation prefixes.
        prefix_exists = visit_mask[:, stop - 1].bool()[:, None]
        prefix_flux.append(torch.where(prefix_exists, coadd, torch.zeros_like(coadd)))
        prefix_error.append(torch.where(prefix_exists, error, torch.zeros_like(error)))
        prefix_mask.append(
            torch.where(prefix_exists, mask, torch.zeros_like(mask))
        )
    return (
        torch.stack(prefix_flux, dim=1),
        torch.stack(prefix_error, dim=1),
        torch.stack(prefix_mask, dim=1),
    )


def _validate_inputs(
    flux: torch.Tensor,
    wavelength_mask: torch.Tensor,
    visit_mask: torch.Tensor,
    log_scaled_error: torch.Tensor,
    visit_flux_scale: torch.Tensor,
) -> None:
    if flux.ndim != 3:
        raise ValueError("Coadd flux must have shape (objects, visits, wavelength)")
    if wavelength_mask.shape != flux.shape:
        raise ValueError("Coadd wavelength mask must match flux")
    if log_scaled_error.shape != flux.shape:
        raise ValueError("Coadd log error must match flux")
    if visit_mask.shape != flux.shape[:2]:
        raise ValueError("Coadd visit mask must match object and visit axes")
    if visit_flux_scale.shape != flux.shape[:2]:
        raise ValueError("Coadd visit scale must match object and visit axes")
