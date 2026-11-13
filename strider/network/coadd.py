"""The inverse-variance coadd of an object's visits, and the per-pixel weights used in matching."""

import math

import torch

__all__ = ["inverse_variance_coadd", "relative_inverse_variance", "edge_taper"]


def inverse_variance_coadd(flux, mask, log_error, scale):
    """Coadd visits on one wavelength grid, weighting each pixel by its inverse variance.

    Parameters
    ----------
    flux, log_error : torch.Tensor
        (visits, pixels): each visit's flux over its visit scale, and the log of its error over the same scale.
    mask : torch.Tensor
        (visits, pixels), 1 where a pixel is measured.
    scale : torch.Tensor
        (visits,) visit scales.

    Returns
    -------
    flux, error, mask : torch.Tensor
        (pixels,) the coadd and its error, in units of the visits' geometric-mean scale, and where it is measured.
    """
    measured = mask.bool() & torch.isfinite(flux) & torch.isfinite(log_error)
    log_scale = torch.log(scale)
    log_ratio = log_scale - log_scale.sum() / len(log_scale)
    flux = torch.where(measured, flux * torch.exp(log_ratio)[:, None], torch.zeros_like(flux))
    log_error = log_error + log_ratio[:, None]
    # precision relative to the most precise visit at each pixel, so no sum overflows
    log_precision = torch.where(measured, -2.0 * log_error, torch.full_like(log_error, -torch.inf))
    reference = log_precision.max(dim=0).values
    covered = torch.isfinite(reference)
    reference = torch.where(covered, reference, torch.zeros_like(reference))
    precision = torch.where(measured, torch.exp(log_precision - reference), torch.zeros_like(log_precision))
    total = precision.sum(0)
    tiny = torch.finfo(flux.dtype).tiny
    coadd = (precision * flux).sum(0) / total.clamp_min(tiny)
    error = torch.exp(-0.5 * reference) / torch.sqrt(total.clamp_min(tiny))
    covered &= total > 0.0
    zero = torch.zeros_like(coadd)
    return torch.where(covered, coadd, zero), torch.where(covered, error, zero), covered.to(flux.dtype)


def relative_inverse_variance(error, mask):
    """Each pixel's inverse variance relative to its spectrum's median, 0 where unmeasured; a spectrum with no
    usable error is weighted uniformly over its mask."""
    measured = mask.bool()
    valid = measured & torch.isfinite(error) & (error > 0.0)
    log_error = torch.where(valid, torch.log(error.clamp_min(torch.finfo(error.dtype).tiny)), torch.nan)
    median = torch.nanmedian(log_error, dim=-1).values
    usable = valid.any(dim=-1) & torch.isfinite(median)
    # the clamp only guards the exponential: errors at template edges can be 1e9 times the median
    precision = torch.exp((-2.0 * (log_error - median[..., None])).clamp(-60.0, 60.0))
    precision = torch.where(valid, precision, torch.zeros_like(precision))
    return torch.where(usable[..., None], precision, measured.to(error.dtype))


def edge_taper(pixels, fraction):
    """A weight rising from 0 to 1 as a cosine over ``fraction`` of the grid at each end, 1 elsewhere."""
    coordinate = torch.linspace(0.0, 1.0, pixels)
    if fraction == 0.0:
        return torch.ones_like(coordinate)
    phase = (torch.minimum(coordinate, 1.0 - coordinate) / fraction).clamp(0.0, 1.0)
    return 0.5 - 0.5 * torch.cos(math.pi * phase)
