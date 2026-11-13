"""Comparing an observed spectrum with references: interpolation, normalisation and token similarity.

A comparison happens on the common support of the two spectra only, weighted by the observed pixels' relative
inverse variance.
"""

import numpy as np
import torch

__all__ = [
    "interpolation",
    "interpolate",
    "interpolate_error",
    "inverse_variance",
    "normalisation",
    "pool_weight",
    "token_cosine",
    "log_mean_exp",
]


def interpolation(source, target):
    """Linear interpolation from the increasing grid ``source`` onto ``target``: lower and upper indices, the
    upper weight and whether each target point lies inside the grid."""
    upper = np.searchsorted(source, target, side="left")
    valid = (upper > 0) & (upper < len(source))
    upper = np.clip(upper, 1, len(source) - 1)
    lower = upper - 1
    span = source[upper] - source[lower]
    weight = np.divide(target - source[lower], span, out=np.zeros_like(target), where=span > 0.0)
    return (
        torch.from_numpy(lower.astype(np.int64)),
        torch.from_numpy(upper.astype(np.int64)),
        torch.from_numpy(weight.astype(np.float32)),
        torch.from_numpy(valid.astype(np.float32)),
    )


def interpolate(values, mask, table):
    """Values and mask interpolated with a table from `interpolation`; a point is measured if both neighbours are."""
    lower, upper, weight, valid = table
    measured = mask[..., lower] * mask[..., upper] * valid
    return (values[..., lower] * (1.0 - weight) + values[..., upper] * weight) * measured, measured


def interpolate_error(error, mask, table):
    """Independent errors propagated through the same interpolation."""
    lower, upper, weight, valid = table
    variance = (error[..., lower] * (1.0 - weight)).square() + (error[..., upper] * weight).square()
    return torch.sqrt(variance.clamp_min(0.0)) * (mask[..., lower] * mask[..., upper] * valid)


def inverse_variance(error, mask):
    """1 / error² where measured, finite and positive, else 0; uniform over the mask where no error is usable."""
    valid = mask.bool() & torch.isfinite(error) & (error > 0.0)
    log_error = torch.log(error.clamp_min(torch.finfo(error.dtype).tiny))
    precision = torch.where(valid, torch.exp((-2.0 * log_error).clamp(-60.0, 60.0)), torch.zeros_like(error))
    return torch.where(valid.any(dim=-1, keepdim=True), precision, mask.to(error.dtype))


def normalisation(values, mask, weight):
    """The scale, then the weighted mean and standard deviation after scaling, of each spectrum (..., pixels)."""
    measured = mask.to(values.dtype)
    tiny = torch.finfo(values.dtype).tiny
    usable = torch.isfinite(weight) & (weight > 0.0)
    reliability = measured * torch.where(usable, weight, torch.zeros_like(weight))
    total = reliability.sum(-1, keepdim=True)
    weighted = (values.abs() * reliability).sum(-1, keepdim=True) / total.clamp_min(tiny)
    largest = (values.abs() * measured).amax(-1, keepdim=True)
    scale = torch.where(total > 0.0, weighted, largest).clamp_min(tiny)
    scaled = values / scale
    support = measured * weight.clamp_min(0.0)
    count = support.sum(-1, keepdim=True).clamp_min(tiny)
    mean = (scaled * support).sum(-1, keepdim=True) / count
    variance = ((scaled - mean).square() * support).sum(-1, keepdim=True) / count
    return scale, mean, torch.sqrt(variance.clamp_min(1e-6))


def pool_weight(weight, mask, tokens):
    """Pixel weights (N, pixels) averaged over the measured pixels of each token, (N, tokens)."""
    size = weight.shape[-1] // tokens
    measured = mask.reshape(*mask.shape[:-1], tokens, size).to(weight.dtype)
    grouped = weight.reshape(*weight.shape[:-1], tokens, size)
    return (grouped * measured).sum(-1) / measured.sum(-1).clamp_min(1.0)


def token_cosine(tokens, support, references, reference_support, supported, minimum_fraction, minimum_shared, weight):
    """Cosine similarity between observed tokens (N, T, d) and reference tokens (P, T, d) on their common support.

    The observed tokens are weighted by ``weight`` (N, T). A pair is valid if the reference is ``supported``, they
    share at least ``minimum_fraction`` of the tokens, and the shared tokens are at least ``minimum_shared`` of the
    smaller support.

    Returns
    -------
    similarity, valid : torch.Tensor
        (N, P), the similarity zero where not valid.
    """
    measured, reference_mask = support.float(), reference_support.float()
    reliability = measured * weight.clamp_min(0.0)
    references = references * reference_mask[..., None]
    # tokens are zero outside their support, so flattening tokens and channels gives the common-support sums
    numerator = (tokens * reliability[..., None]).flatten(1) @ references.flatten(1).T
    observed_energy = (tokens.square().sum(-1) * reliability) @ reference_mask.T
    reference_energy = reliability @ references.square().sum(-1).T
    denominator = torch.sqrt((observed_energy * reference_energy).clamp_min(1.0e-12))
    similarity = (numerator / denominator).clamp(-1.0, 1.0)
    shared = measured @ reference_mask.T
    smaller = torch.minimum(measured.sum(-1, keepdim=True), reference_mask.sum(-1)[None, :])
    valid = supported[None, :].expand_as(shared).clone()
    valid &= shared >= minimum_fraction * tokens.shape[-2]
    valid &= shared / smaller.clamp_min(1.0) >= minimum_shared
    valid &= denominator > 1.0e-8
    return torch.where(valid, similarity, torch.zeros_like(similarity)), valid


def log_mean_exp(values, support, temperature, dim=-1):
    """T log mean exp(v / T) over the supported entries along ``dim``, and whether any is supported (else 0)."""
    values = values.masked_fill(~support, 0.0)
    count = support.sum(dim=dim)
    scaled = (values / temperature).masked_fill(~support, -torch.inf)
    log_mean = torch.logsumexp(scaled, dim=dim) - torch.log(count.clamp_min(1).to(values.dtype))
    valid = count > 0
    log_mean = torch.where(valid, log_mean, torch.zeros_like(log_mean))
    score = temperature * log_mean
    return torch.where(valid, score, torch.zeros_like(score)), valid
