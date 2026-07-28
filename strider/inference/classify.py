"""Single-object and batch inference."""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np
import torch


from strider.inference.timeseries import (
    SpectralEpoch,
    build_strider_inputs_from_spectra,
)
from strider.inference.metadata import InferenceMetadata


def _normalize_state_dict_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Normalize checkpoint keys from compiled and uncompiled PyTorch models."""
    return {k.removeprefix('_orig_mod.'): v for k, v in state.items()}


def _metadata_from_model(model) -> InferenceMetadata:
    """Read metadata attached by the model loader."""
    metadata = getattr(model, "inference_metadata", None)
    if isinstance(metadata, InferenceMetadata):
        return metadata
    raise ValueError("Model is missing inference metadata")


def _with_derived_class_outputs(result: dict, class_names: list[str]) -> dict:
    """Attach the class label and strict P(Ia) alongside the raw probabilities."""
    probs = np.asarray(result["class_probs"], dtype=float)
    result["class_names"] = list(class_names)
    result["pred_class"] = class_names[int(np.argmax(probs))]
    result["p_ia"] = float(probs[class_names.index("Ia")] if "Ia" in class_names else probs[0])
    return result


def _class_names_for_result(model: Any, result: dict) -> list[str]:
    n_classes = int(np.asarray(result["class_probs"]).shape[-1])
    metadata = getattr(model, "inference_metadata", None)
    if isinstance(metadata, InferenceMetadata) and len(metadata.class_names) == n_classes:
        return list(metadata.class_names)
    scheme = getattr(getattr(model, "config", None), "scheme", None)
    class_names = list(getattr(scheme, "class_names", []) or [])
    if len(class_names) == n_classes:
        return class_names
    return [f"class_{idx}" for idx in range(n_classes)]


def _log_from_probs(probs: torch.Tensor) -> torch.Tensor:
    tiny = torch.finfo(probs.dtype).tiny
    return torch.log(probs.clamp_min(tiny))


def _posterior_tensors_from_output(out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return normalized class, z, and joint posterior tensors across families."""
    if "class_log_probs" in out:
        class_log_probs = out["class_log_probs"]
        class_probs = class_log_probs.exp()
    elif "class_probs" in out:
        class_probs = out["class_probs"]
        class_log_probs = _log_from_probs(class_probs)
    else:
        raise KeyError("model output must contain class_probs or class_log_probs")

    if "z_log_posterior" in out:
        z_log_posterior = out["z_log_posterior"]
        z_posterior = z_log_posterior.exp()
    elif "z_posterior" in out:
        z_posterior = out["z_posterior"]
        z_log_posterior = _log_from_probs(z_posterior)
    else:
        raise KeyError("model output must contain z_posterior or z_log_posterior")

    if "joint_log_probs" in out:
        joint_log_probs = out["joint_log_probs"]
    elif "joint_log_posterior" in out:
        joint_log_probs = out["joint_log_posterior"]
    else:
        joint_log_probs = class_log_probs.unsqueeze(-1) + z_log_posterior.unsqueeze(1)

    return {
        "class_probs": class_probs,
        "class_log_probs": class_log_probs,
        "z_posterior": z_posterior,
        "z_log_posterior": z_log_posterior,
        "joint_log_probs": joint_log_probs,
    }


def _normalize_joint_log_probs(joint_log_probs: torch.Tensor) -> torch.Tensor:
    return joint_log_probs - torch.logsumexp(
        joint_log_probs.reshape(joint_log_probs.shape[0], -1),
        dim=-1,
    ).reshape(-1, 1, 1)


def _posterior_tensors_from_joint(joint_log_probs: torch.Tensor) -> dict[str, torch.Tensor]:
    joint_log_probs = _normalize_joint_log_probs(joint_log_probs)
    class_log_probs = torch.logsumexp(joint_log_probs, dim=2)
    z_log_posterior = torch.logsumexp(joint_log_probs, dim=1)
    return {
        "class_probs": class_log_probs.exp(),
        "class_log_probs": class_log_probs,
        "z_posterior": z_log_posterior.exp(),
        "z_log_posterior": z_log_posterior,
        "joint_log_probs": joint_log_probs,
    }


def _z_grid_for_posterior(
    metadata: InferenceMetadata,
    posterior: dict[str, torch.Tensor],
) -> torch.Tensor | None:
    z_posterior = posterior["z_posterior"]
    if metadata.z_grid.size != z_posterior.shape[-1]:
        return None
    return torch.as_tensor(
        metadata.z_grid,
        dtype=z_posterior.dtype,
        device=z_posterior.device,
    )


def _posterior_z_pred(
    posterior: dict[str, torch.Tensor],
    z_grid: torch.Tensor | None,
    out: dict[str, torch.Tensor] | None = None,
) -> float:
    if z_grid is not None:
        return float((posterior["z_posterior"][0] * z_grid).sum().item())
    if out is not None and "z_pred" in out:
        return float(out["z_pred"][0].item())
    return float("nan")


def _as_posterior(post: np.ndarray, is_log: bool) -> tuple[np.ndarray, bool]:
    values = np.asarray(post, dtype=float)
    squeeze = values.ndim == 1
    if squeeze:
        values = values[None, :]
    probabilities = (
        np.exp(values - values.max(axis=1, keepdims=True))
        if is_log
        else values
    )
    probabilities /= probabilities.sum(axis=1, keepdims=True).clip(min=1e-300)
    return probabilities, squeeze


def peak_curvature_sigma(post, z_grid, *, is_log: bool = False):
    """Return the dominant redshift peak and its local curvature width."""
    probabilities, squeeze = _as_posterior(post, is_log)
    z_grid = np.asarray(z_grid, dtype=float)
    logp = np.log(np.clip(probabilities, 1e-300, None))
    n_objects, n_grid = probabilities.shape
    peak = np.clip(np.argmax(logp, axis=1), 1, n_grid - 2)
    rows = np.arange(n_objects)
    y0, y1, y2 = logp[rows, peak - 1], logp[rows, peak], logp[rows, peak + 1]
    x0, x1, x2 = z_grid[peak - 1], z_grid[peak], z_grid[peak + 1]
    h0, h1 = x1 - x0, x2 - x1
    curvature = 2.0 * ((y2 - y1) / h1 - (y1 - y0) / h0) / (h0 + h1)
    spacing = 0.5 * (h0 + h1)
    sigma = np.where(curvature < 0, 1.0 / np.sqrt(np.abs(curvature)), spacing)
    denominator = y0 - 2.0 * y1 + y2
    offset = np.where(
        denominator < 0,
        0.5 * spacing * (y0 - y2) / denominator,
        0.0,
    )
    z_peak = x1 + np.clip(offset, -spacing, spacing)
    if squeeze:
        return float(z_peak[0]), float(sigma[0])
    return z_peak, sigma


def interp_posterior_quantile(
    probs: "np.ndarray", z_grid: "np.ndarray", q: float,
) -> float:
    """Continuous inverse-CDF of a binned redshift posterior.

    Uses the mid-point (Hazen) plotting position — each bin's probability is
    assigned to its CENTER, so the interpolation CDF is ``cumsum - 0.5*p``.
    This removes the z-grid quantization floor (Δz ~ grid spacing) while staying
    UNBIASED: a posterior concentrated in one bin returns that bin's z exactly,
    and a uniform posterior returns the true centre (not half a bin low, which
    is what naive CDF-snapping gives). Works for non-uniform grids (log1p).

    Every STRIDER quantile goes through here, so the quoted z, its p16/p84
    bounds, and the raw/calibrated variants all use one convention.
    """
    probs = np.asarray(probs, dtype=float)
    z_grid = np.asarray(z_grid, dtype=float)
    mid_cdf = np.cumsum(probs) - 0.5 * probs
    return float(np.interp(q, mid_cdf, z_grid))


def detect_redshift_peaks(
    probs: "np.ndarray", z_grid: "np.ndarray", *,
    min_ratio: float = 0.1, min_sep: float = 0.05,
) -> dict[str, float | int]:
    """Multi-peak / alias structure of a redshift posterior.

    Finds the local maxima of the marginal, merges peaks closer than ``min_sep``
    in ``dz/(1+z)`` (keeping the taller), and reports the significant ones
    (height >= ``min_ratio`` x the primary). A strong SECONDARY peak is the
    signature of a competing redshift alias — useful for evidence-map trust and
    for flagging objects whose confident-looking median hides a rival z.

    Returns ``z_n_peaks``, ``z_primary``, ``z_secondary`` (nan if single-peak),
    ``z_secondary_ratio`` (secondary/primary height, 0 if single-peak).
    """
    p = np.asarray(probs, dtype=float)
    z = np.asarray(z_grid, dtype=float)
    total = p.sum()
    if total <= 0 or not np.isfinite(total):
        return {"z_n_peaks": 0, "z_primary": float("nan"),
                "z_secondary": float("nan"), "z_secondary_ratio": 0.0}
    p = p / total
    # Interior local maxima, plus endpoint maxima: a high-z pile-up alias peaks
    # at the last z-bin, and must not be dropped just because an interior bump
    # also exists.
    cand = list(np.where((p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:]))[0] + 1)
    n = p.size
    if n >= 2 and p[0] >= p[1]:
        cand.append(0)
    if n >= 2 and p[-1] > p[-2]:
        cand.append(n - 1)
    if not cand:
        cand = [int(np.argmax(p))]
    idx = np.array(cand)
    idx = idx[np.argsort(p[idx])[::-1]]                  # tallest first
    # greedy merge of peaks closer than min_sep in dz/(1+z); symmetric midpoint
    # denominator so the merge boundary is purely geometric (not height-ordered).
    kept: list[int] = []
    for i in idx:
        if all(abs(z[i] - z[j]) / (1.0 + 0.5 * (z[i] + z[j])) >= min_sep
               for j in kept):
            kept.append(int(i))
    primary_h = p[kept[0]]
    sig = [j for j in kept if p[j] >= min_ratio * primary_h]
    z_sec = float(z[sig[1]]) if len(sig) >= 2 else float("nan")
    ratio = float(p[sig[1]] / primary_h) if len(sig) >= 2 else 0.0
    return {"z_n_peaks": len(sig), "z_primary": float(z[sig[0]]),
            "z_secondary": z_sec, "z_secondary_ratio": ratio}


def summarize_redshift_posterior(
    probs: "np.ndarray",
    z_grid: "np.ndarray",
    *,
    clean_width_norm: float = 0.02,
    ambiguous_split_norm: float = 0.005,
    ambiguous_map_split_norm: float = 0.01,
    ambiguous_secondary_ratio: float = 0.3,
) -> dict[str, float | str]:
    """Characterize a redshift posterior: robust estimate + interval + quality.

    Returns the canonical z-estimation contract:

    * ``z_median`` — the QUOTED point estimate. Robust to tails and, on a
      multimodal posterior, stays in the heavier mode instead of landing in the
      empty valley between modes (where the mean goes). Now continuous/unbiased.
    * ``z_mean`` / ``z_map`` — diagnostics (expectation / peak).
    * ``z_p16`` / ``z_p84`` and ``z_interval_width_norm`` — the credible interval,
      normalized by ``(1+z)``.
    * ``z_skew_norm`` = ``|mean - median| / (1+z)`` — a grid-robust skew /
      bimodality indicator (mean and median diverge when the posterior is
      skewed or multimodal; they agree for a clean unimodal posterior).
    * ``z_quality`` — ``clean`` / ``broad`` / ``ambiguous`` / ``invalid``. Lets a
      cosmology/PV user CUT broken posteriors rather than trust a point value:
        - ``ambiguous``: ``z_skew_norm >= ambiguous_split_norm`` (skewed/multimodal)
        - ``clean``: unimodal and ``z_interval_width_norm <= clean_width_norm``
        - ``broad``: unimodal but wide

    The default thresholds are tied to STRIDER's NMAD scale (~0.005): a split
    above ~NMAD flags a non-trivially-skewed posterior; a normalized 16-84
    width above ~4xNMAD flags a broad one. Both are tunable.
    """
    probs = np.asarray(probs, dtype=float).ravel()
    z_grid = np.asarray(z_grid, dtype=float).ravel()
    total = probs.sum()
    if probs.size != z_grid.size or not np.isfinite(total) or total <= 0:
        nan = float("nan")
        return {
            "z_median": nan, "z_mean": nan, "z_map": nan,
            "z_peak": nan, "z_sigma_peak": nan, "z_ambiguity": nan,
            "z_p16": nan, "z_p84": nan, "z_interval_width": nan,
            "z_interval_width_norm": nan, "z_skew_norm": nan,
            "z_map_split_norm": nan, "z_n_peaks": 0,
            "z_secondary": nan, "z_secondary_ratio": 0.0,
            "z_quality": "invalid",
        }
    p = probs / total
    z_mean = float((p * z_grid).sum())
    z_map = float(z_grid[int(np.argmax(p))])
    z_median = interp_posterior_quantile(p, z_grid, 0.50)
    z_p16 = interp_posterior_quantile(p, z_grid, 0.16)
    z_p84 = interp_posterior_quantile(p, z_grid, 0.84)
    # Laplace 1-sigma of the dominant mode (redshift PRECISION); the ratio of the
    # marginal interval to 2*sigma_peak exposes alias tails beyond the peak.
    z_peak, z_sigma_peak = peak_curvature_sigma(p, z_grid, is_log=False)

    one_plus = 1.0 + z_median
    width = z_p84 - z_p16
    width_norm = width / one_plus
    z_ambiguity = float(width / (2.0 * z_sigma_peak)) if z_sigma_peak > 0 else float("inf")
    skew_norm = abs(z_mean - z_median) / one_plus
    # MAP-vs-median split catches SYMMETRIC bimodal posteriors that skew_norm
    # misses: there mean==median (in the valley) but the peak sits on a mode,
    # so |map - median| is large. (Above grid-quantization noise ~half a bin.)
    map_split_norm = abs(z_map - z_median) / one_plus
    # multi-peak structure: a strong secondary peak = a competing z-alias.
    peaks = detect_redshift_peaks(p, z_grid, min_ratio=0.1)
    sec_ratio = float(peaks["z_secondary_ratio"])

    if (skew_norm >= ambiguous_split_norm
            or map_split_norm >= ambiguous_map_split_norm
            or sec_ratio >= ambiguous_secondary_ratio):
        quality = "ambiguous"
    elif width_norm <= clean_width_norm:
        quality = "clean"
    else:
        quality = "broad"

    return {
        "z_median": z_median, "z_mean": z_mean, "z_map": z_map,
        "z_peak": float(z_peak), "z_sigma_peak": float(z_sigma_peak),
        "z_ambiguity": z_ambiguity,
        "z_p16": z_p16, "z_p84": z_p84,
        "z_interval_width": width, "z_interval_width_norm": width_norm,
        "z_skew_norm": skew_norm, "z_map_split_norm": map_split_norm,
        "z_n_peaks": peaks["z_n_peaks"], "z_secondary": peaks["z_secondary"],
        "z_secondary_ratio": sec_ratio, "z_quality": quality,
    }


def _posterior_z_summaries(
    posterior: dict[str, torch.Tensor],
    z_grid: torch.Tensor | None,
    out: dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    if z_grid is None:
        z_pred = _posterior_z_pred(posterior, z_grid, out)
        return {
            "z_mean": z_pred,
            "z_median": float("nan"),
            "z_map": z_pred,
            "z_p16": float("nan"),
            "z_p84": float("nan"),
        }

    probs = posterior["z_posterior"][0]
    z_mean = float((probs * z_grid).sum().item())
    z_map = float(z_grid[int(torch.argmax(probs).item())].item())
    # Continuous, unbiased quantiles via the shared mid-point inverse-CDF.
    probs_np = probs.detach().cpu().numpy()
    z_grid_np = z_grid.detach().cpu().numpy()

    def quantile(q: float) -> float:
        return interp_posterior_quantile(probs_np, z_grid_np, q)

    return {
        "z_mean": z_mean,
        "z_median": quantile(0.50),
        "z_map": z_map,
        "z_p16": quantile(0.16),
        "z_p84": quantile(0.84),
    }


def _apply_posthoc_z_restrictions(
    posterior: dict[str, torch.Tensor],
    z_grid: torch.Tensor | None,
    *,
    z_prior: float | None,
    z_prior_sigma: float | None,
    z_window: tuple[float, float] | None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    info: dict[str, Any] = {
        "applied": False,
        "z_window_valid": True,
        "z_window_effective": None,
        "z_prior_applied": z_prior is not None,
    }
    if z_prior is None and z_window is None:
        return posterior, info
    if z_grid is None:
        warnings.warn(
            "z prior/window requested but model z grid is unavailable; returning raw posterior",
            RuntimeWarning,
            stacklevel=2,
        )
        info["z_window_valid"] = z_window is None
        return posterior, info

    joint = posterior["joint_log_probs"].clone()
    if z_window is not None:
        lo, hi = float(z_window[0]), float(z_window[1])
        if hi < lo:
            lo, hi = hi, lo
        grid_min = float(z_grid[0].item())
        grid_max = float(z_grid[-1].item())
        eff_lo = max(lo, grid_min)
        eff_hi = min(hi, grid_max)
        in_window = (z_grid >= eff_lo) & (z_grid <= eff_hi)
        if eff_hi < eff_lo or not bool(in_window.any().item()):
            warnings.warn(
                f"z_window={z_window} has no overlap with model z grid "
                f"[{grid_min:.4f}, {grid_max:.4f}]; returning raw posterior",
                RuntimeWarning,
                stacklevel=2,
            )
            info["z_window_valid"] = False
            return posterior, info
        info["z_window_effective"] = (eff_lo, eff_hi)
        joint = joint.masked_fill(~in_window.reshape(1, 1, -1), -torch.inf)
        info["applied"] = True

    if z_prior is not None:
        sigma = float(z_prior_sigma if z_prior_sigma is not None else 0.05)
        if sigma <= 0:
            raise ValueError(f"z_prior_sigma must be positive, got {sigma}")
        prior_log = -0.5 * ((z_grid - float(z_prior)) / sigma) ** 2
        joint = joint + prior_log.reshape(1, 1, -1)
        info["applied"] = True

    return _posterior_tensors_from_joint(joint), info


@torch.no_grad()
def classify_spectral_timeseries(
    model: Any,
    spectra: list,
    phases: np.ndarray | None = None,
    *,
    normalization: str | None = None,
    z_prior: float | None = None,
    z_prior_sigma: float | None = None,
    z_window: tuple[float, float] | None = None,
    peak_normalize: bool = False,
    return_prepared: bool = False,
    device: str = 'cpu',
) -> dict:
    """Classify arbitrary wavelength/flux spectra using the shared STRIDER contract.

    `spectra` may be a list of `(wave, flux)`, `(wave, flux, flux_err)`,
    `(wave, flux, flux_err, phase)`, mappings with `wave`/`flux`/optional
    `flux_err`/`phase`, or `SpectralEpoch` objects. If `phases` is supplied,
    it overrides per-spectrum phase values. Phase-only positional tuples are
    not supported because scalar third items are valid constant-error models.
    Wavelengths may be Angstrom or micron-like.
    """
    metadata = _metadata_from_model(model)
    if normalization is None:
        normalization = metadata.normalization
    patch_size = metadata.patch_size
    n_channels = metadata.n_channels
    prepared = build_strider_inputs_from_spectra(
        spectra,
        phases=phases,
        normalization=normalization,
        z_prior_phys=None,
        z_prior_sigma_phys=z_prior_sigma,
        z_prior_valid=False,
        patch_size=patch_size,
        n_channels=n_channels,
        peak_normalize=peak_normalize,
        device=device,
    )

    out = model(**prepared.inputs)
    posterior_raw = _posterior_tensors_from_output(out)
    z_grid = _z_grid_for_posterior(metadata, posterior_raw)
    posterior_current, restriction_info = _apply_posthoc_z_restrictions(
        posterior_raw,
        z_grid,
        z_prior=z_prior,
        z_prior_sigma=z_prior_sigma,
        z_window=z_window,
    )
    class_probs = posterior_current["class_probs"]
    z_posterior = posterior_current["z_posterior"]
    z_summary = _posterior_z_summaries(posterior_current, z_grid, out)
    # Use the posterior median as the default point estimate. It is less
    # sensitive than the mean to long low-probability redshift tails while still
    # using the full marginalized posterior.
    z_pred = z_summary["z_median"]
    # Quality characterization (ambiguity flag) so callers can CUT broken
    # posteriors for a PV-grade sample. z_quality in {clean, broad, ambiguous,
    # invalid}; 'ambiguous' marks skewed/multimodal posteriors (the catastrophic
    # outlier mode). See summarize_redshift_posterior.
    if z_grid is not None and z_posterior is not None:
        _zq = summarize_redshift_posterior(
            z_posterior[0].detach().cpu().numpy(),
            z_grid.detach().cpu().numpy(),
        )
    else:
        _zq = {"z_quality": "invalid", "z_interval_width_norm": float("nan"),
               "z_skew_norm": float("nan"), "z_map_split_norm": float("nan"),
               "z_n_peaks": 0, "z_secondary": float("nan"),
               "z_secondary_ratio": 0.0}
    result = {
        'class_probs': class_probs[0].cpu().numpy(),
        'fused_class_probs': out.get('fused_class_probs', class_probs)[0].cpu().numpy(),
        'joint_class_probs': out.get('joint_class_probs', class_probs)[0].cpu().numpy(),
        'z_pred': z_pred,
        'z_mean': z_summary["z_mean"],
        'z_median': z_summary["z_median"],
        'z_map': z_summary["z_map"],
        'z_p16': z_summary["z_p16"],
        'z_p84': z_summary["z_p84"],
        'z_quality': _zq["z_quality"],
        'z_interval_width_norm': _zq["z_interval_width_norm"],
        'z_skew_norm': _zq["z_skew_norm"],
        'z_map_split_norm': _zq["z_map_split_norm"],
        'z_n_peaks': _zq["z_n_peaks"],
        'z_secondary': _zq["z_secondary"],
        'z_secondary_ratio': _zq["z_secondary_ratio"],
        'z_posterior': z_posterior[0].cpu().numpy(),
        'z_log_posterior': posterior_current["z_log_posterior"][0].cpu().numpy(),
        'joint_log_probs': posterior_current["joint_log_probs"][0].cpu().numpy(),
        'alpha': float(out['alpha'][0].item()) if 'alpha' in out else float('nan'),
        'n_epochs': prepared.n_epochs,
        'wave_mask': prepared.wave_mask,
        'processing_summary': dict(prepared.processing_summary),
    }
    restrictions_requested = z_window is not None or z_prior is not None
    if restrictions_requested:
        result.update({
            "joint_log_probs_raw": posterior_raw["joint_log_probs"][0].cpu().numpy(),
            "class_log_probs_raw": posterior_raw["class_log_probs"][0].cpu().numpy(),
            "z_log_posterior_raw": posterior_raw["z_log_posterior"][0].cpu().numpy(),
            "z_window_valid": bool(restriction_info["z_window_valid"]),
            "z_prior_applied": bool(restriction_info["z_prior_applied"]),
        })
        if restriction_info["applied"]:
            result.update({
                "joint_log_probs_masked": posterior_current["joint_log_probs"][0].cpu().numpy(),
                "class_log_probs_masked": posterior_current["class_log_probs"][0].cpu().numpy(),
                "z_log_posterior_masked": posterior_current["z_log_posterior"][0].cpu().numpy(),
            })
        if restriction_info["z_window_effective"] is not None:
            result["z_window_effective"] = tuple(float(x) for x in restriction_info["z_window_effective"])
    if return_prepared:
        result['prepared'] = prepared
    return _with_derived_class_outputs(result, _class_names_for_result(model, result))


def classify_spectrum(
    model: Any,
    wave: np.ndarray,
    flux: np.ndarray,
    *,
    flux_err: np.ndarray | None = None,
    phase: float | None = None,
    normalization: str | None = None,
    z_prior: float | None = None,
    z_prior_sigma: float | None = None,
    z_window: tuple[float, float] | None = None,
    peak_normalize: bool = False,
    return_prepared: bool = False,
    device: str = 'cpu',
) -> dict:
    """Classify one raw observed-frame spectrum.

    This is the one-spectrum convenience surface. The phase must be explicit
    because the model's prototypes are phase-aware. Use `phase=0.0` only when
    intentionally treating the spectrum as near peak; callers with real
    time-series information should use `classify_spectral_timeseries`.
    """
    return classify_spectral_timeseries(
        model,
        [SpectralEpoch(wave=wave, flux=flux, flux_err=flux_err, phase=phase)],
        normalization=normalization,
        z_prior=z_prior,
        z_prior_sigma=z_prior_sigma,
        z_window=z_window,
        peak_normalize=peak_normalize,
        return_prepared=return_prepared,
        device=device,
    )
