"""Shared spectral time-series preprocessing for STRIDER inference.

Whatever the spectra came from, they are resampled onto the STRIDER
wavelength grid and packed into model inputs here, so every caller reaches
the model through the same path.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any
import warnings

import numpy as np
import torch

from strider.constants import (
    LOG_WAVE_MIN,
    LOG_WAVE_RANGE,
    MAX_EPOCHS,
    N_WAVE,
    PHASE_NORM,
    WAVE_MAX,
    WAVE_MIN,
    Z_NORM,
)


VALID_NORMALIZATIONS = {"per_epoch", "per_object"}


@dataclass(frozen=True)
class ResampledSpectrum:
    flux: np.ndarray
    wave_mask: np.ndarray
    flux_err: np.ndarray | None = None


@dataclass(frozen=True)
class SpectralEpoch:
    """One observed spectrum before STRIDER-grid resampling.

    `phase` is in rest-frame days relative to peak. Public inference should
    pass it explicitly; phase-aware prototypes make silent phase defaults
    scientifically risky.
    """

    wave: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray | None = None
    phase: float | None = None


@dataclass(frozen=True)
class PreparedTimeSeries:
    inputs: dict[str, torch.Tensor]
    normed_flux: np.ndarray
    diff_flux: np.ndarray
    reliability: np.ndarray | None
    wave_mask: np.ndarray | None
    phases: np.ndarray
    n_epochs: int
    processing_summary: dict[str, int | float]


def strider_wave_grid_aa() -> np.ndarray:
    """Return the STRIDER observed-frame wavelength grid in Angstrom."""
    log_wave = np.linspace(LOG_WAVE_MIN, LOG_WAVE_MIN + LOG_WAVE_RANGE, N_WAVE)
    return np.exp(log_wave).astype(np.float32)


STRIDER_WAVE_AA = strider_wave_grid_aa()


def wave_grid_frac(patch_size: int = 8) -> np.ndarray:
    """Return per-patch wavelength position fractions for the backbone."""
    if N_WAVE % patch_size != 0:
        raise ValueError(f"patch_size={patch_size} must divide N_WAVE={N_WAVE}")
    n_patches = N_WAVE // patch_size
    log_wave = np.linspace(LOG_WAVE_MIN, LOG_WAVE_MIN + LOG_WAVE_RANGE, N_WAVE)
    frac = (log_wave - LOG_WAVE_MIN) / LOG_WAVE_RANGE
    return frac.reshape(n_patches, patch_size).mean(axis=1).astype(np.float32)


def ensure_wave_aa(wave: np.ndarray) -> np.ndarray:
    """Convert micron-like wavelength arrays to Angstrom, otherwise pass through."""
    wave = np.asarray(wave, dtype=np.float32)
    if wave.size == 0:
        return wave
    return wave * 1.0e4 if np.nanmax(wave) < 100 else wave


def resample_to_strider_grid(
    wave_aa: np.ndarray,
    flux: np.ndarray,
    *,
    flux_err: np.ndarray | None = None,
    peak_normalize: bool = False,
    edge_margin: float = 0.05,
) -> ResampledSpectrum | None:
    """Resample one spectrum to the STRIDER grid and return its coverage mask.

    `peak_normalize` is intended for explicit ablations or visual diagnostics,
    not the default classification route.
    """
    wave_aa = ensure_wave_aa(wave_aa)
    flux = np.asarray(flux, dtype=np.float32)
    err = np.asarray(flux_err, dtype=np.float32) if flux_err is not None else None
    if err is not None and err.ndim == 0:
        err = np.full_like(flux, float(err), dtype=np.float32)
    valid = np.isfinite(wave_aa) & np.isfinite(flux)
    if err is not None:
        if err.shape != flux.shape:
            raise ValueError(f"flux_err shape {err.shape} does not match flux shape {flux.shape}")
    valid &= (wave_aa >= WAVE_MIN * (1.0 - edge_margin))
    valid &= (wave_aa <= WAVE_MAX * (1.0 + edge_margin))
    if valid.sum() < 10:
        return None

    wave_valid = wave_aa[valid]
    flux_valid = flux[valid]
    err_valid = err[valid] if err is not None else None
    order = np.argsort(wave_valid)
    wave_valid = wave_valid[order]
    flux_valid = flux_valid[order]
    if err_valid is not None:
        err_valid = err_valid[order]

    resampled = np.interp(STRIDER_WAVE_AA, wave_valid, flux_valid, left=0.0, right=0.0)
    wave_mask = (STRIDER_WAVE_AA >= wave_valid.min()) & (STRIDER_WAVE_AA <= wave_valid.max())
    err_resampled = None
    if err_valid is not None:
        err_resampled = np.zeros(N_WAVE, dtype=np.float32)
        good_err = np.isfinite(err_valid) & (err_valid > 0)
        if good_err.sum() >= 2:
            wave_err = wave_valid[good_err]
            sigma = err_valid[good_err]
            pos = np.searchsorted(wave_err, STRIDER_WAVE_AA[wave_mask], side="left")
            left = np.clip(pos - 1, 0, len(wave_err) - 1)
            right = np.clip(pos, 0, len(wave_err) - 1)
            err_resampled[wave_mask] = np.maximum(sigma[left], sigma[right]).astype(np.float32)
        elif good_err.sum() == 1:
            err_resampled[wave_mask] = float(err_valid[good_err][0])
    if peak_normalize:
        peak = np.abs(resampled[wave_mask]).max() if wave_mask.any() else np.abs(resampled).max()
        if peak > 0:
            resampled = resampled / peak
            if err_resampled is not None:
                err_resampled = err_resampled / peak
    return ResampledSpectrum(resampled.astype(np.float32), wave_mask.astype(np.bool_), err_resampled)


def _coerce_spectral_epoch(
    spec: SpectralEpoch | Mapping[str, Any] | Sequence[Any],
) -> SpectralEpoch:
    """Accept convenient public input forms and return `SpectralEpoch`.

    Supported forms:
      - SpectralEpoch(wave, flux, flux_err=None, phase=None)
      - {"wave": ..., "flux": ..., "flux_err": optional, "phase": optional}
      - (wave, flux)
      - (wave, flux, flux_err)
      - (wave, flux, flux_err, phase)

    Phase in positional form is intentionally not supported because scalar
    third items are valid constant-error models. Use a dict or SpectralEpoch
    when supplying phase without flux_err.
    """
    if isinstance(spec, SpectralEpoch):
        return spec
    if isinstance(spec, Mapping):
        if "wave" not in spec or "flux" not in spec:
            raise ValueError("spectral epoch mappings must contain 'wave' and 'flux'")
        flux_err = spec.get("flux_err", spec.get("error", spec.get("err")))
        phase = spec.get("phase")
        return SpectralEpoch(spec["wave"], spec["flux"], flux_err, phase)
    if not isinstance(spec, Sequence) or isinstance(spec, (str, bytes)):
        raise ValueError(
            "each spectrum must be a SpectralEpoch, mapping, or tuple/list"
        )
    if len(spec) == 2:
        wave, flux = spec
        return SpectralEpoch(wave, flux)
    if len(spec) == 3:
        wave, flux, flux_err = spec
        return SpectralEpoch(wave, flux, flux_err=flux_err)
    if len(spec) == 4:
        wave, flux, flux_err, phase = spec
        return SpectralEpoch(wave, flux, flux_err=flux_err, phase=float(phase))
    raise ValueError(
        "each spectrum must be (wave, flux), (wave, flux, flux_err), "
        "or (wave, flux, flux_err, phase)"
    )


def validate_flux_and_phases(flux: np.ndarray, phases: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    flux = np.asarray(flux, dtype=np.float32)
    phases = np.asarray(phases, dtype=np.float32)
    if flux.ndim != 2 or flux.shape[1] != N_WAVE:
        raise ValueError(f"flux must have shape (n_epochs, {N_WAVE}), got {flux.shape}")
    if phases.ndim != 1:
        raise ValueError(f"phases must be 1D, got shape {phases.shape}")
    n_epochs = int(min(len(phases), flux.shape[0], MAX_EPOCHS))
    if n_epochs <= 0:
        raise ValueError("at least one epoch is required")
    if min(len(phases), flux.shape[0]) > MAX_EPOCHS:
        warnings.warn(
            f"{min(len(phases), flux.shape[0])} epochs supplied but the model was "
            f"trained on at most {MAX_EPOCHS}; using the first {MAX_EPOCHS}. "
            "Select the epochs you want rather than relying on this cut.",
            RuntimeWarning,
            stacklevel=3,
        )
    return flux, phases, n_epochs


def standardize_flux_err(flux_err: np.ndarray | None, n_epochs: int) -> np.ndarray | None:
    if flux_err is None:
        return None
    err = np.asarray(flux_err, dtype=np.float32)
    if err.ndim != 2 or err.shape[1] != N_WAVE or err.shape[0] < n_epochs:
        raise ValueError(
            f"flux_err must have shape (>= {n_epochs}, {N_WAVE}), got {err.shape}"
        )
    return err[:n_epochs].copy()


def standardize_wave_mask(wave_mask: np.ndarray | None, n_epochs: int) -> np.ndarray | None:
    if wave_mask is None:
        return None
    mask = np.asarray(wave_mask, dtype=np.bool_)
    if mask.shape == (N_WAVE,):
        mask = np.broadcast_to(mask, (n_epochs, N_WAVE)).copy()
    elif mask.shape == (n_epochs, N_WAVE):
        mask = mask.copy()
    else:
        raise ValueError(
            f"wave_mask must have shape ({N_WAVE},) or ({n_epochs}, {N_WAVE}), "
            f"got {mask.shape}"
        )
    if not mask.any():
        raise ValueError("wave_mask must contain at least one valid bin")
    if not mask.any(axis=1).all():
        raise ValueError("each real epoch in wave_mask must contain at least one valid bin")
    return mask


def normalize_flux_with_scale(
    flux: np.ndarray,
    n_epochs: int,
    *,
    wave_mask: np.ndarray | None = None,
    normalization: str = "per_object",
) -> tuple[np.ndarray, np.ndarray]:
    if normalization not in VALID_NORMALIZATIONS:
        raise ValueError(
            f"Unknown normalization={normalization!r}; expected one of {sorted(VALID_NORMALIZATIONS)}"
        )

    normed = np.zeros_like(flux, dtype=np.float32)
    scale = np.ones(n_epochs, dtype=np.float32)
    if normalization == "per_object":
        vals = flux[:n_epochs][wave_mask] if wave_mask is not None else flux[:n_epochs].ravel()
        mu = float(vals.mean())
        std = float(vals.std())
        if std > 0:
            normed[:n_epochs] = (flux[:n_epochs] - mu) / std
            scale[:] = std
        if wave_mask is not None:
            normed[:n_epochs][~wave_mask] = 0.0
        return normed, scale

    for t in range(n_epochs):
        vals = flux[t][wave_mask[t]] if wave_mask is not None else flux[t]
        mu = float(vals.mean())
        std = float(vals.std())
        if std > 0:
            normed[t] = (flux[t] - mu) / std
            scale[t] = std
        if wave_mask is not None:
            normed[t, ~wave_mask[t]] = 0.0
    return normed, scale


def build_diff_channel(normed_flux: np.ndarray, n_epochs: int, *, wave_mask: np.ndarray | None = None) -> np.ndarray:
    diff = np.zeros_like(normed_flux, dtype=np.float32)
    for t in range(1, n_epochs):
        diff[t] = normed_flux[t] - normed_flux[t - 1]
        if wave_mask is not None:
            diff[t, ~(wave_mask[t] & wave_mask[t - 1])] = 0.0
    return diff


def build_reliability_channel(
    flux_err: np.ndarray | None,
    n_epochs: int,
    *,
    wave_mask: np.ndarray | None,
    norm_scale: np.ndarray,
) -> np.ndarray:
    reliability = np.zeros((n_epochs, N_WAVE), dtype=np.float32)
    if flux_err is None:
        return reliability
    valid = np.isfinite(flux_err) & (flux_err > 0)
    if wave_mask is not None:
        valid &= wave_mask
    if not valid.any():
        return reliability
    scale_grid = np.broadcast_to(np.clip(norm_scale.astype(np.float32), 1e-20, None)[:, None], flux_err.shape)
    scaled_ivar = np.zeros_like(flux_err, dtype=np.float32)
    scaled_ivar[valid] = (scale_grid[valid] / flux_err[valid]) ** 2
    positive = valid & (scaled_ivar > 0) & np.isfinite(scaled_ivar)
    if not positive.any():
        return reliability
    log_ivar = np.zeros_like(flux_err, dtype=np.float32)
    log_ivar[positive] = np.log(scaled_ivar[positive])
    center = float(np.median(log_ivar[positive]))
    reliability[positive] = np.clip(log_ivar[positive] - center, -3.0, 3.0) / 3.0
    return reliability


def build_ivar_weight_channel(
    flux_err: np.ndarray | None,
    n_epochs: int,
    *,
    wave_mask: np.ndarray | None,
    norm_scale: np.ndarray,
) -> np.ndarray | None:
    """Per-bin IVW weight: 0 = exclude, positive = weight in IVW-NCC.

    Returns None when flux_err is absent so callers can omit the kwarg.
    """
    if flux_err is None:
        return None
    iw = np.zeros((n_epochs, N_WAVE), dtype=np.float32)
    valid = np.isfinite(flux_err) & (flux_err > 0)
    if wave_mask is not None:
        valid &= wave_mask
    if not valid.any():
        return iw
    scale_grid = np.broadcast_to(
        np.clip(norm_scale.astype(np.float32), 1e-20, None)[:, None],
        flux_err.shape,
    )
    raw_ivar = np.zeros_like(flux_err, dtype=np.float32)
    raw_ivar[valid] = (scale_grid[valid] / flux_err[valid]) ** 2
    positive = valid & (raw_ivar > 0) & np.isfinite(raw_ivar)
    if not positive.any():
        return iw
    log_ivar = np.log(raw_ivar[positive])
    center = float(np.median(log_ivar))
    iw[positive] = np.exp(np.clip(log_ivar - center, -3.0, 3.0))
    return iw


def build_strider_inputs(
    flux: np.ndarray,
    phases: np.ndarray,
    *,
    normalization: str = "per_object",
    flux_err: np.ndarray | None = None,
    wave_mask: np.ndarray | None = None,
    z_prior_phys: float | None = None,
    z_prior_sigma_phys: float | None = None,
    z_prior_valid: bool | None = None,
    z_window_phys: tuple[float, float] | None = None,
    z_window_valid: bool | None = None,
    include_invalid_prior: bool = False,
    include_invalid_window: bool = False,
    patch_size: int = 8,
    n_channels: int = 2,
    sort_by_phase: bool = True,
    device: str | torch.device = "cpu",
) -> PreparedTimeSeries:
    """Build model kwargs from STRIDER-grid raw flux and phase arrays."""
    flux, phases, n_epochs = validate_flux_and_phases(flux, phases)
    mask = standardize_wave_mask(wave_mask, n_epochs)
    err = standardize_flux_err(flux_err, n_epochs)
    if int(n_channels) not in {2, 3}:
        raise ValueError("n_channels must be 2 or 3")
    flux = flux[:n_epochs].copy()
    phases = phases[:n_epochs].copy()
    if sort_by_phase:
        order = np.argsort(phases, kind="stable")
        flux = flux[order]
        phases = phases[order]
        if mask is not None:
            mask = mask[order]
        if err is not None:
            err = err[order]

    normed, norm_scale = normalize_flux_with_scale(flux, n_epochs, wave_mask=mask, normalization=normalization)
    diff = build_diff_channel(normed, n_epochs, wave_mask=mask)
    reliability = (
        build_reliability_channel(err, n_epochs, wave_mask=mask, norm_scale=norm_scale)
        if int(n_channels) == 3 else None
    )
    ivar_weight = build_ivar_weight_channel(err, n_epochs, wave_mask=mask, norm_scale=norm_scale)

    # Sized to the epochs supplied; training padded to MAX_EPOCHS to batch.
    spectra = np.zeros((1, n_epochs, int(n_channels), N_WAVE), dtype=np.float32)
    spectra[0, :, 0] = normed[:n_epochs]
    spectra[0, :, 1] = diff[:n_epochs]
    if reliability is not None:
        spectra[0, :, 2] = reliability[:n_epochs]

    phase_arr = (phases[:n_epochs] / PHASE_NORM).astype(np.float32).reshape(1, n_epochs)

    epoch_mask = np.ones((1, n_epochs), dtype=np.bool_)

    inputs: dict[str, torch.Tensor] = {
        "spectra": torch.from_numpy(spectra).to(device),
        "phases": torch.from_numpy(phase_arr).to(device),
        "epoch_mask": torch.from_numpy(epoch_mask).to(device),
        "wave_grid_frac": torch.from_numpy(wave_grid_frac(patch_size)).unsqueeze(0).to(device),
    }
    if mask is not None:
        inputs["wave_mask"] = torch.from_numpy(
            mask[:n_epochs].reshape(1, n_epochs, N_WAVE).copy()
        ).to(device)
    if ivar_weight is not None:
        # Apply wave_mask zeroing one more time for consistency with the
        # training loader (which zeros augmented-masked bins in ivar_weight).
        if mask is not None:
            ivar_weight = ivar_weight * mask.astype(np.float32)
        inputs["ivar_weight"] = torch.from_numpy(
            ivar_weight[:n_epochs].reshape(1, n_epochs, N_WAVE).copy()
        ).to(device)

    prior_valid = (z_prior_phys is not None) if z_prior_valid is None else bool(z_prior_valid)
    if prior_valid:
        sigma = 0.05 if z_prior_sigma_phys is None else float(z_prior_sigma_phys)
        inputs["z_prior"] = torch.tensor(
            [[float(z_prior_phys) / Z_NORM, sigma / Z_NORM]],
            dtype=torch.float32,
            device=device,
        )
        inputs["z_prior_valid"] = torch.ones(1, dtype=torch.float32, device=device)
    elif include_invalid_prior:
        inputs["z_prior"] = torch.zeros(1, 2, dtype=torch.float32, device=device)
        inputs["z_prior_valid"] = torch.zeros(1, dtype=torch.float32, device=device)

    window_valid = (z_window_phys is not None) if z_window_valid is None else bool(z_window_valid)
    if window_valid:
        if z_window_phys is None:
            raise ValueError("z_window_phys is required when z_window_valid is true")
        lo, hi = z_window_phys
        inputs["z_window"] = torch.tensor(
            [[float(lo) / Z_NORM, float(hi) / Z_NORM]],
            dtype=torch.float32,
            device=device,
        )
        inputs["z_window_valid"] = torch.ones(1, dtype=torch.float32, device=device)
    elif include_invalid_window:
        inputs["z_window"] = torch.zeros(1, 2, dtype=torch.float32, device=device)
        inputs["z_window_valid"] = torch.zeros(1, dtype=torch.float32, device=device)

    if mask is None:
        valid_bins = int(n_epochs * N_WAVE)
    else:
        valid_bins = int(mask.sum())
    total_real_bins = int(n_epochs * N_WAVE)
    processing_summary = {
        "n_epochs_used": int(n_epochs),
        "n_resampled_bins": valid_bins,
        "effective_wave_coverage_pct": (
            100.0 * valid_bins / max(total_real_bins, 1)
        ),
        "n_masked_bins": int(total_real_bins - valid_bins),
    }

    return PreparedTimeSeries(
        inputs=inputs,
        normed_flux=normed,
        diff_flux=diff,
        reliability=reliability,
        wave_mask=mask,
        phases=phases,
        n_epochs=n_epochs,
        processing_summary=processing_summary,
    )


def build_strider_inputs_from_spectra(
    spectra: Sequence[SpectralEpoch | Mapping[str, Any] | Sequence[Any]],
    phases: Sequence[float] | np.ndarray | None = None,
    *,
    normalization: str = "per_object",
    z_prior_phys: float | None = None,
    z_prior_sigma_phys: float | None = None,
    z_prior_valid: bool | None = None,
    z_window_phys: tuple[float, float] | None = None,
    z_window_valid: bool | None = None,
    include_invalid_prior: bool = False,
    include_invalid_window: bool = False,
    patch_size: int = 8,
    n_channels: int = 2,
    peak_normalize: bool = False,
    sort_by_phase: bool = True,
    device: str | torch.device = "cpu",
) -> PreparedTimeSeries:
    """Build model kwargs from raw wavelength/flux spectra.

    This is the public-friendly wrapper around the STRIDER-grid contract.
    `spectra` can be a list of mappings/dataclasses/tuples; see
    `_coerce_spectral_epoch` for accepted forms. If `phases` is supplied, it
    overrides any per-spectrum phase fields. If phases are absent, every
    spectrum must carry an explicit phase in rest-frame days.
    """
    if len(spectra) == 0:
        raise ValueError("at least one spectrum is required")

    epochs = [_coerce_spectral_epoch(spec) for spec in spectra[:MAX_EPOCHS]]
    n_epochs = len(epochs)
    if phases is not None:
        phase_arr = np.asarray(phases, dtype=np.float32)
        if phase_arr.ndim != 1:
            raise ValueError(f"phases must be 1D, got shape {phase_arr.shape}")
        if len(phase_arr) < n_epochs:
            raise ValueError(
                f"phases has length {len(phase_arr)} but {n_epochs} spectra were provided"
            )
        phase_arr = phase_arr[:n_epochs]
    else:
        phase_values: list[float] = []
        missing = []
        for i, epoch in enumerate(epochs):
            if epoch.phase is None:
                missing.append(i)
                phase_values.append(np.nan)
            else:
                phase_values.append(float(epoch.phase))
        if missing:
            raise ValueError(
                "STRIDER inference requires phase in rest-frame days. "
                "Pass phase=N for each spectrum (dict/SpectralEpoch or "
                "separate phases= array); "
                f"missing phase for epoch indices {missing}"
            )
        phase_arr = np.asarray(phase_values, dtype=np.float32)

    flux_rows = []
    err_rows = []
    mask_rows = []
    have_err = False
    for epoch in epochs:
        resampled = resample_to_strider_grid(
            epoch.wave,
            epoch.flux,
            flux_err=epoch.flux_err,
            peak_normalize=peak_normalize,
        )
        if resampled is None:
            raise ValueError(
                "spectrum has too few finite samples in the STRIDER wavelength range"
            )
        flux_rows.append(resampled.flux)
        mask_rows.append(resampled.wave_mask)
        if resampled.flux_err is not None:
            have_err = True
            err_rows.append(resampled.flux_err)
        else:
            err_rows.append(np.zeros_like(resampled.flux))

    return build_strider_inputs(
        np.asarray(flux_rows, dtype=np.float32),
        phase_arr,
        normalization=normalization,
        flux_err=np.asarray(err_rows, dtype=np.float32) if have_err else None,
        wave_mask=np.asarray(mask_rows, dtype=np.bool_),
        z_prior_phys=z_prior_phys,
        z_prior_sigma_phys=z_prior_sigma_phys,
        z_prior_valid=z_prior_valid,
        z_window_phys=z_window_phys,
        z_window_valid=z_window_valid,
        include_invalid_prior=include_invalid_prior,
        include_invalid_window=include_invalid_window,
        patch_size=patch_size,
        n_channels=n_channels,
        sort_by_phase=sort_by_phase,
        device=device,
    )


