"""Running a trained STRIDER model.

    from strider import load_model

    model = load_model("models/strider.pt")
    out = model.classify(wavelength, flux, phase)

Two suffixes appear throughout the output keys:

``_spectra_only`` / ``_with_controls``
    Without and with any redshift prior or window, so the prior's effect
    stays auditable. Equal when no control was supplied.
``_uncal`` / ``_cal``
    Raw evidence, and the calibrated probability reported beside it. Labels
    and rankings always come from the raw evidence.

Redshift point estimates stay model-native. The calibrated intervals use a
frozen PIT map, and are skipped after a prior or window because the
calibration was not fitted for that case.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from strider.checkpoint import (
    CheckpointMetadata,
    load_checkpoint,
)


def _ia_subset_probs(
    p: np.ndarray, class_names: list[str], names: tuple[str, ...]
) -> float:
    """Sum probability over the named subset; missing names contribute zero."""
    idxs = [i for i, n in enumerate(class_names) if n in names]
    if not idxs:
        return 0.0
    return float(p[idxs].sum())


@dataclass
class STRIDER:
    """Trained STRIDER classifier with a single ``classify()`` entry point.

    Construct via :func:`strider.load_model`, not directly. The underlying
    model lives on ``self._model`` for advanced users.
    """

    _model: Any                              # underlying torch model
    metadata: CheckpointMetadata               # how to use it
    _device: str = "cpu"

    def classify(
        self,
        wavelength: np.ndarray | Sequence[float],
        flux: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
        phase: float | np.ndarray | Sequence[float],
        *,
        flux_err: np.ndarray | None = None,
        z_prior: tuple[float, float] | None = None,
        z_prior_mean: float | None = None,
        z_prior_sigma: float | None = None,
        z_window: tuple[float, float] | None = None,
        z_min: float | None = None,
        z_max: float | None = None,
        wave_range: tuple[float, float] | None = None,
        wave_min: float | None = None,
        wave_max: float | None = None,
        condition_class: str | None = None,
        wavelength_unit: str = "auto",
        calibrated: bool = True,
        return_joint: bool = False,
        return_inputs: bool = False,
    ) -> dict[str, Any]:
        """Classify one object.

        Returns the flat output keys plus the ``classification``, ``redshift``,
        ``controls`` and ``spectra`` blocks, which are easier to read.

        ``z_prior=(mean, sigma)``
            Soft Gaussian redshift prior.
        ``z_window=(z_min, z_max)``
            Hard redshift range.
        ``wave_range=(min, max)``
            Observed-frame wavelength cut in Angstrom, applied before
            preprocessing.
        ``condition_class="Ia"``
            Also report p(z | class) from that row of the joint evidence
            surface. Does not change the predicted class.
        ``return_joint=True``
            Include the full class × redshift joint log posterior, for
            plotting. Off by default to keep JSON output small.
        """
        wave_aa = _coerce_wavelength_aa(
            np.asarray(wavelength, dtype=np.float32),
            unit=wavelength_unit,
        )
        flux_arr = np.asarray(flux, dtype=np.float32)
        phases = np.atleast_1d(np.asarray(phase, dtype=np.float32))
        flux_err_arr = None if flux_err is None else np.asarray(flux_err, dtype=np.float32)

        z_prior_pair = _resolve_z_prior(
            z_prior=z_prior,
            z_prior_mean=z_prior_mean,
            z_prior_sigma=z_prior_sigma,
        )
        z_window_pair = _resolve_pair(
            name="z_window",
            pair=z_window,
            lo=z_min,
            hi=z_max,
        )
        wave_range_pair = _resolve_pair(
            name="wave_range",
            pair=wave_range,
            lo=wave_min,
            hi=wave_max,
        )

        original_wave_range = _finite_range(wave_aa)
        wave_used = wave_aa
        flux_used = flux_arr
        flux_err_used = flux_err_arr
        if wave_range_pair is not None:
            wave_used, flux_used, flux_err_used = _apply_wave_range(
                wave_aa,
                flux_arr,
                flux_err_arr,
                wave_range_pair,
            )

        if z_prior_pair is None:
            prior_mean = None
            prior_sigma = None
        else:
            prior_mean, prior_sigma = z_prior_pair

        out = self._predict(
            wavelength=wave_used,
            flux=flux_used,
            phase=phases,
            flux_err=flux_err_used,
            z_prior_mean=prior_mean,
            z_prior_sigma=prior_sigma,
            z_window=z_window_pair,
            condition_class=condition_class,
            calibrated=calibrated,
            return_joint=return_joint,
        )
        _add_friendly_output_blocks(
            out,
            metadata=self.metadata,
            calibrated=calibrated,
            z_prior=z_prior_pair,
            z_window=z_window_pair,
            wave_range=wave_range_pair,
            original_wave_range=original_wave_range,
            wave_used=wave_used,
            flux_used=flux_used,
            flux_err_used=flux_err_used,
            phases=phases,
            return_inputs=return_inputs,
        )
        return out

    def _predict(
        self,
        wavelength: np.ndarray | Sequence[float],
        flux: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
        phase: float | np.ndarray | Sequence[float],
        *,
        flux_err: np.ndarray | None = None,
        z_prior_mean: float | None = None,
        z_prior_sigma: float | None = None,
        z_window: tuple[float, float] | None = None,
        condition_class: str | None = None,
        calibrated: bool = True,
        return_joint: bool = False,
    ) -> dict[str, Any]:
        """Classify a single object's spectrum (one or multiple epochs).

        Parameters
        ----------
        wavelength
            Observer-frame wavelength grid in Å, shape ``(n_wave,)``. All
            epochs share this grid.
        flux
            Flux values. Shape ``(n_wave,)`` for a single-epoch spectrum or
            ``(n_epochs, n_wave)`` for a time series.
        phase
            Rest-frame days from B-max. Scalar for single-epoch or
            shape ``(n_epochs,)`` for a time series.
        flux_err
            Optional flux uncertainty, same shape as ``flux``. Used for
            inverse-variance weighting if provided.
        z_prior_mean, z_prior_sigma
            Optional Gaussian prior on redshift. **Both must be passed
            together**, otherwise a ``ValueError`` is raised — no magic
            default ``sigma`` is applied. When passed, a Gaussian
            ``N(mean, sigma**2)`` is multiplied into the joint posterior
            before marginalization. Default: spectra-only.
        z_window
            Optional ``(z_min, z_max)`` to restrict the redshift search range.
            Useful when an external photo-z bounds the plausible range.
            Default: full training range.
        condition_class
            Optional class name. When supplied, the output includes
            ``z_STRIDER_given_class`` and matching posterior fields computed
            from ``p(z | class, spectrum)`` on the joint evidence surface.
        return_joint
            If True, include the full class × redshift joint log posterior
            arrays for plotting. Default: False.
        calibrated
            If True (default), return calibrated probabilities as the
            unsuffixed values. If False, return uncalibrated as the
            unsuffixed values. Both are always present in the output dict
            under ``_uncal`` and ``_cal`` suffixes regardless.

        Returns
        -------
        Output dict with the keys listed in the module docstring. The
        ``_spectra_only`` values are always present; they equal the
        ``_uncal`` values when neither prior nor window was passed.
        """
        from strider.inference.classify import classify_spectral_timeseries
        from strider.inference.timeseries import SpectralEpoch

        wavelength = np.asarray(wavelength, dtype=np.float32)
        flux_arr = np.asarray(flux, dtype=np.float32)
        phases = np.atleast_1d(np.asarray(phase, dtype=np.float32))

        if flux_arr.ndim == 1:
            flux_arr = flux_arr[None, :]
        if flux_arr.shape[0] != phases.size:
            raise ValueError(
                f"flux has {flux_arr.shape[0]} epochs but phase has "
                f"{phases.size}; they must match"
            )
        if flux_arr.shape[1] != wavelength.size:
            raise ValueError(
                f"flux has {flux_arr.shape[1]} wavelength bins but "
                f"wavelength has {wavelength.size}; they must match"
            )

        flux_err_arr = None
        if flux_err is not None:
            flux_err_arr = np.asarray(flux_err, dtype=np.float32)
            if flux_err_arr.ndim == 1:
                flux_err_arr = flux_err_arr[None, :]
            if flux_err_arr.shape != flux_arr.shape:
                raise ValueError("flux_err must have the same shape as flux")

        # Hard contract: prior mean and sigma travel together. The internal
        # classifier silently defaults sigma to 0.05 if mean alone is passed;
        # that is a footgun for collaborators reading a results table later.
        if (z_prior_mean is None) != (z_prior_sigma is None):
            raise ValueError(
                "z_prior_mean and z_prior_sigma must be passed together. "
                "Got mean=%r, sigma=%r. To disable the prior, pass neither."
                % (z_prior_mean, z_prior_sigma)
            )

        epochs = [
            SpectralEpoch(
                wave=wavelength,
                flux=flux_arr[i],
                flux_err=(flux_err_arr[i] if flux_err_arr is not None else None),
                phase=float(phases[i]),
            )
            for i in range(phases.size)
        ]

        raw_out = classify_spectral_timeseries(
            self._model,
            epochs,
            z_prior=z_prior_mean,
            z_prior_sigma=z_prior_sigma,
            z_window=z_window,
            device=self._device,
        )

        return self._format_output(
            raw_out,
            calibrated=calibrated,
            prior_or_window_supplied=(
                z_prior_mean is not None or z_window is not None
            ),
            spectra_only_override=None,
            condition_class=condition_class,
            return_joint=return_joint,
        )

    def _format_output(
        self,
        raw: dict[str, Any],
        *,
        calibrated: bool,
        prior_or_window_supplied: bool,
        spectra_only_override: dict[str, Any] | None = None,
        condition_class: str | None = None,
        return_joint: bool = False,
    ) -> dict[str, Any]:
        """Convert internal classify output to the stable public schema.

        Three sources of the spectra-only posterior, in priority order:

        1. ``spectra_only_override`` — optional explicit no-control result.
        2. ``raw['class_log_probs_raw']`` etc. — set when post-hoc restrictions
           were applied.
        3. ``raw['class_probs']`` itself — when no restrictions were
           requested at all, the model output IS the spectra-only result.
        """
        class_names = list(self.metadata.class_names)
        z_grid = np.asarray(self.metadata.z_grid, dtype=np.float64)

        # Posterior with controls (= model output when no controls passed)
        p_class_with = np.asarray(raw["class_probs"], dtype=np.float64)
        z_marg_with = np.asarray(raw["z_posterior"], dtype=np.float64)
        joint_with = _joint_log_from_output(raw, p_class_with, z_marg_with)

        # Spectra-only posterior, in the priority order documented above.
        if spectra_only_override is not None:
            p_so = np.asarray(
                spectra_only_override["class_probs"], dtype=np.float64
            )
            z_marg_so = np.asarray(
                spectra_only_override["z_posterior"], dtype=np.float64
            )
            joint_so = _joint_log_from_output(
                spectra_only_override,
                p_so,
                z_marg_so,
            )
        elif "class_log_probs_raw" in raw and "z_log_posterior_raw" in raw:
            log_p_so = np.asarray(raw["class_log_probs_raw"], dtype=np.float64)
            p_so = _softmax_from_logp(log_p_so)
            log_z_so = np.asarray(raw["z_log_posterior_raw"], dtype=np.float64)
            z_marg_so = _normalize_log(log_z_so)
            if "joint_log_probs_raw" in raw:
                joint_so = np.asarray(raw["joint_log_probs_raw"], dtype=np.float64)
            else:
                joint_so = _outer_joint_log(p_so, z_marg_so)
        else:
            p_so = p_class_with.copy()
            z_marg_so = z_marg_with.copy()
            joint_so = joint_with.copy()

        # Class calibration reports coherent probabilities but does not change
        # the raw evidence ranking used for labels and selection.
        p_so_cal = self._calibrate_class_probs(p_so)
        p_with_cal = self._calibrate_class_probs(p_class_with)

        conditioned: dict[str, Any] = {}
        if condition_class is not None:
            cond_idx, cond_name = _resolve_class_name(condition_class, class_names)
            z_cond_so = _z_given_class_from_joint(joint_so, cond_idx)
            z_cond_with = _z_given_class_from_joint(joint_with, cond_idx)
            conditioned = {
                "conditioned_class": cond_name,
                "conditioned_class_index": cond_idx,
                "z_marginal_given_class_uncal_spectra_only": z_cond_so,
                "z_marginal_given_class_uncal_with_controls": z_cond_with,
                "z_med_given_class_uncal_spectra_only": _quantile(z_grid, z_cond_so, 0.5),
                "z_med_given_class_uncal_with_controls": _quantile(z_grid, z_cond_with, 0.5),
                "z_map_given_class_uncal_spectra_only": float(z_grid[int(np.argmax(z_cond_so))]),
                "z_map_given_class_uncal_with_controls": float(z_grid[int(np.argmax(z_cond_with))]),
            }

        # Decisions stay on the raw, rare-class-balanced ranking. The frozen
        # Dirichlet map is only for the probabilities reported beside them.
        best_idx = int(np.argmax(p_class_with))
        best_name = class_names[best_idx]

        # Per-class probability families. Each posterior gets four scalars
        # extracted: P(Ia) strict, P(91bg) strict, P(Iax) strict, and the
        # explicit combined diagnostics P(Ia+91bg) and P(Ia+91bg+Iax).
        def per_class(p: np.ndarray) -> dict[str, float]:
            return {
                "p_Ia":        _ia_subset_probs(p, class_names, ("Ia",)),
                "p_91bg":      _ia_subset_probs(p, class_names, ("91bg",)),
                "p_Iax":       _ia_subset_probs(p, class_names, ("Iax",)),
                "p_Ia91bg":    _ia_subset_probs(p, class_names, ("Ia", "91bg")),
                "p_Ia_family": _ia_subset_probs(p, class_names, ("Ia", "91bg", "Iax")),
            }
        per_so      = per_class(p_so)
        per_so_cal  = per_class(p_so_cal)
        per_with    = per_class(p_class_with)
        per_with_cal = per_class(p_with_cal)

        raw_intervals_so = _central_intervals(z_grid, z_marg_so)
        raw_intervals_with = _central_intervals(z_grid, z_marg_with)
        cal_intervals_so, cal_map_so = self._calibrated_z_intervals(
            z_grid,
            z_marg_so,
            raw_p_ia=per_so["p_Ia"],
            allow=True,
        )
        cal_intervals_with, cal_map_with = self._calibrated_z_intervals(
            z_grid,
            z_marg_with,
            raw_p_ia=per_with["p_Ia"],
            allow=not prior_or_window_supplied,
        )
        hpd_conformal = self._conformal_hpd_sets(
            z_grid,
            z_marg_with,
            allow=calibrated and not prior_or_window_supplied,
        )
        gold_threshold = self._gold_ia_threshold()
        gold_selected = bool(per_so["p_Ia"] >= gold_threshold)

        out: dict[str, Any] = {
            # All class probabilities — all 4 (uncal/cal × spectra-only/controls)
            "p_class_uncal_spectra_only":   p_so,
            "p_class_cal_spectra_only":     p_so_cal,
            "p_class_uncal_with_controls":  p_class_with,
            "p_class_cal_with_controls":    p_with_cal,

            # z marginals — all 4
            "z_marginal_uncal_spectra_only":  z_marg_so,
            "z_marginal_uncal_with_controls": z_marg_with,

            # z point estimates — all 4
            "z_med_uncal_spectra_only":  _quantile(z_grid, z_marg_so,  0.5),
            "z_med_uncal_with_controls": _quantile(z_grid, z_marg_with, 0.5),
            "z_map_uncal_spectra_only":  float(z_grid[int(np.argmax(z_marg_so))]),
            "z_map_uncal_with_controls": float(z_grid[int(np.argmax(z_marg_with))]),

            # Central redshift intervals. Point estimates above remain native.
            **_prefixed_intervals(raw_intervals_so, "uncal_spectra_only"),
            **_prefixed_intervals(cal_intervals_so, "cal_spectra_only"),
            **_prefixed_intervals(raw_intervals_with, "uncal_with_controls"),
            **_prefixed_intervals(cal_intervals_with, "cal_with_controls"),
            "redshift_calibration_spectra_only": cal_map_so,
            "redshift_calibration_with_controls": cal_map_with,
            "redshift_hpd_conformal": hpd_conformal,

            # The class with the most evidence.
            "strider_class": best_name,
            "strider_class_index": best_idx,

            # Frozen Ia selection contract.
            "selection_p_Ia_raw": per_so["p_Ia"],
            "gold_Ia_threshold_raw": gold_threshold,
            "gold_Ia_selected": gold_selected,

            # Convenience metadata
            "class_names": class_names,
            "calibrated": calibrated,
            "prior_or_window_supplied": prior_or_window_supplied,
            # Audit trail: whether a requested control was actually honored.
            # False z_window_valid => window had no grid overlap and was ignored
            # (with-controls == spectra-only). Never silently hide this.
            "z_window_valid": bool(raw.get("z_window_valid", True)),
            "z_prior_applied": bool(raw.get("z_prior_applied", False)),
            "z_window_effective": raw.get("z_window_effective"),
        }
        out.update(conditioned)
        if return_joint:
            out["joint_log_probs_uncal_spectra_only"] = _normalize_joint_log_np(joint_so)
            out["joint_log_probs_uncal_with_controls"] = _normalize_joint_log_np(joint_with)
            out["joint_log_probs"] = out["joint_log_probs_uncal_with_controls"]
            out["joint_log_probs_spectra_only"] = out["joint_log_probs_uncal_spectra_only"]

        # Promote per-class probabilities to top-level keys with the same
        # uncal/cal × spectra-only/with-controls grid. Five class groups ×
        # four variants = 20 scalar fields. Verbose but explicit; no hidden
        # combinations.
        for group, scalars in (
            ("uncal_spectra_only",  per_so),
            ("cal_spectra_only",    per_so_cal),
            ("uncal_with_controls", per_with),
            ("cal_with_controls",   per_with_cal),
        ):
            for name, value in scalars.items():
                out[f"{name}_{group}"] = value

        # Convenience aliases use the requested probability view and the
        # actual inference controls.
        suffix = "cal_with_controls" if calibrated else "uncal_with_controls"
        out["p_class"]    = out[f"p_class_{suffix}"]
        # Calibration reshapes the reported intervals, not the posterior.
        out["z_marginal"] = out["z_marginal_uncal_with_controls"]
        out["z_med"]      = out["z_med_uncal_with_controls"]
        out["z_STRIDER"]  = out["z_med"]
        out["z_map"]      = out["z_map_uncal_with_controls"]
        interval_suffix = suffix if calibrated else "uncal_with_controls"
        for quantile in ("p025", "p05", "p16", "p84", "p95", "p975"):
            out[f"z_{quantile}"] = out[f"z_{quantile}_{interval_suffix}"]
        out["z_quality"]  = str(raw.get("z_quality", "unknown"))
        for name in ("p_Ia", "p_91bg", "p_Iax", "p_Ia91bg", "p_Ia_family"):
            out[name] = out[f"{name}_{suffix}"]
        if conditioned:
            out["z_marginal_given_class"] = out["z_marginal_given_class_uncal_with_controls"]
            out["z_med_given_class"] = out["z_med_given_class_uncal_with_controls"]
            out["z_STRIDER_given_class"] = out["z_med_given_class"]
            out["z_map_given_class"] = out["z_map_given_class_uncal_with_controls"]

        return out

    def _calibrate_class_probs(self, p: np.ndarray) -> np.ndarray:
        """Return the frozen reported probability vector."""
        calibration = self.metadata.class_calibration
        if calibration is not None:
            logp = np.log(np.clip(np.asarray(p, dtype=np.float64), 1e-300, None))
            logp -= np.logaddexp.reduce(logp)
            weights = np.asarray(calibration["weights"], dtype=np.float64)
            bias = np.asarray(calibration["bias"], dtype=np.float64)
            logits = weights @ logp + bias
            logits -= logits.max()
            out = np.exp(logits)
            return out / out.sum()

        return p.copy()

    def _gold_ia_threshold(self) -> float:
        selection = self.metadata.selection or {}
        return float(selection.get("gold_threshold_raw_p_ia", 0.9))

    def _calibrated_z_intervals(
        self,
        z_grid: np.ndarray,
        posterior: np.ndarray,
        *,
        raw_p_ia: float,
        allow: bool,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        calibration = self.metadata.redshift_calibration
        if calibration is None:
            return _central_intervals(z_grid, posterior), {
                "applied": False,
                "method": "none",
            }
        if not allow:
            return _central_intervals(z_grid, posterior), {
                "applied": False,
                "method": "pit_recalibration",
                "reason": "not validated after a redshift prior or window",
            }

        use_gold = raw_p_ia >= self._gold_ia_threshold()
        population = "gold_Ia" if use_gold else "full"
        pit_key = "pit_gold_sorted" if use_gold else "pit_full_sorted"
        intervals = _central_intervals(
            z_grid,
            posterior,
            pit_sorted=np.asarray(calibration[pit_key], dtype=np.float64),
        )
        return intervals, {
            "applied": True,
            "method": "pit_recalibration",
            "population": population,
            "fitted_on": self.metadata.calibration_fitted_on,
        }

    def _conformal_hpd_sets(
        self,
        z_grid: np.ndarray,
        posterior: np.ndarray,
        *,
        allow: bool,
    ) -> dict[str, Any]:
        calibration = self.metadata.redshift_calibration
        if calibration is None or not allow:
            return {"applied": False}
        scores = np.asarray(calibration["hpd_scores_sorted"], dtype=np.float64)
        sets = {
            str(level): _hpd_conformal_set(z_grid, posterior, scores, level)
            for level in (0.68, 0.90, 0.95)
        }
        return {
            "applied": True,
            "method": "split_conformal_hpd",
            "fitted_on": self.metadata.calibration_fitted_on,
            "sets": sets,
        }


def _coerce_wavelength_aa(wavelength: np.ndarray, *, unit: str = "auto") -> np.ndarray:
    """Return observed-frame wavelength in Angstrom."""
    unit_key = str(unit).lower().replace("å", "a")
    if unit_key in {"angstrom", "angstroms", "aa", "a"}:
        return wavelength.astype(np.float32, copy=False)
    if unit_key in {"micron", "microns", "um", "micrometer", "micrometers"}:
        return (wavelength * 1.0e4).astype(np.float32, copy=False)
    if unit_key != "auto":
        raise ValueError(
            "wavelength_unit must be 'auto', 'angstrom', or 'micron'; "
            f"got {unit!r}"
        )
    finite = wavelength[np.isfinite(wavelength)]
    if finite.size and float(np.nanmax(finite)) < 100.0:
        return (wavelength * 1.0e4).astype(np.float32, copy=False)
    return wavelength.astype(np.float32, copy=False)


def _resolve_pair(
    *,
    name: str,
    pair: tuple[float, float] | None,
    lo: float | None,
    hi: float | None,
) -> tuple[float, float] | None:
    """Resolve either a tuple pair or separate min/max kwargs."""
    if pair is not None and (lo is not None or hi is not None):
        raise ValueError(
            f"Pass either {name}=(min, max) or separate min/max kwargs, not both."
        )
    if pair is None:
        if lo is None and hi is None:
            return None
        if lo is None or hi is None:
            raise ValueError(f"{name} min and max must be passed together.")
        pair = (lo, hi)
    if len(pair) != 2:
        raise ValueError(f"{name} must contain exactly two values.")
    a, b = float(pair[0]), float(pair[1])
    if not (np.isfinite(a) and np.isfinite(b)):
        raise ValueError(f"{name} values must be finite.")
    return (a, b) if a <= b else (b, a)


def _resolve_z_prior(
    *,
    z_prior: tuple[float, float] | None,
    z_prior_mean: float | None,
    z_prior_sigma: float | None,
) -> tuple[float, float] | None:
    """Resolve Gaussian z prior from tuple or explicit mean/sigma kwargs."""
    if z_prior is not None and (
        z_prior_mean is not None or z_prior_sigma is not None
    ):
        raise ValueError(
            "Pass either z_prior=(mean, sigma) or z_prior_mean/z_prior_sigma, "
            "not both."
        )
    if z_prior is None:
        if z_prior_mean is None and z_prior_sigma is None:
            return None
        if z_prior_mean is None or z_prior_sigma is None:
            raise ValueError("z_prior_mean and z_prior_sigma must be passed together.")
        z_prior = (z_prior_mean, z_prior_sigma)
    if len(z_prior) != 2:
        raise ValueError("z_prior must be a (mean, sigma) pair.")
    mean, sigma = float(z_prior[0]), float(z_prior[1])
    if not np.isfinite(mean):
        raise ValueError("z_prior mean must be finite.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("z_prior sigma must be finite and positive.")
    return mean, sigma


def _finite_range(values: np.ndarray) -> list[float] | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return [float(np.nanmin(finite)), float(np.nanmax(finite))]


def _apply_wave_range(
    wavelength: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray | None,
    wave_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Apply an observed-frame wavelength cut along the final flux axis."""
    lo, hi = wave_range
    keep = np.isfinite(wavelength) & (wavelength >= lo) & (wavelength <= hi)
    if int(keep.sum()) < 2:
        raise ValueError(
            f"wave_range={wave_range} leaves fewer than two wavelength bins."
        )
    flux_out = np.asarray(flux, dtype=np.float32)[..., keep]
    err_out = None
    if flux_err is not None:
        err_out = np.asarray(flux_err, dtype=np.float32)[..., keep]
    return wavelength[keep], flux_out, err_out


def _as_list_or_none(pair: tuple[float, float] | list[float] | None) -> list[float] | None:
    if pair is None:
        return None
    return [float(pair[0]), float(pair[1])]


def _add_friendly_output_blocks(
    out: dict[str, Any],
    *,
    metadata: CheckpointMetadata,
    calibrated: bool,
    z_prior: tuple[float, float] | None,
    z_window: tuple[float, float] | None,
    wave_range: tuple[float, float] | None,
    original_wave_range: list[float] | None,
    wave_used: np.ndarray,
    flux_used: np.ndarray,
    flux_err_used: np.ndarray | None,
    phases: np.ndarray,
    return_inputs: bool,
) -> None:
    """Add compact public-facing views without removing the flat schema."""
    class_names = list(out["class_names"])
    p_class = np.asarray(out["p_class"], dtype=float)
    ranking_scores = np.asarray(out["p_class_uncal_with_controls"], dtype=float)
    best_idx = int(out["strider_class_index"])
    best_name = str(out["strider_class"])
    pred_prob = float(p_class[best_idx]) if p_class.size else float("nan")
    p_ia = float(out["p_Ia"])
    p_non_ia = float(max(0.0, min(1.0, 1.0 - p_ia))) if np.isfinite(p_ia) else float("nan")
    top_idx = (
        np.argsort(ranking_scores)[-3:][::-1]
        if ranking_scores.size
        else np.asarray([], dtype=int)
    )

    out["classification"] = {
        "class": best_name,
        "class_index": best_idx,
        "confidence": pred_prob,
        "p_Ia": p_ia,
        "p_Ia_raw": float(out["selection_p_Ia_raw"]),
        "gold_Ia_selected": bool(out["gold_Ia_selected"]),
        "gold_Ia_threshold_raw": float(out["gold_Ia_threshold_raw"]),
        "ia_selection_context": "spectra_only",
        "p_non_Ia": p_non_ia,
        "p_91bg": float(out["p_91bg"]),
        "p_Iax": float(out["p_Iax"]),
        "p_Ia91bg": float(out["p_Ia91bg"]),
        "p_Ia_family": float(out["p_Ia_family"]),
        "top3": [
            {
                "class": class_names[int(i)],
                "probability": float(p_class[int(i)]),
                "raw_score": float(ranking_scores[int(i)]),
            }
            for i in top_idx
        ],
        "ranking": "raw evidence",
        "probabilities": "calibrated" if calibrated else "raw",
    }

    suffix = "cal" if calibrated else "uncal"
    z_grid = np.asarray(metadata.z_grid, dtype=np.float64)
    interval_cal_so = out["redshift_calibration_spectra_only"]
    interval_cal_with = out["redshift_calibration_with_controls"]
    if not calibrated:
        interval_cal_so = {"applied": False, "reason": "calibrated=False"}
        interval_cal_with = {"applied": False, "reason": "calibrated=False"}
    z_prior_block = None
    if z_prior is not None:
        z_prior_block = {
            "mode": "gaussian",
            "mean": float(z_prior[0]),
            "sigma": float(z_prior[1]),
            "application": "posterior_conditioning",
        }
    out["redshift"] = {
        "z_STRIDER": float(out["z_med"]),
        "z_S": float(out["z_med"]),
        "z_med": float(out["z_med"]),
        "z_map": float(out["z_map"]),
        "z_p16": float(out["z_p16"]),
        "z_p84": float(out["z_p84"]),
        "z_p05": float(out["z_p05"]),
        "z_p95": float(out["z_p95"]),
        "z_p025": float(out["z_p025"]),
        "z_p975": float(out["z_p975"]),
        "quality": out["z_quality"],
        "z_grid": z_grid,
        "posterior": np.asarray(out["z_marginal"], dtype=np.float64),
        "spectra_only": {
            "z_S": float(out["z_med_uncal_spectra_only"]),
            "z_map": float(out["z_map_uncal_spectra_only"]),
            "z_p16": float(out[f"z_p16_{suffix}_spectra_only"]),
            "z_p84": float(out[f"z_p84_{suffix}_spectra_only"]),
            "interval_calibration": interval_cal_so,
        },
        "with_controls": {
            "z_S": float(out["z_med_uncal_with_controls"]),
            "z_map": float(out["z_map_uncal_with_controls"]),
            "z_p16": float(out[f"z_p16_{suffix}_with_controls"]),
            "z_p84": float(out[f"z_p84_{suffix}_with_controls"]),
            "interval_calibration": interval_cal_with,
        },
        "interval_calibration": interval_cal_with,
        "hpd_conformal": out["redshift_hpd_conformal"],
    }
    if "conditioned_class" in out:
        cond_class = str(out["conditioned_class"])
        out["redshift"]["given_class"] = {
            "class": cond_class,
            "class_index": int(out["conditioned_class_index"]),
            "z_STRIDER": float(out["z_med_given_class"]),
            "z_S": float(out["z_med_given_class"]),
            "z_med": float(out["z_med_given_class"]),
            "z_map": float(out["z_map_given_class"]),
            "posterior": np.asarray(
                out["z_marginal_given_class"],
                dtype=np.float64,
            ),
            "spectra_only": {
                "z_S": float(out["z_med_given_class_uncal_spectra_only"]),
                "z_map": float(out["z_map_given_class_uncal_spectra_only"]),
            },
            "with_controls": {
                "z_S": float(out["z_med_given_class_uncal_with_controls"]),
                "z_map": float(out["z_map_given_class_uncal_with_controls"]),
            },
        }
    out["controls"] = {
        "calibrated": bool(calibrated),
        "class_probability_calibration": (
            metadata.class_calibration or {"method": "none"}
        )["method"] if calibrated else "none",
        "redshift_interval_calibration_applied": bool(
            calibrated
            and interval_cal_with.get("applied", False)
        ),
        "z_prior": z_prior_block,
        "z_window": _as_list_or_none(z_window),
        "z_window_valid": bool(out["z_window_valid"]),
        "z_window_effective": out["z_window_effective"],
        "wave_range_aa": _as_list_or_none(wave_range),
        "prior_or_window_supplied": bool(out["prior_or_window_supplied"]),
    }

    wave_model = np.asarray(metadata.wavelength_grid, dtype=np.float64)
    flux_matrix = np.atleast_2d(np.asarray(flux_used, dtype=np.float64))
    valid_flux_fraction = [
        float(np.mean(np.isfinite(epoch))) for epoch in flux_matrix
    ]
    used_range = _finite_range(wave_used)
    if used_range is None:
        model_coverage_fraction = 0.0
    else:
        covered = (wave_model >= used_range[0]) & (wave_model <= used_range[1])
        model_coverage_fraction = float(np.mean(covered))
    spectra_block: dict[str, Any] = {
        "n_epochs": int(np.atleast_1d(phases).size),
        "phase_days": [float(x) for x in np.atleast_1d(phases)],
        "input_wave_range_aa": original_wave_range,
        "used_wave_range_aa": used_range,
        "n_wavelength": int(np.asarray(wave_used).size),
        "model_wave_grid_range_aa": _finite_range(wave_model),
        "model_n_wave": int(metadata.n_wave),
        "quality": {
            "model_wavelength_coverage_fraction": model_coverage_fraction,
            "valid_flux_fraction_per_epoch": valid_flux_fraction,
            "flux_error_supplied": flux_err_used is not None,
        },
    }
    if return_inputs:
        spectra_block.update({
            "wavelength_aa": np.asarray(wave_used, dtype=np.float32),
            "flux": np.asarray(flux_used, dtype=np.float32),
        })
        if flux_err_used is not None:
            spectra_block["flux_err"] = np.asarray(flux_err_used, dtype=np.float32)
    out["spectra"] = spectra_block


def load_model(checkpoint_path: str | Path, device: str = "cpu") -> Any:
    """Load a released STRIDER model and return a ready-to-use model.

    Parameters
    ----------
    checkpoint_path
        Path to either a legacy STRIDER ``.pt`` checkpoint or a current
        self-contained STRIDER model-package directory.
    device
        Torch device ("cpu" or "cuda").

    Returns
    -------
    A model instance ready for ``.classify()``. Current model packages accept
    observer-frame wavelength, FLAM, FLAMERR and observation times. Legacy
    checkpoints retain their historical rest-phase interface.
    """
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        from strider.engine.deployment import load_model_package

        return load_model_package(checkpoint_path, device=device)
    payload = load_checkpoint(checkpoint_path)
    metadata = CheckpointMetadata.from_dict(payload["metadata"])
    model = _build_underlying_model(payload, metadata, device=device)
    return STRIDER(_model=model, metadata=metadata, _device=device)


def _build_underlying_model(
    payload: dict[str, Any], metadata: CheckpointMetadata, device: str
) -> Any:
    """Instantiate the production model from a checkpoint payload."""
    from strider.network.config import STRIDERConfig
    from strider.network.model import ModelConfig, build_model
    from strider.network.template_bank import TemplateBank
    from strider.inference.classify import _normalize_state_dict_keys
    from strider.inference.metadata import (
        attach_inference_metadata,
    )
    from strider.model_info import _CHECKPOINT_FAMILY_TAG

    state_dict = _normalize_state_dict_keys(payload["state_dict"])
    sconfig = STRIDERConfig._apply_overrides(metadata.model_config)
    if metadata.family != _CHECKPOINT_FAMILY_TAG:
        raise ValueError("This checkpoint is not compatible with public STRIDER")
    if "bank_state" not in payload or metadata.detector_config is None:
        raise ValueError("Portable checkpoint is missing its embedded signature bank")

    model_config = ModelConfig.from_dict(metadata.detector_config)
    bank = TemplateBank.from_dict(payload["bank_state"])
    model = build_model(sconfig, model_config, bank)
    model.load_state_dict(state_dict, strict=True)
    model.config = sconfig
    model.detector_config = model_config

    model.eval()
    model.to(device)
    attach_inference_metadata(model, _make_inference_metadata(metadata))
    return model


def _make_inference_metadata(md: "CheckpointMetadata"):
    """Translate checkpoint metadata into the shared inference contract."""
    from strider.inference.metadata import InferenceMetadata

    return InferenceMetadata(
        family="strider",
        class_names=list(md.class_names),
        z_grid=np.asarray(md.z_grid, dtype=np.float32),
        z_min=float(md.z_min),
        z_max=float(md.z_max),
        n_z_bins=int(md.n_z_bins),
        normalization=md.normalization,
        patch_size=int(md.patch_size),
        n_channels=int(md.n_channels),
        peak_normalize_default=False,
    )


def _quantile(grid: np.ndarray, marginal: np.ndarray, q: float) -> float:
    """Continuous q-th quantile of a discrete redshift marginal.

    Uses the mid-point (Hazen) inverse-CDF — each bin's mass sits at its
    centre, so the interpolation CDF is ``cumsum - 0.5*p``. Unbiased and
    continuous: a uniform marginal returns the true centre (not half a bin
    low, which naive grid-snapping gives). This matches
    :func:`strider.inference.classify.interp_posterior_quantile` exactly, so
    all STRIDER outputs use the same continuous convention.
    """
    total = marginal.sum()
    if total <= 0:
        return float(grid[len(grid) // 2])
    p = marginal / total
    mid_cdf = np.cumsum(p) - 0.5 * p
    return float(np.interp(q, mid_cdf, grid))


def _central_intervals(
    grid: np.ndarray,
    marginal: np.ndarray,
    *,
    pit_sorted: np.ndarray | None = None,
) -> dict[str, float]:
    targets = {
        "p025": 0.025,
        "p05": 0.05,
        "p16": 0.16,
        "p84": 0.84,
        "p95": 0.95,
        "p975": 0.975,
    }
    if pit_sorted is None:
        levels = targets
    else:
        levels = {
            name: float(np.quantile(pit_sorted, target))
            for name, target in targets.items()
        }
    return {
        name: _quantile(grid, marginal, level)
        for name, level in levels.items()
    }


def _prefixed_intervals(
    intervals: dict[str, float],
    suffix: str,
) -> dict[str, float]:
    return {f"z_{name}_{suffix}": value for name, value in intervals.items()}


def _hpd_conformal_set(
    grid: np.ndarray,
    marginal: np.ndarray,
    calibration_scores: np.ndarray,
    level: float,
) -> dict[str, Any]:
    p = np.clip(np.asarray(marginal, dtype=np.float64), 0.0, None)
    total = float(p.sum())
    if total <= 0:
        return {
            "nominal_coverage": float(level),
            "posterior_mass": 0.0,
            "components": [],
        }
    p = p / total
    finite_level = min(1.0, float(level) * (1.0 + 1.0 / calibration_scores.size))
    threshold = float(np.quantile(calibration_scores, finite_level))
    bin_scores = np.asarray([p[p > value].sum() for value in p], dtype=np.float64)
    selected = bin_scores <= threshold
    edges = np.diff(np.concatenate(([False], selected, [False])).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    components = [
        {
            "z_low": float(grid[start]),
            "z_high": float(grid[stop - 1]),
            "mass": float(p[start:stop].sum()),
        }
        for start, stop in zip(starts, stops)
    ]
    return {
        "nominal_coverage": float(level),
        "posterior_mass": float(p[selected].sum()),
        "components": components,
    }


def _softmax_from_logp(log_p: np.ndarray) -> np.ndarray:
    """Numerically-safe softmax from log-probabilities."""
    shifted = log_p - log_p.max()
    out = np.exp(shifted)
    return out / out.sum()


def _normalize_log(log_p: np.ndarray) -> np.ndarray:
    """Convert log-probabilities to a normalized probability vector."""
    shifted = log_p - log_p.max()
    out = np.exp(shifted)
    total = out.sum()
    return out / total if total > 0 else out


def _resolve_class_name(name: str, class_names: list[str]) -> tuple[int, str]:
    """Resolve a public class name or numeric class index."""
    requested = str(name).strip()
    if not requested:
        raise ValueError("condition_class must be a non-empty class name.")
    if requested.isdigit():
        idx = int(requested)
        if 0 <= idx < len(class_names):
            return idx, class_names[idx]
    aliases = {
        "sn ia": "Ia",
        "snia": "Ia",
        "normal ia": "Ia",
        "ia-normal": "Ia",
        "ia normal": "Ia",
    }
    target = aliases.get(requested.lower(), requested)
    for idx, class_name in enumerate(class_names):
        if target == class_name:
            return idx, class_name
    for idx, class_name in enumerate(class_names):
        if target.lower() == class_name.lower():
            return idx, class_name
    allowed = ", ".join(class_names)
    raise ValueError(
        f"condition_class={name!r} is not one of the model classes: {allowed}"
    )


def _outer_joint_log(p_class: np.ndarray, z_marginal: np.ndarray) -> np.ndarray:
    """Fallback joint log surface when an older output lacks a joint posterior."""
    p_class = np.asarray(p_class, dtype=np.float64)
    z_marginal = np.asarray(z_marginal, dtype=np.float64)
    log_class = np.log(np.clip(p_class, 1e-300, 1.0))
    log_z = np.log(np.clip(z_marginal, 1e-300, None))
    return log_class[:, None] + log_z[None, :]


def _joint_log_from_output(
    raw: dict[str, Any],
    p_class: np.ndarray,
    z_marginal: np.ndarray,
) -> np.ndarray:
    """Return the class × z joint log posterior from an inference output."""
    for key in ("joint_log_probs", "joint_log_posterior"):
        if key in raw:
            joint = np.asarray(raw[key], dtype=np.float64)
            if joint.ndim == 3 and joint.shape[0] == 1:
                joint = joint[0]
            return joint
    return _outer_joint_log(p_class, z_marginal)


def _normalize_joint_log_np(joint_log: np.ndarray) -> np.ndarray:
    """Normalize a class × z log posterior over all cells."""
    joint = np.asarray(joint_log, dtype=np.float64)
    finite = np.isfinite(joint)
    if not finite.any():
        return joint.copy()
    maxv = float(np.max(joint[finite]))
    shifted = np.full_like(joint, -np.inf, dtype=np.float64)
    shifted[finite] = joint[finite] - maxv
    total = float(np.exp(shifted[finite]).sum())
    if not np.isfinite(total) or total <= 0.0:
        return shifted
    return shifted - np.log(total)


def _z_given_class_from_joint(joint_log: np.ndarray, class_idx: int) -> np.ndarray:
    """Normalize one class row of a joint log posterior into p(z | class)."""
    joint = np.asarray(joint_log, dtype=np.float64)
    if joint.ndim == 3 and joint.shape[0] == 1:
        joint = joint[0]
    if joint.ndim != 2:
        raise ValueError(
            "joint posterior must have shape (n_class, n_z) to condition on class"
        )
    if not (0 <= int(class_idx) < joint.shape[0]):
        raise ValueError(
            f"class index {class_idx} is outside joint posterior shape {joint.shape}"
        )
    row = np.asarray(joint[int(class_idx)], dtype=np.float64)
    finite = np.isfinite(row)
    if not finite.any():
        return np.full(row.shape, 1.0 / max(1, row.size), dtype=np.float64)
    shifted = np.zeros_like(row, dtype=np.float64)
    shifted[finite] = np.exp(row[finite] - float(np.max(row[finite])))
    total = float(shifted.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full(row.shape, 1.0 / max(1, row.size), dtype=np.float64)
    return shifted / total
