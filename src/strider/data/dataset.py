"""PyTorch dataset for native-bin observations and controlled noise views."""

from __future__ import annotations

import hashlib
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from strider.config import project_path

from .classes import class_name_for_source, class_names_for_scheme
from .noise import (
    draw_controlled_background_measurement,
    draw_reported_error_flux,
    repeatable_rng,
)
from .template_support import template_support_policy


OBSERVED_SNR_EDGE_TRIM_FRACTION = 0.05
OBSERVED_SNR_MAX_RELATIVE_ERROR = 3.0


def log_wavelength_grid(minimum: float, maximum: float, bins: int) -> np.ndarray:
    return np.geomspace(float(minimum), float(maximum), int(bins), dtype=np.float64)


def _optional_visit_limit(value: Any) -> int | None:
    """Return ``None`` for an unlimited visit setting, otherwise a positive cap."""
    if value is None or (isinstance(value, str) and value.lower() == "all"):
        return None
    limit = int(value)
    if limit < 1:
        raise ValueError("visit limits must be positive or 'all'")
    return limit


class SundialDataset(Dataset):
    """One object per item, with visits kept as an explicit axis."""

    def __init__(
        self,
        config: dict[str, Any],
        split: str,
        view: str,
        training: bool = False,
        pair_no_source: bool | None = None,
        visit_selection: str = "span",
        visit_repeat: int = 0,
        generated_noise_scale: float | None = None,
        paired_noise_seed: int | None = None,
        include_observed_signal_to_noise: bool = False,
    ) -> None:
        if view not in {
            "generated",
            "original",
            "clean",
            "no_source",
            "residual",
            "reported_error_with_source",
            "reported_error_no_source",
        }:
            raise ValueError(f"Unsupported observation view: {view}")
        self.config = config
        self.split = split
        self.view = view
        self.training = training
        self.include_observed_signal_to_noise = bool(
            include_observed_signal_to_noise
        )
        # Persistent data-loading workers must see the new epoch so controlled
        # noise changes after every pass through the training set.
        self._training_epoch = torch.zeros((), dtype=torch.int64)
        workers = int(config["training"].get("num_workers", 0))
        persistent = bool(config["training"].get("persistent_workers", workers > 0))
        if workers > 0 and persistent:
            self._training_epoch.share_memory_()
        self.seed = int(config["project"]["seed"])
        self.max_visits = _optional_visit_limit(config["data"]["max_visits"])
        configured_pairing = bool(config["training"].get("paired_no_source", False))
        self.pair_no_source = (
            configured_pairing and training if pair_no_source is None else bool(pair_no_source)
        )
        if visit_selection not in {"span", "random"}:
            raise ValueError(f"Unsupported visit selection: {visit_selection}")
        self.visit_selection = visit_selection
        self.visit_repeat = int(visit_repeat)
        self.paired_noise_seed = (
            None if paired_noise_seed is None else int(paired_noise_seed)
        )
        self.generated_noise_scale = (
            None if generated_noise_scale is None else float(generated_noise_scale)
        )
        if self.generated_noise_scale is not None and self.generated_noise_scale < 0.0:
            raise ValueError("generated_noise_scale must be nonnegative")
        self.training_visit_counts = tuple(
            _optional_visit_limit(value)
            for value in config["data"].get("training_visit_counts", [])
        )
        finite_training_counts = [
            value for value in self.training_visit_counts if value is not None
        ]
        if any(value < 1 for value in finite_training_counts):
            raise ValueError("training_visit_counts must be positive or 'all'")
        self.full_visit_training_fraction = float(
            config["training"].get("full_visit_training_fraction", 0.0)
        )
        if not 0.0 <= self.full_visit_training_fraction <= 1.0:
            raise ValueError("full_visit_training_fraction must lie in [0, 1]")
        self.no_source_fraction = float(config["observation"].get("no_source_fraction", 0.0))
        self.reported_error_fraction = float(
            config["training"].get("paired_reported_error_fraction", 0.0)
        )
        if not 0.0 <= self.reported_error_fraction <= 1.0:
            raise ValueError("paired_reported_error_fraction must lie in [0, 1]")
        self.observed_flux_fraction = float(
            config["training"].get("observed_flux_fraction", 0.0)
        )
        if not 0.0 <= self.observed_flux_fraction <= 1.0:
            raise ValueError("observed_flux_fraction must lie in [0, 1]")
        self.include_flux_error_channel = bool(
            config["data"].get("include_flux_error_channel", False)
        )
        self.include_clean_flux_target = bool(
            config["data"].get("include_clean_flux_target", False)
        )
        self.observed_flux_blank_family = str(
            config["training"].get("observed_flux_blank_family", "residual")
        )
        if self.observed_flux_blank_family not in {
            "residual",
            "controlled_background",
        }:
            raise ValueError(
                "observed_flux_blank_family must be 'residual' or "
                "'controlled_background'"
            )
        self.noise_scale_augmentation_fraction = float(
            config["training"].get("noise_scale_augmentation_fraction", 0.0)
        )
        if not 0.0 <= self.noise_scale_augmentation_fraction <= 1.0:
            raise ValueError("noise_scale_augmentation_fraction must lie in [0, 1]")
        noise_scale_range = config["training"].get("noise_scale_range", [1.0, 1.0])
        if not isinstance(noise_scale_range, list) or len(noise_scale_range) != 2:
            raise ValueError("noise_scale_range must contain [minimum, maximum]")
        self.minimum_noise_scale = float(noise_scale_range[0])
        self.maximum_noise_scale = float(noise_scale_range[1])
        if not 0.0 <= self.minimum_noise_scale <= self.maximum_noise_scale:
            raise ValueError("noise_scale_range must be nonnegative and ordered")
        prepared = project_path(config, config["data"]["prepared_dir"])
        objects = pd.read_parquet(prepared / "objects.parquet")
        objects = _apply_class_scheme(objects, config)
        self.objects = objects[objects["split"] == split].reset_index(drop=True)
        runtime_limits = config["data"].get("runtime_object_limits", {})
        self.objects = _apply_runtime_object_limit(
            self.objects,
            split=split,
            runtime_limits=runtime_limits,
            seed=self.seed,
        )
        if "source_id" in self.objects:
            self._source_keys = self.objects["source_id"].astype(str).tolist()
        else:
            self._source_keys = self.objects["snid"].astype(str).tolist()
        self._observation_counts = (
            self.objects["observation_count"].astype(int).tolist()
        )
        self.observations = pd.read_parquet(prepared / "observations.parquet")
        self.template_support_policy = template_support_policy(config["observation"])
        self.require_complete_template_support = self.template_support_policy == "complete"
        if self.require_complete_template_support:
            required = {"native_wavelength_min", "native_wavelength_max"}
            missing = required - set(self.observations.columns)
            if missing:
                raise ValueError(
                    "Prepared observations lack template-support metadata; run data "
                    "preparation again. Missing: " + ", ".join(sorted(missing))
                )
        self.h5_path = prepared / "spectra.h5"
        if not self.h5_path.is_file():
            # Prepared datasets written before the naming cleanup remain readable.
            self.h5_path = prepared / "native_spectra.h5"
        self._store: h5py.File | None = None
        observation_config = config["observation"]
        self.output_wavelength = log_wavelength_grid(
            observation_config["wavelength_min"],
            observation_config["wavelength_max"],
            observation_config["wavelength_bins"],
        )
        if self.require_complete_template_support:
            used_objects = set(self.objects["object_index"].astype(int))
            used = self.observations["object_index"].isin(used_objects)
            used_observations = self.observations.loc[used]
            incomplete = used_observations.loc[
                used_observations["native_wavelength_min"].gt(
                    self.output_wavelength[0]
                )
                | used_observations["native_wavelength_max"].lt(
                    self.output_wavelength[-1]
                )
            ]
            if len(incomplete):
                examples = incomplete["snid"].head().astype(str).tolist()
                raise ValueError(
                    "Prepared split contains template-truncated spectra inside the "
                    "requested observer interval; run data preparation again. SNID(s): "
                    + ", ".join(examples)
                )

    def __len__(self) -> int:
        multiplier = 2 if self.pair_no_source else 1
        return multiplier * len(self.objects)

    def requested_visit_count(self, item: int) -> int:
        """Return the redshift-independent visit limit assigned to one item."""
        object_position = item // 2 if self.pair_no_source else item
        return self._requested_visit_count(
            self._source_keys[object_position],
            self._observation_counts[object_position],
        )

    def set_epoch(self, epoch: int) -> None:
        self._training_epoch.fill_(int(epoch))

    @property
    def training_epoch(self) -> int:
        return int(self._training_epoch.item())

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        if self.pair_no_source:
            object_position = item // 2
            paired_no_source = bool(item % 2)
        else:
            object_position = item
            paired_no_source = False
        object_row = self.objects.iloc[object_position]
        source_key = (
            str(object_row.source_id)
            if "source_id" in object_row.index
            else str(int(object_row.snid))
        )
        first = int(object_row.first_observation)
        count = int(object_row.observation_count)
        observation_rows = self.observations.iloc[first:first + count].copy()
        if not np.all(observation_rows["object_index"].to_numpy() == object_row.object_index):
            raise RuntimeError(f"Observation table is inconsistent for SNID {object_row.snid}")
        chosen = self._choose_visits(observation_rows, source_key)
        native_block = self._read_native_block(chosen)
        observed_signal_to_noise: float | None = None
        observed_snr_valid_bins: int | None = None
        if self.include_observed_signal_to_noise:
            observed_signal_to_noise, observed_snr_valid_bins = (
                _coadded_observed_signal_to_noise(
                    chosen,
                    native_block,
                    wavelength_min=float(self.output_wavelength[0]),
                    wavelength_max=float(self.output_wavelength[-1]),
                )
            )
        if self.paired_noise_seed is not None:
            noise_key = (
                str(object_row.paired_noise_key)
                if "paired_noise_key" in object_row.index
                else source_key
            )
            native_block["paired_standard_normal"] = _paired_standard_normal(
                chosen,
                native_block,
                noise_key=noise_key,
                repeat=self.visit_repeat,
                seed=self.paired_noise_seed,
            )

        force_no_source = paired_no_source or self.view in {
            "no_source",
            "residual",
            "reported_error_no_source",
        }
        if self.training and not self.pair_no_source and self.no_source_fraction > 0:
            draw = repeatable_rng(
                self.seed, source_key, self.training_epoch, "evidence_sufficiency"
            ).random()
            force_no_source = bool(draw < self.no_source_fraction)
        generated_noise_family = "source_free"
        if self.training and self.view == "generated":
            observed_draw = repeatable_rng(
                self.seed, source_key, self.training_epoch, "observed_flux_family"
            ).random()
            if observed_draw < self.observed_flux_fraction:
                generated_noise_family = "original"
            else:
                family_draw = repeatable_rng(
                    self.seed, source_key, self.training_epoch, "generated_noise_family"
                ).random()
                if family_draw < self.reported_error_fraction:
                    generated_noise_family = "reported_error"
        noise_scale = (
            1.0
            if generated_noise_family == "original"
            else self._training_noise_scale(source_key)
        )

        fluxes = []
        masks = []
        clean_signals = []
        flux_error_shapes = []
        times = []
        visit_flux_scales = []
        for row in chosen.itertuples(index=False):
            flux, mask, clean_signal, flux_error_shape = self._read_and_resample(
                row,
                source_key,
                force_no_source,
                generated_noise_family,
                noise_scale,
                native_block,
            )
            fluxes.append(flux)
            masks.append(mask)
            clean_signals.append(clean_signal)
            if flux_error_shape is not None:
                flux_error_shapes.append(flux_error_shape)
            times.append(float(row.mjd))
            visit_flux_scales.append(float(row.background_scale))
        first_mjd = times[0]
        observer_days = [mjd - first_mjd for mjd in times]
        estimated_peak_mjd = float(
            object_row.estimated_peak_mjd
            if "estimated_peak_mjd" in object_row.index
            else np.nan
        )
        peak_date_valid = bool(
            np.isfinite(estimated_peak_mjd) and estimated_peak_mjd > 0.0
        )
        peak_day_offset = estimated_peak_mjd - first_mjd if peak_date_valid else 0.0
        simulation_peak_mjd = float(
            object_row.simulation_peak_mjd
            if "simulation_peak_mjd" in object_row.index
            else object_row.peak_mjd
        )
        simulation_rest_phase_days = [
            (mjd - simulation_peak_mjd) / (1.0 + float(object_row.redshift))
            for mjd in times
        ]
        clean_array = np.stack(clean_signals)
        mask_array = np.stack(masks)
        clean_signal_to_noise = _coadded_clean_signal_to_noise(clean_array, mask_array)
        effective_signal_to_noise = clean_signal_to_noise / max(noise_scale, 1.0e-3)
        if force_no_source:
            evidence_sufficiency_target = 0.0
        else:
            # A smooth target avoids pretending that S/N=0.99 and S/N=1.01
            # are qualitatively different.  It equals 0.5 at coadded S/N=1.
            evidence_sufficiency_target = float(
                1.0
                / (
                    1.0
                    + np.exp(
                        -4.0 * np.log(max(effective_signal_to_noise, 1.0e-6))
                    )
                )
            )
        result = {
            "flux": torch.from_numpy(np.stack(fluxes)),
            "wavelength_mask": torch.from_numpy(mask_array),
            # Dates remain relative to the first retained visit; absolute MJD
            # is never exposed to the model.
            "observer_days": torch.tensor(observer_days, dtype=torch.float32),
            # PEAKMJD is an observer-frame light-curve estimate.  Only its
            # offset from the first retained visit is exposed, and only to the
            # deterministic candidate-phase comparison when that route is on.
            "peak_day_offset": torch.tensor(peak_day_offset, dtype=torch.float32),
            "peak_date_valid": torch.tensor(
                float(peak_date_valid), dtype=torch.float32
            ),
            # This scalar reverses the visit-level preprocessing scale only
            # inside the deterministic relative-amplitude calculation. It is
            # never exposed directly to a learned layer.
            "visit_flux_scale": torch.tensor(
                visit_flux_scales, dtype=torch.float32
            ),
            # This simulation-only value may supervise the phase head or build
            # a bank. measurement_inputs never passes it into the model.
            "simulation_rest_phase_days": torch.tensor(
                simulation_rest_phase_days, dtype=torch.float32
            ),
            "class_index": torch.tensor(int(object_row.class_index), dtype=torch.long),
            "redshift": torch.tensor(float(object_row.redshift), dtype=torch.float32),
            "has_source": torch.tensor(0.0 if force_no_source else 1.0, dtype=torch.float32),
            "evidence_sufficiency_target": torch.tensor(
                evidence_sufficiency_target, dtype=torch.float32
            ),
            "coadded_clean_signal_to_noise": torch.tensor(
                clean_signal_to_noise, dtype=torch.float32
            ),
            "snid": torch.tensor(int(object_row.snid), dtype=torch.long),
        }
        if observed_signal_to_noise is not None and observed_snr_valid_bins is not None:
            # This deployable summary uses only the original FLAM and FLAMERR
            # measurements for the same visits retained by this evaluation
            # item. It is independent of the requested synthetic/clean view and
            # is skipped during training to avoid unnecessary loader overhead.
            result["median_coadded_observed_signal_to_noise"] = torch.tensor(
                observed_signal_to_noise, dtype=torch.float32
            )
            result["observed_snr_valid_wavelength_bins"] = torch.tensor(
                observed_snr_valid_bins, dtype=torch.long
            )
        if self.include_flux_error_channel:
            result["flux_error_shape"] = torch.from_numpy(
                np.stack(flux_error_shapes)
            )
        if self.include_clean_flux_target:
            # This is a simulation-only supervision target. measurement_inputs
            # deliberately excludes it from every runtime model call.
            result["clean_flux_target"] = torch.from_numpy(clean_array)
        return result

    def observed_signal_to_noise_record(
        self,
        item: int,
        *,
        edge_trim_fraction: float = OBSERVED_SNR_EDGE_TRIM_FRACTION,
        maximum_relative_error: float = OBSERVED_SNR_MAX_RELATIVE_ERROR,
    ) -> dict[str, float | int]:
        """Return measured S/N without constructing model input tensors."""
        record, _ = self.observed_signal_to_noise_records(
            item,
            edge_trim_fraction=edge_trim_fraction,
            maximum_relative_error=maximum_relative_error,
            include_epoch_history=False,
        )
        return record

    def observed_signal_to_noise_records(
        self,
        item: int,
        *,
        edge_trim_fraction: float = OBSERVED_SNR_EDGE_TRIM_FRACTION,
        maximum_relative_error: float = OBSERVED_SNR_MAX_RELATIVE_ERROR,
        include_epoch_history: bool = True,
    ) -> tuple[
        dict[str, float | int],
        list[dict[str, float | int | bool]],
    ]:
        """Return final and per-observational-epoch measured S/N records."""
        if self.pair_no_source:
            raise ValueError("S/N catalog generation requires unpaired dataset items")
        object_row = self.objects.iloc[item]
        source_key = (
            str(object_row.source_id)
            if "source_id" in object_row.index
            else str(int(object_row.snid))
        )
        first = int(object_row.first_observation)
        count = int(object_row.observation_count)
        rows = self.observations.iloc[first:first + count].copy()
        chosen = self._choose_visits(rows, source_key)
        native_block = self._read_native_block(chosen)
        epochs = _observed_signal_to_noise_series(
            chosen,
            native_block,
            wavelength_min=float(self.output_wavelength[0]),
            wavelength_max=float(self.output_wavelength[-1]),
            edge_trim_fraction=edge_trim_fraction,
            maximum_relative_error=maximum_relative_error,
            include_epoch_history=include_epoch_history,
        )
        if epochs:
            final = epochs[-1]
            value = float(final["median_coadded_observed_signal_to_noise"])
            valid_bins = int(final["observed_snr_valid_wavelength_bins"])
        else:
            value = float("nan")
            valid_bins = 0
        record: dict[str, float | int] = {
            "snid": int(object_row.snid),
            "visit_count": int(len(chosen)),
            "median_coadded_observed_signal_to_noise": float(value),
            "observed_snr_valid_wavelength_bins": int(valid_bins),
        }
        epoch_records: list[dict[str, float | int | bool]] = []
        if include_epoch_history:
            for epoch in epochs:
                epoch_records.append({"snid": int(object_row.snid), **epoch})
        return record, epoch_records

    def _choose_visits(self, rows: pd.DataFrame, source_key: str) -> pd.DataFrame:
        rows = rows.sort_values("mjd").reset_index(drop=True)
        visit_limit = self._requested_visit_count(source_key, len(rows))
        if len(rows) <= visit_limit:
            return rows
        if self.training or self.visit_selection == "random":
            rng = repeatable_rng(
                self.seed,
                source_key,
                self.training_epoch if self.training else self.visit_repeat,
                "random_visits",
            )
            indices = np.sort(rng.choice(len(rows), size=visit_limit, replace=False))
        else:
            # Deterministic evaluation uses the full observed time span.
            positions = np.linspace(0, len(rows) - 1, visit_limit)
            indices = np.unique(np.rint(positions).astype(int))
        return rows.iloc[indices].reset_index(drop=True)

    def _requested_visit_count(self, source_key: str, available_visits: int) -> int:
        configured_limit = (
            available_visits
            if self.max_visits is None
            else min(available_visits, self.max_visits)
        )
        if not self.training or not self.training_visit_counts:
            return configured_limit
        full_draw = repeatable_rng(
            self.seed, source_key, self.training_epoch, "full_visit_training"
        ).random()
        if full_draw < self.full_visit_training_fraction:
            return configured_limit
        count_rng = repeatable_rng(
            self.seed, source_key, self.training_epoch, "visit_count"
        )
        choice = self.training_visit_counts[
            int(count_rng.integers(len(self.training_visit_counts)))
        ]
        requested = available_visits if choice is None else choice
        return min(
            configured_limit,
            requested,
        )

    def _training_noise_scale(self, source_key: str) -> float:
        if self.generated_noise_scale is not None:
            return self.generated_noise_scale
        if (
            not self.training
            or self.view != "generated"
            or self.noise_scale_augmentation_fraction == 0.0
        ):
            return 1.0
        rng = repeatable_rng(
            self.seed, source_key, self.training_epoch, "noise_scale"
        )
        if rng.random() >= self.noise_scale_augmentation_fraction:
            return 1.0
        return float(rng.uniform(self.minimum_noise_scale, self.maximum_noise_scale))

    def _read_and_resample(
        self,
        row: Any,
        source_key: str,
        force_no_source: bool,
        generated_noise_family: str,
        noise_scale: float,
        native_block: dict[str, np.ndarray | int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        start = int(row.first_bin) - int(native_block["first_bin"])
        stop = start + int(row.bin_count)
        wavelength_min = np.asarray(native_block["wavelength_min"])[start:stop]
        wavelength_max = np.asarray(native_block["wavelength_max"])[start:stop]
        wavelength = 0.5 * (wavelength_min + wavelength_max)
        clean = np.asarray(native_block["clean_flux"])[start:stop]
        flux_error = np.asarray(native_block["flux_error"])[start:stop]
        native_error = flux_error
        scale = max(float(row.background_scale), np.finfo(np.float32).tiny)

        if self.view == "generated" and generated_noise_family == "original":
            if (
                force_no_source
                and self.observed_flux_blank_family == "controlled_background"
            ):
                rng = repeatable_rng(
                    self.seed,
                    source_key,
                    int(row.observation_index),
                    self.training_epoch,
                    "observed_flux_controlled_blank",
                )
                native_flux, native_error = draw_controlled_background_measurement(
                    wavelength_angstrom=wavelength,
                    clean_flux=clean,
                    background_scale=scale,
                    include_source=False,
                    settings=self.config["observation"]["generated_noise"],
                    rng=rng,
                    noise_scale=1.0,
                )
            else:
                observed = np.asarray(native_block["observed_flux"])[start:stop]
                native_flux = observed - clean if force_no_source else observed
        elif self.view == "generated" and generated_noise_family == "reported_error":
            rng = repeatable_rng(
                self.seed,
                source_key,
                int(row.observation_index),
                self.training_epoch,
                "training_reported_error",
            )
            native_flux = draw_reported_error_flux(
                clean,
                flux_error,
                include_source=not force_no_source,
                rng=rng,
                noise_scale=noise_scale,
            )
            native_error = float(noise_scale) * flux_error
        elif self.view == "original" and not force_no_source:
            native_flux = np.asarray(native_block["observed_flux"])[start:stop]
        elif self.view == "residual":
            native_flux = np.asarray(native_block["observed_flux"])[start:stop] - clean
        elif self.view == "clean":
            native_flux = np.zeros_like(clean) if force_no_source else clean
        elif self.view in {"reported_error_with_source", "reported_error_no_source"}:
            if self.paired_noise_seed is not None:
                standard_normal = np.asarray(
                    native_block["paired_standard_normal"]
                )[start:stop]
                signal = (
                    clean
                    if self.view == "reported_error_with_source"
                    else np.zeros_like(clean)
                )
                native_flux = signal + noise_scale * flux_error * standard_normal
            else:
                rng_key: tuple[object, ...] = (
                    self.seed,
                    source_key,
                    int(row.observation_index),
                    "reported_error_control",
                )
                # Preserve the historical deterministic control at its default
                # scale.  Explicit noise sweeps add a repeat key so several paired
                # draws can be compared without changing the draw between scales.
                if self.generated_noise_scale is not None:
                    rng_key += (self.visit_repeat,)
                rng = repeatable_rng(*rng_key)
                native_flux = draw_reported_error_flux(
                    clean,
                    flux_error,
                    include_source=self.view == "reported_error_with_source",
                    rng=rng,
                    noise_scale=noise_scale,
                )
            native_error = float(noise_scale) * flux_error
        else:
            noise_key = (
                "controlled_background"
                if self.view in {"generated", "no_source"}
                else self.view
            )
            rng = repeatable_rng(
                self.seed,
                source_key,
                int(row.observation_index),
                self.training_epoch if self.training else self.visit_repeat,
                noise_key,
            )
            native_flux, native_error = draw_controlled_background_measurement(
                wavelength_angstrom=wavelength,
                clean_flux=clean,
                background_scale=scale,
                include_source=not force_no_source,
                settings=self.config["observation"]["generated_noise"],
                rng=rng,
                noise_scale=noise_scale,
            )

        finite_wavelength = np.isfinite(wavelength)
        wavelength = wavelength[finite_wavelength]
        wavelength_min = wavelength_min[finite_wavelength]
        wavelength_max = wavelength_max[finite_wavelength]
        native_flux = native_flux[finite_wavelength] / scale
        native_error = native_error[finite_wavelength] / scale
        if len(wavelength) < 2:
            empty = np.zeros(self.output_wavelength.shape, dtype=np.float32)
            error_shape = empty if self.include_flux_error_channel else None
            return empty, empty, empty, error_shape
        order = np.argsort(wavelength)
        wavelength = wavelength[order]
        wavelength_min = wavelength_min[order]
        wavelength_max = wavelength_max[order]
        native_flux = native_flux[order]
        native_error = native_error[order]
        clean_scaled = clean[finite_wavelength][order] / scale
        finite_flux = np.isfinite(native_flux) & np.isfinite(clean_scaled)
        finite_error = np.isfinite(native_error) & (native_error > 0.0)
        if self.require_complete_template_support:
            if (
                float(row.native_wavelength_min) > self.output_wavelength[0]
                or float(row.native_wavelength_max) < self.output_wavelength[-1]
            ):
                raise ValueError(
                    f"SNID {int(row.snid)} lacks complete template support over the requested observer interval"
                )
            in_detector = (
                wavelength_max >= self.output_wavelength[0]
            ) & (wavelength_min <= self.output_wavelength[-1])
            if not finite_flux[in_detector].all():
                raise ValueError(
                    f"SNID {int(row.snid)} has non-finite flux inside the requested observer interval"
                )
            inside = np.ones(self.output_wavelength.shape, dtype=bool)
        else:
            inside = (
                (self.output_wavelength >= wavelength[0])
                & (self.output_wavelength <= wavelength[-1])
            )
            native_flux = np.where(finite_flux, native_flux, 0.0)
            clean_scaled = np.where(finite_flux, clean_scaled, 0.0)
        output = np.zeros(self.output_wavelength.shape, dtype=np.float32)
        output[inside] = np.interp(
            self.output_wavelength[inside], wavelength, native_flux
        ).astype(np.float32)
        clean_output = np.zeros(self.output_wavelength.shape, dtype=np.float32)
        clean_output[inside] = np.interp(
            self.output_wavelength[inside], wavelength, clean_scaled
        ).astype(np.float32)
        error_output = None
        output_mask = inside.copy()
        if self.include_flux_error_channel:
            # Propagate variance through the same linear interpolation used for
            # flux.  Interpolating FLAMERR or log(FLAMERR) directly would give
            # incorrect inverse-variance weights for the coadd.
            error_output = np.zeros(self.output_wavelength.shape, dtype=np.float32)
            target_wavelength = self.output_wavelength[inside]
            upper = np.searchsorted(wavelength, target_wavelength, side="left")
            upper = np.clip(upper, 1, len(wavelength) - 1)
            lower = upper - 1
            denominator = wavelength[upper] - wavelength[lower]
            interpolation_valid = (
                finite_error[lower]
                & finite_error[upper]
                & np.isfinite(denominator)
                & (denominator > 0.0)
            )
            fraction = np.divide(
                target_wavelength - wavelength[lower],
                denominator,
                out=np.zeros_like(target_wavelength, dtype=np.float64),
                where=denominator > 0.0,
            )
            variance = (
                np.square(1.0 - fraction) * np.square(native_error[lower])
                + np.square(fraction) * np.square(native_error[upper])
            )
            propagated_error = np.sqrt(variance)
            interpolation_valid &= (
                np.isfinite(propagated_error) & (propagated_error > 0.0)
            )
            inside_indices = np.flatnonzero(inside)
            valid_indices = inside_indices[interpolation_valid]
            error_output[valid_indices] = np.log(
                propagated_error[interpolation_valid]
            ).astype(np.float32)
            output_mask[inside_indices[~interpolation_valid]] = False
        output[~output_mask] = 0.0
        clean_output[~output_mask] = 0.0
        if error_output is not None:
            error_output[~output_mask] = 0.0
        return output, output_mask.astype(np.float32), clean_output, error_output

    def _read_native_block(self, rows: pd.DataFrame) -> dict[str, np.ndarray | int]:
        """Read all chosen visits in one HDF5 slice to avoid repeated decompression."""
        first_bin = int(rows["first_bin"].min())
        last_bin = int((rows["first_bin"] + rows["bin_count"]).max())
        store = self._open_store()
        block: dict[str, np.ndarray | int] = {"first_bin": first_bin}
        for name in (
            "wavelength_min",
            "wavelength_max",
            "observed_flux",
            "flux_error",
            "clean_flux",
        ):
            block[name] = np.asarray(store[name][first_bin:last_bin], dtype=np.float32)
        return block

    def _open_store(self) -> h5py.File:
        if self._store is None:
            self._store = h5py.File(self.h5_path, "r")
        return self._store

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_store"] = None
        return state


def _paired_standard_normal(
    rows: pd.DataFrame,
    native_block: dict[str, np.ndarray | int],
    *,
    noise_key: str,
    repeat: int,
    seed: int,
) -> np.ndarray:
    """Reproduce the frozen-v2 native-bin Gaussian draw for selected visits.

    One object-level generator is advanced through visits in chronological
    order.  The same standard-normal realization is then multiplied by each
    requested FLAMERR scale, so a noise sweep changes amplitude only.
    """
    text = f"{seed}:{noise_key}:{repeat}"
    number = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
    rng = np.random.default_rng(number)
    first_bin = int(native_block["first_bin"])
    flux_error = np.asarray(native_block["flux_error"])
    standard_normal = np.zeros(flux_error.shape, dtype=np.float32)
    for row in rows.sort_values("mjd").itertuples(index=False):
        start = int(row.first_bin) - first_bin
        stop = start + int(row.bin_count)
        errors = flux_error[start:stop]
        good = np.isfinite(errors) & (errors > 0.0)
        visit_draw = standard_normal[start:stop]
        visit_draw[good] = rng.normal(size=int(good.sum())).astype(np.float32)
    return standard_normal


def _apply_runtime_object_limit(
    objects: pd.DataFrame,
    *,
    split: str,
    runtime_limits: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    """Apply a deterministic development cap; zero means no runtime cap."""
    runtime_limit = runtime_limits.get(split)
    if runtime_limit is None or int(runtime_limit) <= 0:
        return objects
    limit = int(runtime_limit)
    if len(objects) <= limit:
        return objects
    return (
        objects.sample(n=limit, random_state=seed)
        .sort_index()
        .reset_index(drop=True)
    )


def collate_objects(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    maximum_visits = max(item["flux"].shape[0] for item in items)
    wavelength_bins = items[0]["flux"].shape[1]
    batch_size = len(items)
    flux = torch.zeros(batch_size, maximum_visits, wavelength_bins, dtype=torch.float32)
    wavelength_mask = torch.zeros_like(flux)
    flux_error_shape = (
        torch.zeros_like(flux) if "flux_error_shape" in items[0] else None
    )
    observer_days = torch.zeros(batch_size, maximum_visits, dtype=torch.float32)
    visit_flux_scale = torch.ones(
        batch_size, maximum_visits, dtype=torch.float32
    )
    simulation_rest_phase_days = torch.zeros(
        batch_size, maximum_visits, dtype=torch.float32
    )
    visit_mask = torch.zeros(batch_size, maximum_visits, dtype=torch.float32)
    for batch_index, item in enumerate(items):
        visits = item["flux"].shape[0]
        flux[batch_index, :visits] = item["flux"]
        wavelength_mask[batch_index, :visits] = item["wavelength_mask"]
        if flux_error_shape is not None:
            flux_error_shape[batch_index, :visits] = item["flux_error_shape"]
        observer_days[batch_index, :visits] = item["observer_days"]
        visit_flux_scale[batch_index, :visits] = item["visit_flux_scale"]
        simulation_rest_phase_days[batch_index, :visits] = item[
            "simulation_rest_phase_days"
        ]
        visit_mask[batch_index, :visits] = 1.0
    result = {
        "flux": flux,
        "wavelength_mask": wavelength_mask,
        "observer_days": observer_days,
        "peak_day_offset": torch.stack([item["peak_day_offset"] for item in items]),
        "peak_date_valid": torch.stack([item["peak_date_valid"] for item in items]),
        "visit_flux_scale": visit_flux_scale,
        "simulation_rest_phase_days": simulation_rest_phase_days,
        "visit_mask": visit_mask,
        "class_index": torch.stack([item["class_index"] for item in items]),
        "redshift": torch.stack([item["redshift"] for item in items]),
        "has_source": torch.stack([item["has_source"] for item in items]),
        "evidence_sufficiency_target": torch.stack(
            [item["evidence_sufficiency_target"] for item in items]
        ),
        "coadded_clean_signal_to_noise": torch.stack(
            [item["coadded_clean_signal_to_noise"] for item in items]
        ),
        "snid": torch.stack([item["snid"] for item in items]),
    }
    if "median_coadded_observed_signal_to_noise" in items[0]:
        result["median_coadded_observed_signal_to_noise"] = torch.stack(
            [item["median_coadded_observed_signal_to_noise"] for item in items]
        )
        result["observed_snr_valid_wavelength_bins"] = torch.stack(
            [item["observed_snr_valid_wavelength_bins"] for item in items]
        )
    if flux_error_shape is not None:
        result["flux_error_shape"] = flux_error_shape
    if "clean_flux_target" in items[0]:
        clean_flux_target = torch.zeros_like(flux)
        for batch_index, item in enumerate(items):
            visits = item["clean_flux_target"].shape[0]
            clean_flux_target[batch_index, :visits] = item[
                "clean_flux_target"
            ]
        result["clean_flux_target"] = clean_flux_target
    return result


def _coadded_clean_signal_to_noise(clean: np.ndarray, mask: np.ndarray) -> float:
    valid = mask > 0
    visit_count = valid.sum(axis=0)
    present = visit_count > 0
    if not np.any(present):
        return 0.0
    mean_clean = (clean * valid).sum(axis=0) / np.maximum(visit_count, 1)
    per_wavelength = np.abs(mean_clean[present]) * np.sqrt(visit_count[present])
    return float(np.median(per_wavelength))


def _coadded_observed_signal_to_noise(
    rows: pd.DataFrame,
    native_block: dict[str, np.ndarray | int],
    *,
    wavelength_min: float | None = None,
    wavelength_max: float | None = None,
    edge_trim_fraction: float = OBSERVED_SNR_EDGE_TRIM_FRACTION,
    maximum_relative_error: float = OBSERVED_SNR_MAX_RELATIVE_ERROR,
) -> tuple[float, int]:
    """Return median per-bin S/N of the inverse-variance coadded observations.

    This uses only observed FLAM and its reported FLAMERR. It aligns native
    wavelength bins across the selected visits, computes the usual
    inverse-variance coadd and propagated error in every wavelength bin, then
    takes the signed median S/N across a fixed interior observer-frame band.
    The outer five percent at either end of the configured log-wavelength range
    is excluded by default, as are bins whose propagated error is more than
    three times the median in-band error. Both masks use wavelength or reported
    error only—never flux, class, redshift, or clean simulation truth. The signed
    statistic has a noise-only expectation near zero; taking an absolute value
    would introduce a positive low-signal bias.
    """
    series = _observed_signal_to_noise_series(
        rows,
        native_block,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        edge_trim_fraction=edge_trim_fraction,
        maximum_relative_error=maximum_relative_error,
        include_epoch_history=False,
    )
    if not series:
        return float("nan"), 0
    final = series[-1]
    return (
        float(final["median_coadded_observed_signal_to_noise"]),
        int(final["observed_snr_valid_wavelength_bins"]),
    )


def _observed_signal_to_noise_series(
    rows: pd.DataFrame,
    native_block: dict[str, np.ndarray | int],
    *,
    wavelength_min: float | None = None,
    wavelength_max: float | None = None,
    edge_trim_fraction: float = OBSERVED_SNR_EDGE_TRIM_FRACTION,
    maximum_relative_error: float = OBSERVED_SNR_MAX_RELATIVE_ERROR,
    include_epoch_history: bool = True,
) -> list[dict[str, float | int | bool]]:
    """Return measured S/N for each epoch and each cumulative history.

    Epochs are ordered by MJD. The single-epoch value uses only that spectrum;
    the cumulative value applies the same inverse-variance coadd used by
    :func:`_coadded_observed_signal_to_noise` to all spectra through that epoch.
    """
    if rows.empty:
        return []
    ordered = rows.sort_values("mjd").reset_index(drop=True)
    first_bin = int(native_block["first_bin"])
    reference_row = ordered.loc[ordered["bin_count"].astype(int).idxmax()]
    reference_start = int(reference_row.first_bin) - first_bin
    reference_stop = reference_start + int(reference_row.bin_count)
    reference = 0.5 * (
        np.asarray(native_block["wavelength_min"], dtype=np.float64)[
            reference_start:reference_stop
        ]
        + np.asarray(native_block["wavelength_max"], dtype=np.float64)[
            reference_start:reference_stop
        ]
    )
    if not 0.0 <= edge_trim_fraction < 0.5:
        raise ValueError("edge_trim_fraction must lie in [0, 0.5)")
    if maximum_relative_error <= 1.0:
        raise ValueError("maximum_relative_error must exceed one")
    declared_minimum = float(reference[0] if wavelength_min is None else wavelength_min)
    declared_maximum = float(reference[-1] if wavelength_max is None else wavelength_max)
    if declared_minimum <= 0.0 or declared_maximum <= declared_minimum:
        raise ValueError("Observed S/N wavelength bounds must be positive and ordered")
    log_minimum = np.log(declared_minimum)
    log_maximum = np.log(declared_maximum)
    interior_minimum = float(
        np.exp(log_minimum + edge_trim_fraction * (log_maximum - log_minimum))
    )
    interior_maximum = float(
        np.exp(log_minimum + (1.0 - edge_trim_fraction) * (log_maximum - log_minimum))
    )
    lookup = {
        round(float(wavelength), 3): index
        for index, wavelength in enumerate(reference)
    }
    visits: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, int]] = []
    all_errors: list[np.ndarray] = []
    for epoch_index, row in enumerate(ordered.itertuples(index=False), start=1):
        start = int(row.first_bin) - first_bin
        stop = start + int(row.bin_count)
        wavelength = 0.5 * (
            np.asarray(native_block["wavelength_min"], dtype=np.float64)[start:stop]
            + np.asarray(native_block["wavelength_max"], dtype=np.float64)[start:stop]
        )
        flux = np.asarray(native_block["observed_flux"], dtype=np.float64)[start:stop]
        error = np.asarray(native_block["flux_error"], dtype=np.float64)[start:stop]
        indices = np.asarray(
            [lookup.get(round(float(value), 3), -1) for value in wavelength],
            dtype=np.int64,
        )
        valid = (
            (indices >= 0)
            & np.isfinite(flux)
            & np.isfinite(error)
            & (error > 0.0)
        )
        if valid.any():
            visits.append(
                (
                    indices[valid],
                    flux[valid],
                    error[valid],
                    float(row.mjd),
                    int(getattr(row, "observation_index", epoch_index - 1)),
                )
            )
            all_errors.append(error[valid])
        else:
            visits.append(
                (
                    np.asarray([], dtype=np.int64),
                    np.asarray([], dtype=np.float64),
                    np.asarray([], dtype=np.float64),
                    float(row.mjd),
                    int(getattr(row, "observation_index", epoch_index - 1)),
                )
            )
    # Scaling cancels algebraically but avoids extreme inverse variances for
    # physical FLAM units around 1e-20.
    unit = (
        float(np.median(np.concatenate(all_errors)))
        if all_errors
        else float("nan")
    )
    precision = np.zeros(len(reference), dtype=np.float64)
    weighted_flux = np.zeros(len(reference), dtype=np.float64)
    first_mjd = float(visits[0][3])
    if not include_epoch_history:
        if np.isfinite(unit) and unit > 0.0:
            for indices, flux, error, _, _ in visits:
                if not len(indices):
                    continue
                scaled_flux = flux / unit
                scaled_error = error / unit
                weight = 1.0 / np.square(scaled_error)
                np.add.at(precision, indices, weight)
                np.add.at(weighted_flux, indices, weight * scaled_flux)
        value, valid_bins = _observed_signal_to_noise_from_coadd(
            reference,
            precision,
            weighted_flux,
            interior_minimum=interior_minimum,
            interior_maximum=interior_maximum,
            maximum_relative_error=maximum_relative_error,
        )
        return [
            {
                "median_coadded_observed_signal_to_noise": value,
                "observed_snr_valid_wavelength_bins": valid_bins,
            }
        ]
    result: list[dict[str, float | int | bool]] = []
    for epoch_index, (indices, flux, error, mjd, observation_index) in enumerate(
        visits,
        start=1,
    ):
        epoch_precision = np.zeros(len(reference), dtype=np.float64)
        epoch_weighted_flux = np.zeros(len(reference), dtype=np.float64)
        if np.isfinite(unit) and unit > 0.0 and len(indices):
            scaled_flux = flux / unit
            scaled_error = error / unit
            weight = 1.0 / np.square(scaled_error)
            np.add.at(epoch_precision, indices, weight)
            np.add.at(epoch_weighted_flux, indices, weight * scaled_flux)
            np.add.at(precision, indices, weight)
            np.add.at(weighted_flux, indices, weight * scaled_flux)
        epoch_value, epoch_valid_bins = _observed_signal_to_noise_from_coadd(
            reference,
            epoch_precision,
            epoch_weighted_flux,
            interior_minimum=interior_minimum,
            interior_maximum=interior_maximum,
            maximum_relative_error=maximum_relative_error,
        )
        cumulative_value, cumulative_valid_bins = (
            _observed_signal_to_noise_from_coadd(
                reference,
                precision,
                weighted_flux,
                interior_minimum=interior_minimum,
                interior_maximum=interior_maximum,
                maximum_relative_error=maximum_relative_error,
            )
        )
        result.append(
            {
                "observation_epoch": epoch_index,
                "observation_index": observation_index,
                "mjd": mjd,
                "observer_days": mjd - first_mjd,
                "visit_count": epoch_index,
                "total_visit_count": len(visits),
                "median_epoch_observed_signal_to_noise": epoch_value,
                "epoch_observed_snr_valid_wavelength_bins": epoch_valid_bins,
                "median_coadded_observed_signal_to_noise": cumulative_value,
                "observed_snr_valid_wavelength_bins": cumulative_valid_bins,
                "is_final_epoch": epoch_index == len(visits),
            }
        )
    return result


def _observed_signal_to_noise_from_coadd(
    reference: np.ndarray,
    precision: np.ndarray,
    weighted_flux: np.ndarray,
    *,
    interior_minimum: float,
    interior_maximum: float,
    maximum_relative_error: float,
) -> tuple[float, int]:
    """Summarize aligned inverse-variance accumulators with fixed quality masks."""
    valid_bins = (
        np.isfinite(precision)
        & (precision > 0.0)
        & (reference >= interior_minimum)
        & (reference <= interior_maximum)
    )
    if not valid_bins.any():
        return float("nan"), 0
    coadded_error = np.full(len(reference), np.nan, dtype=np.float64)
    coadded_error[valid_bins] = 1.0 / np.sqrt(precision[valid_bins])
    median_error = float(np.median(coadded_error[valid_bins]))
    if not np.isfinite(median_error) or median_error <= 0.0:
        return float("nan"), 0
    valid_bins &= coadded_error <= maximum_relative_error * median_error
    if not valid_bins.any():
        return float("nan"), 0
    per_bin = weighted_flux[valid_bins] / np.sqrt(precision[valid_bins])
    finite = np.isfinite(per_bin)
    if not finite.any():
        return float("nan"), 0
    return float(np.median(per_bin[finite])), int(finite.sum())


def _apply_class_scheme(
    objects: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Map stored simulation labels to the class grouping used by this run."""
    scheme = str(config["data"].get("class_scheme", "normal_ia_binary"))
    class_names = class_names_for_scheme(scheme)
    declared = tuple(str(name) for name in config["model"]["classes"])
    if declared != class_names:
        raise ValueError(
            f"model.classes {declared} do not match {scheme}: {class_names}"
        )
    required = {"gentype", "template_index"}
    if not required.issubset(objects.columns):
        return objects
    mapped = [
        class_name_for_source(gentype, template, scheme)
        for gentype, template in zip(objects["gentype"], objects["template_index"])
    ]
    result = objects.copy()
    result["class_name"] = mapped
    result = result[result["class_name"].notna()].copy()
    class_to_index = {name: index for index, name in enumerate(class_names)}
    result["class_index"] = result["class_name"].map(class_to_index).astype(np.int64)
    return result
