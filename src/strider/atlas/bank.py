"""Validated storage contract for phase-neutral ONIR profiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class OnirBank:
    mean_profiles: np.ndarray
    mean_profile_mask: np.ndarray
    medoid_profiles: np.ndarray
    medoid_profile_mask: np.ndarray
    prototype_profiles: np.ndarray
    prototype_profile_mask: np.ndarray
    prototype_support_counts: np.ndarray
    support_counts: np.ndarray
    class_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    rest_wavelengths: np.ndarray
    feature_radii: np.ndarray
    window_mask: np.ndarray
    overlap_weights: np.ndarray
    rest_grid: np.ndarray
    source_path: str

    def validate(self) -> None:
        classes, features, window = self.mean_profiles.shape
        expected = (classes, features, window)
        if self.medoid_profiles.shape != expected:
            raise ValueError("Mean and medoid ONIR profiles have different shapes")
        if self.mean_profile_mask.shape != expected or self.medoid_profile_mask.shape != expected:
            raise ValueError("ONIR mean or medoid profile mask has an inconsistent shape")
        if self.support_counts.shape != (classes, features):
            raise ValueError("ONIR support_counts shape is inconsistent")
        if self.prototype_profiles.ndim != 4:
            raise ValueError("ONIR prototype profiles must have class, feature, prototype, window axes")
        if self.prototype_profiles.shape[:2] != (classes, features):
            raise ValueError("ONIR prototype class or feature dimension is inconsistent")
        if self.prototype_profiles.shape[-1] != window:
            raise ValueError("ONIR prototype window dimension is inconsistent")
        if self.prototype_support_counts.shape != self.prototype_profiles.shape[:-1]:
            raise ValueError("ONIR prototype support shape is inconsistent")
        if self.prototype_profile_mask.shape != self.prototype_profiles.shape:
            raise ValueError("ONIR prototype profile mask has an inconsistent shape")
        if len(self.class_names) != classes or len(self.feature_names) != features:
            raise ValueError("ONIR names do not match profile dimensions")
        if self.rest_wavelengths.shape != (features,):
            raise ValueError("ONIR wavelength count does not match feature count")
        if self.feature_radii.shape != (features,):
            raise ValueError("ONIR radius count does not match feature count")
        if self.window_mask.shape != (features, window):
            raise ValueError("ONIR window mask shape is inconsistent")
        if self.overlap_weights.shape != (features,):
            raise ValueError("ONIR overlap weight count does not match feature count")
        if not np.all(np.diff(self.rest_wavelengths) > 0):
            raise ValueError("ONIR wavelengths are not strictly increasing")
        expected_mean = np.broadcast_to(
            self.window_mask[None, :, :], self.mean_profile_mask.shape
        )
        mean_active = self.support_counts > 0
        if not np.array_equal(
            self.mean_profile_mask[mean_active], expected_mean[mean_active]
        ):
            raise ValueError("ONIR mean-profile masks differ from feature geometry")
        if not np.array_equal(
            self.medoid_profile_mask[mean_active], expected_mean[mean_active]
        ):
            raise ValueError("ONIR medoid-profile masks differ from feature geometry")
        expected_prototype = np.broadcast_to(
            self.window_mask[None, :, None, :], self.prototype_profile_mask.shape
        )
        prototype_active = self.prototype_support_counts > 0
        if not np.array_equal(
            self.prototype_profile_mask[prototype_active],
            expected_prototype[prototype_active],
        ):
            raise ValueError(
                "ONIR prototype masks differ from feature geometry; rebuild the bank"
            )


def load_onir_bank(path: str | Path) -> OnirBank:
    bank_path = Path(path).expanduser().resolve()
    with np.load(bank_path, allow_pickle=False) as values:
        profile_key = "mean_profiles" if "mean_profiles" in values else "clean_mean_profiles"
        mean_profiles = values[profile_key].astype(np.float32)
        geometry_mask = values["window_mask"].astype(bool)
        if "prototype_profiles" in values:
            prototype_profiles = values["prototype_profiles"].astype(np.float32)
            prototype_support = values["prototype_support_counts"].astype(np.int64)
        else:
            prototype_profiles = mean_profiles[:, :, None, :]
            prototype_support = values["support_counts"].astype(np.int64)[:, :, None]
        mean_profile_mask = (
            values["mean_profile_mask"].astype(bool)
            if "mean_profile_mask" in values
            else np.broadcast_to(geometry_mask[None, :, :], mean_profiles.shape).copy()
        )
        medoid_profile_mask = (
            values["medoid_profile_mask"].astype(bool)
            if "medoid_profile_mask" in values
            else mean_profile_mask.copy()
        )
        prototype_profile_mask = (
            values["prototype_profile_mask"].astype(bool)
            if "prototype_profile_mask" in values
            else np.broadcast_to(
                geometry_mask[None, :, None, :], prototype_profiles.shape
            ).copy()
        )
        bank = OnirBank(
            mean_profiles=mean_profiles,
            mean_profile_mask=mean_profile_mask,
            medoid_profiles=values["medoid_profiles"].astype(np.float32),
            medoid_profile_mask=medoid_profile_mask,
            prototype_profiles=prototype_profiles,
            prototype_profile_mask=prototype_profile_mask,
            prototype_support_counts=prototype_support,
            support_counts=values["support_counts"].astype(np.int64),
            class_names=tuple(str(value) for value in values["class_names"]),
            feature_names=tuple(str(value) for value in values["feature_names"]),
            rest_wavelengths=values["rest_wavelengths"].astype(np.float32),
            feature_radii=values["feature_radii"].astype(np.int64),
            window_mask=geometry_mask,
            overlap_weights=values["overlap_weights"].astype(np.float32),
            rest_grid=values["rest_grid"].astype(np.float32),
            source_path=str(bank_path),
        )
    bank.validate()
    return bank
