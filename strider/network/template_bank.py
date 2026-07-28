"""Embedded spectral-signature bank loader and validator.

Each class is its own anchor: cells without own-class training support
contribute zero to the match surface rather than borrowing another class's
signatures. Falling 91bg/Iax back to Ia would corrupt the gold-Ia cosmology
sample, because STRIDER's strict Ia definition treats them as contaminants.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F


# Support levels recorded in the bank's `support_flag`.
SUPPORT_UNSUPPORTED = 0
SUPPORT_KMEDOIDS = 2


@dataclass(frozen=True)
class TemplateBank:
    """Loaded train-only ONIR template bank.

    Construct with `from_dict`; validation runs at construction.
    """
    prototype_windows: np.ndarray   # (C, P, nF, K, W) float32, NaN where !active
    prototype_active: np.ndarray    # (C, P, nF, K) bool
    prototype_counts: np.ndarray    # (C, P, nF) int64
    n_subsampled: np.ndarray        # (C, P, nF) int64
    k_actual: np.ndarray            # (C, P, nF) int32
    support_flag: np.ndarray        # (C, P, nF) int32 in {0, 1, 2}
    class_names: np.ndarray         # (C,) str
    phase_bins: np.ndarray          # (P+1,) float32 — bin edges (days)
    feature_names: np.ndarray       # (nF,) str
    feature_rest_waves: np.ndarray  # (nF,) float64
    window_half_bins: int
    source_path: str = ''
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict, source_path: str = '') -> 'TemplateBank':
        # build_metadata is a JSON-stringified 0-d array in the NPZ.
        meta: dict = {}
        if 'build_metadata' in d:
            try:
                import json
                meta = json.loads(str(d['build_metadata']))
            except (json.JSONDecodeError, TypeError, ValueError):
                meta = {'raw': str(d['build_metadata'])}
        bank = cls(
            prototype_windows=d['prototype_windows'],
            prototype_active=d['prototype_active'].astype(bool),
            prototype_counts=d['prototype_counts'].astype(np.int64),
            n_subsampled=d['n_subsampled'].astype(np.int64),
            k_actual=d['k_actual'].astype(np.int32),
            support_flag=d['support_flag'].astype(np.int32),
            class_names=np.asarray(d['class_names']),
            phase_bins=d['phase_bins'].astype(np.float32),
            feature_names=np.asarray(d['feature_names']),
            feature_rest_waves=d['feature_rest_waves'].astype(np.float64),
            window_half_bins=int(np.asarray(d['window_half_bins']).item()),
            source_path=source_path,
            metadata=meta,
        )
        bank.validate()
        return bank

    @property
    def n_classes(self) -> int:
        return self.prototype_windows.shape[0]

    @property
    def n_phase_bins(self) -> int:
        return self.prototype_windows.shape[1]

    @property
    def n_features(self) -> int:
        return self.prototype_windows.shape[2]

    @property
    def k_max(self) -> int:
        return self.prototype_windows.shape[3]

    @property
    def phase_bin_centers(self) -> np.ndarray:
        return 0.5 * (self.phase_bins[1:] + self.phase_bins[:-1])

    def validate(self) -> None:
        C, P, nF, K, W = self.prototype_windows.shape
        if C == 0 or nF == 0 or K == 0 or W == 0:
            raise ValueError(f'bank has zero-size dimension: {self.prototype_windows.shape}')
        if self.prototype_active.shape != (C, P, nF, K):
            raise ValueError(
                f'prototype_active shape {self.prototype_active.shape} != {(C, P, nF, K)}'
            )
        for name, arr, expected in [
            ('prototype_counts', self.prototype_counts, (C, P, nF)),
            ('n_subsampled', self.n_subsampled, (C, P, nF)),
            ('k_actual', self.k_actual, (C, P, nF)),
            ('support_flag', self.support_flag, (C, P, nF)),
        ]:
            if arr.shape != expected:
                raise ValueError(f'{name} shape {arr.shape} != {expected}')
        if self.class_names.shape != (C,):
            raise ValueError(f'class_names shape {self.class_names.shape} != ({C},)')
        if self.feature_names.shape != (nF,):
            raise ValueError(f'feature_names shape {self.feature_names.shape} != ({nF},)')
        if self.feature_rest_waves.shape != (nF,):
            raise ValueError(
                f'feature_rest_waves shape {self.feature_rest_waves.shape} != ({nF},)'
            )
        if self.phase_bins.shape != (P + 1,):
            raise ValueError(f'phase_bins shape {self.phase_bins.shape} != ({P + 1},)')
        if self.support_flag.max() > SUPPORT_KMEDOIDS or self.support_flag.min() < SUPPORT_UNSUPPORTED:
            raise ValueError(
                f'support_flag range {self.support_flag.min()}..{self.support_flag.max()} '
                f'outside [0, {SUPPORT_KMEDOIDS}]'
            )
        # Cross-field consistency: k_actual=0 iff support_flag=UNSUPPORTED.
        # Catches a corrupted bank before it reaches the detector.
        if not np.array_equal(self.support_flag == SUPPORT_UNSUPPORTED, self.k_actual == 0):
            n_bad = int((((self.support_flag == SUPPORT_UNSUPPORTED) != (self.k_actual == 0))).sum())
            raise ValueError(
                f'k_actual/support_flag inconsistent in {n_bad} cells '
                f'(must agree that "no prototypes" ↔ "unsupported")'
            )
        # NaN locations must match ~prototype_active.
        nan_present = np.isnan(self.prototype_windows).any(axis=-1)
        if not np.array_equal(nan_present, ~self.prototype_active):
            n_inconsistent = int((nan_present != ~self.prototype_active).sum())
            raise ValueError(
                f'NaN locations inconsistent with prototype_active '
                f'({n_inconsistent} mismatched cells)'
            )

    def resampled_to(self, window_size: int) -> np.ndarray:
        """Linear-resample prototypes to a target W along the last axis.

        NaN-padded (inactive) entries are propagated as NaN; downstream
        consumers gate on `prototype_active`.
        """
        C, P, nF, K, W = self.prototype_windows.shape
        if window_size == W:
            return self.prototype_windows.copy()
        # F.interpolate doesn't preserve NaN, so zero them, resample,
        # then re-NaN positions according to active mask.
        flat = self.prototype_windows.reshape(-1, W)
        zeroed = np.nan_to_num(flat, nan=0.0).astype(np.float32)
        t = torch.from_numpy(zeroed).unsqueeze(1)  # (N, 1, W)
        resampled = F.interpolate(t, size=window_size, mode='linear',
                                   align_corners=True).squeeze(1).numpy()
        out = resampled.reshape(C, P, nF, K, window_size)
        inactive = ~self.prototype_active
        out[inactive] = np.nan
        return out
