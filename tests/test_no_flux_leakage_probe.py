"""Regression probes for truth leakage through non-flux model inputs.

Each test states an expected value, measures it, and asserts pass/fail.
Run: pytest tests/test_no_flux_leakage_probe.py -s
Uses STRIDER_CONFIG when set, otherwise the local binary development store.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

from strider.config import load_config, project_path
from strider.data.dataset import SundialDataset
from strider.data.noise import draw_controlled_background_flux, repeatable_rng
from strider.model import measurement_inputs

DEFAULT_CONFIG = "configs/experiments/local_ia_binary_20k.yaml"
PROBE_OBJECTS = 1200
PERMUTATION_REPEATS = 99


def _config():
    config_path = Path(os.environ.get("STRIDER_CONFIG", DEFAULT_CONFIG))
    try:
        config = load_config(config_path)
    except Exception as error:  # pragma: no cover
        pytest.skip(f"config unavailable: {error}")
    prepared = project_path(config, config["data"]["prepared_dir"])
    if not (prepared / "objects.parquet").is_file():
        pytest.skip(f"prepared data unavailable: {prepared}")
    return config


def _mask_probe(config, objects=PROBE_OBJECTS):
    """Return (valid output bins per object, true redshift) with flux never read."""
    dataset = SundialDataset(
        config, "train", "generated", training=False, pair_no_source=False
    )
    counts, redshifts = [], []
    for item in _representative_positions(len(dataset), objects):
        record = dataset[item]
        covered = record["wavelength_mask"].numpy().max(axis=0) > 0
        counts.append(float(covered.sum()))
        redshifts.append(float(record["redshift"]))
    return np.asarray(counts), np.asarray(redshifts)


def _representative_positions(length, count):
    """Return a deterministic sample spread across the active population."""
    sample_size = min(int(count), int(length))
    if sample_size == length:
        return np.arange(length, dtype=np.int64)
    rng = np.random.default_rng(73031)
    return np.sort(rng.choice(length, size=sample_size, replace=False))


def _single_mad_reduction(feature, redshift, seed, bins=10):
    if np.ptp(feature) == 0.0:
        return 0.0
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(feature))
    split = 2 * len(order) // 3
    train, test = order[:split], order[split:]
    edges = np.unique(np.quantile(feature[train], np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return 0.0
    train_bin = np.digitize(feature[train], edges[1:-1])
    test_bin = np.digitize(feature[test], edges[1:-1])
    fallback = float(np.median(redshift[train]))
    medians = {
        key: float(np.median(redshift[train][train_bin == key]))
        for key in np.unique(train_bin)
    }
    predicted = np.asarray([medians.get(key, fallback) for key in test_bin])
    baseline = np.mean(np.abs(redshift[test] - fallback))
    measured = np.mean(np.abs(redshift[test] - predicted))
    return 100.0 * (1.0 - measured / baseline)


def _mad_reduction(feature, redshift, bins=10, repeats=21):
    """Return the median held-out gain of a binned one-feature predictor."""
    reductions = [
        _single_mad_reduction(feature, redshift, 73031 + repeat, bins)
        for repeat in range(repeats)
    ]
    return float(np.median(reductions))


def _permuted_reduction_limit(feature, redshift, quantile=0.95):
    """Return the finite-sample null limit for the same binned predictor."""
    rng = np.random.default_rng(190831)
    reductions = [
        _mad_reduction(feature, rng.permutation(redshift))
        for _ in range(PERMUTATION_REPEATS)
    ]
    return float(np.quantile(reductions, quantile))


def _safe_spearman(first, second):
    if np.ptp(first) == 0.0 or np.ptp(second) == 0.0:
        return 0.0
    value = float(spearmanr(first, second)[0])
    return 0.0 if not np.isfinite(value) else value


def test_positive_control_noise_generator_moments():
    """Control: the background envelope must reproduce its analytic moments."""
    settings = {
        "background_shape_strength": 0.20,
        "background_scale_log_std": 0.20,
    }
    shape_strength = settings["background_shape_strength"]
    scale_log_std = settings["background_scale_log_std"]
    draws = []
    for index in range(4000):
        rng = repeatable_rng(73031, "control", index, 0, "generated")
        draws.append(
            [
                rng.uniform(-shape_strength, shape_strength),
                rng.uniform(0.0, shape_strength),
                np.exp(rng.normal(0.0, scale_log_std)),
            ]
        )
    draws = np.asarray(draws)
    assert abs(draws[:, 0].mean()) < 0.01
    assert abs(draws[:, 0].std() - shape_strength / np.sqrt(3.0)) < 0.01
    assert abs(draws[:, 1].mean() - shape_strength / 2.0) < 0.01
    expected_scale = np.exp(0.5 * np.square(scale_log_std))
    assert abs(draws[:, 2].mean() - expected_scale) < 0.02


def test_generated_noise_is_independent_across_visits():
    """v2 failure mode: repeated visits must not share one wavelength shape."""
    wavelength = np.linspace(7500.0, 20000.0, 300)
    clean = np.zeros_like(wavelength)
    settings = {
        "background_shape_strength": 0.20,
        "background_scale_log_std": 0.20,
        "source_variance_fraction": 0.0,
    }
    visits = np.stack(
        [
            draw_controlled_background_flux(
                wavelength,
                clean,
                1.0,
                False,
                settings,
                repeatable_rng(73031, "one_object", index, 0, "generated"),
            )
            for index in range(12)
        ]
    )
    correlation = np.corrcoef(visits)
    off_diagonal = np.abs(correlation[np.triu_indices(len(visits), 1)])
    assert off_diagonal.max() < 0.35, "visits share a wavelength pattern (v2 regression)"


def test_measurement_inputs_is_a_whitelist():
    allowed = {
        "flux",
        "wavelength_mask",
        "visit_mask",
        "observer_days",
        "visit_flux_scale",
        "peak_day_offset",
        "peak_date_valid",
    }
    import torch

    batch = {name: torch.zeros(2) for name in allowed}
    batch.update(
        {
            "redshift": torch.zeros(2),
            "class_index": torch.zeros(2),
            "simulation_rest_phase_days": torch.zeros(2),
            "SIM_FLAM": torch.zeros(2),
            "simulation_peak_mjd": torch.zeros(2),
            "SIM_PEAKMJD": torch.zeros(2),
        }
    )
    assert set(measurement_inputs(batch)) == allowed


def test_wavelength_mask_does_not_carry_redshift():
    """The mask alone must not predict redshift."""
    config = _config()
    counts, redshifts = _mask_probe(config)
    rho = _safe_spearman(counts, redshifts)
    reduction = _mad_reduction(counts, redshifts)
    print(f"\n  spearman(valid output bins, z) = {rho:+.4f}")
    print(f"  median redshift MAD reduction from mask alone = {reduction:.1f}%")
    assert abs(rho) < 0.10, f"mask leaks redshift: rho={rho:+.4f}"


def test_no_source_amplitude_does_not_carry_redshift():
    """Noise amplitude must not materially beat its finite-sample null."""
    config = _config()
    dataset = SundialDataset(
        config, "train", "generated", training=True, pair_no_source=True
    )
    dataset.set_epoch(0)
    amplitudes, redshifts = [], []
    # Relabelling can remove unsupported objects, so the first N rows are not a
    # comparable population across class schemes. Probe a deterministic sample
    # distributed over every eligible training object instead.
    for position in _representative_positions(len(dataset) // 2, PROBE_OBJECTS):
        partner = dataset[2 * position + 1]
        mask = partner["wavelength_mask"].numpy() > 0
        if not mask.any():
            continue
        amplitudes.append(float(np.std(partner["flux"].numpy()[mask])))
        redshifts.append(float(partner["redshift"]))
    rho = _safe_spearman(np.asarray(amplitudes), np.asarray(redshifts))
    amplitudes = np.asarray(amplitudes)
    redshifts = np.asarray(redshifts)
    reduction = _mad_reduction(amplitudes, redshifts)
    null_limit = _permuted_reduction_limit(amplitudes, redshifts)
    allowed_reduction = max(5.0, null_limit)
    print(f"\n  spearman(blank amplitude, z) = {rho:+.4f}")
    print(f"  median redshift MAD reduction from blank amplitude = {reduction:.1f}%")
    print(f"  shuffled-null 95% reduction limit = {null_limit:.1f}%")
    assert abs(rho) < 0.10, f"blank amplitude leaks redshift: rho={rho:+.4f}"
    assert reduction < 10.0, "blank amplitude has a large redshift association"
    assert reduction <= allowed_reduction, (
        "blank amplitude improves redshift recovery beyond its shuffled null: "
        f"measured={reduction:.1f}%, allowed={allowed_reduction:.1f}%"
    )


def test_paired_partners_share_dates_and_masks():
    import torch

    config = _config()
    dataset = SundialDataset(
        config, "train", "generated", training=True, pair_no_source=True
    )
    dataset.set_epoch(0)
    for position in range(200):
        source = dataset[2 * position]
        partner = dataset[2 * position + 1]
        assert torch.equal(source["observer_days"], partner["observer_days"])
        assert torch.equal(source["wavelength_mask"], partner["wavelength_mask"])
