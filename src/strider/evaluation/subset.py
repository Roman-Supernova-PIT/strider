"""Shared object subsets for diagnostic comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd


def stratified_positions(
    objects: pd.DataFrame,
    max_objects: int,
    seed: int,
    redshift_edges: list[float],
) -> list[int]:
    """Choose one deterministic class-redshift sample from a prepared split."""
    target = min(len(objects), int(max_objects))
    if target < 1:
        raise ValueError("max_objects must be positive and the split must be nonempty")
    if target == len(objects):
        return list(range(target))

    class_index = objects["class_index"].to_numpy(dtype=np.int64)
    redshift_bin = np.digitize(
        objects["redshift"].to_numpy(dtype=np.float64),
        np.asarray(redshift_edges, dtype=np.float64)[1:-1],
    )
    labels = class_index.astype(str) + ":" + redshift_bin.astype(str)
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    expected = np.asarray([len(group) for group in groups], dtype=np.float64)
    expected *= target / expected.sum()
    quotas = np.floor(expected).astype(int)

    if target >= len(groups):
        quotas = np.maximum(quotas, 1)
    while quotas.sum() > target:
        candidates = np.flatnonzero(quotas > 1)
        remove = candidates[np.argmax(quotas[candidates] - expected[candidates])]
        quotas[remove] -= 1
    while quotas.sum() < target:
        capacity = np.asarray([len(group) for group in groups]) - quotas
        candidates = np.flatnonzero(capacity > 0)
        add = candidates[np.argmax(expected[candidates] - quotas[candidates])]
        quotas[add] += 1

    rng = np.random.default_rng(int(seed))
    selected = []
    for group, quota in zip(groups, quotas, strict=True):
        selected.extend(rng.permutation(group)[:quota].tolist())
    return sorted(int(position) for position in selected)
