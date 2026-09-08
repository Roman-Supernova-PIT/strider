"""Metrics for source-free and evidence-sufficiency controls."""

from __future__ import annotations

import numpy as np
import pandas as pd


def blank_redshift_metrics(
    predictions: pd.DataFrame,
    seed: int,
    tolerance: float = 0.1,
    permutations: int = 200,
) -> dict[str, float | int]:
    predicted = predictions["predicted_redshift"].to_numpy(dtype=np.float64)
    truth = predictions["true_redshift"].to_numpy(dtype=np.float64)
    difference = np.abs(predicted - truth)
    lock = float(np.mean(difference <= tolerance))
    correlation = _safe_correlation(predicted, truth)

    rng = np.random.default_rng(int(seed))
    chance = np.asarray(
        [np.mean(np.abs(rng.permutation(predicted) - truth) <= tolerance) for _ in range(permutations)]
    )
    chance_std = float(chance.std(ddof=1)) if len(chance) > 1 else 0.0
    return {
        "N": int(len(predictions)),
        "blank_redshift_lock": lock,
        "blank_median_absolute_delta_z": float(np.median(difference)),
        "blank_redshift_truth_correlation": correlation,
        "prior_chance_lock_mean": float(chance.mean()),
        "prior_chance_lock_std": chance_std,
        "blank_lock_z_score": (
            float((lock - chance.mean()) / chance_std) if chance_std > 0.0 else 0.0
        ),
    }


def sufficiency_auc(source: pd.DataFrame, blank: pd.DataFrame) -> float:
    source_score = source["evidence_score"].to_numpy(dtype=np.float64)
    blank_score = blank["evidence_score"].to_numpy(dtype=np.float64)
    scores = np.concatenate([source_score, blank_score])
    ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=np.float64)
    source_rank_sum = ranks[: len(source_score)].sum()
    return float(
        (source_rank_sum - len(source_score) * (len(source_score) + 1) / 2)
        / (len(source_score) * len(blank_score))
    )


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or np.ptp(first) == 0.0 or np.ptp(second) == 0.0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])
