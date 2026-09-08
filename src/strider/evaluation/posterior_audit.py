"""Post-process saved redshift posteriors without rerunning the model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strider.config import project_path
from strider.model.redshift_scan import build_redshift_grid, redshift_cell_widths

from .evaluate import _posterior_basin_candidates
from .metrics import metrics_by_redshift, source_metrics


def audit_saved_posteriors(
    config: dict[str, Any],
    *,
    predictions_path: Path | None = None,
    split: str | None = None,
    view: str = "original",
    output_tag: str = "basin",
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Add distinct-solution summaries to a saved evaluation prediction table."""
    configured_output = project_path(config, config["project"]["output_dir"])
    output_dir = Path(output_dir) if output_dir is not None else configured_output
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_split = split or str(config["evaluation"].get("split", "test"))
    source_path = (
        predictions_path
        if predictions_path is not None
        else configured_output / f"{evaluation_split}_predictions_{view}.parquet"
    )
    source_path = Path(source_path)
    predictions = pd.read_parquet(source_path)
    if "redshift_probability" not in predictions:
        raise ValueError(
            f"{source_path} has no redshift_probability column; rerun evaluation "
            "with save_redshift_probability enabled"
        )

    model = config["model"]
    grid = build_redshift_grid(
        float(model["redshift_min"]),
        float(model["redshift_max"]),
        int(model["redshift_bins"]),
        str(model.get("redshift_spacing", "linear")),
    ).astype(np.float64)
    cell_width = redshift_cell_widths(grid).astype(np.float64)
    enriched = append_posterior_basin_candidates(predictions, grid, cell_width)

    destination = output_dir / (
        f"{evaluation_split}_predictions_{view}_{output_tag}.parquet"
    )
    summary_path = output_dir / (
        f"{evaluation_split}_posterior_audit_{view}_{output_tag}.json"
    )
    enriched.to_parquet(destination, index=False)
    threshold = float(config["evaluation"]["outlier_delta_z"])
    report = source_metrics(enriched, threshold)
    report["redshift_groups"] = metrics_by_redshift(
        enriched,
        threshold,
        config["evaluation"].get(
            "ia_redshift_edges", config["data"]["redshift_edges"]
        ),
    )
    summary = {
        "source_predictions": str(source_path),
        "enriched_predictions": str(destination),
        "objects": int(len(enriched)),
        "metrics": report,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def append_posterior_basin_candidates(
    predictions: pd.DataFrame,
    redshift_grid: np.ndarray,
    cell_width: np.ndarray,
) -> pd.DataFrame:
    """Apply the released basin summary to a saved posterior table."""
    enriched = predictions.copy()
    candidate_rows = []
    for stored_probability in enriched["redshift_probability"]:
        probability = np.asarray(stored_probability, dtype=np.float64)
        if probability.shape != redshift_grid.shape:
            raise ValueError(
                "Saved redshift probability does not match the configured grid"
            )
        candidate_rows.append(
            _posterior_basin_candidates(redshift_grid, probability, cell_width)
        )
    enriched["posterior_primary_redshift"] = [
        row[0]["median_redshift"] for row in candidate_rows
    ]
    enriched["posterior_primary_peak_redshift"] = [
        row[0]["peak_redshift"] for row in candidate_rows
    ]
    if "posterior_median_redshift" not in enriched and "predicted_redshift" in enriched:
        enriched["posterior_median_redshift"] = enriched["predicted_redshift"]
    if "full_posterior_lower_68" not in enriched and "redshift_lower_68" in enriched:
        enriched["full_posterior_lower_68"] = enriched["redshift_lower_68"]
    if "full_posterior_upper_68" not in enriched and "redshift_upper_68" in enriched:
        enriched["full_posterior_upper_68"] = enriched["redshift_upper_68"]
    enriched["predicted_redshift"] = enriched["posterior_primary_peak_redshift"]
    enriched["z_strider"] = enriched["posterior_primary_peak_redshift"]
    enriched["redshift_point_estimator"] = "primary_basin_peak"
    enriched["posterior_primary_lower_68"] = [
        row[0]["lower_68"] for row in candidate_rows
    ]
    enriched["posterior_primary_upper_68"] = [
        row[0]["upper_68"] for row in candidate_rows
    ]
    enriched["redshift_lower_68"] = enriched["posterior_primary_lower_68"]
    enriched["redshift_upper_68"] = enriched["posterior_primary_upper_68"]
    enriched["redshift_68_interval_width"] = (
        enriched["redshift_upper_68"] - enriched["redshift_lower_68"]
    )
    enriched["posterior_primary_basin_lower"] = [
        row[0]["basin_lower"] for row in candidate_rows
    ]
    enriched["posterior_primary_basin_upper"] = [
        row[0]["basin_upper"] for row in candidate_rows
    ]
    enriched["posterior_primary_basin_mass"] = [
        row[0]["mass"] for row in candidate_rows
    ]
    enriched["posterior_primary_is_largest_mass_basin"] = [
        row[0]["is_largest_mass_basin"] for row in candidate_rows
    ]
    enriched["posterior_primary_competitor_peak_redshift"] = [
        row[0]["strongest_competitor_peak_redshift"] for row in candidate_rows
    ]
    enriched["posterior_primary_log_peak_to_competitor_saddle_ratio"] = [
        row[0]["log_peak_to_strongest_competitor_saddle_ratio"]
        for row in candidate_rows
    ]
    enriched["posterior_candidate_count"] = [len(row) for row in candidate_rows]
    enriched["posterior_candidate_redshifts"] = [
        [candidate["median_redshift"] for candidate in row]
        for row in candidate_rows
    ]
    enriched["posterior_candidate_peak_redshifts"] = [
        [candidate["peak_redshift"] for candidate in row]
        for row in candidate_rows
    ]
    enriched["posterior_candidate_masses"] = [
        [candidate["mass"] for candidate in row] for row in candidate_rows
    ]
    enriched["posterior_candidate_is_largest_mass_basin"] = [
        [candidate["is_largest_mass_basin"] for candidate in row]
        for row in candidate_rows
    ]
    enriched["posterior_candidate_lower_68"] = [
        [candidate["lower_68"] for candidate in row] for row in candidate_rows
    ]
    enriched["posterior_candidate_upper_68"] = [
        [candidate["upper_68"] for candidate in row] for row in candidate_rows
    ]
    return enriched
