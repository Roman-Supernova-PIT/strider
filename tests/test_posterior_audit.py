import numpy as np
import pandas as pd

from strider.evaluation.posterior_audit import append_posterior_basin_candidates
from strider.model.redshift_scan import build_redshift_grid, redshift_cell_widths


def test_saved_posterior_audit_recovers_primary_and_alternative_basins() -> None:
    grid = build_redshift_grid(0.05, 3.0, 500, "log1p").astype(np.float64)
    width = redshift_cell_widths(grid).astype(np.float64)
    coordinate = np.log1p(grid)
    density = np.exp(-0.5 * ((coordinate - np.log1p(1.1)) / 0.012) ** 2)
    density += 0.25 * np.exp(
        -0.5 * ((coordinate - np.log1p(2.0)) / 0.05) ** 2
    )
    mass = density * width
    mass /= mass.sum()
    predictions = pd.DataFrame(
        {
            "redshift_probability": [mass.astype(np.float32)],
            "predicted_redshift": [1.5],
            "redshift_lower_68": [0.8],
            "redshift_upper_68": [2.2],
        }
    )

    enriched = append_posterior_basin_candidates(predictions, grid, width)

    assert enriched.loc[0, "posterior_candidate_count"] == 2
    assert abs(enriched.loc[0, "posterior_primary_redshift"] - 1.1) < 0.02
    assert abs(enriched.loc[0, "z_strider"] - 1.1) < 0.02
    assert enriched.loc[0, "predicted_redshift"] == enriched.loc[0, "z_strider"]
    assert enriched.loc[0, "posterior_median_redshift"] == 1.5
    assert enriched.loc[0, "full_posterior_lower_68"] == 0.8
    assert enriched.loc[0, "redshift_point_estimator"] == "primary_basin_peak"
    candidates = enriched.loc[0, "posterior_candidate_redshifts"]
    assert abs(candidates[1] - 2.0) < 0.04
