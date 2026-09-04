from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.plot_paired_noise import (
    _classification_summary,
    _normal_noise_summary,
    _paired,
    _plot_classification,
    _plot_nominal_redshift,
    _plot_redshift,
    _plot_reliability,
    _redshift_summary,
    _reliability_summary,
    _wide_predictions,
)


def _model_predictions(model: str) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "snid": [1, 2, 1, 2],
            "noise_scale": [0.5, 0.5, 1.0, 1.0],
            "noise_repeat": [0, 0, 0, 0],
            "true_redshift": [0.4, 1.2, 0.4, 1.2],
            "predicted_redshift": [0.42, 1.4, 0.43, 1.5],
            "p_ia": [0.95, 0.93, 0.91, 0.20],
            "is_ia": [True, False, True, False],
        }
    )
    if model == "v3":
        frame["predicted_redshift"] += 0.01
    return frame


def test_paired_noise_summaries_keep_redshift_and_classification_distinct() -> None:
    paired = _paired(_model_predictions("v2"), _model_predictions("v3"))
    redshift = _redshift_summary(paired)
    classification = _classification_summary(paired)
    reliability = _reliability_summary(paired)

    assert paired["is_ia"].sum() == 2
    assert set(redshift["model"]) == {"v2", "v3"}
    v2_low_z = redshift[
        redshift["model"].eq("v2")
        & redshift["redshift_bin"].eq("0.00–0.75")
        & np.isclose(redshift["noise_scale"], 1.0)
    ].iloc[0]
    assert np.isclose(v2_low_z["median_p_ia"], 0.91)
    assert np.isclose(v2_low_z["nmad_delta_z"], 0.0)
    assert v2_low_z["outlier_fraction_abs_delta_z_gt_0p1"] == 0.0
    nominal_redshift = _normal_noise_summary(redshift)
    assert set(nominal_redshift["model"]) == {"v2", "v3"}
    assert "sigma_delta_z" not in nominal_redshift
    assert "nmad_delta_z" in nominal_redshift
    nominal_v2 = classification[
        classification["model"].eq("v2")
        & classification["scope"].eq("all")
        & np.isclose(classification["noise_scale"], 1.0)
    ].iloc[0]
    assert nominal_v2["purity_p_ia_ge_0p9"] == 1.0
    assert nominal_v2["completeness_p_ia_ge_0p9"] == 1.0
    assert set(reliability["model"]) == {"v2", "v3"}


def test_pairing_rejects_disagreeing_truth_labels() -> None:
    v2 = _model_predictions("v2")
    v3 = _model_predictions("v3")
    v3.loc[0, "is_ia"] = False

    try:
        _paired(v2, v3)
    except ValueError as error:
        assert "Ia label" in str(error)
    else:
        raise AssertionError("truth-label mismatch was not rejected")


def test_v3_only_summary_uses_the_same_metric_contract() -> None:
    v3_only = _wide_predictions(_model_predictions("v3"), "v3")
    redshift = _redshift_summary(v3_only)
    classification = _classification_summary(v3_only)
    reliability = _reliability_summary(v3_only)

    assert set(redshift["model"]) == {"v3"}
    assert set(classification["model"]) == {"v3"}
    assert set(reliability["model"]) == {"v3"}


def test_comparison_figures_render(tmp_path: Path) -> None:
    paired = _paired(_model_predictions("v2"), _model_predictions("v3"))
    redshift = _redshift_summary(paired)
    classification = _classification_summary(paired)
    reliability = _reliability_summary(paired)

    outputs = {
        "redshift": tmp_path / "redshift.png",
        "classification": tmp_path / "classification.png",
        "reliability": tmp_path / "reliability.png",
        "scatter": tmp_path / "scatter.png",
    }
    _plot_redshift(redshift, outputs["redshift"])
    _plot_classification(classification, outputs["classification"])
    _plot_reliability(reliability, outputs["reliability"])
    _plot_nominal_redshift(paired, outputs["scatter"])

    assert all(path.stat().st_size > 0 for path in outputs.values())


def test_v3_only_figures_render(tmp_path: Path) -> None:
    v3_only = _wide_predictions(_model_predictions("v3"), "v3")
    redshift = _redshift_summary(v3_only)

    redshift_output = tmp_path / "redshift_v3.png"
    scatter_output = tmp_path / "scatter_v3.png"
    _plot_redshift(redshift, redshift_output)
    _plot_nominal_redshift(v3_only, scatter_output)

    assert redshift_output.stat().st_size > 0
    assert scatter_output.stat().st_size > 0
