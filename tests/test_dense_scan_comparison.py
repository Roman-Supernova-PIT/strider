import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.compare_dense_scan_runs import compare_runs


def _write_run(
    root: Path,
    name: str,
    *,
    view: str,
    metric_shift: float,
    blank_shift: float,
    epoch_seconds: float,
) -> Path:
    run = root / name
    run.mkdir()
    config = {
        "project": {"name": name},
        "model": {"dense_scan_view": view},
    }
    (run / "config.resolved.yaml").write_text(yaml.safe_dump(config))
    history = [{"epoch": 1, "epoch_seconds": epoch_seconds}]
    (run / "training_history.json").write_text(json.dumps(history))
    metrics = {
        "class_balanced_accuracy_present": 0.90 + metric_shift,
        "Ia_f1": 0.88 + metric_shift,
        "Ia_median_absolute_delta_z": 0.04 - metric_shift,
        "Ia_outlier_fraction_abs_delta_z_gt_0_10": 0.12 - metric_shift,
    }
    evaluation = {
        "checkpoint_epoch": 1,
        "views": {name: metrics for name in ("original", "generated", "clean")},
    }
    (run / "test_evaluation_summary.json").write_text(json.dumps(evaluation))
    routes = {
        "rows": [
            {
                "view": "no_source",
                "route": "combined",
                "blank_redshift_lock": 0.10 + blank_shift,
                "blank_lock_z_score": 0.5,
            }
        ]
    }
    (run / "route_check_summary.json").write_text(json.dumps(routes))
    alias = run / "alias_audit"
    alias.mkdir()
    pd.DataFrame(
        [
            {
                "cohort": "alias",
                "route": "combined",
                "fraction_route_prefers_true": 0.7 + metric_shift,
            }
        ]
    ).to_csv(alias / "test_original_route_summary.csv", index=False)
    return run


def test_detail_scan_comparison_passes_small_matched_changes(tmp_path: Path) -> None:
    blend = _write_run(
        tmp_path,
        "blend",
        view="blend",
        metric_shift=0.0,
        blank_shift=0.0,
        epoch_seconds=100.0,
    )
    detail = _write_run(
        tmp_path,
        "detail",
        view="detail",
        metric_shift=-0.002,
        blank_shift=0.005,
        epoch_seconds=80.0,
    )

    report = compare_runs(blend, detail)

    assert report["gate"]["all_available_checks_passed"] is True
    assert report["gate"]["runtime"]["fraction_saved"] == pytest.approx(0.2)
    assert report["alias_audit"][0]["detail_minus_blend"] == pytest.approx(-0.002)
    assert "second-seed" in report["gate"]["status"]


def test_detail_scan_comparison_rejects_science_regression(tmp_path: Path) -> None:
    blend = _write_run(
        tmp_path,
        "blend",
        view="blend",
        metric_shift=0.0,
        blank_shift=0.0,
        epoch_seconds=100.0,
    )
    detail = _write_run(
        tmp_path,
        "detail",
        view="detail",
        metric_shift=-0.03,
        blank_shift=0.0,
        epoch_seconds=70.0,
    )

    report = compare_runs(blend, detail)

    assert report["gate"]["all_available_checks_passed"] is False
    assert report["gate"]["status"] == "keep the learned blend"
