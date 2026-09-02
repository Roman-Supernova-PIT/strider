#!/usr/bin/env python3
"""Compare matched blend and detail-only dense-scan runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


SCIENCE_METRICS = (
    ("balanced accuracy", "class_balanced_accuracy_present", "higher"),
    ("Ia F1", "Ia_f1", "higher"),
    ("Ia median |dz|", "Ia_median_absolute_delta_z", "lower"),
    ("Ia out>0.1", "Ia_outlier_fraction_abs_delta_z_gt_0_10", "lower"),
)
VIEWS = ("original", "generated", "clean")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blend", type=Path, required=True, help="Blend run directory")
    parser.add_argument(
        "--detail", type=Path, required=True, help="Detail-only run directory"
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _run_summary(run_dir: Path) -> dict[str, Any]:
    evaluation = _read_json(run_dir / "test_evaluation_summary.json")
    history_path = run_dir / "training_history.json"
    history = _read_json(history_path) if history_path.is_file() else []
    config_path = run_dir / "config.resolved.yaml"
    config = yaml.safe_load(config_path.read_text()) if config_path.is_file() else {}
    epoch_seconds = [
        float(row["epoch_seconds"])
        for row in history
        if row.get("epoch_seconds") is not None
    ]
    return {
        "run_dir": str(run_dir.resolve()),
        "name": config.get("project", {}).get("name", run_dir.name),
        "dense_scan_view": config.get("model", {}).get("dense_scan_view", "legacy"),
        "checkpoint_epoch": int(evaluation["checkpoint_epoch"]),
        "epochs_completed": len(history),
        "median_epoch_minutes": (
            float(np.median(epoch_seconds) / 60.0) if epoch_seconds else None
        ),
        "evaluation": evaluation,
        "route_check": _optional_json(run_dir / "route_check_summary.json"),
        "alias_audit": _optional_alias_audit(run_dir),
    }


def _optional_json(path: Path) -> dict[str, Any] | None:
    return _read_json(path) if path.is_file() else None


def _optional_alias_audit(run_dir: Path) -> list[dict[str, Any]] | None:
    path = run_dir / "alias_audit" / "test_original_route_summary.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path).replace({np.nan: None}).to_dict(orient="records")


def _metric(view: dict[str, Any], name: str) -> float:
    if name in view:
        return float(view[name])
    if name == "Ia_outlier_fraction_abs_delta_z_gt_0_10":
        return float(view["Ia_outlier_fraction"])
    raise KeyError(name)


def _science_rows(
    blend: dict[str, Any], detail: dict[str, Any]
) -> list[dict[str, Any]]:
    blend_views = blend["evaluation"]["views"]
    detail_views = detail["evaluation"]["views"]
    rows = []
    for view_name in VIEWS:
        if view_name not in blend_views or view_name not in detail_views:
            continue
        for label, key, preferred in SCIENCE_METRICS:
            blend_value = _metric(blend_views[view_name], key)
            detail_value = _metric(detail_views[view_name], key)
            rows.append(
                {
                    "view": view_name,
                    "metric": label,
                    "preferred": preferred,
                    "blend": blend_value,
                    "detail": detail_value,
                    "detail_minus_blend": detail_value - blend_value,
                }
            )
    return rows


def _route_row(run: dict[str, Any], view: str, route: str) -> dict[str, Any] | None:
    report = run["route_check"]
    if report is None:
        return None
    return next(
        (
            row
            for row in report["rows"]
            if row["view"] == view and row["route"] == route
        ),
        None,
    )


def _blank_control(
    blend: dict[str, Any], detail: dict[str, Any]
) -> dict[str, Any] | None:
    blend_row = _route_row(blend, "no_source", "combined")
    detail_row = _route_row(detail, "no_source", "combined")
    if blend_row is None or detail_row is None:
        return None
    return {
        "blend_lock": float(blend_row["blank_redshift_lock"]),
        "detail_lock": float(detail_row["blank_redshift_lock"]),
        "detail_minus_blend_lock": float(detail_row["blank_redshift_lock"])
        - float(blend_row["blank_redshift_lock"]),
        "blend_z_score": float(blend_row["blank_lock_z_score"]),
        "detail_z_score": float(detail_row["blank_lock_z_score"]),
    }


def _alias_rows(
    blend: dict[str, Any], detail: dict[str, Any]
) -> list[dict[str, Any]]:
    if blend["alias_audit"] is None or detail["alias_audit"] is None:
        return []
    blend_rows = {
        (str(row["cohort"]), str(row["route"])): row
        for row in blend["alias_audit"]
    }
    detail_rows = {
        (str(row["cohort"]), str(row["route"])): row
        for row in detail["alias_audit"]
    }
    rows = []
    for key in sorted(blend_rows.keys() & detail_rows.keys()):
        blend_value = float(blend_rows[key]["fraction_route_prefers_true"])
        detail_value = float(detail_rows[key]["fraction_route_prefers_true"])
        rows.append(
            {
                "cohort": key[0],
                "route": key[1],
                "blend_prefers_true": blend_value,
                "detail_prefers_true": detail_value,
                "detail_minus_blend": detail_value - blend_value,
            }
        )
    return rows


def _gate(
    science: list[dict[str, Any]],
    blank: dict[str, Any] | None,
    blend: dict[str, Any],
    detail: dict[str, Any],
) -> dict[str, Any]:
    checks = []
    for row in science:
        tolerance = {
            "balanced accuracy": 0.01,
            "Ia F1": 0.01,
            "Ia median |dz|": 0.01,
            "Ia out>0.1": 0.02,
        }[row["metric"]]
        difference = float(row["detail_minus_blend"])
        passed = (
            difference >= -tolerance
            if row["preferred"] == "higher"
            else difference <= tolerance
        )
        checks.append(
            {
                "name": f"{row['view']} {row['metric']}",
                "passed": passed,
                "tolerance": tolerance,
                "detail_minus_blend": difference,
            }
        )
    if blank is not None:
        checks.append(
            {
                "name": "combined blank redshift lock",
                "passed": blank["detail_minus_blend_lock"] <= 0.02
                and abs(blank["detail_z_score"]) <= 2.5,
                "tolerance": 0.02,
                "detail_minus_blend": blank["detail_minus_blend_lock"],
                "detail_z_score": blank["detail_z_score"],
            }
        )
    runtime = None
    if (
        blend["median_epoch_minutes"] is not None
        and detail["median_epoch_minutes"] is not None
    ):
        runtime = {
            "blend_minutes": blend["median_epoch_minutes"],
            "detail_minutes": detail["median_epoch_minutes"],
            "fraction_saved": 1.0
            - detail["median_epoch_minutes"] / blend["median_epoch_minutes"],
        }
    complete = bool(science) and blank is not None
    passed = complete and all(check["passed"] for check in checks)
    if not complete:
        decision = "incomplete: run full test evaluation and route checks"
    elif passed and runtime is None:
        decision = "science gate passed; runtime comparison pending"
    elif passed and runtime["fraction_saved"] > 0.0:
        decision = "promote detail-only to a second-seed confirmation"
    elif passed:
        decision = "science gate passed, but detail-only did not reduce runtime"
    else:
        decision = "keep the learned blend"
    return {
        "status": decision,
        "all_available_checks_passed": passed,
        "checks": checks,
        "runtime": runtime,
        "note": "These are provisional pilot tolerances, not publication cuts.",
    }


def compare_runs(blend_dir: Path, detail_dir: Path) -> dict[str, Any]:
    blend = _run_summary(blend_dir)
    detail = _run_summary(detail_dir)
    science = _science_rows(blend, detail)
    blank = _blank_control(blend, detail)
    aliases = _alias_rows(blend, detail)
    report = {
        "comparison": "detail-only minus learned whole/detail blend",
        "blend": {key: value for key, value in blend.items() if key not in {"evaluation", "route_check", "alias_audit"}},
        "detail": {key: value for key, value in detail.items() if key not in {"evaluation", "route_check", "alias_audit"}},
        "science": science,
        "blank_control": blank,
        "alias_audit": aliases,
    }
    report["gate"] = _gate(science, blank, blend, detail)
    return report


def _print_report(report: dict[str, Any]) -> None:
    print("\nDense redshift-scan comparison")
    print("  detail-only minus learned whole/detail blend")
    print(
        "  view       metric                 blend    detail     delta"
    )
    for row in report["science"]:
        print(
            f"  {row['view']:<10} {row['metric']:<21} "
            f"{row['blend']:>7.4f}  {row['detail']:>7.4f}  "
            f"{row['detail_minus_blend']:>+8.4f}"
        )
    blank = report["blank_control"]
    if blank is not None:
        print(
            "  blank lock (combined)      "
            f"{blank['blend_lock']:>7.4f}  {blank['detail_lock']:>7.4f}  "
            f"{blank['detail_minus_blend_lock']:>+8.4f}"
        )
    runtime = report["gate"]["runtime"]
    if runtime is not None:
        print(
            "  median epoch minutes       "
            f"{runtime['blend_minutes']:>7.1f}  {runtime['detail_minutes']:>7.1f}  "
            f"{100.0 * runtime['fraction_saved']:>+7.1f}%"
        )
    print(f"\n  decision: {report['gate']['status']}")


def main() -> None:
    arguments = _arguments()
    report = compare_runs(arguments.blend, arguments.detail)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    _print_report(report)
    print(f"  results: {arguments.output}")


if __name__ == "__main__":
    main()
