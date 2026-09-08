"""Compare the trained evidence routes without changing the checkpoint."""

from __future__ import annotations

import json
from typing import Any

import torch
from torch.utils.data import Subset

from strider.config import project_path
from strider.data.dataset import SundialDataset

from .checkpoint import load_trained_model
from .controls import blank_redshift_metrics
from .evaluate import _predict
from .loader import inference_loader
from .metrics import source_metrics
from .subset import stratified_positions


ROUTES = (
    "onir",
    "named_shape",
    "onir_shape",
    "with_onir_masked",
    "without_onir_masked",
    "without_onir_spectral",
    "without_onir",
    "spectral",
    "combined",
)


def run_route_check(
    config: dict[str, Any],
    max_objects: int = 800,
    views: tuple[str, ...] = ("generated", "clean", "no_source"),
    split: str | None = None,
) -> dict[str, Any]:
    output_dir = project_path(config, config["project"]["output_dir"])
    model, checkpoint, device = load_trained_model(config)
    routes = list(ROUTES)
    if model.dense_scan is not None:
        routes.insert(routes.index("without_onir_masked"), "global_spectrum")
        routes.insert(routes.index("global_spectrum"), "dense")
        routes.insert(routes.index("spectral"), "without_dense")
    if model.continuum_removal is not None:
        dense_position = routes.index("dense") + 1
        dense_view = getattr(model, "dense_scan_view", "blend")
        component_routes = (
            ["dense_detail"]
            if dense_view == "detail"
            else ["dense_whole", "dense_detail"]
        )
        routes[dense_position:dense_position] = component_routes
    evaluation_split = split or str(
        config["evaluation"].get("split", "calibration")
    )
    threshold = float(config["evaluation"]["outlier_delta_z"])
    rows: list[dict[str, Any]] = []
    reference = SundialDataset(config, evaluation_split, views[0], training=False)
    positions = stratified_positions(
        reference.objects,
        max_objects,
        int(config["project"]["seed"]),
        config["data"]["redshift_edges"],
    )
    source_ids = [reference._source_keys[position] for position in positions]

    print("\nEvidence routes", flush=True)
    print(
        f"  checkpoint epoch {checkpoint['epoch']} | {evaluation_split} | up to {max_objects:,} objects",
        flush=True,
    )
    print(
        "  view          route                   balanced   Ia F1   Ia |dz|   out>0.1   blank lock",
        flush=True,
    )
    for view in views:
        dataset = SundialDataset(config, evaluation_split, view, training=False)
        selected_ids = [dataset._source_keys[position] for position in positions]
        if selected_ids != source_ids:
            raise RuntimeError("Evidence routes do not share the same object order")
        selected = Subset(dataset, positions)
        loader = inference_loader(selected, config)
        for route in routes:
            predictions = _predict(model, loader, device, logit_source=route)
            row: dict[str, Any] = {"view": view, "route": route, "N": len(predictions)}
            if view == "no_source":
                row.update(
                    blank_redshift_metrics(
                        predictions,
                        int(config["project"]["seed"]),
                    )
                )
            else:
                metrics = source_metrics(predictions, threshold)
                row.update(
                    {
                        "balanced_accuracy": metrics[
                            "class_balanced_accuracy_present"
                        ],
                        "Ia_f1": metrics["Ia_f1"],
                        "Ia_median_absolute_delta_z": metrics[
                            "Ia_median_absolute_delta_z"
                        ],
                        "Ia_outlier_fraction": metrics["Ia_outlier_fraction"],
                    }
                )
            rows.append(row)
            print(_row_text(row), flush=True)

    dense_contribution = _dense_contribution(rows)
    if dense_contribution:
        print("\n  Dense contribution: spectral minus without-dense", flush=True)
        print(
            "  view          delta balanced   delta Ia F1   delta Ia |dz|   delta blank lock",
            flush=True,
        )
        for row in dense_contribution:
            print(_dense_contribution_text(row), flush=True)

    onir_contribution = _onir_contribution(rows)
    if onir_contribution:
        print(
            "\n  ONIR audit: profile score, support mask, and named regions",
            flush=True,
        )
        print(
            "  comparison                  view          delta balanced   delta Ia F1   delta Ia |dz|   delta outlier   delta blank lock",
            flush=True,
        )
        for row in onir_contribution:
            print(_onir_contribution_text(row), flush=True)

    continuum_fraction = _continuum_fraction_by_redshift(model)
    if continuum_fraction:
        text = "  ".join(
            f"z={row['redshift']:.1f}: {100.0 * row['continuum_fraction']:.1f}%"
            for row in continuum_fraction
        )
        print(f"\n  Continuum-subtracted fraction\n  {text}", flush=True)

    report = {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "split": evaluation_split,
        "objects": len(positions),
        "subset": "shared class-redshift stratified sample",
        "rows": rows,
        "dense_contribution": dense_contribution,
        "onir_contribution": onir_contribution,
        "continuum_fraction_by_redshift": continuum_fraction,
    }
    with (output_dir / "route_check_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"  results {output_dir / 'route_check_summary.json'}", flush=True)
    return report


def _continuum_fraction_by_redshift(model: Any) -> list[dict[str, float]]:
    """Report the learned whole/detail mixture at interpretable redshifts."""
    dense = getattr(model, "dense_scan", None)
    if dense is None or not hasattr(dense, "detail_intercept"):
        return []
    detail_only = getattr(model, "dense_scan_view", "blend") == "detail"
    grid = model.redshift_grid.detach()
    requested = [0.0, 1.0, 2.0, 3.0]
    rows = []
    for redshift in requested:
        index = int((grid - redshift).abs().argmin())
        coordinate = dense.redshift_coordinate[index]
        if detail_only:
            weight = torch.ones((), device=coordinate.device)
        else:
            relative_weight = torch.sigmoid(
                dense.detail_intercept + dense.detail_redshift_slope * coordinate
            )
            weight = (
                float(getattr(dense, "maximum_detail_weight", 1.0))
                * relative_weight
            )
        rows.append(
            {
                "redshift": float(grid[index].detach().cpu()),
                "continuum_fraction": float(weight.detach().cpu()),
            }
        )
    return rows


def _dense_contribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_view_route = {
        (str(row["view"]), str(row["route"])): row
        for row in rows
    }
    views = list(dict.fromkeys(str(row["view"]) for row in rows))
    contribution = []
    for view in views:
        spectral = by_view_route.get((view, "spectral"))
        without = by_view_route.get((view, "without_dense"))
        if spectral is None or without is None:
            continue
        row: dict[str, Any] = {"view": view}
        if view == "no_source":
            row["delta_blank_redshift_lock"] = (
                float(spectral["blank_redshift_lock"])
                - float(without["blank_redshift_lock"])
            )
        else:
            row.update(
                {
                    "delta_balanced_accuracy": float(spectral["balanced_accuracy"])
                    - float(without["balanced_accuracy"]),
                    "delta_Ia_f1": float(spectral["Ia_f1"])
                    - float(without["Ia_f1"]),
                    "delta_Ia_median_absolute_delta_z": float(
                        spectral["Ia_median_absolute_delta_z"]
                    )
                    - float(without["Ia_median_absolute_delta_z"]),
                }
            )
        contribution.append(row)
    return contribution


def _onir_contribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Contrast the three distinct roles currently bundled into ONIR."""
    by_view_route = {
        (str(row["view"]), str(row["route"])): row
        for row in rows
    }
    comparisons = (
        ("profile score", "with_onir_masked", "without_onir_masked"),
        ("support mask", "without_onir_masked", "without_onir"),
        ("named-region shape", "without_onir_spectral", "global_spectrum"),
        ("temporal evolution", "without_onir", "without_onir_spectral"),
    )
    views = list(dict.fromkeys(str(row["view"]) for row in rows))
    contribution = []
    for label, included_name, excluded_name in comparisons:
        for view in views:
            included = by_view_route.get((view, included_name))
            excluded = by_view_route.get((view, excluded_name))
            if included is None or excluded is None:
                continue
            row: dict[str, Any] = {"comparison": label, "view": view}
            if view == "no_source":
                row["delta_blank_redshift_lock"] = (
                    float(included["blank_redshift_lock"])
                    - float(excluded["blank_redshift_lock"])
                )
            else:
                row.update(
                    {
                        "delta_balanced_accuracy": float(included["balanced_accuracy"])
                        - float(excluded["balanced_accuracy"]),
                        "delta_Ia_f1": float(included["Ia_f1"])
                        - float(excluded["Ia_f1"]),
                        "delta_Ia_median_absolute_delta_z": float(
                            included["Ia_median_absolute_delta_z"]
                        )
                        - float(excluded["Ia_median_absolute_delta_z"]),
                        "delta_Ia_outlier_fraction": float(
                            included["Ia_outlier_fraction"]
                        )
                        - float(excluded["Ia_outlier_fraction"]),
                    }
                )
            contribution.append(row)
    return contribution


def _onir_contribution_text(row: dict[str, Any]) -> str:
    if row["view"] == "no_source":
        return (
            f"  {row['comparison']:<27} {row['view']:<13} "
            f"{'-':>14} {'-':>13} {'-':>16} {'-':>15} "
            f"{100.0 * row['delta_blank_redshift_lock']:>+16.1f}%"
        )
    return (
        f"  {row['comparison']:<27} {row['view']:<13} "
        f"{100.0 * row['delta_balanced_accuracy']:>+13.1f}% "
        f"{100.0 * row['delta_Ia_f1']:>+12.1f}% "
        f"{row['delta_Ia_median_absolute_delta_z']:>+16.4f} "
        f"{100.0 * row['delta_Ia_outlier_fraction']:>+14.1f}% {'-':>17}"
    )


def _dense_contribution_text(row: dict[str, Any]) -> str:
    if row["view"] == "no_source":
        return (
            f"  {row['view']:<13} {'-':>14} {'-':>13} {'-':>16} "
            f"{100.0 * row['delta_blank_redshift_lock']:>+16.1f}%"
        )
    return (
        f"  {row['view']:<13} "
        f"{100.0 * row['delta_balanced_accuracy']:>+13.1f}% "
        f"{100.0 * row['delta_Ia_f1']:>+12.1f}% "
        f"{row['delta_Ia_median_absolute_delta_z']:>+16.4f} {'-':>17}"
    )


def _row_text(row: dict[str, Any]) -> str:
    if row["view"] == "no_source":
        return (
            f"  {row['view']:<13} {row['route']:<22} "
            f"{'-':>8} {'-':>7} {'-':>9} {'-':>9} "
            f"{100.0 * row['blank_redshift_lock']:>10.1f}%"
        )
    return (
        f"  {row['view']:<13} {row['route']:<22} "
        f"{100.0 * row['balanced_accuracy']:>7.1f}% "
        f"{100.0 * row['Ia_f1']:>6.1f}% "
        f"{row['Ia_median_absolute_delta_z']:>9.4f} "
        f"{100.0 * row['Ia_outlier_fraction']:>8.1f}% {'-':>10}"
    )
