"""Small summaries for training and evaluation logs.

The full numerical records remain in JSON and Parquet files. These functions
only control what a person sees while following a run in the terminal.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


VIEW_LABELS = {
    "generated": "clean + noise",
    "original": "observed",
    "clean": "clean",
    "no_source": "noise only",
    "residual": "observed - clean",
    "reported_error_with_source": "clean + SNANA noise",
    "reported_error_no_source": "SNANA noise only",
}


def training_start(
    *,
    device: str,
    parameters: int,
    trainable_parameters: int,
    train_objects: int,
    validation_objects: int,
    validation_views: list[str],
    epochs: int,
    batch_size: int,
    workers: int,
    output_dir: Path,
    start_epoch: int,
    classes: list[str],
    redshift_bins: int,
    visit_limit: int | str | None,
    full_history_fraction: float,
    job_epoch_limit: int | None,
) -> None:
    print("\nSTRIDER training", flush=True)
    task = (
        "normal Ia / all other transients"
        if classes == ["Ia", "other"]
        else f"{len(classes)} classes"
    )
    print(f"  task {task} | {redshift_bins} redshift bins", flush=True)
    print(
        f"  device {device} | {parameters:,} parameters"
        + (
            f" ({trainable_parameters:,} trained)"
            if trainable_parameters != parameters
            else ""
        )
        + " | "
        f"batch {batch_size} | {workers} workers",
        flush=True,
    )
    print(
        f"  {train_objects:,} training objects | "
        f"{validation_objects:,} validation objects | "
        f"views {', '.join(_view_label(name) for name in validation_views)}",
        flush=True,
    )
    if visit_limit is None or str(visit_limit).strip().lower() == "all":
        full_percent = 100.0 * full_history_fraction
        print(
            "  visits all available | training: "
            f"{full_percent:.0f}% complete histories, "
            f"{100.0 - full_percent:.0f}% shorter histories",
            flush=True,
        )
    if start_epoch:
        print(f"  continuing at epoch {start_epoch + 1} of {epochs}", flush=True)
    else:
        print(f"  {epochs} epochs | output {output_dir}", flush=True)
    if job_epoch_limit is not None:
        print(
            f"  this scheduler job will pause after {job_epoch_limit} complete "
            "epoch(s)",
            flush=True,
        )
    print(
        "  route values: s named features | t temporal change | "
        "c spectral context | d complete-spectrum scan | "
        "detail continuum-subtracted share",
        flush=True,
    )
    print(
        "  epoch | loss: train/select | z<2: macro F1 | Ia: precision recall F1 | "
        "Ia |dz| scatter | evidence: transient/noise | route values | lr | time",
        flush=True,
    )


def training_epoch(
    record: dict[str, Any],
    total_epochs: int,
    seconds: float,
    saved: bool,
    saved_science: bool = False,
    saved_redshift: bool = False,
    saved_macro_redshift: bool = False,
) -> None:
    saved_roles = []
    if saved:
        saved_roles.append("posterior")
    if saved_science:
        saved_roles.append("science")
    if saved_redshift:
        saved_roles.append("redshift")
    if saved_macro_redshift:
        saved_roles.append("macro-z")
    status = f" * {','.join(saved_roles)}" if saved_roles else ""
    unusable = record["train"].get("unsupported_source_fraction", 0.0)
    coverage = f" | unusable={100.0 * unusable:.3f}%" if unusable > 0.0 else ""
    validation = record.get("validation", {})
    phase_fraction = validation.get("phase_supervised_visit_fraction", 0.0)
    phase = (
        f" | phase |error| {validation['phase_median_absolute_error_days']:.2f}d"
        if phase_fraction > 0.0
        else ""
    )
    scale_names = {
        "shape": "s",
        "temporal": "t",
        "context": "c",
        "dense": "d",
        "dense_detail": "detail",
        "flux_evolution": "relative_flux",
        "background_scaled_amplitude": "amplitude",
        "phase": "p",
    }
    scales = (
        " ".join(
            f"{scale_names.get(name, name)}={value:.2f}"
            for name, value in record.get("learned_scales", {}).items()
        )
        or "-"
    )
    macro_f1 = _percent_short(validation.get("z_lt_2_macro_f1_present", float("nan")))
    precision = _percent_short(
        validation.get(
            "ia_z_lt_2_precision",
            validation.get("ia_precision", float("nan")),
        )
    )
    recall = _percent_short(
        validation.get(
            "ia_z_lt_2_recall",
            validation.get("ia_recall", float("nan")),
        )
    )
    f1 = _percent_short(
        validation.get("ia_z_lt_2_f1", validation.get("ia_f1", float("nan")))
    )
    ia_delta = _number(
        validation.get(
            "ia_z_lt_2_median_absolute_delta_z",
            validation.get("ia_median_absolute_delta_z", float("nan")),
        ),
        4,
    )
    ia_scatter = _number(
        validation.get("ia_z_lt_2_population_scatter_delta_z", float("nan")),
        4,
    )
    source_evidence = _number(
        validation.get("source_mean_evidence_sufficiency", float("nan")), 2
    )
    no_source_evidence = _number(
        validation.get("no_source_mean_evidence_sufficiency", float("nan")), 2
    )
    print(
        f"  {record['epoch']:>3}/{total_epochs:<3} | "
        f"{record['train']['loss']:>7.4f}/{record['selection_score']:<7.4f} | "
        f"{macro_f1:>6} | {precision:>6} {recall:>6} {f1:>6} | "
        f"{ia_delta:>7} {ia_scatter:>7} | "
        f"{source_evidence:>4}/{no_source_evidence:<4} | "
        f"{scales:<20} | {record['learning_rate']:.1e} | "
        f"{_duration(seconds):>6}{coverage}{phase}{status}",
        flush=True,
    )
    _training_class_table(validation)


def _training_class_table(validation: dict[str, Any]) -> None:
    rows = validation.get("z_lt_2_metrics_by_class", [])
    if len(rows) <= 2:
        return
    print(
        "        z<2 class                   N      P      R     F1  "
        "median |dz|  scatter  out>0.1",
        flush=True,
    )
    for row in rows:
        print(
            f"        {row['class_name']:<23} {int(row['N']):>5,} "
            f"{_percent_short(row['precision']):>6} "
            f"{_percent_short(row['recall']):>6} "
            f"{_percent_short(row['f1']):>6} "
            f"{_number(row['median_absolute_delta_z'], 4):>11} "
            f"{_number(row['population_scatter_delta_z'], 4):>8} "
            f"{_percent_short(row['outlier_fraction_abs_delta_z_gt_0_1']):>8}",
            flush=True,
        )


def training_end(report: dict[str, Any]) -> None:
    if report.get("stop_reason") == "job_epoch_limit":
        print("\nTraining paused at an epoch boundary", flush=True)
        print(
            f"  completed epoch {report['epochs_completed']} of "
            f"{report['configured_epochs']} | resume state saved",
            flush=True,
        )
    else:
        print("\nTraining complete", flush=True)
    print(
        f"  best epoch {report['best_epoch']} of {report['epochs_completed']} | "
        f"selection loss {report['best_selection_score']:.4f}",
        flush=True,
    )
    print(f"  model {report['checkpoint']}", flush=True)
    metric_view = report.get("checkpoint_metric_view", "generated")
    if "best_science_epoch" in report:
        print(
            f"  science epoch {report['best_science_epoch']} | "
            f"{metric_view} z<2 macro F1 "
            f"{100.0 * report['best_science_score']:.1f}% | "
            f"model {report['science_checkpoint']}",
            flush=True,
        )
    if "best_redshift_epoch" in report:
        print(
            f"  redshift epoch {report['best_redshift_epoch']} | "
            f"{metric_view} Ia outlier fraction "
            f"{100.0 * report['best_redshift_score']:.1f}% | "
            f"model {report['redshift_checkpoint']}",
            flush=True,
        )
    if "best_macro_redshift_epoch" in report:
        print(
            f"  macro-z epoch {report['best_macro_redshift_epoch']} | "
            f"{metric_view} class-balanced outlier fraction "
            f"{100.0 * report['best_macro_redshift_score']:.1f}% | "
            f"model {report['macro_redshift_checkpoint']}",
            flush=True,
        )


def evaluation_start(*, split: str, checkpoint_epoch: int, device: str) -> None:
    print(f"\nSTRIDER evaluation — {split}", flush=True)
    print(f"  checkpoint epoch {checkpoint_epoch} | device {device}", flush=True)


def evaluation_view(name: str, report: dict[str, Any], outlier_delta_z: float) -> None:
    name = _view_label(name)
    count = int(report["N"])
    if "class_accuracy" not in report:
        print(f"\n{name} — {count:,} control objects", flush=True)
        print(
            "  "
            f"mean evidence score={report['mean_evidence_sufficiency']:.3f} | "
            f"evidence>=0.5={_percent(report['fraction_sufficient_evidence_above_0_5'])} | "
            f"largest joint cell={report['mean_largest_joint_probability']:.3f}",
            flush=True,
        )
        print(
            "  "
            f"largest class={report['mean_largest_class_probability']:.3f} | "
            "largest class>0.5="
            f"{_percent(report['fraction_largest_class_probability_above_0_5'])}",
            flush=True,
        )
        print(
            "  "
            f"median |delta z to simulated label|="
            f"{report['median_absolute_delta_z_to_simulation']:.4f} | "
            f"within 0.1={_percent(report['fraction_within_delta_z_0_1_of_simulation'])}",
            flush=True,
        )
        if "mean_runtime_phase_largest_probability" in report:
            print(
                "  "
                "mean largest phase-bin probability="
                f"{report['mean_runtime_phase_largest_probability']:.3f}",
                flush=True,
            )
        return

    print(f"\n{name} — {count:,} objects", flush=True)
    ia_f1 = f" | Ia F1={_percent(report['Ia_f1'])}" if "Ia_f1" in report else ""
    print(
        "  "
        f"balanced accuracy={_percent(report['class_balanced_accuracy_present'])} | "
        f"macro F1={_percent(report['class_macro_f1_present'])} | "
        f"Ia precision={_percent(report['Ia_precision'])} | "
        f"Ia recall={_percent(report['Ia_recall'])}{ia_f1} | "
        f"mean evidence score={report['mean_evidence_sufficiency']:.3f}",
        flush=True,
    )
    print(
        "  "
        f"median delta z={report['median_delta_z']:+.4f} | "
        f"scatter={report['population_scatter_delta_z']:.4f} | "
        f"median |delta z|={report['median_absolute_delta_z']:.4f} | "
        f"|delta z|>{outlier_delta_z:g}={_percent(report['outlier_fraction'])} | "
        f"primary-basin 68% coverage="
        f"{_percent(report['posterior_68_interval_coverage'])}",
        flush=True,
    )
    estimator = report.get("redshift_point_estimator_comparison", {})
    if estimator:
        labels = {
            "primary_basin_peak": "STRIDER basin peak",
            "median": "posterior median",
            "density_mode": "density peak",
            "primary_basin_median": "primary-basin median",
            "joint_primary_basin_median": "joint-basin median",
            "mean": "posterior mean",
        }
        values = [
            f"{labels.get(name, name.replace('_', ' '))}="
            f"{metrics['median_absolute_delta_z']:.4f}"
            for name, metrics in estimator.items()
        ]
        print(
            "  point-estimate median |delta z|: " + " | ".join(values),
            flush=True,
        )
    if "Ia_median_phase_absolute_error_days" in report:
        print(
            "  "
            "Ia phase: median |error|="
            f"{report['Ia_median_phase_absolute_error_days']:.2f} d | "
            "ordering="
            f"{_percent(report['Ia_mean_phase_order_accuracy'])} | "
            "68% coverage="
            f"{_percent(report['Ia_mean_phase_68_interval_coverage'])}",
            flush=True,
        )
    _evaluation_class_table(report)
    _evaluation_ia_redshift_table(report)


def _evaluation_class_table(report: dict[str, Any]) -> None:
    rows = report.get("metrics_by_class", [])
    if not rows:
        return
    print(
        "  class       N   precision recall    F1   median dz  scatter  "
        "median |dz|  outliers  coverage  evidence",
        flush=True,
    )
    for row in rows:
        print(
            f"  {row['class_name']:<9} {row['N']:>5,} "
            f"{_percent(row['precision']):>9} {_percent(row['recall']):>6} "
            f"{_percent(row['f1']):>6} {row['median_delta_z']:>+10.4f} "
            f"{row['population_scatter_delta_z']:>8.4f} "
            f"{row['median_absolute_delta_z']:>11.4f} "
            f"{_percent(row['outlier_fraction']):>9} "
            f"{_percent(row['posterior_68_interval_coverage']):>9} "
            f"{row['mean_evidence_sufficiency']:>8.3f}",
            flush=True,
        )


def _evaluation_ia_redshift_table(report: dict[str, Any]) -> None:
    rows = report.get("ia_redshift_groups", [])
    if not rows:
        return
    print("  Ia by redshift", flush=True)
    print(
        "  redshift       N  precision recall    F1  median |dz|  scatter  "
        "outliers  evidence",
        flush=True,
    )
    for row in rows:
        print(
            f"  {row['redshift_min']:.1f}-{row['redshift_max']:<5.1f} "
            f"{int(row.get('Ia_N', 0)):>5,} "
            f"{_percent(row.get('Ia_precision', float('nan'))):>9} "
            f"{_percent(row.get('Ia_recall', float('nan'))):>6} "
            f"{_percent(row.get('Ia_f1', float('nan'))):>6} "
            f"{row.get('Ia_median_absolute_delta_z', float('nan')):>11.4f} "
            f"{row.get('Ia_population_scatter_delta_z', float('nan')):>8.4f} "
            f"{_percent(row.get('Ia_outlier_fraction', float('nan'))):>9} "
            f"{row.get('Ia_mean_evidence_sufficiency', float('nan')):>8.3f}",
            flush=True,
        )


def evaluation_end(output_dir: Path) -> None:
    print(f"\nEvaluation finished | results {output_dir}", flush=True)


def timing_only_result(report: dict[str, Any], output_dir: Path) -> None:
    metrics = report["evaluation"]
    print(f"\nTiming-only check — {report['evaluation_split']}", flush=True)
    print(
        f"  {metrics['N']:,} objects | "
        f"balanced accuracy={_percent(metrics['class_balanced_accuracy_present'])} | "
        f"median |delta z|={metrics['median_absolute_delta_z']:.4f} | "
        f"|delta z|>0.1={_percent(metrics['outlier_fraction_abs_delta_z_gt_0_10'])}",
        flush=True,
    )
    print(f"  results {output_dir / 'timing_only_summary.json'}", flush=True)


def _percent(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{100.0 * value:.1f}%"


def _percent_short(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{100.0 * value:.1f}"


def _number(value: float, places: int) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.{places}f}"


def _duration(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.0f}s"
    minutes = int(round(seconds / 60.0))
    if minutes < 60:
        return f"{minutes}m"
    return f"{minutes // 60}h{minutes % 60:02d}m"


def _view_label(name: str) -> str:
    return VIEW_LABELS.get(name, name.replace("_", " "))
