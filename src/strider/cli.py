"""STRIDER command-line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_config, project_path
from .data.prepare import build_native_store, source_pair_assignments
from .data.snana import discover_source_pairs, inspect_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    temporal_example = subparsers.add_parser("temporal-example")
    temporal_example.add_argument("--output", required=True, type=Path)
    temporal_example.add_argument("--epochs", type=int, default=40)
    temporal_example.add_argument("--training-objects", type=int, default=900)
    temporal_example.add_argument("--test-objects", type=int, default=300)
    temporal_example.add_argument("--seed", type=int, default=4)
    temporal_example.add_argument("--batch-size", type=int, default=100)
    temporal_example.add_argument("--redshift-bins", type=int, default=15)
    temporal_example.add_argument("--training-gap-min", type=int, default=4)
    temporal_example.add_argument("--training-gap-max", type=int, default=18)
    temporal_example.add_argument("--test-gap-min", type=int)
    temporal_example.add_argument("--test-gap-max", type=int)
    temporal_example.add_argument("--feature-noise-std", type=float, default=0.03)
    temporal_example.add_argument("--intrinsic-variation-std", type=float, default=0.0)
    temporal_example.add_argument("--minimum-training-visits", type=int, default=5)
    temporal_example.add_argument(
        "--binary",
        action="store_true",
        help="use normal Ia against fourteen contaminant families",
    )
    commands = (
        "inspect",
        "class-support",
        "wavelength-support",
        "prepare",
        "prepare-external-test",
        "benchmark",
        "train",
        "evaluate",
        "observed-snr",
        "fit-calibration",
        "noise-check",
        "measurement-controls",
        "route-check",
        "time-controls",
        "plot-training",
        "plot-examples",
        "evidence-maps",
        "evidence-gifs",
        "evidence-growth",
        "visit-controls",
        "paired-controls",
        "timing-baseline",
        "metadata-baseline",
        "posterior-audit",
        "build-onir",
        "build-reference",
        "export-model",
    )
    for name in commands:
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        if name == "train":
            command.add_argument(
                "--resume",
                action="store_true",
                help="continue from the last completed epoch in this run directory",
            )
        if name == "export-model":
            command.add_argument(
                "--replace",
                action="store_true",
                help=(
                    "atomically replace an earlier package, preserving it in a "
                    "uniquely named backup directory"
                ),
            )
        if name == "evaluate":
            command.add_argument(
                "--split",
                choices=("selection", "calibration", "test"),
                help="evaluate another prepared split without changing the checkpoint config",
            )
            command.add_argument(
                "--views",
                nargs="+",
                choices=(
                    "original",
                    "generated",
                    "clean",
                    "no_source",
                    "residual",
                    "reported_error_with_source",
                    "reported_error_no_source",
                ),
                help="evaluate only these views; defaults to the full configured audit",
            )
            command.add_argument(
                "--external-prepared-dir",
                type=Path,
                help="versioned external-test store; checkpoint config remains unchanged",
            )
            command.add_argument(
                "--output-dir",
                type=Path,
                help="separate destination required for an external-test evaluation",
            )
        if name == "prepare-external-test":
            command.add_argument("--source-dir", required=True, type=Path)
            command.add_argument("--prepared-dir", required=True, type=Path)
            command.add_argument("--dataset-tag", required=True)
            command.add_argument(
                "--blocks",
                nargs="+",
                type=int,
                default=list(range(1, 11)),
            )
        if name == "fit-calibration":
            command.add_argument(
                "--source-predictions",
                type=Path,
                help="reserved calibration-split source-view parquet",
            )
            command.add_argument(
                "--blank-predictions",
                type=Path,
                help="matched calibration-split no_source parquet",
            )
            command.add_argument(
                "--output",
                type=Path,
                help="calibration artifact; defaults to OUTPUT_DIR/calibration.json",
            )
            command.add_argument("--folds", type=int, default=2)
            command.add_argument(
                "--coverage-levels",
                nargs="+",
                type=float,
                default=[0.68, 0.90],
            )
            command.add_argument("--minimum-stratum-size", type=int, default=200)
            command.add_argument("--gold-purity", type=float, default=0.95)
        if name == "observed-snr":
            command.add_argument(
                "--split",
                choices=("selection", "calibration", "test"),
                default="test",
            )
            command.add_argument(
                "--predictions",
                type=Path,
                help="prediction parquet/CSV fixing the exact SNID cohort",
            )
            command.add_argument(
                "--output",
                type=Path,
                help=(
                    "object-level parquet/CSV with coadded and best-epoch S/N; "
                    "defaults inside the configured run"
                ),
            )
            command.add_argument(
                "--epochs-output",
                type=Path,
                help=(
                    "optional long-form parquet/CSV with single-epoch and "
                    "cumulative S/N for every chronological observational epoch"
                ),
            )
            command.add_argument(
                "--external-prepared-dir",
                type=Path,
                help="versioned external-test store used by the prediction table",
            )
            command.add_argument(
                "--edge-trim-fraction",
                type=float,
                default=0.05,
                help="fraction removed from each end of the log-wavelength range",
            )
            command.add_argument(
                "--maximum-relative-error",
                type=float,
                default=3.0,
                help="exclude bins above this multiple of median in-band coadd error",
            )
        if name == "paired-controls":
            command.add_argument(
                "--max-objects",
                type=int,
                help="evaluate a larger fixed subset without changing the checkpoint config",
            )
        if name == "noise-check":
            command.add_argument(
                "--split",
                choices=("selection", "calibration", "test"),
                help="evaluate another prepared split without changing the checkpoint config",
            )
            command.add_argument(
                "--scales",
                nargs="+",
                type=float,
                default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5],
            )
            command.add_argument("--max-objects", type=int)
            command.add_argument(
                "--noise-family",
                choices=("controlled-background", "reported-error"),
                default="controlled-background",
            )
            command.add_argument(
                "--object-list",
                type=Path,
                help="CSV manifest selecting exact SNIDs for a paired comparison",
            )
            command.add_argument(
                "--paired-noise-seed",
                type=int,
                help="repeat frozen-v2 object-level native-bin Gaussian draws",
            )
            command.add_argument(
                "--save-predictions",
                action="store_true",
                help="write per-object source and blank predictions for every draw",
            )
            command.add_argument(
                "--ia-only",
                action="store_true",
                help="evaluate every true Ia in the selected split",
            )
            command.add_argument(
                "--objects-per-redshift-bin",
                type=int,
                help="select the same number of true Ia objects in every requested interval",
            )
            command.add_argument(
                "--redshift-edges",
                nargs="+",
                type=float,
                help="redshift interval edges for a balanced Ia noise sweep",
            )
            command.add_argument("--repeats", type=int, default=1)
            command.add_argument(
                "--output-tag",
                help="append a short tag to noise diagnostic filenames",
            )
        if name == "measurement-controls":
            command.add_argument(
                "--split",
                choices=("selection", "calibration", "test"),
                help="prepared split; defaults to evaluation.split",
            )
            command.add_argument(
                "--view",
                choices=("original", "generated", "clean"),
                default="original",
            )
        if name == "route-check":
            command.add_argument("--max-objects", type=int, default=800)
            command.add_argument(
                "--split",
                choices=("selection", "calibration", "test"),
                help="audit another prepared split without changing the checkpoint config",
            )
            command.add_argument(
                "--views",
                nargs="+",
                default=["generated", "clean", "no_source"],
            )
        if name == "evidence-maps":
            command.add_argument(
                "--layout",
                choices=("summary", "diagnostic"),
                help="concise evidence summary or the full route/ONIR diagnostic",
            )
            command.add_argument(
                "--split",
                choices=("selection", "calibration", "test"),
                help="plot another prepared split without changing the checkpoint config",
            )
            command.add_argument(
                "--view",
                choices=("original", "generated", "clean", "no_source"),
                help="plot another spectral view without changing the checkpoint config",
            )
            command.add_argument(
                "--object-list",
                type=Path,
                help="CSV manifest selecting exact objects without changing the checkpoint config",
            )
            command.add_argument(
                "--objects-per-redshift",
                type=int,
                help="representative objects per configured redshift",
            )
            command.add_argument(
                "--competing-peak-ratio",
                type=float,
                help="minimum secondary-to-dominant peak ratio shown in the figure",
            )
        if name == "evidence-gifs":
            command.add_argument(
                "--layout",
                choices=("summary", "diagnostic"),
                help="concise accumulation story or full diagnostic animation",
            )
            command.add_argument(
                "--split",
                choices=("selection", "calibration", "test"),
                help="animate another prepared split without changing the checkpoint config",
            )
            command.add_argument(
                "--view",
                choices=("original", "generated", "clean", "no_source"),
                help="animate another spectral view without changing the checkpoint config",
            )
            command.add_argument(
                "--object-list",
                type=Path,
                help="CSV manifest selecting exact animation objects",
            )
            command.add_argument(
                "--max-frames",
                type=int,
                help="maximum cumulative visit frames per object",
            )
        if name == "evidence-growth":
            command.add_argument(
                "--exponents",
                nargs="+",
                type=float,
                default=[0.0, 0.15, 0.25, 0.35, 0.5],
            )
            command.add_argument(
                "--visit-counts",
                nargs="+",
                type=int,
                default=[1, 4, 16, 32],
            )
            command.add_argument("--max-objects", type=int, default=500)
            command.add_argument("--repeats", type=int, default=1)
        if name == "posterior-audit":
            command.add_argument(
                "--predictions",
                type=Path,
                help="saved prediction parquet; defaults to the configured run output",
            )
            command.add_argument(
                "--split",
                choices=("selection", "calibration", "test"),
                help="prepared split used by the saved predictions",
            )
            command.add_argument("--view", default="original")
            command.add_argument("--output-tag", default="basin")
            command.add_argument(
                "--output-dir",
                type=Path,
                help="destination directory, useful for downloaded prediction files",
            )
    arguments = parser.parse_args()
    if arguments.command == "temporal-example":
        from .evaluation.temporal_example import run_temporal_example

        print(
            json.dumps(
                run_temporal_example(
                    output=arguments.output,
                    epochs=arguments.epochs,
                    training_objects=arguments.training_objects,
                    test_objects=arguments.test_objects,
                    seed=arguments.seed,
                    batch_size=arguments.batch_size,
                    redshift_bins=arguments.redshift_bins,
                    training_gap_min=arguments.training_gap_min,
                    training_gap_max=arguments.training_gap_max,
                    test_gap_min=arguments.test_gap_min,
                    test_gap_max=arguments.test_gap_max,
                    feature_noise_std=arguments.feature_noise_std,
                    intrinsic_variation_std=arguments.intrinsic_variation_std,
                    minimum_training_visits=arguments.minimum_training_visits,
                    binary=arguments.binary,
                ),
                indent=2,
            )
        )
        return
    config = load_config(arguments.config)

    if arguments.command == "inspect":
        if config["data"].get("source_products") is None:
            pairs = discover_source_pairs(config["data"]["source_dir"])
            report = inspect_pairs(pairs)
        else:
            assignments, _ = source_pair_assignments(config["data"])
            parts = []
            for product, split, pair in assignments:
                part = inspect_pairs([pair])
                part.insert(0, "split", split)
                part.insert(0, "source_product", product)
                parts.append(part)
            report = pd.concat(parts, ignore_index=True)
        print(report.to_string(index=False))
        print(f"\nFound {len(report)} assigned HEAD/SPEC pairs.")
        return
    if arguments.command == "prepare":
        print(json.dumps(build_native_store(config), indent=2))
        return
    if arguments.command == "prepare-external-test":
        from .data.external_test import prepare_external_test

        print(
            json.dumps(
                prepare_external_test(
                    config,
                    source_dir=arguments.source_dir,
                    prepared_dir=arguments.prepared_dir,
                    blocks=arguments.blocks,
                    dataset_tag=arguments.dataset_tag,
                ),
                indent=2,
            )
        )
        return
    if arguments.command == "class-support":
        prepared = project_path(config, config["data"]["prepared_dir"])
        path = prepared / "class_support_by_redshift.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Prepare the data before reading {path}")
        support = pd.read_csv(path)
        for split, rows in support.groupby("split", sort=False):
            print(f"\n{split}")
            print(rows.to_string(index=False))
            ia_only = rows[(rows["n_ia"] > 0) & (rows["n_non_ia"] == 0)]
            if len(ia_only):
                intervals = ", ".join(
                    f"{row.redshift_min:.1f}--{row.redshift_max:.1f}"
                    for row in ia_only.itertuples(index=False)
                )
                print(f"Ia present with no non-Ia support in: {intervals}")
            else:
                print("Every populated Ia redshift interval also contains non-Ia objects.")
        return
    if arguments.command == "wavelength-support":
        from .data.wavelength_support import wavelength_support_report

        report = wavelength_support_report(config)
        prepared = project_path(config, config["data"]["prepared_dir"])
        with (prepared / "wavelength_support.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(report, handle, indent=2)
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            raise RuntimeError(
                "Simulation-template wavelength support is associated with class or redshift"
            )
        return
    if arguments.command == "benchmark":
        from .training.benchmark import benchmark

        report = benchmark(config)
        output_dir = project_path(config, config["project"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "benchmark_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(json.dumps(report, indent=2))
        if not report["configured_batch_passed"]:
            raise RuntimeError(
                f"Configured training batch {report['configured_batch_size']} did not pass"
            )
        return
    if arguments.command == "train":
        from .training.trainer import train

        train(config, resume=arguments.resume)
        return
    if arguments.command == "evaluate":
        from .evaluation.evaluate import evaluate

        evaluate(
            config,
            split=arguments.split,
            views=arguments.views,
            external_prepared_dir=arguments.external_prepared_dir,
            output_dir=arguments.output_dir,
        )
        return
    if arguments.command == "observed-snr":
        from .evaluation.signal_to_noise import write_observed_snr_catalog

        print(
            json.dumps(
                write_observed_snr_catalog(
                    config,
                    split=arguments.split,
                    predictions=arguments.predictions,
                    output=arguments.output,
                    epochs_output=arguments.epochs_output,
                    external_prepared_dir=arguments.external_prepared_dir,
                    edge_trim_fraction=arguments.edge_trim_fraction,
                    maximum_relative_error=arguments.maximum_relative_error,
                ),
                indent=2,
            )
        )
        return
    if arguments.command == "fit-calibration":
        from .calibration import fit_calibration

        print(
            json.dumps(
                fit_calibration(
                    config,
                    source_predictions=arguments.source_predictions,
                    blank_predictions=arguments.blank_predictions,
                    output=arguments.output,
                    folds=arguments.folds,
                    coverage_levels=tuple(arguments.coverage_levels),
                    minimum_stratum_size=arguments.minimum_stratum_size,
                    gold_purity=arguments.gold_purity,
                ),
                indent=2,
            )
        )
        return
    if arguments.command == "noise-check":
        from .evaluation.noise_check import run_noise_check

        run_noise_check(
            config,
            arguments.scales,
            arguments.max_objects,
            noise_family=arguments.noise_family,
            objects_per_redshift_bin=arguments.objects_per_redshift_bin,
            redshift_edges=arguments.redshift_edges,
            repeats=arguments.repeats,
            split=arguments.split,
            output_tag=arguments.output_tag,
            object_list=arguments.object_list,
            paired_noise_seed=arguments.paired_noise_seed,
            save_predictions=arguments.save_predictions,
            ia_only=arguments.ia_only,
        )
        return
    if arguments.command == "measurement-controls":
        from .evaluation.measurement_controls import run_measurement_controls

        print(
            json.dumps(
                run_measurement_controls(
                    config,
                    split=arguments.split,
                    view=arguments.view,
                ),
                indent=2,
            )
        )
        return
    if arguments.command == "route-check":
        from .evaluation.route_check import run_route_check

        run_route_check(
            config,
            max_objects=arguments.max_objects,
            views=tuple(arguments.views),
            split=arguments.split,
        )
        return
    if arguments.command == "posterior-audit":
        from .evaluation.posterior_audit import audit_saved_posteriors

        print(
            json.dumps(
                audit_saved_posteriors(
                    config,
                    predictions_path=arguments.predictions,
                    split=arguments.split,
                    view=arguments.view,
                    output_tag=arguments.output_tag,
                    output_dir=arguments.output_dir,
                ),
                indent=2,
            )
        )
        return
    if arguments.command == "time-controls":
        from .evaluation.time_controls import run_time_controls

        print(json.dumps(run_time_controls(config), indent=2))
        return
    if arguments.command == "plot-training":
        from .training.plot_history import plot_training_history

        print(json.dumps(plot_training_history(config), indent=2))
        return
    if arguments.command == "plot-examples":
        from .evaluation.plot_examples import plot_examples

        print(json.dumps(plot_examples(config), indent=2))
        return
    if arguments.command == "evidence-maps":
        from .evaluation.evidence_maps import write_evidence_maps

        print(
            json.dumps(
                write_evidence_maps(
                    config,
                    split=arguments.split,
                    view=arguments.view,
                    object_list=arguments.object_list,
                    objects_per_redshift=arguments.objects_per_redshift,
                    competing_peak_ratio=arguments.competing_peak_ratio,
                    layout=arguments.layout,
                ),
                indent=2,
            )
        )
        return
    if arguments.command == "evidence-gifs":
        from .evaluation.evidence_animation import write_evidence_gifs

        print(
            json.dumps(
                write_evidence_gifs(
                    config,
                    split=arguments.split,
                    view=arguments.view,
                    object_list=arguments.object_list,
                    layout=arguments.layout,
                    maximum_frames=arguments.max_frames,
                ),
                indent=2,
            )
        )
        return
    if arguments.command == "evidence-growth":
        from .evaluation.evidence_growth import run_evidence_growth_sweep

        run_evidence_growth_sweep(
            config,
            exponents=arguments.exponents,
            visit_counts=arguments.visit_counts,
            max_objects=arguments.max_objects,
            repeats=arguments.repeats,
        )
        return
    if arguments.command == "visit-controls":
        from .evaluation.visit_controls import run_visit_controls

        print(json.dumps(run_visit_controls(config), indent=2))
        return
    if arguments.command == "paired-controls":
        from .evaluation.paired_controls import run_paired_controls

        print(
            json.dumps(
                run_paired_controls(config, max_objects=arguments.max_objects),
                indent=2,
            )
        )
        return
    if arguments.command == "timing-baseline":
        from .evaluation.timing_baseline import run_timing_baseline
        from .reporting import timing_only_result

        report = run_timing_baseline(config)
        timing_only_result(
            report,
            project_path(config, config["project"]["output_dir"]),
        )
        return
    if arguments.command == "metadata-baseline":
        from .evaluation.metadata_baseline import run_metadata_baseline

        run_metadata_baseline(config)
        return
    if arguments.command == "build-onir":
        from .atlas import build_onir_bank

        print(json.dumps(build_onir_bank(config), indent=2))
        return
    if arguments.command == "build-reference":
        from .atlas.roman_reference import build_roman_reference_bank

        print(json.dumps(build_roman_reference_bank(config), indent=2))
        return
    if arguments.command == "export-model":
        from .model_package import export_model_package

        print(
            json.dumps(
                export_model_package(config, replace=arguments.replace),
                indent=2,
            )
        )
        return
    raise RuntimeError(f"Unsupported command: {arguments.command}")


if __name__ == "__main__":
    main()
