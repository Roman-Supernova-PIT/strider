"""Gradient-boosted baselines that never read spectral flux."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, r2_score, recall_score

from strider.config import project_path
from strider.data.dataset import SundialDataset


def run_metadata_baseline(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config["project"]["seed"])
    evaluation_split = str(config["evaluation"].get("split", "calibration"))
    training = SundialDataset(
        config, "train", "generated", training=False, pair_no_source=False
    )
    evaluation = SundialDataset(
        config, evaluation_split, "generated", training=False, pair_no_source=False
    )
    train_features = _metadata_features(training)
    evaluation_features = _metadata_features(evaluation)
    train_class = training.objects["class_index"].to_numpy(dtype=np.int64)
    true_class = evaluation.objects["class_index"].to_numpy(dtype=np.int64)
    train_redshift = training.objects["redshift"].to_numpy(dtype=np.float64)
    true_redshift = evaluation.objects["redshift"].to_numpy(dtype=np.float64)

    reports = {}
    for name, columns in {
        "cadence": list(range(8)),
        "cadence_and_support": list(range(12)),
    }.items():
        classifier = HistGradientBoostingClassifier(
            max_iter=150,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=seed,
        )
        redshift_model = HistGradientBoostingRegressor(
            max_iter=150,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=seed,
        )
        classifier.fit(train_features[:, columns], train_class)
        redshift_model.fit(train_features[:, columns], train_redshift)
        predicted_class = classifier.predict(evaluation_features[:, columns])
        predicted_redshift = redshift_model.predict(evaluation_features[:, columns])
        reports[name] = _metrics(
            true_class,
            predicted_class,
            true_redshift,
            predicted_redshift,
        )

    report = {
        "training_split": "train",
        "evaluation_split": evaluation_split,
        "training_objects": len(training.objects),
        "evaluation_objects": len(evaluation.objects),
        "Ia_fraction": float(np.mean(true_class == 0)),
        "features": [
            "visit_count",
            "span",
            "mean_time",
            "time_std",
            "mean_gap",
            "gap_std",
            "minimum_gap",
            "maximum_gap",
            "mean_native_minimum",
            "mean_native_maximum",
            "minimum_native_minimum",
            "maximum_native_maximum",
        ],
        "models": reports,
    }
    output_dir = project_path(config, config["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "metadata_baseline_summary.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    _print_report(report, path)
    return report


def _metadata_features(dataset: SundialDataset) -> np.ndarray:
    features = []
    for row in dataset.objects.itertuples(index=False):
        first = int(row.first_observation)
        count = int(row.observation_count)
        visits = dataset.observations.iloc[first:first + count].sort_values("mjd")
        if dataset.max_visits is not None and len(visits) > dataset.max_visits:
            positions = np.unique(
                np.rint(np.linspace(0, len(visits) - 1, dataset.max_visits)).astype(int)
            )
            visits = visits.iloc[positions]
        time = visits["mjd"].to_numpy(dtype=np.float64)
        time -= time[0]
        gaps = np.diff(time)
        features.append(
            [
                len(time),
                time[-1],
                time.mean(),
                time.std(),
                gaps.mean() if len(gaps) else 0.0,
                gaps.std() if len(gaps) else 0.0,
                gaps.min() if len(gaps) else 0.0,
                gaps.max() if len(gaps) else 0.0,
                visits["native_wavelength_min"].mean(),
                visits["native_wavelength_max"].mean(),
                visits["native_wavelength_min"].min(),
                visits["native_wavelength_max"].max(),
            ]
        )
    return np.asarray(features, dtype=np.float64)


def _metrics(
    true_class: np.ndarray,
    predicted_class: np.ndarray,
    true_redshift: np.ndarray,
    predicted_redshift: np.ndarray,
) -> dict[str, float | int]:
    ia = true_class == 0
    delta = predicted_redshift - true_redshift
    low_redshift_ia = ia & (true_redshift < 1.5)
    return {
        "N": int(len(true_class)),
        "class_accuracy": float(accuracy_score(true_class, predicted_class)),
        "class_balanced_accuracy": float(balanced_accuracy_score(true_class, predicted_class)),
        "class_macro_f1": float(f1_score(true_class, predicted_class, average="macro")),
        "Ia_precision": float(precision_score(ia, predicted_class == 0, zero_division=0)),
        "Ia_recall": float(recall_score(ia, predicted_class == 0, zero_division=0)),
        "Ia_f1": float(f1_score(ia, predicted_class == 0, zero_division=0)),
        "redshift_r2": float(r2_score(true_redshift, predicted_redshift)),
        "median_absolute_delta_z": float(np.median(np.abs(delta))),
        "outlier_fraction_abs_delta_z_gt_0_1": float(np.mean(np.abs(delta) > 0.1)),
        "Ia_low_z_median_absolute_delta_z_over_1_plus_z": (
            float(np.median(np.abs(delta[low_redshift_ia]) / (1.0 + true_redshift[low_redshift_ia])))
            if np.any(low_redshift_ia)
            else float("nan")
        ),
    }


def _print_report(report: dict[str, Any], path: Any) -> None:
    print("\nMetadata-only baseline", flush=True)
    print(
        f"  {report['training_objects']:,} training | "
        f"{report['evaluation_objects']:,} {report['evaluation_split']} objects",
        flush=True,
    )
    print("  input                  balanced   Ia F1    Ia |dz|/(1+z), z<1.5   z R2", flush=True)
    for name, metrics in report["models"].items():
        print(
            f"  {name:<22} "
            f"{100.0 * metrics['class_balanced_accuracy']:>7.1f}% "
            f"{100.0 * metrics['Ia_f1']:>7.1f}% "
            f"{metrics['Ia_low_z_median_absolute_delta_z_over_1_plus_z']:>20.4f} "
            f"{metrics['redshift_r2']:>7.3f}",
            flush=True,
        )
    print(f"  results {path}", flush=True)
