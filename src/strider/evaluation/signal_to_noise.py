"""Build deployable, object-level observed spectral S/N catalogs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strider.config import project_path, resolved_config_sha256
from strider.data.dataset import (
    OBSERVED_SNR_EDGE_TRIM_FRACTION,
    OBSERVED_SNR_MAX_RELATIVE_ERROR,
    SundialDataset,
)
from strider.data.external_test import (
    config_for_prepared_external_test,
    prepared_external_test_provenance,
)


OBSERVED_SNR_COLUMN = "median_coadded_observed_signal_to_noise"
BEST_EPOCH_SNR_COLUMN = "best_epoch_observed_signal_to_noise"


def write_observed_snr_catalog(
    config: dict[str, Any],
    *,
    split: str = "test",
    predictions: str | Path | None = None,
    output: str | Path | None = None,
    epochs_output: str | Path | None = None,
    external_prepared_dir: str | Path | None = None,
    edge_trim_fraction: float = OBSERVED_SNR_EDGE_TRIM_FRACTION,
    maximum_relative_error: float = OBSERVED_SNR_MAX_RELATIVE_ERROR,
) -> dict[str, Any]:
    """Write measured S/N for the exact objects and visits used in evaluation.

    The object catalogue records the final coadded S/N and the best single
    observational epoch. When ``epochs_output`` is supplied, also write a
    long-form table containing one row per object per chronological epoch. It
    records both the single-epoch S/N and the cumulative coadded S/N through
    that epoch.
    """
    if split not in {"selection", "calibration", "test"}:
        raise ValueError("split must be selection, calibration, or test")
    if not 0.0 <= edge_trim_fraction < 0.5:
        raise ValueError("edge_trim_fraction must lie in [0, 0.5)")
    if maximum_relative_error <= 1.0:
        raise ValueError("maximum_relative_error must exceed one")
    if external_prepared_dir is not None and split != "test":
        raise ValueError("A prepared external-test store can only use the test split")
    if external_prepared_dir is not None and output is None:
        raise ValueError("External-test S/N generation requires an explicit output")
    data_config = config
    external_provenance: dict[str, str] | None = None
    if external_prepared_dir is not None:
        external_provenance = prepared_external_test_provenance(
            external_prepared_dir
        )
        data_config = config_for_prepared_external_test(
            config,
            external_prepared_dir,
        )
    run_dir = project_path(config, config["project"]["output_dir"])
    output_path = (
        Path(output)
        if output is not None
        else run_dir / f"{split}_observed_signal_to_noise.parquet"
    )
    epochs_output_path = (
        None if epochs_output is None else Path(epochs_output)
    )
    requested: pd.DataFrame | None = None
    requested_snids: set[int] | None = None
    if predictions is not None:
        requested = _read_table(Path(predictions))
        required = {"snid"}
        missing = required - set(requested.columns)
        if missing:
            raise ValueError(f"Predictions lack required columns: {sorted(missing)}")
        if requested["snid"].duplicated().any():
            raise ValueError("Prediction SNIDs must be unique")
        requested_snids = set(requested["snid"].astype(int))

    dataset = SundialDataset(
        data_config,
        split,
        "original",
        training=False,
        pair_no_source=False,
    )
    dataset_snids = set(dataset.objects["snid"].astype(int))
    if requested_snids is not None:
        missing_snids = requested_snids - dataset_snids
        if missing_snids:
            examples = ", ".join(str(value) for value in sorted(missing_snids)[:5])
            raise ValueError(
                f"Predictions contain {len(missing_snids)} SNIDs outside the prepared "
                f"{split} split, including {examples}"
            )

    rows: list[dict[str, float | int]] = []
    epoch_rows: list[dict[str, float | int | bool]] = []
    for index in range(len(dataset.objects)):
        snid = int(dataset.objects.iloc[index].snid)
        if requested_snids is not None and snid not in requested_snids:
            continue
        record, object_epoch_rows = dataset.observed_signal_to_noise_records(
            index,
            edge_trim_fraction=edge_trim_fraction,
            maximum_relative_error=maximum_relative_error,
        )
        best_epoch = _annotate_and_summarize_best_epoch(object_epoch_rows)
        record.update(best_epoch)
        if epochs_output_path is not None:
            epoch_rows.extend(object_epoch_rows)
        rows.append(record)
        if len(rows) % 1000 == 0:
            print(f"  measured S/N for {len(rows):,} objects", flush=True)
    catalog = pd.DataFrame(rows)
    if requested is not None:
        requested_order = requested[["snid"]].copy()
        requested_order["snid"] = requested_order["snid"].astype(int)
        catalog = requested_order.merge(catalog, on="snid", how="left", validate="one_to_one")
        if catalog[OBSERVED_SNR_COLUMN].isna().any():
            raise ValueError("Observed S/N catalog is incomplete for the prediction table")
        if "visit_count" in requested:
            expected_visits = requested[["snid", "visit_count"]].copy()
            expected_visits["snid"] = expected_visits["snid"].astype(int)
            checked = catalog.merge(
                expected_visits,
                on="snid",
                suffixes=("_snr", "_prediction"),
                validate="one_to_one",
            )
            mismatch = checked["visit_count_snr"] != checked["visit_count_prediction"]
            if mismatch.any():
                raise ValueError(
                    f"Observed S/N used different visits for {int(mismatch.sum())} objects"
                )
    config_digest = resolved_config_sha256(config)
    data_config_digest = resolved_config_sha256(data_config)
    if requested is not None and "config_sha256" in requested:
        if not requested["config_sha256"].astype(str).eq(config_digest).all():
            raise ValueError("Prediction checkpoint configuration does not match S/N config")
    if requested is not None and "data_config_sha256" in requested:
        if not requested["data_config_sha256"].astype(str).eq(data_config_digest).all():
            raise ValueError("Prediction data configuration does not match S/N data")
    catalog.insert(0, "data_split", split)
    catalog.insert(1, "config_sha256", config_digest)
    if external_provenance is not None:
        catalog.insert(2, "data_config_sha256", data_config_digest)
        catalog.insert(3, "dataset_tag", external_provenance["dataset_tag"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_table(catalog, output_path)
    epoch_summary: dict[str, Any] | None = None
    if epochs_output_path is not None:
        epoch_catalog = pd.DataFrame(epoch_rows)
        if requested is not None and not epoch_catalog.empty:
            order = {
                int(snid): position
                for position, snid in enumerate(requested["snid"].astype(int))
            }
            epoch_catalog["_prediction_order"] = (
                epoch_catalog["snid"].astype(int).map(order)
            )
            if epoch_catalog["_prediction_order"].isna().any():
                raise ValueError("Epoch S/N table includes an object outside predictions")
            epoch_catalog = (
                epoch_catalog.sort_values(["_prediction_order", "observation_epoch"])
                .drop(columns="_prediction_order")
                .reset_index(drop=True)
            )
        _validate_final_epoch_rows(catalog, epoch_catalog)
        epoch_catalog.insert(0, "data_split", split)
        epoch_catalog.insert(1, "config_sha256", config_digest)
        if external_provenance is not None:
            epoch_catalog.insert(2, "data_config_sha256", data_config_digest)
            epoch_catalog.insert(3, "dataset_tag", external_provenance["dataset_tag"])
        epochs_output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_table(epoch_catalog, epochs_output_path)
        epoch_summary = _write_epoch_summary(
            epoch_catalog,
            epochs_output_path,
            split=split,
            config_digest=config_digest,
            configured_minimum=float(data_config["observation"]["wavelength_min"]),
            configured_maximum=float(data_config["observation"]["wavelength_max"]),
            edge_trim_fraction=edge_trim_fraction,
            maximum_relative_error=maximum_relative_error,
        )
    finite = np.isfinite(catalog[OBSERVED_SNR_COLUMN].to_numpy(dtype=float))
    snr_values = catalog.loc[finite, OBSERVED_SNR_COLUMN].to_numpy(dtype=float)
    valid_bin_values = catalog.loc[
        finite, "observed_snr_valid_wavelength_bins"
    ].to_numpy(dtype=float)
    configured_minimum = float(data_config["observation"]["wavelength_min"])
    configured_maximum = float(data_config["observation"]["wavelength_max"])
    log_minimum = np.log(configured_minimum)
    log_maximum = np.log(configured_maximum)
    used_minimum = float(
        np.exp(log_minimum + edge_trim_fraction * (log_maximum - log_minimum))
    )
    used_maximum = float(
        np.exp(
            log_minimum
            + (1.0 - edge_trim_fraction) * (log_maximum - log_minimum)
        )
    )
    summary = {
        "format_version": "strider-observed-snr-v2",
        "output": str(output_path),
        "split": split,
        "config_sha256": resolved_config_sha256(config),
        "objects": int(len(catalog)),
        "finite_objects": int(finite.sum()),
        "snr_quantiles_0_05_0_25_0_50_0_75_0_95": (
            np.quantile(snr_values, [0.05, 0.25, 0.50, 0.75, 0.95]).tolist()
            if len(snr_values)
            else []
        ),
        "valid_native_bin_quantiles_0_05_0_50_0_95": (
            np.quantile(valid_bin_values, [0.05, 0.50, 0.95]).tolist()
            if len(valid_bin_values)
            else []
        ),
        "finite_best_epoch_objects": int(
            np.isfinite(catalog[BEST_EPOCH_SNR_COLUMN].to_numpy(dtype=float)).sum()
        ),
        "best_epoch_snr_quantiles_0_05_0_25_0_50_0_75_0_95": (
            np.quantile(
                catalog.loc[
                    np.isfinite(
                        catalog[BEST_EPOCH_SNR_COLUMN].to_numpy(dtype=float)
                    ),
                    BEST_EPOCH_SNR_COLUMN,
                ].to_numpy(dtype=float),
                [0.05, 0.25, 0.50, 0.75, 0.95],
            ).tolist()
            if np.isfinite(
                catalog[BEST_EPOCH_SNR_COLUMN].to_numpy(dtype=float)
            ).any()
            else []
        ),
        "definition": (
            "signed median across native wavelength bins of the inverse-variance "
            "coadded observed FLAM divided by its propagated FLAMERR"
        ),
        "best_epoch_definition": (
            "maximum signed median native-bin S/N among the object's retained "
            "observational epochs; ties select the earliest epoch"
        ),
        "visits": "exact deterministic evaluation visits for each object",
        "configured_wavelength_angstrom": [configured_minimum, configured_maximum],
        "used_wavelength_angstrom": [used_minimum, used_maximum],
        "log_wavelength_edge_trim_fraction_per_side": edge_trim_fraction,
        "maximum_relative_coadded_error": maximum_relative_error,
        "error_quality_mask": (
            "exclude bins with propagated coadd error above the stated multiple "
            "of the object's median in-band propagated error"
        ),
        "uses_simulation_clean_flux": False,
        "uses_class_or_redshift_truth": False,
    }
    if external_provenance is not None:
        summary["external_test"] = {
            **external_provenance,
            "data_config_sha256": data_config_digest,
        }
    if epoch_summary is not None:
        summary["epochs_output"] = epoch_summary["output"]
        summary["epoch_rows"] = epoch_summary["rows"]
        summary["epoch_summary"] = epoch_summary["summary"]
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary


def _annotate_and_summarize_best_epoch(
    epochs: list[dict[str, float | int | bool]],
) -> dict[str, float | int]:
    """Mark and summarize the highest finite single-epoch observed S/N."""
    for epoch in epochs:
        epoch["is_best_epoch"] = False
    values = np.asarray(
        [epoch["median_epoch_observed_signal_to_noise"] for epoch in epochs],
        dtype=np.float64,
    )
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return {
            BEST_EPOCH_SNR_COLUMN: float("nan"),
            "best_observation_epoch": -1,
            "best_observation_index": -1,
            "best_epoch_mjd": float("nan"),
            "best_epoch_observer_days": float("nan"),
            "best_epoch_observed_snr_valid_wavelength_bins": 0,
        }
    # np.argmax returns the first maximum, giving a deterministic earliest-epoch
    # tie break because the input rows are in chronological order.
    best_position = int(finite[np.argmax(values[finite])])
    best = epochs[best_position]
    best["is_best_epoch"] = True
    return {
        BEST_EPOCH_SNR_COLUMN: float(values[best_position]),
        "best_observation_epoch": int(best["observation_epoch"]),
        "best_observation_index": int(best["observation_index"]),
        "best_epoch_mjd": float(best["mjd"]),
        "best_epoch_observer_days": float(best["observer_days"]),
        "best_epoch_observed_snr_valid_wavelength_bins": int(
            best["epoch_observed_snr_valid_wavelength_bins"]
        ),
    }


def _validate_final_epoch_rows(
    catalog: pd.DataFrame,
    epoch_catalog: pd.DataFrame,
) -> None:
    """Require the last cumulative epoch to match the object-level catalog."""
    if epoch_catalog.empty:
        if len(catalog):
            raise ValueError("Epoch S/N output is empty for a non-empty object catalog")
        return
    final = epoch_catalog.loc[epoch_catalog["is_final_epoch"].astype(bool)].copy()
    if final["snid"].duplicated().any() or len(final) != len(catalog):
        raise ValueError("Epoch S/N output must contain one final row per object")
    compared = catalog[
        [
            "snid",
            "visit_count",
            OBSERVED_SNR_COLUMN,
            "observed_snr_valid_wavelength_bins",
        ]
    ].merge(
        final[
            [
                "snid",
                "visit_count",
                OBSERVED_SNR_COLUMN,
                "observed_snr_valid_wavelength_bins",
            ]
        ],
        on="snid",
        suffixes=("_object", "_epoch"),
        validate="one_to_one",
    )
    if len(compared) != len(catalog):
        raise ValueError("Epoch S/N output does not cover the object catalog")
    if not np.array_equal(
        compared["visit_count_object"].to_numpy(dtype=int),
        compared["visit_count_epoch"].to_numpy(dtype=int),
    ):
        raise ValueError("Final epoch visit counts do not match the object catalog")
    if not np.array_equal(
        compared["observed_snr_valid_wavelength_bins_object"].to_numpy(dtype=int),
        compared["observed_snr_valid_wavelength_bins_epoch"].to_numpy(dtype=int),
    ):
        raise ValueError("Final epoch valid-bin counts do not match the object catalog")
    if not np.allclose(
        compared[f"{OBSERVED_SNR_COLUMN}_object"].to_numpy(dtype=float),
        compared[f"{OBSERVED_SNR_COLUMN}_epoch"].to_numpy(dtype=float),
        equal_nan=True,
    ):
        raise ValueError("Final cumulative epoch S/N does not match the object catalog")
    best = epoch_catalog.loc[epoch_catalog["is_best_epoch"].astype(bool)].copy()
    finite_best = catalog[np.isfinite(catalog[BEST_EPOCH_SNR_COLUMN])]
    if best["snid"].duplicated().any() or len(best) != len(finite_best):
        raise ValueError("Epoch S/N output must identify one best row per finite object")
    compared_best = finite_best[
        ["snid", BEST_EPOCH_SNR_COLUMN, "best_observation_epoch"]
    ].merge(
        best[
            [
                "snid",
                "median_epoch_observed_signal_to_noise",
                "observation_epoch",
            ]
        ],
        on="snid",
        validate="one_to_one",
    )
    if len(compared_best) != len(finite_best):
        raise ValueError("Best-epoch S/N output does not cover the object catalog")
    if not np.allclose(
        compared_best[BEST_EPOCH_SNR_COLUMN].to_numpy(dtype=float),
        compared_best["median_epoch_observed_signal_to_noise"].to_numpy(dtype=float),
    ):
        raise ValueError("Best single-epoch S/N does not match the object catalog")
    if not np.array_equal(
        compared_best["best_observation_epoch"].to_numpy(dtype=int),
        compared_best["observation_epoch"].to_numpy(dtype=int),
    ):
        raise ValueError("Best observation epoch does not match the object catalog")


def _write_epoch_summary(
    catalog: pd.DataFrame,
    output_path: Path,
    *,
    split: str,
    config_digest: str,
    configured_minimum: float,
    configured_maximum: float,
    edge_trim_fraction: float,
    maximum_relative_error: float,
) -> dict[str, Any]:
    """Write provenance and compact diagnostics for a long-form epoch catalog."""
    log_minimum = np.log(configured_minimum)
    log_maximum = np.log(configured_maximum)
    used_minimum = float(
        np.exp(log_minimum + edge_trim_fraction * (log_maximum - log_minimum))
    )
    used_maximum = float(
        np.exp(
            log_minimum
            + (1.0 - edge_trim_fraction) * (log_maximum - log_minimum)
        )
    )
    epoch_values = catalog["median_epoch_observed_signal_to_noise"].to_numpy(
        dtype=float
    )
    cumulative_values = catalog[OBSERVED_SNR_COLUMN].to_numpy(dtype=float)
    epoch_counts = catalog.groupby("snid", sort=False).size().to_numpy(dtype=float)

    def quantiles(values: np.ndarray) -> list[float]:
        finite_values = values[np.isfinite(values)]
        if not len(finite_values):
            return []
        return np.quantile(finite_values, [0.05, 0.25, 0.50, 0.75, 0.95]).tolist()

    summary: dict[str, Any] = {
        "format_version": "strider-observed-snr-by-epoch-v2",
        "output": str(output_path),
        "split": split,
        "config_sha256": config_digest,
        "objects": int(catalog["snid"].nunique()),
        "rows": int(len(catalog)),
        "finite_single_epoch_rows": int(np.isfinite(epoch_values).sum()),
        "finite_cumulative_rows": int(np.isfinite(cumulative_values).sum()),
        "epochs_per_object_quantiles_0_05_0_25_0_50_0_75_0_95": quantiles(
            epoch_counts
        ),
        "single_epoch_snr_quantiles_0_05_0_25_0_50_0_75_0_95": quantiles(
            epoch_values
        ),
        "cumulative_snr_quantiles_0_05_0_25_0_50_0_75_0_95": quantiles(
            cumulative_values
        ),
        "single_epoch_definition": (
            "signed median native-bin S/N from observed FLAM and FLAMERR for one "
            "chronological observational epoch"
        ),
        "cumulative_definition": (
            "signed median native-bin S/N after inverse-variance coadding observed "
            "FLAM and FLAMERR through the stated chronological epoch"
        ),
        "visit_order": "chronological MJD order after deterministic evaluation selection",
        "configured_wavelength_angstrom": [configured_minimum, configured_maximum],
        "used_wavelength_angstrom": [used_minimum, used_maximum],
        "log_wavelength_edge_trim_fraction_per_side": edge_trim_fraction,
        "maximum_relative_coadded_error": maximum_relative_error,
        "uses_simulation_clean_flux": False,
        "uses_class_or_redshift_truth": False,
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Prediction/SNR tables must be parquet or CSV")


def _write_table(values: pd.DataFrame, path: Path) -> None:
    if path.suffix.lower() == ".parquet":
        values.to_parquet(path, index=False)
        return
    if path.suffix.lower() == ".csv":
        values.to_csv(path, index=False, float_format="%.7g")
        return
    raise ValueError("Observed S/N output must be parquet or CSV")
