from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.plot_paper_sundial_redshift_snr import _attach_snr_catalog
from strider.data.dataset import (
    _coadded_observed_signal_to_noise,
    _observed_signal_to_noise_series,
)
from strider.evaluation import signal_to_noise
from strider.evaluation.signal_to_noise import (
    _annotate_and_summarize_best_epoch,
)


def _native_observations(
    fluxes: list[np.ndarray],
    errors: list[np.ndarray],
) -> tuple[pd.DataFrame, dict[str, np.ndarray | int]]:
    bins = len(fluxes[0])
    wavelength_edges = np.linspace(7500.0, 18175.0, bins + 1)
    rows = []
    for visit in range(len(fluxes)):
        rows.append(
            {
                "first_bin": visit * bins,
                "bin_count": bins,
                "mjd": 62000.0 + visit,
            }
        )
    return pd.DataFrame(rows), {
        "first_bin": 0,
        "wavelength_min": np.tile(wavelength_edges[:-1], len(fluxes)),
        "wavelength_max": np.tile(wavelength_edges[1:], len(fluxes)),
        "observed_flux": np.concatenate(fluxes),
        "flux_error": np.concatenate(errors),
    }


def test_observed_snr_is_signed_inverse_variance_coadd() -> None:
    rows, block = _native_observations(
        [np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 2.0, 3.0])],
        [np.ones(3), np.ones(3)],
    )

    value, valid_bins = _coadded_observed_signal_to_noise(
        rows,
        block,
        wavelength_min=7500.0,
        wavelength_max=18175.0,
        edge_trim_fraction=0.0,
        maximum_relative_error=100.0,
    )

    assert valid_bins == 3
    assert value == pytest.approx(2.0 * np.sqrt(2.0))


def test_observed_snr_ignores_fixed_edges_and_high_error_bins() -> None:
    central = np.ones(10)
    flux = central.copy()
    flux[[0, 9]] = 1.0e6
    flux[5] = 1.0e4
    error = np.ones(10)
    error[5] = 100.0
    rows, block = _native_observations([flux], [error])

    value, valid_bins = _coadded_observed_signal_to_noise(
        rows,
        block,
        wavelength_min=7500.0,
        wavelength_max=18175.0,
        edge_trim_fraction=0.10,
        maximum_relative_error=3.0,
    )

    assert valid_bins < 10
    assert value == pytest.approx(1.0)


def test_observed_snr_series_tracks_single_and_cumulative_epochs() -> None:
    rows, block = _native_observations(
        [np.asarray([1.0, 2.0, 3.0]), np.asarray([3.0, 4.0, 5.0])],
        [np.ones(3), np.ones(3)],
    )

    series = _observed_signal_to_noise_series(
        rows,
        block,
        wavelength_min=7500.0,
        wavelength_max=18175.0,
        edge_trim_fraction=0.0,
        maximum_relative_error=100.0,
    )
    final_value, final_bins = _coadded_observed_signal_to_noise(
        rows,
        block,
        wavelength_min=7500.0,
        wavelength_max=18175.0,
        edge_trim_fraction=0.0,
        maximum_relative_error=100.0,
    )

    assert [row["observation_epoch"] for row in series] == [1, 2]
    assert [row["observer_days"] for row in series] == [0.0, 1.0]
    assert series[0]["median_epoch_observed_signal_to_noise"] == pytest.approx(2.0)
    assert series[1]["median_epoch_observed_signal_to_noise"] == pytest.approx(4.0)
    assert series[1]["median_coadded_observed_signal_to_noise"] == pytest.approx(
        3.0 * np.sqrt(2.0)
    )
    assert series[1]["median_coadded_observed_signal_to_noise"] == pytest.approx(
        final_value
    )
    assert series[1]["observed_snr_valid_wavelength_bins"] == final_bins == 3
    assert series[1]["is_final_epoch"] is True


def test_best_epoch_summary_marks_maximum_and_uses_earliest_tie() -> None:
    epochs = [
        {
            "observation_epoch": 1,
            "observation_index": 17,
            "mjd": 62001.0,
            "observer_days": 0.0,
            "median_epoch_observed_signal_to_noise": 4.0,
            "epoch_observed_snr_valid_wavelength_bins": 100,
        },
        {
            "observation_epoch": 2,
            "observation_index": 19,
            "mjd": 62005.0,
            "observer_days": 4.0,
            "median_epoch_observed_signal_to_noise": 4.0,
            "epoch_observed_snr_valid_wavelength_bins": 101,
        },
        {
            "observation_epoch": 3,
            "observation_index": 21,
            "mjd": 62009.0,
            "observer_days": 8.0,
            "median_epoch_observed_signal_to_noise": 2.0,
            "epoch_observed_snr_valid_wavelength_bins": 102,
        },
    ]

    result = _annotate_and_summarize_best_epoch(epochs)

    assert result[signal_to_noise.BEST_EPOCH_SNR_COLUMN] == 4.0
    assert result["best_observation_epoch"] == 1
    assert result["best_observation_index"] == 17
    assert [epoch["is_best_epoch"] for epoch in epochs] == [True, False, False]


def test_observed_snr_catalog_merge_is_one_to_one(tmp_path: Path) -> None:
    predictions = pd.DataFrame({"snid": [10, 20], "true_redshift": [0.4, 1.2]})
    catalog = pd.DataFrame(
        {
            "snid": [20, 10],
            "median_coadded_observed_signal_to_noise": [2.0, 0.5],
        }
    )
    path = tmp_path / "snr.parquet"
    catalog.to_parquet(path, index=False)

    result = _attach_snr_catalog(
        predictions,
        path,
        "median_coadded_observed_signal_to_noise",
    )

    assert result["snid"].tolist() == [10, 20]
    assert result["median_coadded_observed_signal_to_noise"].tolist() == [0.5, 2.0]


def test_observed_snr_catalog_merge_rejects_missing_objects(tmp_path: Path) -> None:
    predictions = pd.DataFrame({"snid": [10, 20]})
    catalog = pd.DataFrame(
        {"snid": [10], "median_coadded_observed_signal_to_noise": [0.5]}
    )
    path = tmp_path / "snr.csv"
    catalog.to_csv(path, index=False)

    with pytest.raises(ValueError, match="lacks measured values"):
        _attach_snr_catalog(
            predictions,
            path,
            "median_coadded_observed_signal_to_noise",
        )


def test_observed_snr_catalog_freezes_prediction_cohort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDataset:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.objects = pd.DataFrame({"snid": [10, 20, 30]})

        def observed_signal_to_noise_records(
            self,
            item: int,
            **_kwargs: object,
        ) -> tuple[
            dict[str, float | int],
            list[dict[str, float | int | bool]],
        ]:
            snid = int(self.objects.iloc[item].snid)
            visits = snid // 10
            epochs = [
                {
                    "snid": snid,
                    "observation_epoch": epoch,
                    "observation_index": 10 * item + epoch,
                    "mjd": 62000.0 + epoch,
                    "observer_days": float(epoch - 1),
                    "visit_count": epoch,
                    "total_visit_count": visits,
                    "median_epoch_observed_signal_to_noise": snid / 20.0,
                    "epoch_observed_snr_valid_wavelength_bins": 100 + item,
                    "median_coadded_observed_signal_to_noise": snid / 20.0,
                    "observed_snr_valid_wavelength_bins": 100 + item,
                    "is_final_epoch": epoch == visits,
                }
                for epoch in range(1, visits + 1)
            ]
            return (
                {
                    "snid": snid,
                    "visit_count": visits,
                    "median_coadded_observed_signal_to_noise": snid / 20.0,
                    "observed_snr_valid_wavelength_bins": 100 + item,
                },
                epochs,
            )

    monkeypatch.setattr(signal_to_noise, "SundialDataset", FakeDataset)
    monkeypatch.setattr(signal_to_noise, "resolved_config_sha256", lambda _config: "abc")
    predictions = pd.DataFrame({"snid": [30, 10], "visit_count": [3, 1]})
    prediction_path = tmp_path / "predictions.csv"
    output_path = tmp_path / "observed_snr.csv"
    predictions.to_csv(prediction_path, index=False)
    config = {
        "project": {"output_dir": str(tmp_path)},
        "observation": {"wavelength_min": 7500.0, "wavelength_max": 18175.0},
    }

    summary = signal_to_noise.write_observed_snr_catalog(
        config,
        predictions=prediction_path,
        output=output_path,
    )
    catalog = pd.read_csv(output_path)

    assert catalog["snid"].tolist() == [30, 10]
    assert catalog[signal_to_noise.OBSERVED_SNR_COLUMN].tolist() == [1.5, 0.5]
    assert catalog[signal_to_noise.BEST_EPOCH_SNR_COLUMN].tolist() == [1.5, 0.5]
    assert summary["objects"] == 2
    assert summary["uses_simulation_clean_flux"] is False
    assert summary["uses_class_or_redshift_truth"] is False
    assert output_path.with_suffix(".summary.json").is_file()


def test_observed_snr_catalog_writes_epoch_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDataset:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.objects = pd.DataFrame({"snid": [10, 20, 30]})

        def observed_signal_to_noise_records(
            self,
            item: int,
            **_kwargs: object,
        ) -> tuple[
            dict[str, float | int],
            list[dict[str, float | int | bool]],
        ]:
            snid = int(self.objects.iloc[item].snid)
            final_snr = snid / 10.0
            epochs: list[dict[str, float | int | bool]] = []
            for epoch in (1, 2):
                    epochs.append(
                        {
                            "snid": snid,
                            "observation_epoch": epoch,
                        "observation_index": item * 2 + epoch - 1,
                        "mjd": 62000.0 + epoch,
                        "observer_days": float(epoch - 1),
                        "visit_count": epoch,
                        "total_visit_count": 2,
                        "median_epoch_observed_signal_to_noise": (
                            final_snr / 2.0 if epoch == 1 else final_snr
                        ),
                        "epoch_observed_snr_valid_wavelength_bins": 100,
                        "median_coadded_observed_signal_to_noise": (
                            final_snr / 2.0 if epoch == 1 else final_snr
                        ),
                        "observed_snr_valid_wavelength_bins": 100,
                        "is_final_epoch": epoch == 2,
                    }
                )
            return (
                {
                    "snid": snid,
                    "visit_count": 2,
                    "median_coadded_observed_signal_to_noise": final_snr,
                    "observed_snr_valid_wavelength_bins": 100,
                },
                epochs,
            )

    monkeypatch.setattr(signal_to_noise, "SundialDataset", FakeDataset)
    monkeypatch.setattr(signal_to_noise, "resolved_config_sha256", lambda _config: "abc")
    predictions = pd.DataFrame({"snid": [30, 10], "visit_count": [2, 2]})
    prediction_path = tmp_path / "predictions.csv"
    output_path = tmp_path / "observed_snr.csv"
    epochs_path = tmp_path / "observed_snr_by_epoch.csv"
    predictions.to_csv(prediction_path, index=False)
    config = {
        "project": {"output_dir": str(tmp_path)},
        "observation": {"wavelength_min": 7500.0, "wavelength_max": 18175.0},
    }

    summary = signal_to_noise.write_observed_snr_catalog(
        config,
        predictions=prediction_path,
        output=output_path,
        epochs_output=epochs_path,
    )
    epochs = pd.read_csv(epochs_path)
    objects = pd.read_csv(output_path)

    assert epochs[["snid", "observation_epoch"]].values.tolist() == [
        [30, 1],
        [30, 2],
        [10, 1],
        [10, 2],
    ]
    assert epochs.loc[epochs["is_final_epoch"], "visit_count"].tolist() == [2, 2]
    assert epochs.loc[epochs["is_best_epoch"], "observation_epoch"].tolist() == [2, 2]
    assert objects[signal_to_noise.BEST_EPOCH_SNR_COLUMN].tolist() == [3.0, 1.0]
    assert objects["best_observation_epoch"].tolist() == [2, 2]
    assert summary["epochs_output"] == str(epochs_path)
    assert summary["epoch_rows"] == 4
    assert epochs_path.with_suffix(".summary.json").is_file()
