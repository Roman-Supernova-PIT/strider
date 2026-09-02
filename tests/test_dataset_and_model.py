from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from strider.config import load_config
from strider.data.dataset import (
    SundialDataset,
    _paired_standard_normal,
    collate_objects,
)
from strider.evaluation.noise_check import (
    _amplitude_summary,
    _balanced_ia_indices,
    _class_probability_diagnostics,
    _ia_probability_metrics,
    _mean_input_rms,
)
from strider.evaluation.measurement_controls import apply_measurement_control
from strider.model import Strider, measurement_inputs
from strider.training.losses import training_loss
from strider.training.trainer import (
    _loader,
    _optimizer_parameter_groups,
    _run_epoch,
    _validation_view_weights,
    _weighted_validation_score,
)


ROOT = Path(__file__).resolve().parents[1]


def test_ia_probability_metrics_require_negative_examples_for_purity() -> None:
    ia_only = pd.DataFrame(
        {"true_class_name": ["Ia", "Ia"], "p_Ia": [0.95, 0.40]}
    )
    mixed = pd.DataFrame(
        {
            "true_class_name": ["Ia", "Ia", "other", "other"],
            "p_Ia": [0.95, 0.40, 0.92, 0.10],
        }
    )

    ia_result = _ia_probability_metrics(ia_only, threshold=0.9)
    mixed_result = _ia_probability_metrics(mixed, threshold=0.9)

    assert np.isnan(ia_result["purity"])
    assert ia_result["completeness"] == 0.5
    assert mixed_result["purity"] == 0.5
    assert mixed_result["completeness"] == 0.5


def test_class_probability_diagnostics_measure_noise_confidence_and_reliability() -> None:
    predictions = pd.DataFrame(
        {
            "true_class_name": ["Ia", "Ia", "other", "other"],
            "p_Ia": [0.95, 0.75, 0.20, 0.10],
        }
    )

    result = _class_probability_diagnostics(predictions, bins=10)

    assert np.isclose(result["Ia_mean_probability"], 0.85)
    assert np.isclose(result["Ia_median_probability"], 0.85)
    assert np.isclose(result["Ia_fraction_p_ge_0_9"], 0.5)
    assert np.isclose(result["non_Ia_mean_probability"], 0.15)
    assert np.isclose(result["class_brier_score"], 0.02875)
    assert np.isclose(result["class_expected_calibration_error_15_bin"], 0.15)
    assert np.isclose(result["class_mean_score_minus_prevalence"], 0.0)


def test_class_probability_diagnostics_allow_ia_only_confidence_sweeps() -> None:
    predictions = pd.DataFrame(
        {"true_class_name": ["Ia", "Ia"], "p_Ia": [0.9, 0.5]}
    )

    result = _class_probability_diagnostics(predictions)

    assert np.isclose(result["Ia_mean_probability"], 0.7)
    assert np.isnan(result["non_Ia_mean_probability"])
    assert np.isclose(result["class_brier_score"], 0.13)


def _config_with_test_store(tmp_path: Path, config_name: str) -> dict:
    """Create a small prepared store so dataset tests never read local research data."""
    config = load_config(ROOT / config_name)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    object_rows = []
    observation_rows = []
    arrays = {
        name: []
        for name in (
            "wavelength_min",
            "wavelength_max",
            "observed_flux",
            "flux_error",
            "clean_flux",
        )
    }
    observation_index = 0
    first_bin = 0
    native_edges = np.geomspace(7500.0, 20000.0, 65, dtype=np.float32)
    native_position = np.linspace(0.0, 2.0 * np.pi, 64, dtype=np.float32)
    for object_index, split in enumerate(("train", "train", "test", "test")):
        redshift = 0.35 + 0.4 * object_index
        peak_mjd = 62000.0 + 100.0 * object_index
        first_observation = observation_index
        for visit in range(3):
            phase_change = 0.08 * visit * np.cos(2.0 * native_position)
            clean = (
                0.5 * np.sin(native_position + 0.3 * object_index)
                + phase_change
                + 0.15 * (object_index % 2)
            ).astype(np.float32)
            observed = (clean + 0.12 * np.cos(5.0 * native_position + visit)).astype(
                np.float32
            )
            flux_error = np.full(64, 0.8 + 0.05 * visit, dtype=np.float32)
            for name, values in (
                ("wavelength_min", native_edges[:-1]),
                ("wavelength_max", native_edges[1:]),
                ("observed_flux", observed),
                ("flux_error", flux_error),
                ("clean_flux", clean),
            ):
                arrays[name].append(values)
            observation_rows.append(
                {
                    "observation_index": observation_index,
                    "object_index": object_index,
                    "source_id": f"test:{1000 + object_index}",
                    "source_product": "test",
                    "snid": 1000 + object_index,
                    "mjd": peak_mjd + (-12.0 + 14.0 * visit) * (1.0 + redshift),
                    "exposure_seconds": 900.0,
                    "first_bin": first_bin,
                    "bin_count": 64,
                    "native_wavelength_min": float(native_edges[0]),
                    "native_wavelength_max": float(native_edges[-1]),
                    "background_scale": float(np.quantile(flux_error, 0.25)),
                }
            )
            observation_index += 1
            first_bin += 64
        object_rows.append(
            {
                "object_index": object_index,
                "source_id": f"test:{1000 + object_index}",
                "source_product": "test",
                "snid": 1000 + object_index,
                "split": split,
                "block": 1,
                "model": "TEST",
                "gentype": 10 if object_index % 2 == 0 else 20,
                "template_index": 0,
                "redshift": redshift,
                "estimated_peak_mjd": peak_mjd + 0.5,
                "simulation_peak_mjd": peak_mjd,
                "class_index": object_index % 2,
                "class_name": "Ia" if object_index % 2 == 0 else "other",
                "first_observation": first_observation,
                "observation_count": 3,
            }
        )
    pd.DataFrame(object_rows).to_parquet(prepared / "objects.parquet", index=False)
    pd.DataFrame(observation_rows).to_parquet(
        prepared / "observations.parquet", index=False
    )
    with h5py.File(prepared / "spectra.h5", "w") as store:
        for name, pieces in arrays.items():
            store.create_dataset(name, data=np.concatenate(pieces).astype(np.float32))
    config["data"]["prepared_dir"] = str(prepared)
    return config


def test_prepared_item_and_model_forward(tmp_path: Path) -> None:
    config = _config_with_test_store(tmp_path, "configs/local_pilot.yaml")
    dataset = SundialDataset(config, "train", "generated", training=True)
    batch = collate_objects([dataset[0], dataset[1]])
    assert "simulation_rest_phase_days" in batch
    assert "median_coadded_observed_signal_to_noise" not in batch
    assert batch["flux"].ndim == 3
    assert batch["flux"].shape[-1] == config["observation"]["wavelength_bins"]
    model = Strider(config)
    model_batch = measurement_inputs(batch)
    assert set(model_batch) == {
        "flux",
        "wavelength_mask",
        "visit_mask",
        "observer_days",
        "visit_flux_scale",
        "peak_day_offset",
        "peak_date_valid",
    }
    assert "simulation_rest_phase_days" not in model_batch
    assert batch["peak_date_valid"].tolist() == [1.0, 1.0]
    expected_peak_offsets = []
    for object_index in range(2):
        redshift = 0.35 + 0.4 * object_index
        expected_peak_offsets.append(0.5 + 12.0 * (1.0 + redshift))
    assert torch.allclose(
        batch["peak_day_offset"],
        torch.tensor(expected_peak_offsets),
    )
    output = model(model_batch)
    assert output["joint_logits"].shape == (
        2,
        len(config["model"]["classes"]),
        config["model"]["redshift_bins"],
    )
    loss, pieces = training_loss(
        output,
        batch,
        model.redshift_grid,
        model.redshift_cell_width,
        model.redshift_prior,
        config["training"]["evidence_sufficiency_loss_weight"],
    )
    assert torch.isfinite(loss)
    assert pieces["joint_loss"] > 0


def test_optional_flux_error_shape_reaches_model_without_labels(tmp_path: Path) -> None:
    config = _config_with_test_store(tmp_path, "configs/local_pilot.yaml")
    config["data"]["include_flux_error_channel"] = True
    config["model"]["use_flux_error_channel"] = True
    config["model"]["flux_error_initial_scale"] = 0.2
    dataset = SundialDataset(config, "train", "original", training=True)
    batch = collate_objects([dataset[0], dataset[1]])

    assert batch["flux_error_shape"].shape == batch["flux"].shape
    assert torch.isfinite(batch["flux_error_shape"]).all()
    model_batch = measurement_inputs(batch)
    assert "flux_error_shape" in model_batch
    assert "redshift" not in model_batch

    model = Strider(config)
    fused = model._measurement_batch(model_batch)
    expected = (
        batch["flux"]
        + 0.2 * batch["flux_error_shape"]
    ) * batch["wavelength_mask"]
    assert torch.allclose(fused["flux"], expected, atol=1.0e-6)
    assert model(model_batch)["joint_logits"].shape[:2] == (2, 2)


def test_clean_coadd_target_is_training_only(tmp_path: Path) -> None:
    config = _config_with_test_store(tmp_path, "configs/local_pilot.yaml")
    config["data"]["include_flux_error_channel"] = True
    config["data"]["include_clean_flux_target"] = True
    dataset = SundialDataset(config, "train", "generated", training=True)
    batch = collate_objects([dataset[0], dataset[1]])

    assert batch["clean_flux_target"].shape == batch["flux"].shape
    assert batch["flux_error_shape"].shape == batch["flux"].shape
    assert torch.isfinite(batch["clean_flux_target"]).all()
    assert torch.isfinite(batch["flux_error_shape"]).all()
    assert "clean_flux_target" not in measurement_inputs(batch)


def test_measurement_controls_ablate_and_shuffle_without_mutating_batch() -> None:
    flux = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    error = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]
    )
    batch = {
        "flux": flux,
        "flux_error_shape": error,
        "wavelength_mask": torch.ones_like(flux),
        "snid": torch.tensor([7]),
    }

    flux_only = apply_measurement_control(batch, "flux_only")
    error_only = apply_measurement_control(batch, "error_only")
    shuffled = apply_measurement_control(batch, "shuffled_error")

    assert torch.count_nonzero(flux_only["flux_error_shape"]) == 0
    assert torch.count_nonzero(error_only["flux"]) == 0
    assert torch.equal(batch["flux"], flux)
    assert torch.equal(batch["flux_error_shape"], error)
    assert not torch.equal(shuffled["flux_error_shape"], error)
    assert torch.equal(
        shuffled["flux_error_shape"].sort(dim=-1).values,
        error.sort(dim=-1).values,
    )


def test_all_visit_setting_retains_every_observation(tmp_path: Path) -> None:
    config = _config_with_test_store(tmp_path, "configs/local_pilot.yaml")
    config["data"]["max_visits"] = "all"
    config["data"]["training_visit_counts"] = [1]
    config["training"]["full_visit_training_fraction"] = 1.0

    training = SundialDataset(config, "train", "clean", training=True)
    evaluation = SundialDataset(
        config,
        "test",
        "clean",
        training=False,
        include_observed_signal_to_noise=True,
    )

    assert training.requested_visit_count(0) == 3
    assert training[0]["flux"].shape[0] == 3
    assert evaluation.requested_visit_count(0) == 3
    evaluation_item = evaluation[0]
    assert evaluation_item["flux"].shape[0] == 3
    assert torch.isfinite(
        evaluation_item["median_coadded_observed_signal_to_noise"]
    )
    assert evaluation_item["observed_snr_valid_wavelength_bins"] > 0


def test_all_visit_training_accumulates_memory_sized_microbatches(tmp_path: Path) -> None:
    config = _config_with_test_store(tmp_path, "configs/local_pilot.yaml")
    config["data"]["max_visits"] = "all"
    config["data"]["training_visit_counts"] = ["all"]
    config["training"].update(
        {
            "batch_size": 2,
            "num_workers": 0,
            "batch_by_visit_count": True,
            "maximum_visits_per_batch": 3,
            "maximum_squared_visits_per_batch": 9,
        }
    )
    dataset = SundialDataset(
        config,
        "train",
        "clean",
        training=True,
        pair_no_source=False,
    )
    loader = _loader(dataset, config["training"], shuffle=True)
    assert all(len(batch) == 1 for batch in loader.batch_sampler)

    model = Strider(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    report = _run_epoch(
        model,
        loader,
        torch.device("cpu"),
        model.redshift_cell_width,
        model.redshift_prior,
        float(config["training"]["evidence_sufficiency_loss_weight"]),
        float(config["training"].get("no_source_redshift_loss_weight", 0.0)),
        float(config["training"].get("no_source_class_loss_weight", 0.0)),
        float(config.get("onir", {}).get("drift_loss_weight", 0.0)),
        torch.ones(len(config["model"]["classes"])),
        float(config["training"].get("phase_loss_weight", 0.0)),
        optimizer,
        "float32",
        1.0,
        optimizer_step_objects=2,
    )
    assert np.isfinite(report["loss"])


def test_phase_target_trains_a_head_without_entering_model_inputs(tmp_path: Path) -> None:
    config = _config_with_test_store(tmp_path, "configs/local_pilot.yaml")
    config["model"].update(
        {
            "phase_auxiliary_bins": 15,
            "phase_auxiliary_min_days": -20.0,
            "phase_auxiliary_max_days": 50.0,
        }
    )
    dataset = SundialDataset(config, "train", "clean", training=True)
    batch = collate_objects([dataset[0], dataset[1]])
    model = Strider(config).eval()
    model_batch = measurement_inputs(batch)

    output = model(model_batch)
    assert output["phase_logits"].shape == (
        2,
        3,
        2,
        config["model"]["redshift_bins"],
        15,
    )
    changed_target = dict(batch)
    changed_target["simulation_rest_phase_days"] = (
        batch["simulation_rest_phase_days"] + 7.0
    )
    assert torch.equal(output["phase_logits"], model(measurement_inputs(changed_target))["phase_logits"])

    loss, pieces = training_loss(
        output,
        batch,
        model.redshift_grid,
        model.redshift_cell_width,
        model.redshift_prior,
        config["training"]["evidence_sufficiency_loss_weight"],
        phase_loss_weight=0.25,
    )
    assert torch.isfinite(loss)
    assert pieces["phase_loss"] > 0.0
    assert pieces["phase_supervised_visit_fraction"] > 0.0


def test_candidate_phase_consistency_uses_only_measurement_inputs(tmp_path: Path) -> None:
    config = _config_with_test_store(tmp_path, "configs/local_pilot.yaml")
    config["model"].update(
        {
            "phase_auxiliary_bins": 15,
            "phase_auxiliary_min_days": -20.0,
            "phase_auxiliary_max_days": 50.0,
            "candidate_phase_consistency": True,
            "candidate_phase_initial_scale": 0.1,
            "candidate_phase_minimum_visits": 2,
            "candidate_phase_use_peak_date": True,
            "candidate_phase_peak_uncertainty_days": 5.0,
            "candidate_phase_peak_quadrature_points": 3,
            "candidate_phase_peak_outlier_fraction": 0.1,
        }
    )
    dataset = SundialDataset(config, "train", "clean", training=True)
    batch = collate_objects([dataset[0], dataset[1]])
    model = Strider(config).eval()

    output = model(measurement_inputs(batch))
    assert output["phase_consistency_joint_logits"].shape == output["joint_logits"].shape
    assert "simulation_rest_phase_days" not in measurement_inputs(batch)
    assert "simulation_peak_mjd" not in measurement_inputs(batch)

    changed = dict(batch)
    changed["simulation_rest_phase_days"] = batch["simulation_rest_phase_days"] + 20.0
    changed_output = model(measurement_inputs(changed))
    assert torch.equal(
        output["phase_consistency_joint_logits"],
        changed_output["phase_consistency_joint_logits"],
    )


def test_missing_measured_peak_makes_peak_route_abstain(tmp_path: Path) -> None:
    config = _config_with_test_store(tmp_path, "configs/local_pilot.yaml")
    prepared = Path(config["data"]["prepared_dir"])
    objects = pd.read_parquet(prepared / "objects.parquet")
    objects["estimated_peak_mjd"] = -9.0
    objects.to_parquet(prepared / "objects.parquet", index=False)
    config["model"].update(
        {
            "phase_auxiliary_bins": 15,
            "phase_auxiliary_min_days": -20.0,
            "phase_auxiliary_max_days": 50.0,
            "candidate_phase_consistency": True,
            "candidate_phase_initial_scale": 0.5,
            "candidate_phase_minimum_visits": 2,
            "candidate_phase_use_peak_date": True,
        }
    )
    batch = collate_objects(
        [SundialDataset(config, "train", "clean", training=True)[0]]
    )
    output = Strider(config).eval()(measurement_inputs(batch))

    assert batch["peak_date_valid"].item() == 0.0
    assert batch["peak_day_offset"].item() == 0.0
    assert torch.equal(
        output["phase_consistency_joint_logits"],
        torch.zeros_like(output["phase_consistency_joint_logits"]),
    )


def test_generated_observation_is_repeatable_within_epoch(tmp_path: Path) -> None:
    config = _config_with_test_store(tmp_path, "configs/local_pilot.yaml")
    dataset = SundialDataset(config, "train", "generated", training=True)
    dataset.set_epoch(3)
    first = dataset[0]["flux"]
    second = dataset[0]["flux"]
    assert torch.equal(first, second)
    dataset.set_epoch(4)
    third = dataset[0]["flux"]
    assert not torch.equal(first, third)


def test_generated_observation_uses_one_fixed_detector_mask(tmp_path: Path) -> None:
    config = _config_with_test_store(tmp_path, "configs/local_pilot.yaml")
    dataset = SundialDataset(config, "train", "generated", training=True)
    first = dataset[0]["wavelength_mask"]
    second = dataset[1]["wavelength_mask"]
    assert torch.all(first)
    assert torch.all(second)


def test_no_source_view_has_no_clean_signal(tmp_path: Path) -> None:
    config = _config_with_test_store(tmp_path, "configs/local_pilot.yaml")
    no_source = SundialDataset(config, "test", "no_source", training=False)[0]
    generated = SundialDataset(config, "test", "generated", training=False)[0]
    assert no_source["evidence_sufficiency_target"].item() == 0.0
    assert no_source["has_source"].item() == 0.0
    assert generated["has_source"].item() == 1.0
    assert 0.0 <= generated["evidence_sufficiency_target"].item() <= 1.0
    assert not torch.equal(no_source["flux"], generated["flux"])


def test_source_removal_control_keeps_the_same_generated_noise(tmp_path: Path) -> None:
    config = _config_with_test_store(tmp_path, "configs/local_pilot.yaml")
    source = SundialDataset(config, "test", "generated", training=False)[0]
    no_source = SundialDataset(config, "test", "no_source", training=False)[0]
    clean = SundialDataset(config, "test", "clean", training=False)[0]
    assert torch.allclose(
        source["flux"] - no_source["flux"],
        clean["flux"],
        atol=5.0e-7,
        rtol=1.0e-5,
    )


def test_evaluation_repeat_redraws_generated_noise(tmp_path: Path) -> None:
    config = _config_with_test_store(tmp_path, "configs/local_pilot.yaml")
    first = SundialDataset(
        config, "test", "generated", training=False, visit_repeat=0
    )[0]
    second = SundialDataset(
        config, "test", "generated", training=False, visit_repeat=1
    )[0]
    assert not torch.equal(first["flux"], second["flux"])


def test_evaluation_can_set_generated_noise_scale(tmp_path: Path) -> None:
    config = _config_with_test_store(tmp_path, "configs/local_pilot.yaml")
    clean = SundialDataset(config, "test", "clean", training=False)[0]
    generated = SundialDataset(
        config,
        "test",
        "generated",
        training=False,
        generated_noise_scale=0.0,
    )[0]
    blank = SundialDataset(
        config,
        "test",
        "no_source",
        training=False,
        generated_noise_scale=0.0,
    )[0]

    assert torch.allclose(generated["flux"], clean["flux"])
    assert torch.count_nonzero(blank["flux"]) == 0


def test_evaluation_noise_scale_must_be_nonnegative(tmp_path: Path) -> None:
    config = _config_with_test_store(tmp_path, "configs/local_pilot.yaml")
    try:
        SundialDataset(
            config,
            "test",
            "generated",
            training=False,
            generated_noise_scale=-0.1,
        )
    except ValueError as error:
        assert "generated_noise_scale" in str(error)
    else:
        raise AssertionError("Negative generated noise scale was accepted")


def test_clean_view_can_form_a_true_no_source_twin(tmp_path: Path) -> None:
    config = _config_with_test_store(
        tmp_path, "configs/experiments/cadence_controlled.yaml"
    )
    dataset = SundialDataset(
        config, "train", "clean", training=True, pair_no_source=True
    )
    source = dataset[0]
    no_source = dataset[1]
    assert torch.count_nonzero(source["flux"]) > 0
    assert torch.count_nonzero(no_source["flux"]) == 0
    assert no_source["has_source"].item() == 0.0


def test_residual_and_reported_error_controls_are_distinct(tmp_path: Path) -> None:
    config = _config_with_test_store(tmp_path, "configs/local_pilot.yaml")
    residual = SundialDataset(config, "test", "residual", training=False)[0]
    fresh = SundialDataset(config, "test", "reported_error_no_source", training=False)[0]
    assert residual["evidence_sufficiency_target"].item() == 0.0
    assert fresh["evidence_sufficiency_target"].item() == 0.0
    assert not torch.equal(residual["flux"], fresh["flux"])


def test_reported_error_source_removal_keeps_the_same_noise(tmp_path: Path) -> None:
    config = _config_with_test_store(tmp_path, "configs/local_pilot.yaml")
    source = SundialDataset(
        config, "test", "reported_error_with_source", training=False
    )[0]
    no_source = SundialDataset(
        config, "test", "reported_error_no_source", training=False
    )[0]
    clean = SundialDataset(config, "test", "clean", training=False)[0]
    assert torch.allclose(
        source["flux"] - no_source["flux"],
        clean["flux"],
        atol=5.0e-7,
        rtol=1.0e-5,
    )


def test_reported_error_sweep_uses_paired_draws_across_scales(tmp_path: Path) -> None:
    config = _config_with_test_store(tmp_path, "configs/local_pilot.yaml")
    clean = SundialDataset(config, "test", "clean", training=False)[0]
    half = SundialDataset(
        config,
        "test",
        "reported_error_with_source",
        training=False,
        generated_noise_scale=0.5,
        visit_repeat=0,
    )[0]
    nominal = SundialDataset(
        config,
        "test",
        "reported_error_with_source",
        training=False,
        generated_noise_scale=1.0,
        visit_repeat=0,
    )[0]
    second_draw = SundialDataset(
        config,
        "test",
        "reported_error_with_source",
        training=False,
        generated_noise_scale=1.0,
        visit_repeat=1,
    )[0]

    assert torch.allclose(
        2.0 * (half["flux"] - clean["flux"]),
        nominal["flux"] - clean["flux"],
        atol=5.0e-7,
        rtol=1.0e-5,
    )
    assert not torch.equal(nominal["flux"], second_draw["flux"])


def test_noise_sweep_records_the_realized_input_scale(tmp_path: Path) -> None:
    config = _config_with_test_store(tmp_path, "configs/local_pilot.yaml")
    half = _mean_input_rms(
        config,
        "test",
        "reported_error_no_source",
        0.5,
        indices=[0],
        repeat=0,
    )
    nominal = _mean_input_rms(
        config,
        "test",
        "reported_error_no_source",
        1.0,
        indices=[0],
        repeat=0,
    )

    assert np.isclose(nominal, 2.0 * half, rtol=1.0e-5)


def test_noise_sweep_selects_ia_in_requested_redshift_interval(tmp_path: Path) -> None:
    config = _config_with_test_store(tmp_path, "configs/local_pilot.yaml")
    indices = _balanced_ia_indices(
        config,
        "test",
        "reported_error_with_source",
        [1.0, 2.0],
        1,
    )
    dataset = SundialDataset(
        config, "test", "reported_error_with_source", training=False
    )

    assert indices is not None
    selected = dataset.objects.iloc[indices]
    assert selected["class_name"].tolist() == ["Ia"]
    assert selected["redshift"].between(1.0, 2.0, inclusive="left").all()


def test_noise_amplitude_summary_matches_reported_quantities() -> None:
    predictions = pd.DataFrame(
        {
            "true_class_name": ["Ia", "Ia", "Ia", "other"],
            "true_redshift": [0.2, 0.4, 1.4, 0.5],
            "predicted_redshift": [0.3, 0.2, 1.5, 0.7],
            "p_Ia": [0.95, 0.80, 0.99, 0.1],
        }
    )
    rows = _amplitude_summary(predictions, 1.5, [0.0, 1.0, 2.0])

    assert len(rows) == 2
    assert rows[0]["noise_percent"] == 150.0
    assert rows[0]["n_ia"] == 2
    assert np.isclose(rows[0]["median_delta_z"], -0.05)
    assert np.isclose(rows[0]["scatter_delta_z"], 0.15)
    assert np.isclose(rows[0]["median_abs_delta_z"], 0.15)
    assert rows[0]["fraction_p_ia_ge_0p9"] == 0.5


def test_paired_standard_normal_matches_frozen_v2_contract() -> None:
    rows = pd.DataFrame(
        {
            "mjd": [20.0, 10.0],
            "first_bin": [3, 0],
            "bin_count": [2, 3],
        }
    )
    errors = np.asarray([1.0, 0.0, 2.0, 3.0, 4.0], dtype=np.float32)
    actual = _paired_standard_normal(
        rows,
        {"first_bin": 0, "flux_error": errors},
        noise_key="product:123",
        repeat=1,
        seed=20260803,
    )

    import hashlib

    number = int.from_bytes(
        hashlib.sha256(b"20260803:product:123:1").digest()[:8], "little"
    )
    rng = np.random.default_rng(number)
    expected = np.zeros(5, dtype=np.float32)
    expected[[0, 2]] = rng.normal(size=2).astype(np.float32)
    expected[[3, 4]] = rng.normal(size=2).astype(np.float32)
    np.testing.assert_array_equal(actual, expected)


def test_paired_examples_share_times_and_masks(tmp_path: Path) -> None:
    config = _config_with_test_store(
        tmp_path, "configs/experiments/cadence_controlled.yaml"
    )
    dataset = SundialDataset(config, "train", "generated", training=True)
    source = dataset[0]
    no_source = dataset[1]
    assert source["snid"].item() == no_source["snid"].item()
    assert source["has_source"].item() == 1.0
    assert no_source["has_source"].item() == 0.0
    assert torch.equal(source["observer_days"], no_source["observer_days"])
    assert torch.equal(source["wavelength_mask"], no_source["wavelength_mask"])
    assert not torch.equal(source["flux"], no_source["flux"])


def test_paired_examples_use_the_same_noise_realization(tmp_path: Path) -> None:
    config = _config_with_test_store(
        tmp_path, "configs/experiments/cadence_controlled.yaml"
    )
    dataset = SundialDataset(config, "train", "generated", training=True)
    clean = SundialDataset(
        config, "train", "clean", training=True, pair_no_source=False
    )
    dataset.set_epoch(2)
    clean.set_epoch(2)
    source = dataset[0]
    no_source = dataset[1]
    clean_source = clean[0]
    assert torch.allclose(
        source["flux"] - no_source["flux"],
        clean_source["flux"],
        atol=5.0e-7,
        rtol=1.0e-5,
    )


def test_noise_scale_augmentation_keeps_source_and_blank_paired(tmp_path: Path) -> None:
    config = _config_with_test_store(
        tmp_path, "configs/experiments/cadence_controlled.yaml"
    )
    config["training"]["noise_scale_augmentation_fraction"] = 1.0
    config["training"]["noise_scale_range"] = [0.0, 0.0]
    paired = SundialDataset(config, "train", "generated", training=True)
    clean = SundialDataset(
        config, "train", "clean", training=True, pair_no_source=False
    )

    source = paired[0]
    no_source = paired[1]
    assert torch.equal(source["flux"], clean[0]["flux"])
    assert torch.count_nonzero(no_source["flux"]) == 0
    assert source["evidence_sufficiency_target"].item() > 0.999
    assert no_source["evidence_sufficiency_target"].item() == 0.0


def test_reported_error_training_pair_uses_the_same_noise(tmp_path: Path) -> None:
    config = _config_with_test_store(
        tmp_path, "configs/experiments/cadence_controlled.yaml"
    )
    config["training"]["paired_reported_error_fraction"] = 1.0
    paired = SundialDataset(config, "train", "generated", training=True)
    clean = SundialDataset(
        config, "train", "clean", training=True, pair_no_source=False
    )
    paired.set_epoch(2)
    clean.set_epoch(2)

    source = paired[0]
    no_source = paired[1]
    assert torch.allclose(
        source["flux"] - no_source["flux"],
        clean[0]["flux"],
        atol=5.0e-7,
        rtol=1.0e-5,
    )


def test_reported_error_pair_fraction_must_be_a_probability(tmp_path: Path) -> None:
    config = _config_with_test_store(tmp_path, "configs/local_pilot.yaml")
    config["training"]["paired_reported_error_fraction"] = 1.1
    try:
        SundialDataset(config, "train", "generated", training=True)
    except ValueError as error:
        assert "paired_reported_error_fraction" in str(error)
    else:
        raise AssertionError("Invalid reported-error fraction was accepted")


def test_observed_flux_training_pair_uses_flux_and_residual(tmp_path: Path) -> None:
    config = _config_with_test_store(
        tmp_path, "configs/experiments/cadence_controlled.yaml"
    )
    config["training"]["observed_flux_fraction"] = 1.0
    paired = SundialDataset(config, "train", "generated", training=True)
    original = SundialDataset(
        config, "train", "original", training=True, pair_no_source=False
    )
    clean = SundialDataset(
        config, "train", "clean", training=True, pair_no_source=False
    )
    paired.set_epoch(2)
    original.set_epoch(2)
    clean.set_epoch(2)

    source = paired[0]
    no_source = paired[1]
    assert torch.equal(source["flux"], original[0]["flux"])
    assert torch.allclose(
        source["flux"] - no_source["flux"],
        clean[0]["flux"],
        atol=5.0e-7,
        rtol=1.0e-5,
    )


def test_observed_flux_can_keep_residual_as_an_unseen_control(tmp_path: Path) -> None:
    config = _config_with_test_store(
        tmp_path, "configs/experiments/cadence_controlled.yaml"
    )
    config["training"]["observed_flux_fraction"] = 1.0
    config["training"]["observed_flux_blank_family"] = "controlled_background"
    paired = SundialDataset(config, "train", "generated", training=True)
    original = SundialDataset(
        config, "train", "original", training=True, pair_no_source=False
    )
    residual = SundialDataset(
        config, "train", "residual", training=True, pair_no_source=False
    )
    paired.set_epoch(2)
    original.set_epoch(2)
    residual.set_epoch(2)

    source = paired[0]
    no_source = paired[1]
    assert torch.equal(source["flux"], original[0]["flux"])
    assert not torch.allclose(no_source["flux"], residual[0]["flux"])
    assert no_source["evidence_sufficiency_target"].item() == 0.0

    first_blank = no_source["flux"].clone()
    paired.set_epoch(3)
    assert not torch.equal(first_blank, paired[1]["flux"])


def test_observed_flux_fraction_must_be_a_probability(tmp_path: Path) -> None:
    config = _config_with_test_store(tmp_path, "configs/local_pilot.yaml")
    config["training"]["observed_flux_fraction"] = 1.1
    try:
        SundialDataset(config, "train", "generated", training=True)
    except ValueError as error:
        assert "observed_flux_fraction" in str(error)
    else:
        raise AssertionError("Invalid observed-flux fraction was accepted")


def test_observed_flux_blank_family_must_be_supported(tmp_path: Path) -> None:
    config = _config_with_test_store(tmp_path, "configs/local_pilot.yaml")
    config["training"]["observed_flux_blank_family"] = "unknown"
    try:
        SundialDataset(config, "train", "generated", training=True)
    except ValueError as error:
        assert "observed_flux_blank_family" in str(error)
    else:
        raise AssertionError("Invalid observed-flux blank family was accepted")


def test_paired_examples_receive_the_same_requested_visit_count(tmp_path: Path) -> None:
    config = _config_with_test_store(
        tmp_path, "configs/experiments/cadence_controlled.yaml"
    )
    dataset = SundialDataset(config, "train", "generated", training=True)
    for epoch in range(4):
        dataset.set_epoch(epoch)
        assert dataset.requested_visit_count(0) == dataset.requested_visit_count(1)


def test_paired_noise_family_changes_across_training_epochs(tmp_path: Path) -> None:
    config = _config_with_test_store(
        tmp_path, "configs/experiments/cadence_controlled.yaml"
    )
    dataset = SundialDataset(config, "train", "generated", training=True)
    views = []
    for epoch in range(8):
        dataset.set_epoch(epoch)
        views.append(dataset[1]["flux"])
    assert any(not torch.equal(views[0], value) for value in views[1:])


def test_no_source_redshift_loss_prefers_broad_distribution(tmp_path: Path) -> None:
    config = _config_with_test_store(
        tmp_path, "configs/experiments/cadence_controlled.yaml"
    )
    dataset = SundialDataset(config, "train", "generated", training=True)
    batch = collate_objects([dataset[0], dataset[1]])
    model = Strider(config)
    output = model(measurement_inputs(batch))
    loss, pieces = training_loss(
        output,
        batch,
        model.redshift_grid,
        model.redshift_cell_width,
        model.redshift_prior,
        config["training"]["evidence_sufficiency_loss_weight"],
        config["training"]["no_source_redshift_loss_weight"],
        config["training"]["no_source_class_loss_weight"],
    )
    assert torch.isfinite(loss)
    assert pieces["no_source_redshift_loss"] >= 0.0
    assert pieces["no_source_class_loss"] >= 0.0


def test_alias_ranking_targets_the_strongest_distant_solution() -> None:
    logits = torch.zeros(2, 2, 4, requires_grad=True)
    logits.data[0, 0, 1] = 2.0
    logits.data[0, 1, 3] = 1.0
    logits.data[1, 1, 2] = 0.5
    logits.data[1, 0, 0] = 1.5
    outputs = {
        "joint_logits": logits,
        "evidence_sufficiency_logit": torch.zeros(2, requires_grad=True),
    }
    batch = {
        "has_source": torch.ones(2),
        "class_index": torch.tensor([0, 1]),
        "redshift": torch.tensor([0.1, 0.2]),
        "evidence_sufficiency_target": torch.ones(2),
    }

    loss, pieces = training_loss(
        outputs,
        batch,
        redshift_grid=torch.tensor([0.0, 0.1, 0.2, 0.3]),
        redshift_cell_width=torch.full((4,), 0.1),
        redshift_prior="flat_z",
        evidence_sufficiency_weight=0.0,
        alias_ranking_loss_weight=0.1,
        alias_ranking_minimum_delta_z=0.15,
        alias_ranking_margin=0.25,
    )

    assert torch.isfinite(loss)
    assert pieces["alias_ranking_loss"] > 0.0
    assert pieces["alias_ranking_margin_success_fraction"] == 0.5
    loss.backward()
    assert logits.grad[0, 0, 1] < 0.0
    assert logits.grad[1, 0, 0] > 0.0


def test_no_source_loss_is_finite_when_some_scan_cells_are_unavailable() -> None:
    logits = torch.zeros(2, 2, 4, requires_grad=True)
    support = torch.tensor(
        [
            [[True, True, False, False], [True, True, True, False]],
            [[False, True, True, False], [False, True, False, False]],
        ]
    )
    outputs = {
        "joint_logits": logits.masked_fill(~support, -1.0e4),
        "joint_support": support,
        "evidence_sufficiency_logit": torch.zeros(2, requires_grad=True),
    }
    batch = {
        "has_source": torch.tensor([1.0, 0.0]),
        "class_index": torch.tensor([0, 1]),
        "redshift": torch.tensor([0.1, 0.2]),
        "evidence_sufficiency_target": torch.tensor([1.0, 0.0]),
    }
    loss, pieces = training_loss(
        outputs,
        batch,
        redshift_grid=torch.tensor([0.0, 0.1, 0.2, 0.3]),
        redshift_cell_width=torch.full((4,), 0.1),
        redshift_prior="flat_z",
        evidence_sufficiency_weight=0.5,
        no_source_redshift_weight=0.5,
        no_source_class_weight=0.25,
    )
    assert torch.isfinite(loss)
    assert all(np.isfinite(value) for value in pieces.values())
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_source_without_onir_coverage_is_skipped_by_observation_losses() -> None:
    outputs = {
        "joint_logits": torch.zeros(1, 2, 3),
        "joint_support": torch.tensor([[[True, False, True], [True, True, True]]]),
        "evidence_sufficiency_logit": torch.zeros(1),
    }
    batch = {
        "has_source": torch.ones(1),
        "class_index": torch.zeros(1, dtype=torch.long),
        "redshift": torch.tensor([0.1]),
        "evidence_sufficiency_target": torch.ones(1),
    }
    loss, pieces = training_loss(
        outputs,
        batch,
        redshift_grid=torch.tensor([0.0, 0.1, 0.2]),
        redshift_cell_width=torch.full((3,), 0.1),
        redshift_prior="flat_z",
        evidence_sufficiency_weight=0.5,
        no_source_redshift_weight=0.5,
        no_source_class_weight=0.25,
    )
    assert torch.isfinite(loss)
    assert pieces["joint_loss"] == 0.0
    assert pieces["unsupported_source_fraction"] == 1.0
    assert pieces["evidence_sufficiency_loss"] == 0.0
    assert pieces["no_source_redshift_loss"] == 0.0
    assert pieces["no_source_class_loss"] == 0.0


def test_multi_view_validation_uses_declared_weights() -> None:
    weights = _validation_view_weights(
        {"validation_view_weights": {"generated": 1.0, "clean": 3.0}}
    )
    score = _weighted_validation_score(
        {
            "generated": {"loss": 4.0},
            "clean": {"loss": 2.0},
        },
        weights,
    )
    assert score == 2.5


def test_auxiliary_phase_loss_is_part_of_checkpoint_selection() -> None:
    score = _weighted_validation_score(
        {
            "generated": {"loss": 5.0, "phase_loss": 2.0},
            "clean": {"loss": 3.0, "phase_loss": 1.0},
        },
        {"generated": 1.0, "clean": 1.0},
    )

    assert score == 4.0


def test_weight_decay_excludes_scalar_evidence_scales(tmp_path: Path) -> None:
    config = _config_with_test_store(tmp_path, "configs/local_pilot.yaml")
    model = Strider(config)
    groups = _optimizer_parameter_groups(model, 1.0e-4)
    no_decay = {id(parameter) for parameter in groups[1]["params"]}

    assert groups[0]["weight_decay"] == 1.0e-4
    assert groups[1]["weight_decay"] == 0.0
    for name, parameter in model.named_parameters():
        if parameter.ndim <= 1 or name.endswith(".bias"):
            assert id(parameter) in no_decay


def test_validation_views_must_include_generated() -> None:
    try:
        _validation_view_weights({"validation_view_weights": {"clean": 1.0}})
    except ValueError as error:
        assert "include generated" in str(error)
    else:
        raise AssertionError("missing generated validation view was accepted")


def test_padded_visits_do_not_change_predictions() -> None:
    config = load_config(ROOT / "configs/experiments/spectral_evolution.yaml")
    model = Strider(config).eval()
    wavelength_bins = config["observation"]["wavelength_bins"]
    flux = torch.randn(1, 2, wavelength_bins)
    original = {
        "flux": flux,
        "wavelength_mask": torch.ones_like(flux),
        "visit_mask": torch.ones(1, 2),
        "observer_days": torch.tensor([[0.0, 14.0]]),
    }
    padded = {
        "flux": torch.cat([flux, torch.randn(1, 2, wavelength_bins)], dim=1),
        "wavelength_mask": torch.cat(
            [torch.ones_like(flux), torch.zeros(1, 2, wavelength_bins)], dim=1
        ),
        "visit_mask": torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
        "observer_days": torch.tensor([[0.0, 14.0, -900.0, 4000.0]]),
    }
    first = model(original)
    second = model(padded)
    assert torch.allclose(first["joint_logits"], second["joint_logits"], atol=1e-6)
    assert torch.allclose(
        first["evidence_sufficiency_logit"],
        second["evidence_sufficiency_logit"],
        atol=1e-6,
    )
