from pathlib import Path

from strider.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_experiment_configuration_inherits_base() -> None:
    config = load_config(ROOT / "configs/experiments/no_phase.yaml")
    assert config["model"]["use_phase"] is False
    assert config["model"]["redshift_bins"] == 60
    assert config["data"]["max_objects"]["test"] == 500
    assert config["_project_root"] == str(ROOT)


def test_unknown_config_key_is_rejected_with_a_suggestion(tmp_path: Path) -> None:
    config_path = tmp_path / "misspelled.yaml"
    config_path.write_text("project:\n  naem: typo\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    try:
        load_config(config_path)
    except ValueError as error:
        assert "project.naem" in str(error)
        assert "name" in str(error)
    else:
        raise AssertionError("Misspelled config key was accepted")


def test_nersc_science_arms_change_one_information_route() -> None:
    spectral = load_config(ROOT / "configs/nersc/spectral.yaml")
    background_scaled = load_config(
        ROOT / "configs/nersc/spectral_scaled.yaml"
    )
    temporal = load_config(ROOT / "configs/nersc/temporal.yaml")

    assert spectral["model"]["temporal_mode"] == "none"
    assert spectral["onir"]["input_normalization"] == "per_visit"
    assert background_scaled["model"]["temporal_mode"] == "none"
    assert background_scaled["onir"]["input_normalization"] == "none"
    assert temporal["model"]["temporal_mode"] == "spectral_evolution"
    assert temporal["onir"]["input_normalization"] == "per_visit"
    assert spectral["observation"]["wavelength_max"] == 18175.0
    assert spectral["observation"]["template_support_policy"] == "complete"
    assert spectral["training"]["paired_reported_error_fraction"] == 0.0
    assert spectral["evaluation"]["visit_control_counts"] == [1, 2, 4, 8, 16, 24, 32]


def test_all_visit_configs_remove_the_scientific_cap() -> None:
    for name in (
        "ia_binary_full_all_visits.yaml",
        "grouped_7_full_all_visits.yaml",
        "multiclass_15_full_all_visits.yaml",
    ):
        config = load_config(ROOT / "configs/nersc" / name)
        assert config["data"]["max_visits"] == "all"
        assert config["training"]["batch_by_visit_count"] is True
        assert config["training"]["full_visit_training_fraction"] == 0.5
        assert config["training"]["maximum_visits_per_batch"] == 512
        assert config["training"]["maximum_squared_visits_per_batch"] == 16384
        assert config["evaluation"]["visit_control_counts"][-1] == "all"


def test_whole_spectrum_balance_pilots_bound_long_sequence_memory() -> None:
    for name in (
        "ia_binary_20k_whole_spectrum_minimum_50.yaml",
        "ia_binary_20k_whole_spectrum_minimum_75.yaml",
    ):
        config = load_config(ROOT / "configs/nersc" / name)
        assert config["training"]["batch_size"] == 16
        assert config["training"]["maximum_visits_per_batch"] == 512
        assert config["training"]["maximum_squared_visits_per_batch"] == 16384


def test_retained_template_support_uses_separate_outputs() -> None:
    fixed = load_config(ROOT / "configs/nersc/ia_binary_20k.yaml")
    retained = load_config(
        ROOT / "configs/nersc/ia_binary_20k_retained_support.yaml"
    )

    assert fixed["observation"]["template_support_policy"] == "complete"
    assert retained["observation"]["template_support_policy"] == "retain"
    assert retained["data"]["prepared_dir"] != fixed["data"]["prepared_dir"]
    assert retained["onir"]["bank_path"] != fixed["onir"]["bank_path"]
    assert retained["project"]["output_dir"] != fixed["project"]["output_dir"]


def test_factored_background_arm_changes_only_input_normalization() -> None:
    reference = load_config(ROOT / "configs/nersc/factored_20k_check.yaml")
    background = load_config(ROOT / "configs/nersc/factored_20k_background.yaml")

    assert reference["onir"]["input_normalization"] == "per_visit"
    assert background["onir"]["input_normalization"] == "none"
    assert background["model"] == reference["model"]
    assert background["training"] == reference["training"]
    assert background["data"] == reference["data"]


def test_nersc_development_job_uses_the_small_pipeline_data() -> None:
    pipeline_test = load_config(
        ROOT / "configs/nersc/dev_small.yaml"
    )
    development_test = load_config(
        ROOT / "configs/nersc/dev.yaml"
    )

    # The dependent GPU job must read the data and bank made by the small CPU jobs.
    assert development_test["data"]["prepared_dir"] == pipeline_test["data"]["prepared_dir"]
    assert development_test["data"]["max_objects"] == pipeline_test["data"]["max_objects"]
    assert development_test["onir"]["bank_path"] == pipeline_test["onir"]["bank_path"]


def test_training_source_check_has_no_validation_or_test_input() -> None:
    config = load_config(ROOT / "configs/nersc/training_data_test.yaml")

    assert config["data"]["source_products"] is None
    assert set(config["data"]["split_blocks"]) == {"train"}
    assert config["data"]["max_objects"]["train"] == 5000


def test_five_epoch_run_reuses_the_20k_data_and_bank() -> None:
    full = load_config(ROOT / "configs/nersc/spectral_20k.yaml")
    short = load_config(ROOT / "configs/nersc/spectral_20k_5epoch.yaml")

    assert short["training"]["epochs"] == 5
    assert short["data"] == full["data"]
    assert short["onir"] == full["onir"]


def test_debug_training_limits_runtime_data_and_has_its_own_bank() -> None:
    full = load_config(ROOT / "configs/nersc/spectral_20k.yaml")
    debug = load_config(ROOT / "configs/nersc/spectral_debug.yaml")

    assert debug["training"]["epochs"] == 5
    assert debug["data"]["prepared_dir"] == full["data"]["prepared_dir"]
    assert debug["data"]["runtime_object_limits"]["train"] == 1500
    assert debug["onir"]["bank_path"] != full["onir"]["bank_path"]


def test_longer_debug_run_reuses_the_debug_data_and_bank() -> None:
    short = load_config(ROOT / "configs/nersc/spectral_debug.yaml")
    longer = load_config(ROOT / "configs/nersc/spectral_debug_20epoch.yaml")

    assert longer["training"]["epochs"] == 20
    assert longer["data"] == short["data"]
    assert longer["onir"] == short["onir"]


def test_noise_debug_run_changes_only_the_paired_noise_families() -> None:
    baseline = load_config(ROOT / "configs/nersc/spectral_debug_20epoch.yaml")
    varied = load_config(ROOT / "configs/nersc/spectral_debug_noise.yaml")

    assert varied["data"] == baseline["data"]
    assert varied["onir"] == baseline["onir"]
    assert varied["model"] == baseline["model"]
    assert varied["training"]["paired_reported_error_fraction"] == 0.5


def test_no_schedule_run_changes_only_information_strength_inputs() -> None:
    baseline = load_config(ROOT / "configs/nersc/spectral_debug_noise.yaml")
    varied = load_config(ROOT / "configs/nersc/spectral_debug_no_schedule.yaml")

    assert varied["data"] == baseline["data"]
    assert varied["onir"] == baseline["onir"]
    assert varied["training"] == baseline["training"]
    assert varied["model"]["evidence_use_visit_count_and_span"] is False


def test_one_gpu_jobs_match_the_shared_node_limits() -> None:
    for name in (
        "development_test.sh",
        "train_model.sh",
        "evaluate_model.sh",
        "noise_check.sh",
    ):
        script = (ROOT / "nersc" / name).read_text(encoding="utf-8")
        assert "#SBATCH --gpus=1" in script
        assert "#SBATCH --cpus-per-task=32" in script
        assert "#SBATCH --mem-per-gpu=55G" in script
        assert "#SBATCH --mem=" not in script


def test_moderate_noise_mix_keeps_most_training_at_nominal_noise() -> None:
    baseline = load_config(ROOT / "configs/nersc/ia_binary_20k.yaml")
    varied = load_config(ROOT / "configs/nersc/ia_binary_20k_noise_mix.yaml")

    assert varied["data"] == baseline["data"]
    assert varied["model"] == baseline["model"]
    assert varied["onir"] == baseline["onir"]
    assert varied["training"]["noise_scale_augmentation_fraction"] == 0.10
    assert varied["training"]["noise_scale_range"] == [0.25, 1.25]


def test_20k_runs_use_the_tested_batch_size() -> None:
    standard = load_config(ROOT / "configs/nersc/spectral_20k.yaml")
    large = load_config(ROOT / "configs/nersc/spectral_20k_large.yaml")

    assert standard["training"]["batch_size"] == 16
    assert large["training"]["batch_size"] == 16
    assert 16 in standard["training"]["benchmark_batch_sizes"]


def test_50k_run_uses_all_training_seeds_and_the_accepted_inputs() -> None:
    config = load_config(ROOT / "configs/nersc/spectral_50k.yaml")

    assert config["data"]["max_objects"]["train"] == 50000
    assert config["data"]["training_sample_by_block"] is True
    assert config["data"]["source_products"]["flat_training"]["split_blocks"]["train"] == [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    ]
    assert config["onir"]["bank_path"].endswith("/onir/full_flat.npz")
    assert config["model"]["temporal_mode"] == "none"
    assert config["model"]["evidence_use_visit_count_and_span"] is False
    assert config["training"]["epochs"] == 12
    assert config["training"]["paired_reported_error_fraction"] == 0.5


def test_50k_test_reuses_the_larger_data_and_model_settings() -> None:
    full = load_config(ROOT / "configs/nersc/spectral_50k.yaml")
    test = load_config(ROOT / "configs/nersc/spectral_50k_test.yaml")

    assert test["data"]["prepared_dir"] == full["data"]["prepared_dir"]
    assert test["data"]["max_objects"] == full["data"]["max_objects"]
    assert test["onir"] == full["onir"]
    assert test["model"] == full["model"]
    assert test["training"]["epochs"] == 2
    assert test["data"]["runtime_object_limits"]["train"] == 3000


def test_temporal_20k_updates_only_the_time_branch() -> None:
    temporal = load_config(ROOT / "configs/nersc/temporal_20k.yaml")

    assert temporal["model"]["temporal_mode"] == "spectral_evolution"
    assert temporal["training"]["temporal_only"] is True
    assert temporal["training"]["initial_checkpoint"].endswith(
        "/spectral_20k/best_model.pt"
    )


def test_candidate_phase_uses_the_current_full_binary_baseline() -> None:
    full = load_config(ROOT / "configs/nersc/ia_binary_full.yaml")
    phase = load_config(
        ROOT / "configs/nersc/ia_binary_20k_candidate_phase.yaml"
    )

    assert phase["data"]["prepared_dir"] == full["data"]["prepared_dir"]
    assert phase["onir"]["bank_path"] == full["onir"]["bank_path"]
    assert phase["training"]["initial_checkpoint"].endswith(
        "/ia_binary_full/best_model.pt"
    )
    assert phase["data"]["runtime_object_limits"]["train"] == 20000


def test_main_phase_experiment_has_a_matched_checkpoint_control() -> None:
    control = load_config(
        ROOT / "configs/nersc/ia_binary_20k_phase_control_main.yaml"
    )
    phase = load_config(
        ROOT / "configs/nersc/ia_binary_20k_phase_consistency_main.yaml"
    )

    assert control["training"]["initial_checkpoint"] == phase["training"][
        "initial_checkpoint"
    ]
    assert control["data"] == phase["data"]
    assert control["onir"] == phase["onir"]
    assert control["training"]["learning_rate"] == phase["training"][
        "learning_rate"
    ]
    assert phase["model"]["candidate_phase_consistency"] is True
    assert phase["model"]["phase_auxiliary_bins"] == 31
    assert phase["training"]["phase_loss_weight"] == 0.25


def test_peak_date_phase_experiment_changes_only_the_phase_comparison() -> None:
    reference = load_config(
        ROOT / "configs/nersc/ia_binary_20k_phase_consistency_main.yaml"
    )
    peak_date = load_config(
        ROOT / "configs/nersc/ia_binary_20k_peak_date_phase_main.yaml"
    )

    assert peak_date["data"] == reference["data"]
    assert peak_date["onir"] == reference["onir"]
    assert peak_date["training"] == reference["training"]
    assert peak_date["model"]["candidate_phase_use_peak_date"] is True
    assert peak_date["model"]["candidate_phase_peak_uncertainty_days"] == 15.0
    assert peak_date["model"]["candidate_phase_peak_quadrature_points"] == 5
    assert peak_date["model"]["candidate_phase_peak_outlier_fraction"] == 0.1
    assert peak_date["model"]["candidate_phase_minimum_coverage_fraction"] == 0.5


def test_dense_scan_uses_the_current_full_binary_baseline() -> None:
    full = load_config(ROOT / "configs/nersc/ia_binary_full.yaml")
    dense = load_config(ROOT / "configs/nersc/ia_binary_20k_dense_scan.yaml")

    assert dense["data"]["prepared_dir"] == full["data"]["prepared_dir"]
    assert dense["onir"]["bank_path"] == full["onir"]["bank_path"]
    assert dense["training"]["initial_checkpoint"].endswith(
        "/ia_binary_full/best_model.pt"
    )
    assert dense["model"]["dense_rest_frame_scan"] is True
    assert dense["model"]["dense_scan_initial_scale"] == 0.0
    assert dense["data"]["runtime_object_limits"]["train"] == 20000


def test_dual_dense_scan_changes_only_the_added_detail_route() -> None:
    dense = load_config(ROOT / "configs/nersc/ia_binary_20k_dense_scan.yaml")
    dual = load_config(ROOT / "configs/nersc/ia_binary_20k_dense_dual.yaml")

    assert dual["data"] == dense["data"]
    assert dual["onir"] == dense["onir"]
    assert dual["training"] == dense["training"]
    assert dual["training"]["initial_checkpoint"] == dense["training"][
        "initial_checkpoint"
    ]
    assert dual["model"]["dense_continuum_detail"] is True
    assert dual["model"]["dense_initial_detail_weight"] == 0.5
    assert dual["model"]["dense_continuum_sigma_km_s"] == 12000.0


def test_whole_spectrum_pilots_are_matched_except_for_the_mixture_bound() -> None:
    reference = load_config(
        ROOT / "configs/nersc/ia_binary_20k_phase_control_main.yaml"
    )
    half = load_config(
        ROOT / "configs/nersc/ia_binary_20k_whole_spectrum_minimum_50.yaml"
    )
    primary = load_config(
        ROOT / "configs/nersc/ia_binary_20k_whole_spectrum_minimum_75.yaml"
    )

    for candidate in (half, primary):
        assert candidate["data"] == reference["data"]
        assert candidate["onir"] == reference["onir"]
        assert candidate["training"] == reference["training"]
    assert half["model"]["dense_minimum_whole_weight"] == 0.5
    assert half["model"]["dense_initial_detail_weight"] == 0.25
    assert primary["model"]["dense_minimum_whole_weight"] == 0.75
    assert primary["model"]["dense_initial_detail_weight"] == 0.125


def test_all_visit_whole_spectrum_gate_combines_both_changes() -> None:
    config = load_config(
        ROOT
        / "configs/nersc/ia_binary_20k_all_visits_whole_spectrum_50.yaml"
    )

    assert config["data"]["max_visits"] == "all"
    assert config["data"]["runtime_object_limits"]["train"] == 20_000
    assert config["training"]["batch_by_visit_count"] is True
    assert config["training"]["full_visit_training_fraction"] == 0.5
    assert config["model"]["dense_minimum_whole_weight"] == 0.5
    assert config["model"]["dense_initial_detail_weight"] == 0.25


def test_all_visit_detail_scan_changes_only_the_dense_view() -> None:
    control = load_config(
        ROOT / "configs/nersc/ia_binary_20k_all_visits_scan_control.yaml"
    )
    detail = load_config(
        ROOT / "configs/nersc/ia_binary_20k_all_visits_detail_scan.yaml"
    )

    assert detail["data"] == control["data"]
    assert detail["onir"] == control["onir"]
    assert detail["training"] == control["training"]
    assert detail["evaluation"] == control["evaluation"]
    assert detail["data"]["max_visits"] == "all"
    assert control["model"]["dense_scan_view"] == "blend"
    assert detail["model"]["dense_scan_view"] == "detail"
    assert detail["model"]["dense_continuum_detail"] is True
    assert detail["training"]["batch_by_visit_count"] is True
    control_model = dict(control["model"])
    detail_model = dict(detail["model"])
    control_model.pop("dense_scan_view")
    detail_model.pop("dense_scan_view")
    assert detail_model == control_model


def test_all_visit_dense_scan_seed2_configs_change_only_run_identity() -> None:
    pairs = (
        (
            "ia_binary_20k_all_visits_scan_control.yaml",
            "ia_binary_20k_all_visits_scan_control_seed2.yaml",
        ),
        (
            "ia_binary_20k_all_visits_detail_scan.yaml",
            "ia_binary_20k_all_visits_detail_scan_seed2.yaml",
        ),
    )

    for first_name, second_name in pairs:
        first = load_config(ROOT / "configs/nersc" / first_name)
        second = load_config(ROOT / "configs/nersc" / second_name)

        for section in ("data", "observation", "model", "onir", "training", "evaluation"):
            assert first[section] == second[section]
        assert first["project"]["seed"] == 73031
        assert second["project"]["seed"] == 73032
        assert first["project"]["name"] != second["project"]["name"]
        assert first["project"]["output_dir"] != second["project"]["output_dir"]


def test_coadd_first_pilot_uses_measurement_errors_without_exposing_clean_flux() -> None:
    coadd = load_config(ROOT / "configs/nersc/ia_binary_20k_coadd_only.yaml")
    denoise = load_config(ROOT / "configs/nersc/ia_binary_20k_coadd_denoise.yaml")
    complete = load_config(ROOT / "configs/nersc/ia_binary_20k_coadd_first.yaml")

    assert coadd["model"]["dense_scan_input_mode"] == "inverse_variance_coadd"
    assert coadd["model"]["dense_scan_view"] == "detail"
    assert coadd["model"].get("coadd_reconstruction", False) is False
    assert coadd["model"].get("use_flux_error_channel", False) is False
    assert coadd["data"]["include_flux_error_channel"] is True
    assert coadd["data"].get("include_clean_flux_target", False) is False
    assert denoise["model"]["coadd_reconstruction"] is True
    assert denoise["data"]["include_clean_flux_target"] is True
    assert denoise["training"]["coadd_reconstruction_loss_weight"] == 0.05
    assert complete["model"]["relative_amplitude_mode"] == "object_normalized_flux"
    assert complete["model"]["relative_brightness_evolution"] is True


def test_roman_reference_candidate_is_selection_only_and_measurement_faithful() -> None:
    config = load_config(
        ROOT / "configs/nersc/ia_binary_20k_roman_reference.yaml"
    )

    assert config["model"]["architecture"] == "roman_reference"
    assert config["data"]["include_flux_error_channel"] is True
    assert config["data"].get("include_clean_flux_target", False) is False
    assert config["model"]["use_flux_error_channel"] is False
    assert config["model"]["dense_rest_frame_scan"] is False
    assert config["reference"]["phase_edges_days"] == [
        -20.0,
        -5.0,
        10.0,
        25.0,
        45.0,
        80.0,
    ]
    assert config["reference"]["sequence_visits"] == 6
    assert config["training"]["selection_split"] == "selection"


def test_reference_pilot_preview_is_not_the_final_sundial_evaluation() -> None:
    pilot = load_config(
        ROOT / "configs/nersc/ia_binary_20k_roman_reference.yaml"
    )
    full = load_config(
        ROOT / "configs/nersc/ia_binary_full_roman_reference.yaml"
    )

    assert pilot["data"]["max_objects"]["test"] == 150
    assert full["data"]["max_objects"]["test"] == 0
    assert pilot["data"]["prepared_dir"] != full["data"]["prepared_dir"]


def test_full_attention_candidate_uses_every_flat_redshift_object() -> None:
    pilot = load_config(
        ROOT / "configs/nersc/ia_binary_20k_roman_reference_attention.yaml"
    )
    full = load_config(ROOT / "configs/nersc/strider_attention_full.yaml")

    assert full["model"] == pilot["model"]
    assert full["reference"] == pilot["reference"]
    assert full["data"]["prepared_dir"].endswith("/ia_binary_full")
    assert full["data"]["max_objects"] == {
        "train": 0,
        "selection": 0,
        "calibration": 0,
        "test": 0,
    }
    assert full["data"]["training_sample_by_block"] is False


def test_full_candidate_starts_fresh_without_a_runtime_subset() -> None:
    full = load_config(ROOT / "configs/nersc/ia_binary_full.yaml")
    candidate = load_config(
        ROOT / "configs/nersc/ia_binary_full_from_scratch.yaml"
    )

    assert candidate["data"]["prepared_dir"] == full["data"]["prepared_dir"]
    assert candidate["onir"]["bank_path"] == full["onir"]["bank_path"]
    assert "runtime_object_limits" not in candidate["data"]
    assert candidate["training"]["initial_checkpoint"] is None
    assert candidate["training"]["learning_rate_schedule"] == "cosine"
    assert candidate["training"]["warmup_epochs"] == 5
    assert candidate["model"]["dense_rest_frame_scan"] is True
    assert candidate["model"]["dense_continuum_detail"] is True


def test_spectral_fingerprint_candidate_replaces_the_flat_context_route() -> None:
    config = load_config(
        ROOT / "configs/nersc/ia_binary_20k_spectral_fingerprint.yaml"
    )

    assert config["model"]["full_spectrum_context"] is False
    assert config["model"]["dense_rest_frame_scan"] is True
    assert config["model"]["dense_continuum_detail"] is True
    assert config["model"]["relative_brightness_evolution"] is True
    assert config["training"]["initial_checkpoint"].endswith(
        "/ia_binary_full/best_model.pt"
    )


def test_final_candidate_keeps_context_and_adds_relative_brightness() -> None:
    dual = load_config(ROOT / "configs/nersc/ia_binary_20k_dense_dual.yaml")
    candidate = load_config(ROOT / "configs/nersc/ia_binary_20k_candidate.yaml")

    assert candidate["model"]["full_spectrum_context"] is True
    assert candidate["model"]["dense_rest_frame_scan"] is True
    assert candidate["model"]["dense_continuum_detail"] is True
    assert candidate["model"]["relative_brightness_evolution"] is True
    assert candidate["model"]["relative_brightness_initial_scale"] == 0.0
    assert candidate["data"] == dual["data"]
    assert candidate["onir"] == dual["onir"]
    assert candidate["training"] == dual["training"]


def test_relative_flux_candidate_uses_one_object_normalization() -> None:
    reference = load_config(ROOT / "configs/nersc/ia_binary_20k_dense_dual.yaml")
    candidate = load_config(
        ROOT / "configs/nersc/ia_binary_20k_relative_flux.yaml"
    )

    assert candidate["model"]["full_spectrum_context"] is True
    assert candidate["model"]["dense_rest_frame_scan"] is True
    assert candidate["model"]["relative_brightness_evolution"] is True
    assert candidate["model"]["relative_amplitude_mode"] == "object_normalized_flux"
    assert candidate["model"]["relative_brightness_initial_scale"] == 0.5
    assert candidate["data"] == reference["data"]
    assert candidate["onir"] == reference["onir"]
    assert candidate["training"] == reference["training"]


def test_time_check_uses_the_same_20k_data() -> None:
    spectral = load_config(ROOT / "configs/nersc/spectral_20k.yaml")
    timing = load_config(ROOT / "configs/nersc/time_check_20k.yaml")

    assert timing["data"]["prepared_dir"] == spectral["data"]["prepared_dir"]
    assert timing["data"]["max_objects"] == spectral["data"]["max_objects"]


def test_full_onir_bank_has_separate_input_and_output_paths() -> None:
    full = load_config(ROOT / "configs/nersc/onir_full.yaml")
    bounded = load_config(ROOT / "configs/nersc/spectral_20k.yaml")

    assert full["data"]["prepared_dir"].endswith("/train")
    assert full["onir"]["bank_path"].endswith("full_flat.npz")
    assert full["onir"]["bank_path"] != bounded["onir"]["bank_path"]


def test_factored_comparison_changes_only_the_attention_path() -> None:
    spectral = load_config(ROOT / "configs/nersc/spectral_full_bank.yaml")
    factored = load_config(ROOT / "configs/nersc/factored_20k.yaml")

    assert spectral["onir"]["bank_path"].endswith("/onir/full_flat.npz")
    assert factored["onir"]["bank_path"] == spectral["onir"]["bank_path"]
    assert factored["data"] == spectral["data"]
    assert factored["training"] == spectral["training"]
    assert spectral["model"]["architecture"] == "encoded_onir"
    assert factored["model"]["architecture"] == "factored_onir"
    assert factored["model"]["factored_attention_heads"] == 4
    assert factored["model"]["temporal_initial_scale"] == 0.0


def test_phase_run_adds_an_auxiliary_target_to_the_factored_model() -> None:
    factored = load_config(ROOT / "configs/nersc/factored_20k.yaml")
    phase = load_config(ROOT / "configs/nersc/factored_phase_20k.yaml")

    assert phase["data"] == factored["data"]
    assert phase["onir"] == factored["onir"]
    assert phase["model"]["architecture"] == "factored_onir"
    assert phase["model"]["phase_auxiliary_bins"] == 15
    assert phase["model"]["phase_auxiliary_min_days"] == -20.0
    assert phase["model"]["phase_auxiliary_max_days"] == 50.0
    assert phase["training"]["phase_loss_weight"] == 0.25
    assert phase["evaluation"]["ia_redshift_edges"][4:9] == [
        1.4,
        1.6,
        1.8,
        2.0,
        2.2,
    ]


def test_context_run_changes_only_the_full_spectrum_class_route() -> None:
    factored = load_config(ROOT / "configs/nersc/factored_20k.yaml")
    context = load_config(ROOT / "configs/nersc/factored_context_20k.yaml")

    assert context["data"] == factored["data"]
    assert context["onir"] == factored["onir"]
    assert context["model"]["architecture"] == "factored_onir"
    assert context["model"]["full_spectrum_context"] is True
    assert context["model"]["context_patch_size"] == 8
    assert context["model"]["context_attention_layers"] == 2
    assert context["evaluation"]["ia_redshift_edges"][4:9] == [
        1.4,
        1.6,
        1.8,
        2.0,
        2.2,
    ]


def test_spectrotemporal_run_uses_safe_inputs() -> None:
    config = load_config(ROOT / "configs/nersc/spectrotemporal_20k.yaml")

    assert config["model"]["architecture"] == "factored_onir"
    assert config["model"]["context_visit_attention"] is True
    assert config["model"]["context_token_dim"] == 96
    assert config["model"]["context_input_normalization"] == "per_visit"
    assert config["model"]["context_minimum_support"] == 0.5
    assert config["model"].get("phase_auxiliary_bins", 0) == 0
    assert config["model"]["evidence_use_visit_count_and_span"] is False
    assert config["onir"]["bank_view"] == "clean"
    assert config["onir"]["input_normalization"] == "per_visit"
    assert config["observation"]["generated_noise"]["source_variance_fraction"] == 0.0
    assert config["training"]["paired_reported_error_fraction"] == 0.0


def test_phase_comparison_changes_only_the_auxiliary_target() -> None:
    base = load_config(ROOT / "configs/nersc/spectrotemporal_20k.yaml")
    phase = load_config(ROOT / "configs/nersc/spectrotemporal_phase_20k.yaml")

    assert phase["data"] == base["data"]
    assert phase["observation"] == base["observation"]
    assert phase["onir"] == base["onir"]
    assert phase["model"]["phase_auxiliary_bins"] == 15
    assert phase["training"]["phase_loss_weight"] == 0.25


def test_noise_comparison_changes_only_the_training_noise_mix() -> None:
    base = load_config(ROOT / "configs/nersc/spectrotemporal_20k.yaml")
    varied = load_config(ROOT / "configs/nersc/spectrotemporal_noise_20k.yaml")

    assert varied["data"] == base["data"]
    assert varied["observation"] == base["observation"]
    assert varied["model"] == base["model"]
    assert varied["onir"] == base["onir"]
    assert base["training"]["paired_reported_error_fraction"] == 0.0
    assert varied["training"]["paired_reported_error_fraction"] == 0.5


def test_binary_run_groups_every_contaminant_outside_normal_ia() -> None:
    binary = load_config(ROOT / "configs/nersc/ia_binary_20k.yaml")

    assert binary["data"]["class_scheme"] == "normal_ia_binary"
    assert binary["model"]["classes"] == ["Ia", "other"]
    assert binary["data"]["max_objects"]["train"] == 20000
    assert binary["training"]["paired_reported_error_fraction"] == 0.5
    assert binary["model"]["context_visit_attention"] is True
    assert "ia_binary_20k" in binary["data"]["prepared_dir"]
    assert "ia_binary_20k" in binary["onir"]["bank_path"]


def test_binary_debug_keeps_the_baseline_model_and_data() -> None:
    baseline = load_config(ROOT / "configs/nersc/ia_binary_20k.yaml")
    debug = load_config(ROOT / "configs/nersc/ia_binary_debug.yaml")

    assert debug["model"] == baseline["model"]
    assert debug["onir"] == baseline["onir"]
    assert debug["data"]["prepared_dir"] == baseline["data"]["prepared_dir"]
    assert debug["data"]["runtime_object_limits"] == {
        "train": 3000,
        "selection": 1000,
        "calibration": 500,
        "test": 150,
    }
    assert debug["training"]["epochs"] == 3
    assert debug["training"]["warmup_epochs"] == 1
    assert debug["project"]["output_dir"] != baseline["project"]["output_dir"]


def test_binary_noise_range_changes_only_training_augmentation() -> None:
    baseline = load_config(ROOT / "configs/nersc/ia_binary_20k.yaml")
    augmented = load_config(
        ROOT / "configs/nersc/ia_binary_20k_noise_range.yaml"
    )

    assert augmented["data"] == baseline["data"]
    assert augmented["model"] == baseline["model"]
    assert augmented["onir"] == baseline["onir"]
    assert augmented["training"]["noise_scale_augmentation_fraction"] == 0.25
    assert augmented["training"]["noise_scale_range"] == [0.0, 1.0]


def test_full_binary_run_changes_scale_without_changing_the_model() -> None:
    sample = load_config(ROOT / "configs/nersc/ia_binary_20k.yaml")
    full = load_config(ROOT / "configs/nersc/ia_binary_full.yaml")

    assert full["model"] == sample["model"]
    assert full["observation"] == sample["observation"]
    assert full["data"]["class_scheme"] == "normal_ia_binary"
    assert full["data"]["max_objects"]["train"] == 0
    assert full["training"]["epochs"] == 50


def test_grouped_8_pilot_uses_the_fresh_candidate_architecture() -> None:
    candidate = load_config(
        ROOT / "configs/nersc/ia_binary_full_from_scratch.yaml"
    )
    grouped = load_config(ROOT / "configs/nersc/grouped_8_20k.yaml")

    assert grouped["data"]["class_scheme"] == "grouped_8"
    assert grouped["model"]["classes"] == [
        "Ia",
        "91bg",
        "Iax",
        "H-rich CC",
        "stripped-envelope CC",
        "SLSN",
        "PISN",
        "other",
    ]
    assert grouped["model"]["dense_rest_frame_scan"] is True
    assert grouped["model"]["dense_continuum_detail"] is True
    assert grouped["model"]["relative_brightness_evolution"] is False
    assert grouped["training"]["initial_checkpoint"] is None
    assert grouped["training"]["learning_rate_schedule"] == "cosine"
    assert grouped["data"]["max_objects"]["train"] == 20000
    assert grouped["observation"] == candidate["observation"]


def test_grouped_8_full_uses_the_selected_binary_observation_recipe() -> None:
    binary = load_config(
        ROOT / "configs/nersc/ia_binary_full_flam_anchor_control.yaml"
    )
    grouped = load_config(ROOT / "configs/nersc/grouped_8_full.yaml")

    assert grouped["data"]["class_scheme"] == "grouped_8"
    assert grouped["data"]["prepared_dir"] == binary["data"]["prepared_dir"]
    assert grouped["data"]["max_objects"] == binary["data"]["max_objects"]
    assert grouped["model"]["classes"] == [
        "Ia",
        "91bg",
        "Iax",
        "H-rich CC",
        "stripped-envelope CC",
        "SLSN",
        "PISN",
        "other",
    ]
    for setting in (
        "dense_rest_frame_scan",
        "dense_continuum_detail",
        "relative_brightness_evolution",
    ):
        assert grouped["model"][setting] == binary["model"][setting]
    assert grouped["observation"] == binary["observation"]
    assert grouped["training"] == binary["training"]
    assert grouped["training"]["observed_flux_fraction"] == 0.0
    assert grouped["training"]["paired_reported_error_fraction"] == 0.5
    assert grouped["onir"]["bank_path"].endswith("grouped_8_full.npz")


def test_final_multiclass_runs_change_only_the_output_labels_and_bank() -> None:
    binary = load_config(ROOT / "configs/nersc/ia_binary_full_main.yaml")
    grouped = load_config(ROOT / "configs/nersc/grouped_7_full.yaml")
    detailed = load_config(ROOT / "configs/nersc/multiclass_15_full.yaml")

    expected_grouped_classes = [
        "Ia",
        "91bg",
        "Iax",
        "H-rich CC",
        "stripped-envelope CC",
        "SLSN",
        "other",
    ]
    expected_detailed_classes = [
        "Ia",
        "91bg",
        "Iax",
        "IIP",
        "IIL",
        "IIb",
        "IIn",
        "Ib",
        "Ic",
        "Ic-BL",
        "SLSN",
        "TDE",
        "ILOT",
        "KN",
        "PISN",
    ]

    assert grouped["data"]["class_scheme"] == "grouped_7"
    assert grouped["model"]["classes"] == expected_grouped_classes
    assert detailed["data"]["class_scheme"] == "hourglass_15"
    assert detailed["model"]["classes"] == expected_detailed_classes

    for comparison in (grouped, detailed):
        assert comparison["data"]["prepared_dir"] == binary["data"]["prepared_dir"]
        assert comparison["data"]["max_objects"] == binary["data"]["max_objects"]
        assert comparison["observation"] == binary["observation"]
        assert comparison["training"] == binary["training"]
        assert comparison["evaluation"] == binary["evaluation"]

        comparison_model = dict(comparison["model"])
        binary_model = dict(binary["model"])
        comparison_model.pop("classes")
        binary_model.pop("classes")
        assert comparison_model == binary_model

        comparison_onir = dict(comparison["onir"])
        binary_onir = dict(binary["onir"])
        comparison_onir.pop("bank_path")
        binary_onir.pop("bank_path")
        assert comparison_onir == binary_onir

    assert grouped["onir"]["bank_path"].endswith("grouped_7_full.npz")
    assert detailed["onir"]["bank_path"].endswith(
        "multiclass_15_full.npz"
    )


def test_observed_flam_comparison_changes_only_the_training_mix() -> None:
    baseline = load_config(ROOT / "configs/nersc/ia_binary_20k.yaml")
    control = load_config(ROOT / "configs/nersc/ia_binary_20k_flam_control.yaml")
    mixed = load_config(ROOT / "configs/nersc/ia_binary_20k_flam.yaml")

    assert mixed["data"] == control["data"]
    assert mixed["model"] == control["model"]
    assert mixed["onir"] == control["onir"]
    assert control["onir"]["bank_path"] != baseline["onir"]["bank_path"]
    assert control["training"]["observed_flux_fraction"] == 0.0
    assert mixed["training"]["observed_flux_fraction"] == 1.0 / 3.0
    assert mixed["training"]["validation_view_weights"] == {
        "generated": 2.0,
        "original": 1.0,
        "residual": 1.0,
        "reported_error_with_source": 1.0,
        "reported_error_no_source": 1.0,
    }
    assert mixed["project"]["output_dir"] != control["project"]["output_dir"]


def test_final_candidate_flam_mix_preserves_an_unseen_residual_control() -> None:
    reference = load_config(
        ROOT / "configs/nersc/ia_binary_20k_recipe_reference.yaml"
    )
    mixed = load_config(
        ROOT / "configs/nersc/ia_binary_20k_flam_mix_candidate.yaml"
    )

    assert mixed["data"] == reference["data"]
    assert mixed["model"] == reference["model"]
    assert mixed["onir"] == reference["onir"]
    assert mixed["training"]["observed_flux_fraction"] == 0.20
    assert mixed["training"]["paired_reported_error_fraction"] == 0.50
    assert mixed["training"]["observed_flux_blank_family"] == "controlled_background"
    assert mixed["training"]["noise_scale_augmentation_fraction"] == 1.0
    assert mixed["training"]["noise_scale_range"] == [0.8, 1.2]
    assert mixed["training"]["initial_checkpoint"] is None
    assert mixed["project"]["output_dir"] != reference["project"]["output_dir"]


def test_flam_stress_uses_only_stored_sources_and_keeps_residual_unseen() -> None:
    reference = load_config(
        ROOT / "configs/nersc/ia_binary_20k_recipe_reference.yaml"
    )
    stress = load_config(ROOT / "configs/nersc/ia_binary_20k_flam_stress.yaml")

    assert stress["data"] == reference["data"]
    assert stress["model"] == reference["model"]
    assert stress["onir"] == reference["onir"]
    assert stress["training"]["observed_flux_fraction"] == 1.0
    assert stress["training"]["observed_flux_blank_family"] == "controlled_background"
    assert stress["training"]["paired_reported_error_fraction"] == 0.0
    assert stress["training"]["noise_scale_augmentation_fraction"] == 0.0
    assert stress["training"]["initial_checkpoint"] is None
    assert stress["project"]["output_dir"] != reference["project"]["output_dir"]


def test_flam_anchor_uses_balanced_observation_domains_and_flam_selection() -> None:
    reference = load_config(
        ROOT / "configs/nersc/ia_binary_20k_recipe_reference.yaml"
    )
    anchor = load_config(ROOT / "configs/nersc/ia_binary_20k_flam_anchor.yaml")

    assert anchor["data"] == reference["data"]
    assert anchor["model"] == reference["model"]
    assert anchor["onir"] == reference["onir"]
    assert anchor["observation"] == reference["observation"]
    assert anchor["training"]["observed_flux_fraction"] == 0.50
    assert anchor["training"]["paired_reported_error_fraction"] == 0.50
    assert anchor["training"]["observed_flux_blank_family"] == "controlled_background"
    assert anchor["training"]["noise_scale_augmentation_fraction"] == 1.0
    assert anchor["training"]["noise_scale_range"] == [0.8, 1.2]
    assert anchor["training"]["validation_view_weights"] == {
        "original": 2.0,
        "generated": 1.0,
    }
    assert anchor["training"]["initial_checkpoint"] is None


def test_flam_anchor_replicate_changes_only_seed_and_output_identity() -> None:
    first = load_config(ROOT / "configs/nersc/ia_binary_20k_flam_anchor.yaml")
    second = load_config(
        ROOT / "configs/nersc/ia_binary_20k_flam_anchor_seed2.yaml"
    )

    assert first["data"] == second["data"]
    assert first["observation"] == second["observation"]
    assert first["model"] == second["model"]
    assert first["onir"] == second["onir"]
    assert first["training"] == second["training"]
    assert first["evaluation"] == second["evaluation"]
    assert first["project"]["seed"] == 73031
    assert second["project"]["seed"] == 73032
    assert first["project"]["output_dir"] != second["project"]["output_dir"]


def test_flam_anchor_control_changes_only_the_training_source_fraction() -> None:
    anchor = load_config(ROOT / "configs/nersc/ia_binary_20k_flam_anchor.yaml")
    control = load_config(
        ROOT / "configs/nersc/ia_binary_20k_flam_anchor_control.yaml"
    )

    assert anchor["data"] == control["data"]
    assert anchor["observation"] == control["observation"]
    assert anchor["model"] == control["model"]
    assert anchor["onir"] == control["onir"]
    assert anchor["evaluation"] == control["evaluation"]
    assert anchor["project"]["seed"] == control["project"]["seed"]

    anchor_training = dict(anchor["training"])
    control_training = dict(control["training"])
    assert anchor_training.pop("observed_flux_fraction") == 0.50
    assert control_training.pop("observed_flux_fraction") == 0.0
    assert anchor_training == control_training


def test_full_flam_runs_scale_the_matched_pilot_without_changing_the_model() -> None:
    full = load_config(ROOT / "configs/nersc/ia_binary_full_from_scratch.yaml")
    anchor = load_config(ROOT / "configs/nersc/ia_binary_full_flam_anchor.yaml")
    control = load_config(
        ROOT / "configs/nersc/ia_binary_full_flam_anchor_control.yaml"
    )

    assert anchor["data"] == full["data"]
    assert anchor["observation"] == full["observation"]
    assert anchor["model"] == full["model"]
    assert anchor["onir"] == full["onir"]
    assert anchor["training"]["initial_checkpoint"] is None
    assert anchor["training"]["observed_flux_fraction"] == 0.50
    assert anchor["training"]["paired_reported_error_fraction"] == 0.50
    assert anchor["training"]["observed_flux_blank_family"] == "controlled_background"
    assert anchor["training"]["validation_view_weights"] == {
        "original": 2.0,
        "generated": 1.0,
    }

    assert anchor["data"] == control["data"]
    assert anchor["observation"] == control["observation"]
    assert anchor["model"] == control["model"]
    assert anchor["onir"] == control["onir"]
    assert anchor["evaluation"] == control["evaluation"]
    assert anchor["project"]["seed"] == control["project"]["seed"]

    anchor_training = dict(anchor["training"])
    control_training = dict(control["training"])
    assert anchor_training.pop("observed_flux_fraction") == 0.50
    assert control_training.pop("observed_flux_fraction") == 0.0
    assert anchor_training == control_training


def test_full_flam_only_is_the_matched_stored_observation_endpoint() -> None:
    anchor = load_config(ROOT / "configs/nersc/ia_binary_full_flam_anchor.yaml")
    flam_only = load_config(ROOT / "configs/nersc/ia_binary_full_flam_only.yaml")

    assert anchor["data"] == flam_only["data"]
    assert anchor["observation"] == flam_only["observation"]
    assert anchor["model"] == flam_only["model"]
    assert anchor["onir"] == flam_only["onir"]
    assert anchor["evaluation"] == flam_only["evaluation"]
    assert anchor["project"]["seed"] == flam_only["project"]["seed"]

    anchor_training = dict(anchor["training"])
    flam_training = dict(flam_only["training"])
    assert anchor_training.pop("observed_flux_fraction") == 0.50
    assert flam_training.pop("observed_flux_fraction") == 1.0
    assert anchor_training == flam_training


def test_local_binary_20k_uses_the_same_science_model() -> None:
    nersc = load_config(ROOT / "configs/nersc/ia_binary_20k.yaml")
    local = load_config(ROOT / "configs/experiments/local_ia_binary_20k.yaml")

    assert local["model"] == nersc["model"]
    assert local["observation"] == nersc["observation"]
    assert local["data"]["class_scheme"] == "normal_ia_binary"
    assert local["data"]["max_objects"] == nersc["data"]["max_objects"]
    assert local["training"]["epochs"] == 10
    assert local["training"]["num_workers"] == 0


def test_local_factored_check_uses_the_small_prepared_sample() -> None:
    config = load_config(ROOT / "configs/experiments/local_factored_check.yaml")

    assert config["model"]["architecture"] == "factored_onir"
    assert config["data"]["runtime_object_limits"]["train"] == 300
    assert config["observation"]["wavelength_bins"] == 256
    assert config["model"]["redshift_bins"] == 80
