from pathlib import Path

import numpy as np
import torch

from strider.atlas.roman_reference import RomanReferenceBank
from strider.config import load_config
from strider.data.classes import HOURGLASS_15_CLASSES
from strider.model.strider import Strider
from strider.training.visit_batches import visit_limited_batch_size


ROOT = Path(__file__).resolve().parents[1]


def test_binary_speed_benchmark_keeps_complete_histories_in_larger_batches() -> None:
    config = load_config(ROOT / "configs/nersc/binary_speed.yaml")

    assert config["data"]["prepared_dir"].endswith("/ia_binary_full")
    assert config["data"]["max_objects"] == {
        "train": 0,
        "selection": 0,
        "calibration": 0,
        "test": 0,
    }
    assert config["training"]["maximum_visits_per_batch"] is None
    assert config["training"]["maximum_squared_visits_per_batch"] is None
    assert config["training"]["benchmark_batch_sizes"] == [2, 4, 8, 16]
    assert visit_limited_batch_size(16, 145, None, None) == 16


def test_wide_speed_benchmark_changes_only_chunking_and_output() -> None:
    standard = load_config(ROOT / "configs/nersc/binary_speed.yaml")
    wide = load_config(ROOT / "configs/nersc/binary_speed_wide.yaml")

    assert standard["reference"]["redshift_chunk_size"] == 12
    assert wide["reference"]["redshift_chunk_size"] == 24
    assert wide["training"] == standard["training"]
    assert wide["data"] == standard["data"]
    assert wide["model"] == standard["model"]
    assert wide["project"]["output_dir"] != standard["project"]["output_dir"]


def test_binary_test_changes_only_runtime_and_batching() -> None:
    reference = load_config(
        ROOT / "configs/nersc/ia_binary_20k_roman_reference_attention.yaml"
    )
    test = load_config(ROOT / "configs/nersc/binary_test.yaml")

    assert test["data"] == reference["data"]
    assert test["model"] == reference["model"]
    assert test["reference"]["bank_path"] == reference["reference"]["bank_path"]
    assert test["reference"]["redshift_chunk_size"] == 24
    assert test["training"]["epochs"] == 2
    assert test["training"]["batch_size"] == 16
    assert test["training"]["maximum_visits_per_batch"] is None
    assert test["training"]["maximum_squared_visits_per_batch"] is None
    assert test["project"]["output_dir"].endswith("/binary_test")


def test_binary_full_run_uses_all_flat_redshift_objects() -> None:
    benchmark = load_config(ROOT / "configs/nersc/binary_speed_wide.yaml")
    binary = load_config(ROOT / "configs/nersc/binary.yaml")

    assert binary["data"] == benchmark["data"]
    assert binary["model"] == benchmark["model"]
    assert binary["reference"] == benchmark["reference"]
    assert binary["training"]["batch_size"] == 16
    assert binary["training"]["maximum_visits_per_batch"] is None
    assert binary["training"]["maximum_squared_visits_per_batch"] is None
    assert binary["project"]["output_dir"].endswith("/binary")


def test_classes_test_keeps_all_reference_classes_separate() -> None:
    binary = load_config(ROOT / "configs/nersc/binary_test.yaml")
    classes = load_config(ROOT / "configs/nersc/classes_test.yaml")

    assert classes["project"]["name"] == "classes_test"
    assert classes["project"]["output_dir"].endswith("/classes_test")
    assert classes["data"]["prepared_dir"].endswith("/spectral_20k")
    assert classes["data"]["class_scheme"] == "hourglass_15"
    assert classes["model"]["classes"] == [
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
    assert classes["reference"] == binary["reference"]
    assert classes["training"] == binary["training"]


def test_temporal_test_changes_only_the_sequence_combination() -> None:
    classes = load_config(ROOT / "configs/nersc/classes_test.yaml")
    temporal = load_config(ROOT / "configs/nersc/temporal_test.yaml")

    assert temporal["project"]["name"] == "temporal_test"
    assert temporal["project"]["output_dir"].endswith("/temporal_test")
    assert temporal["data"] == classes["data"]
    assert temporal["model"] == classes["model"]
    assert temporal["training"] == classes["training"]

    changed = {
        key
        for key in temporal["reference"] | classes["reference"]
        if temporal["reference"].get(key) != classes["reference"].get(key)
    }
    assert changed == {
        "sequence_combination",
        "temporal_hidden_dim",
        "temporal_attention_heads",
        "temporal_layers",
        "temporal_feedforward_multiplier",
        "temporal_initial_correction_scale",
    }
    assert temporal["reference"]["sequence_combination"] == (
        "temporal_transformer"
    )
    assert temporal["reference"]["redshift_chunk_size"] == 24
    assert temporal["reference"]["sequence_visits"] == 6
    assert temporal["training"]["batch_size"] == 16


def test_brightness_test_adds_only_relative_flux_evolution() -> None:
    temporal = load_config(ROOT / "configs/nersc/temporal_test.yaml")
    brightness = load_config(ROOT / "configs/nersc/brightness_test.yaml")

    assert brightness["project"]["name"] == "brightness_test"
    assert brightness["project"]["output_dir"].endswith("/brightness_test")
    assert brightness["data"] == temporal["data"]
    assert brightness["model"] == temporal["model"]
    assert brightness["training"] == temporal["training"]

    changed = {
        key
        for key in brightness["reference"] | temporal["reference"]
        if brightness["reference"].get(key) != temporal["reference"].get(key)
    }
    assert changed == {"relative_flux_evolution"}
    assert brightness["reference"]["relative_flux_evolution"] is True


def test_no_snr_control_changes_only_the_temporal_quality_input() -> None:
    brightness = load_config(ROOT / "configs/nersc/brightness_test.yaml")
    control = load_config(ROOT / "configs/nersc/brightness_no_snr_test.yaml")

    assert control["project"]["name"] == "brightness_no_snr_test"
    assert control["project"]["output_dir"].endswith(
        "/brightness_no_snr_test"
    )
    assert control["data"] == brightness["data"]
    assert control["observation"] == brightness["observation"]
    assert control["model"] == brightness["model"]
    assert control["training"] == brightness["training"]
    assert control["evaluation"] == brightness["evaluation"]

    changed = {
        key
        for key in control["reference"] | brightness["reference"]
        if control["reference"].get(key) != brightness["reference"].get(key)
    }
    assert changed == {"temporal_use_signal_to_noise"}
    assert control["reference"]["temporal_use_signal_to_noise"] is False


def test_full_brightness_runs_use_all_flat_redshift_objects() -> None:
    binary = load_config(ROOT / "configs/nersc/binary_brightness.yaml")
    classes = load_config(ROOT / "configs/nersc/classes_brightness.yaml")

    for config in (binary, classes):
        assert config["data"]["prepared_dir"].endswith("/ia_binary_full")
        assert config["data"]["max_objects"] == {
            "train": 0,
            "selection": 0,
            "calibration": 0,
            "test": 0,
        }
        assert config["data"]["training_sample_by_block"] is False
        assert config["training"]["batch_size"] == 16
        assert config["training"]["maximum_visits_per_batch"] is None
        assert config["training"]["maximum_squared_visits_per_batch"] is None
        assert config["reference"]["redshift_chunk_size"] == 24
        assert config["reference"]["sequence_combination"] == (
            "temporal_transformer"
        )
        assert config["reference"]["relative_flux_evolution"] is True

    assert binary["project"]["output_dir"].endswith("/binary_brightness")
    assert binary["data"]["class_scheme"] == "normal_ia_binary"
    assert binary["model"]["classes"] == ["Ia", "other"]
    assert classes["project"]["output_dir"].endswith("/classes_brightness")
    assert classes["data"]["class_scheme"] == "hourglass_15"
    assert classes["model"]["classes"] == list(HOURGLASS_15_CLASSES)


def test_brightness_model_is_invariant_to_one_object_flux_gain(
    tmp_path: Path,
) -> None:
    config = _reference_test_config(tmp_path, redshift_chunk_size=5)
    config["reference"].update(
        {
            "sequence_combination": "temporal_transformer",
            "temporal_hidden_dim": 16,
            "temporal_attention_heads": 2,
            "temporal_layers": 1,
            "temporal_feedforward_multiplier": 2,
            "temporal_initial_correction_scale": 0.10,
            "relative_flux_evolution": True,
        }
    )
    torch.manual_seed(1_103)
    model = Strider(config).eval()
    batch = _measurement_batch()
    brighter = dict(batch)
    brighter["flux"] = 17.0 * batch["flux"]
    brighter["flux_error_shape"] = batch["flux_error_shape"] + np.log(17.0)

    with torch.no_grad():
        reference = model(batch)
        scaled = model(brighter)

    assert torch.equal(reference["joint_support"], scaled["joint_support"])
    assert torch.allclose(
        reference["joint_logits"],
        scaled["joint_logits"],
        atol=2.0e-5,
        rtol=2.0e-5,
    )


def test_redshift_chunk_width_preserves_reference_outputs_and_gradients(
    tmp_path: Path,
) -> None:
    narrow_config = _reference_test_config(tmp_path, redshift_chunk_size=3)
    wide_config = _reference_test_config(tmp_path, redshift_chunk_size=7)
    torch.manual_seed(813)
    narrow = Strider(narrow_config).train()
    wide = Strider(wide_config).train()
    wide.load_state_dict(narrow.state_dict())
    batch = _measurement_batch()

    narrow_output = narrow(batch)
    wide_output = wide(batch)
    assert torch.equal(
        narrow_output["joint_support"],
        wide_output["joint_support"],
    )
    assert torch.allclose(
        narrow_output["joint_logits"],
        wide_output["joint_logits"],
        atol=2.0e-5,
        rtol=2.0e-5,
    )

    narrow_loss = narrow_output["joint_logits"][
        narrow_output["joint_support"]
    ].square().mean()
    wide_loss = wide_output["joint_logits"][
        wide_output["joint_support"]
    ].square().mean()
    narrow_loss.backward()
    wide_loss.backward()
    for (narrow_name, narrow_parameter), (wide_name, wide_parameter) in zip(
        narrow.named_parameters(),
        wide.named_parameters(),
        strict=True,
    ):
        assert narrow_name == wide_name
        if narrow_parameter.grad is None or wide_parameter.grad is None:
            assert narrow_parameter.grad is None
            assert wide_parameter.grad is None
            continue
        assert torch.allclose(
            narrow_parameter.grad,
            wide_parameter.grad,
            atol=2.0e-5,
            rtol=2.0e-5,
        ), narrow_name


def _reference_test_config(
    tmp_path: Path,
    *,
    redshift_chunk_size: int,
) -> dict:
    rest_bins = 32
    coordinate = np.linspace(0.0, 4.0 * np.pi, rest_bins, dtype=np.float32)
    coadd = np.asarray(
        [
            [
                np.sin(coordinate + 0.13 * class_index + 0.05 * profile)
                for profile in range(2)
            ]
            for class_index in range(len(HOURGLASS_15_CLASSES))
        ],
        dtype=np.float32,
    )
    phase = np.stack(
        [coadd + 0.03 * phase_index for phase_index in range(3)],
        axis=1,
    )
    bank_path = RomanReferenceBank(
        class_names=HOURGLASS_15_CLASSES,
        rest_wavelength=np.geomspace(2500.0, 10000.0, rest_bins).astype(
            np.float32
        ),
        phase_edges_days=np.asarray([-20.0, 0.0, 25.0, 80.0], dtype=np.float32),
        coadd_full_profiles=coadd,
        coadd_continuum_removed_profiles=(
            coadd - coadd.mean(axis=-1, keepdims=True)
        ),
        coadd_profile_masks=np.ones_like(coadd, dtype=bool),
        coadd_support_counts=np.full(coadd.shape[:-1], 20, dtype=np.int64),
        phase_full_profiles=phase,
        phase_continuum_removed_profiles=(
            phase - phase.mean(axis=-1, keepdims=True)
        ),
        phase_profile_masks=np.ones_like(phase, dtype=bool),
        phase_support_counts=np.full(phase.shape[:-1], 20, dtype=np.int64),
        metadata={"source_split": "train", "truth_used_at_runtime": False},
    ).save(tmp_path / "reference.npz")
    config = load_config(ROOT / "configs/experiments/encoded_onir_named_clean.yaml")
    config["data"].update(
        {
            "class_scheme": "normal_ia_binary",
            "include_flux_error_channel": True,
        }
    )
    config["model"].update(
        {
            "architecture": "roman_reference",
            "classes": ["Ia", "other"],
            "dropout": 0.0,
            "dense_rest_frame_scan": False,
            "dense_continuum_detail": False,
            "full_spectrum_context": False,
            "temporal_mode": "none",
            "phase_auxiliary_bins": 0,
            "candidate_phase_consistency": False,
            "use_flux_error_channel": False,
            "coadd_maximum_relative_error": 3.0,
            "coadd_edge_trim_fraction": 0.05,
            "redshift_bins": 13,
        }
    )
    config["reference"] = {
        "bank_path": str(bank_path),
        "continuum_width_km_s": 12_000.0,
        "minimum_profile_support": 5,
        "minimum_rest_fraction": 0.10,
        "minimum_shared_fraction": 0.50,
        "prototype_temperature": 0.08,
        "fine_class_temperature": 0.10,
        "phase_temperature": 0.10,
        "initial_continuum_removed_fraction": 0.60,
        "initial_coadd_scale": 0.75,
        "initial_sequence_scale": 0.20,
        "evidence_scale": 10.0,
        "redshift_chunk_size": redshift_chunk_size,
        "sequence_visits": 3,
        "minimum_sequence_visits": 2,
        "spectral_encoder": "multiscale_attention",
        "token_dim": 8,
        "token_pool_size": 2,
        "minimum_encoder_support": 0.5,
        "attention_heads": 2,
        "sequence_combination": "continuous_time_attention",
        "time_attention_hidden_dim": 8,
    }
    return config


def _measurement_batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(191)
    flux = torch.randn(2, 4, 256, generator=generator)
    error = 0.5 + torch.rand(2, 4, 256, generator=generator)
    return {
        "flux": flux,
        "wavelength_mask": torch.ones_like(flux),
        "visit_mask": torch.ones(2, 4),
        "observer_days": torch.tensor(
            [[0.0, 8.0, 20.0, 42.0], [0.0, 11.0, 19.0, 35.0]]
        ),
        "flux_error_shape": torch.log(error),
        "visit_flux_scale": torch.tensor(
            [[2.0, 3.0, 4.0, 5.0], [1.5, 2.5, 3.5, 4.5]]
        ),
    }
