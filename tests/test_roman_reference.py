from pathlib import Path
import json

import numpy as np
import pytest
import torch

from scripts.plot_roman_reference_bank import make_gallery
from strider.atlas.roman_reference import (
    RomanReferenceBank,
    _normalise_rows,
    _paired_spherical_clusters,
    _redshift_bin_index,
)
from strider.config import load_config, resolved_config
from strider.data.classes import HOURGLASS_15_CLASSES
from strider.model.coadd import relative_inverse_variance
from strider.model.roman_reference import (
    CandidateTemporalTransformer,
    _common_support_correlation,
    _common_support_token_cosine,
    _scale_invariant_normalize,
)
from strider.model.spectral_tokens import MaskAwareMultiscaleAttentionEncoder
from strider.model.strider import Strider


ROOT = Path(__file__).resolve().parents[1]


def _test_bank(
    path: Path,
    rest_bins: int = 64,
    flux_scale: float = 1.0,
) -> Path:
    generator = np.random.default_rng(413)
    classes = len(HOURGLASS_15_CLASSES)
    coadd_shape = (classes, 2, rest_bins)
    phase_shape = (classes, 3, 2, rest_bins)
    rest = np.geomspace(2500.0, 10000.0, rest_bins).astype(np.float32)
    coordinate = np.linspace(0.0, 6.0 * np.pi, rest_bins, dtype=np.float32)
    coadd = np.empty(coadd_shape, dtype=np.float32)
    phase = np.empty(phase_shape, dtype=np.float32)
    for class_index in range(classes):
        for prototype in range(2):
            coadd[class_index, prototype] = np.sin(
                coordinate + 0.19 * class_index + 0.07 * prototype
            ) + 0.05 * generator.normal(size=rest_bins)
        for phase_index in range(3):
            for prototype in range(2):
                phase[class_index, phase_index, prototype] = np.sin(
                    coordinate
                    + 0.19 * class_index
                    + 0.11 * phase_index
                    + 0.07 * prototype
                ) + 0.05 * generator.normal(size=rest_bins)
    coadd *= flux_scale
    phase *= flux_scale
    bank = RomanReferenceBank(
        class_names=HOURGLASS_15_CLASSES,
        rest_wavelength=rest,
        phase_edges_days=np.asarray([-20.0, 0.0, 25.0, 80.0], dtype=np.float32),
        coadd_full_profiles=coadd,
        coadd_continuum_removed_profiles=(coadd - coadd.mean(axis=-1, keepdims=True)),
        coadd_profile_masks=np.ones(coadd_shape, dtype=bool),
        coadd_support_counts=np.full(coadd_shape[:-1], 20, dtype=np.int64),
        phase_full_profiles=phase,
        phase_continuum_removed_profiles=(phase - phase.mean(axis=-1, keepdims=True)),
        phase_profile_masks=np.ones(phase_shape, dtype=bool),
        phase_support_counts=np.full(phase_shape[:-1], 20, dtype=np.int64),
        metadata={"source_split": "train", "truth_used_at_runtime": False},
    )
    return bank.save(path)


def test_roman_reference_bank_round_trip(tmp_path: Path) -> None:
    path = _test_bank(tmp_path / "reference.npz")
    bank = RomanReferenceBank.load(path)

    assert bank.class_names == HOURGLASS_15_CLASSES
    assert bank.coadd_full_profiles.shape == (15, 2, 64)
    assert bank.phase_full_profiles.shape == (15, 3, 2, 64)
    assert bank.metadata["source_split"] == "train"
    assert bank.metadata["truth_used_at_runtime"] is False


def test_roman_reference_pilot_configs_are_matched() -> None:
    direct = load_config(ROOT / "configs/nersc/ia_binary_20k_roman_reference.yaml")
    learned = load_config(
        ROOT / "configs/nersc/ia_binary_20k_roman_reference_learned.yaml"
    )

    assert direct["reference"]["bank_path"] == learned["reference"]["bank_path"]
    assert direct["data"] == learned["data"]
    assert direct["training"] == learned["training"]
    assert direct["reference"]["spectral_encoder"] == "direct"
    assert direct["reference"]["sequence_combination"] == "mean"
    assert learned["reference"]["spectral_encoder"] == "shared_cnn"
    assert learned["reference"]["sequence_combination"] == "continuous_time_attention"
    assert "train_profiles" not in direct["reference"]
    assert "drift_loss_weight" not in direct["reference"]
    assert direct["project"]["output_dir"] != learned["project"]["output_dir"]


def test_attention_reference_config_is_a_matched_third_candidate() -> None:
    direct = load_config(ROOT / "configs/nersc/ia_binary_20k_roman_reference.yaml")
    attention = load_config(
        ROOT / "configs/nersc/ia_binary_20k_roman_reference_attention.yaml"
    )

    assert direct["data"] == attention["data"]
    assert direct["training"] == attention["training"]
    assert direct["reference"]["bank_path"] == attention["reference"]["bank_path"]
    assert attention["reference"]["spectral_encoder"] == "multiscale_attention"
    assert attention["reference"]["token_dim"] == 32
    assert attention["reference"]["attention_heads"] == 4
    assert attention["reference"]["sequence_combination"] == (
        "continuous_time_attention"
    )


def test_temporal_transformer_is_matched_to_classes_test() -> None:
    classes = load_config(ROOT / "configs/nersc/classes_test.yaml")
    temporal = load_config(ROOT / "configs/nersc/temporal_test.yaml")

    assert temporal["data"] == classes["data"]
    assert temporal["model"] == classes["model"]
    assert temporal["training"] == classes["training"]
    assert temporal["reference"]["bank_path"] == classes["reference"]["bank_path"]
    assert temporal["reference"]["spectral_encoder"] == "multiscale_attention"
    assert temporal["reference"]["sequence_combination"] == "temporal_transformer"
    assert temporal["reference"]["temporal_hidden_dim"] == 32
    assert temporal["reference"]["temporal_attention_heads"] == 4
    assert temporal["reference"]["temporal_layers"] == 1


def test_edge_test_changes_only_measurement_edges_and_epoch_count() -> None:
    brightness = load_config(ROOT / "configs/nersc/brightness_test.yaml")
    edge = load_config(ROOT / "configs/nersc/edge_test.yaml")
    builder = load_config(ROOT / "configs/nersc/edge_reference.yaml")

    assert edge["data"] == brightness["data"]
    assert edge["observation"] == brightness["observation"]
    assert edge["model"]["classes"] == brightness["model"]["classes"]
    assert edge["reference"]["spectral_encoder"] == brightness["reference"][
        "spectral_encoder"
    ]
    assert edge["reference"]["sequence_combination"] == brightness[
        "reference"
    ]["sequence_combination"]
    assert edge["reference"]["relative_flux_evolution"] is True
    assert edge["training"]["epochs"] == 4
    assert edge["model"]["coadd_maximum_relative_error"] == 3.0
    assert edge["model"]["coadd_edge_trim_fraction"] == 0.0
    assert edge["reference"]["maximum_relative_coadd_error"] == 3.0
    assert edge["reference"]["edge_trim_fraction"] == 0.0
    assert edge["reference"]["edge_taper_fraction"] == 0.05
    assert edge["reference"]["bank_path"] == builder["reference"]["bank_path"]
    assert builder["data"]["prepared_dir"].endswith("/ia_binary_full")
    assert builder["reference"]["maximum_training_objects_per_class"] == 2000


def test_uncertainty_test_replaces_broad_quality_cut_with_numerical_floor() -> None:
    edge = load_config(ROOT / "configs/nersc/edge_test.yaml")
    uncertainty = load_config(ROOT / "configs/nersc/uncertainty_test.yaml")
    builder = load_config(ROOT / "configs/nersc/uncertainty_reference.yaml")

    assert uncertainty["data"] == edge["data"]
    assert uncertainty["observation"] == edge["observation"]
    assert uncertainty["training"] == edge["training"]
    assert uncertainty["model"]["classes"] == edge["model"]["classes"]
    assert uncertainty["model"]["coadd_edge_trim_fraction"] == 0.0
    assert uncertainty["model"]["coadd_maximum_relative_error"] is None
    assert uncertainty["reference"]["maximum_relative_coadd_error"] is None
    assert (
        uncertainty["reference"]["spectral_uncertainty_weighting"]
        == "inverse_variance"
    )
    assert np.isclose(
        uncertainty["reference"]["minimum_relative_spectral_precision"],
        torch.finfo(torch.float32).eps,
    )
    assert uncertainty["reference"]["bank_path"] == builder["reference"][
        "bank_path"
    ]
    assert builder["model"]["coadd_maximum_relative_error"] is None
    assert builder["reference"]["maximum_relative_coadd_error"] is None
    assert np.isclose(
        builder["reference"]["minimum_relative_spectral_precision"],
        torch.finfo(torch.float32).eps,
    )
    assert builder["reference"]["edge_taper_fraction"] == 0.05


def test_v4_test_combines_only_the_selected_changes() -> None:
    uncertainty = load_config(ROOT / "configs/nersc/uncertainty_8_test.yaml")
    candidate = load_config(ROOT / "configs/nersc/v4_test.yaml")

    assert candidate["data"] == uncertainty["data"]
    assert candidate["observation"] == uncertainty["observation"]
    assert candidate["model"] == uncertainty["model"]
    assert candidate["training"] == {
        **uncertainty["training"],
        "epochs": 2,
    }
    expected_reference = {
        **uncertainty["reference"],
        "temporal_use_signal_to_noise": False,
    }
    assert candidate["reference"] == expected_reference
    assert candidate["reference"]["sequence_visits"] == 8
    assert candidate["reference"]["relative_flux_evolution"] is True
    assert (
        candidate["reference"]["spectral_uncertainty_weighting"]
        == "inverse_variance"
    )
    assert np.isclose(
        candidate["reference"]["minimum_relative_spectral_precision"],
        torch.finfo(torch.float32).eps,
    )
    assert candidate["reference"]["edge_trim_fraction"] == 0.0
    assert candidate["reference"]["edge_taper_fraction"] == 0.05
    assert candidate["reference"]["temporal_use_signal_to_noise"] is False


def test_reference_candidate_alias_preserves_the_active_checkpoint_digest() -> None:
    historical = load_config(ROOT / "configs/nersc/v4_test.yaml")
    public_name = load_config(ROOT / "configs/nersc/reference_candidate_gate.yaml")

    assert resolved_config(public_name) == resolved_config(historical)


def test_full_reference_builder_uses_complete_training_role() -> None:
    builder = load_config(ROOT / "configs/nersc/roman_reference_bank_full.yaml")
    direct = load_config(ROOT / "configs/nersc/ia_binary_20k_roman_reference.yaml")

    assert builder["data"]["prepared_dir"].endswith("/ia_binary_full")
    assert builder["reference"]["bank_path"] == direct["reference"]["bank_path"]
    assert builder["reference"]["maximum_training_objects_per_class"] == 2000


def test_all_source_reference_builder_has_a_separate_output() -> None:
    capped = load_config(ROOT / "configs/nersc/roman_reference_bank_full.yaml")
    all_source = load_config(
        ROOT / "configs/nersc/roman_reference_bank_all_source.yaml"
    )

    assert all_source["data"]["prepared_dir"].endswith("/ia_binary_full")
    assert all_source["reference"]["maximum_training_objects_per_class"] == 0
    assert all_source["reference"]["bank_path"] != capped["reference"]["bank_path"]
    assert all_source["reference"]["coadd_profiles_per_class"] == 6
    assert all_source["reference"]["phase_profiles_per_cell"] == 3


def test_reference_redshift_bins_include_the_upper_edge() -> None:
    edges = np.asarray([0.0, 0.5, 1.0, 1.5])

    assert _redshift_bin_index(0.0, edges) == 0
    assert _redshift_bin_index(0.5, edges) == 1
    assert _redshift_bin_index(1.5, edges) == 2
    assert _redshift_bin_index(-0.1, edges) is None
    assert _redshift_bin_index(1.6, edges) is None
    assert _redshift_bin_index(float("nan"), edges) is None


def test_reference_gallery_writes_simple_inspection_figures(tmp_path: Path) -> None:
    bank_path = _test_bank(tmp_path / "reference.npz", rest_bins=32)
    audit_path = bank_path.with_suffix(".audit.json")
    counts = {
        name: [5 + index, 7 + index, 6 + index]
        for index, name in enumerate(HOURGLASS_15_CLASSES)
    }
    audit_path.write_text(
        json.dumps(
            {
                "redshift_edges": [0.0, 0.5, 1.0, 1.5],
                "training_objects_used_by_fine_class_and_redshift": counts,
            }
        ),
        encoding="utf-8",
    )

    result = make_gallery(bank_path, tmp_path / "gallery")

    assert set(result["files"]) == {
        "coadded_profiles",
        "phase_profiles",
        "redshift_coverage",
    }
    for paths in result["files"].values():
        assert {Path(path).suffix for path in paths} == {".png", ".pdf"}
        assert all(Path(path).is_file() for path in paths)
    assert Path(result["summary"]).is_file()


def test_reference_clustering_merges_weak_profiles() -> None:
    coordinate = np.linspace(0.0, 2.0 * np.pi, 32, dtype=np.float32)
    full = np.stack([np.sin(coordinate + 0.03 * index) for index in range(14)])
    removed = full - full.mean(axis=1, keepdims=True)
    masks = np.ones_like(full, dtype=bool)

    _, _, _, support = _paired_spherical_clusters(
        full,
        removed,
        masks,
        prototype_count=6,
        minimum_bin_fraction=0.5,
        minimum_profile_support=5,
    )

    assert support.sum() == 14
    assert np.all(support[support > 0] >= 5)


def test_token_similarity_uses_only_shared_support() -> None:
    candidates = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [3.0, 3.0]],
            [[1.0, 1.0], [2.0, 0.0], [0.0, 0.0]],
        ]
    )
    candidate_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    profiles = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]],
            [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        ]
    )
    profile_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])

    similarity, support = _common_support_token_cosine(
        candidates,
        candidate_mask,
        profiles,
        profile_mask,
        torch.tensor([True, True]),
        0.5,
        0.5,
    )

    assert support.all()
    assert similarity[0, 0] == pytest.approx(1.0)
    expected = torch.nn.functional.cosine_similarity(
        candidates[1, :2].flatten(), profiles[1, :2].flatten(), dim=0
    )
    assert similarity[1, 1] == pytest.approx(float(expected))


def test_direct_similarity_is_invariant_to_physical_flux_scale() -> None:
    candidates = torch.tensor([[1.0, 2.0, 4.0, 7.0]]) * 1.0e-19
    profiles = torch.tensor([[1.0, 2.0, 4.0, 7.0], [7.0, 4.0, 2.0, 1.0]]) * 1.0e-22
    mask = torch.ones_like(candidates)
    similarity, support = _common_support_correlation(
        candidates,
        mask,
        profiles,
        torch.ones_like(profiles),
        torch.tensor([True, True]),
        0.5,
        0.5,
    )

    assert support.all()
    assert similarity[0, 0] == pytest.approx(1.0, abs=1.0e-5)
    assert similarity[0, 1] < -0.8


def test_direct_similarity_softly_downweights_uncertain_bins() -> None:
    candidates = torch.tensor([[0.0, 1.0, 2.0, 30.0]])
    profiles = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    mask = torch.ones_like(candidates)
    supported = torch.tensor([True])

    unweighted, _ = _common_support_correlation(
        candidates,
        mask,
        profiles,
        torch.ones_like(profiles),
        supported,
        0.5,
        0.5,
    )
    weighted, weighted_support = _common_support_correlation(
        candidates,
        mask,
        profiles,
        torch.ones_like(profiles),
        supported,
        0.5,
        0.5,
        torch.tensor([[1.0, 1.0, 1.0, 1.0e-6]]),
    )

    assert weighted_support.all()
    assert weighted[0, 0] > unweighted[0, 0]
    assert weighted[0, 0] > 0.99


def test_token_similarity_softly_downweights_uncertain_tokens() -> None:
    candidates = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [20.0, -20.0]]]
    )
    profiles = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0]]]
    )
    mask = torch.ones(1, 4)
    supported = torch.tensor([True])

    unweighted, _ = _common_support_token_cosine(
        candidates,
        mask,
        profiles,
        mask,
        supported,
        0.5,
        0.5,
    )
    weighted, weighted_support = _common_support_token_cosine(
        candidates,
        mask,
        profiles,
        mask,
        supported,
        0.5,
        0.5,
        torch.tensor([[1.0, 1.0, 1.0, 1.0e-6]]),
    )

    assert weighted_support.all()
    assert weighted[0, 0] > unweighted[0, 0]
    assert weighted[0, 0] > 0.99


def test_reference_similarities_remain_physically_bounded() -> None:
    generator = torch.Generator().manual_seed(913)
    candidates = torch.randn(7, 64, generator=generator).bfloat16()
    profiles = torch.randn(9, 64, generator=generator).bfloat16()
    candidate_mask = (torch.rand(7, 64, generator=generator) > 0.15).float()
    profile_mask = (torch.rand(9, 64, generator=generator) > 0.15).float()
    supported = torch.ones(9, dtype=torch.bool)

    correlation, correlation_support = _common_support_correlation(
        candidates,
        candidate_mask,
        profiles,
        profile_mask,
        supported,
        0.25,
        0.50,
    )
    assert correlation.dtype == torch.float32
    assert correlation[correlation_support].abs().max() <= 1.0

    candidate_tokens = torch.randn(7, 64, 4, generator=generator).bfloat16()
    profile_tokens = torch.randn(9, 64, 4, generator=generator).bfloat16()
    cosine, cosine_support = _common_support_token_cosine(
        candidate_tokens,
        candidate_mask,
        profile_tokens,
        profile_mask,
        supported,
        0.25,
        0.50,
    )
    assert cosine.dtype == torch.float32
    assert cosine[cosine_support].abs().max() <= 1.0


def test_reference_clustering_normalization_handles_flam_units() -> None:
    values = (
        np.asarray(
            [[1.0, 2.0, 4.0, 7.0], [2.0, 4.0, 8.0, 14.0]],
            dtype=np.float32,
        )
        * 1.0e-19
    )
    normalized = _normalise_rows(values, np.ones_like(values, dtype=bool))

    assert np.linalg.norm(normalized[0]) == pytest.approx(1.0)
    assert np.allclose(normalized[0], normalized[1], atol=1.0e-6)


@pytest.mark.parametrize(
    ("spectral_encoder", "sequence_combination"),
    [
        ("direct", "mean"),
        ("shared_cnn", "continuous_time_attention"),
        ("multiscale_attention", "continuous_time_attention"),
        ("multiscale_attention", "temporal_transformer"),
    ],
)
def test_roman_reference_model_uses_measurements_only(
    tmp_path: Path,
    spectral_encoder: str,
    sequence_combination: str,
) -> None:
    config = load_config(ROOT / "configs/experiments/encoded_onir_named_clean.yaml")
    bank_path = _test_bank(tmp_path / "reference.npz", flux_scale=1.0e-20)
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
            "dense_rest_frame_scan": False,
            "dense_continuum_detail": False,
            "full_spectrum_context": False,
            "temporal_mode": "none",
            "phase_auxiliary_bins": 0,
            "candidate_phase_consistency": False,
            "use_flux_error_channel": False,
            "coadd_maximum_relative_error": 3.0,
            "coadd_edge_trim_fraction": 0.05,
            "redshift_bins": 31,
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
        "redshift_chunk_size": 7,
        "sequence_visits": 3,
        "minimum_sequence_visits": 2,
        "spectral_encoder": spectral_encoder,
        "token_dim": 8,
        "token_pool_size": 2,
        "minimum_encoder_support": 0.5,
        "attention_heads": 2,
        "sequence_combination": sequence_combination,
        "time_attention_hidden_dim": 16,
        "temporal_hidden_dim": 16,
        "temporal_attention_heads": 2,
        "temporal_layers": 1,
        "temporal_feedforward_multiplier": 2,
        "temporal_initial_correction_scale": 0.10,
    }
    model = Strider(config).eval()
    generator = torch.Generator().manual_seed(714)
    flux = torch.randn(2, 4, 256, generator=generator)
    error = 0.5 + torch.rand(2, 4, 256, generator=generator)
    batch = {
        "flux": flux,
        "wavelength_mask": torch.ones_like(flux),
        "visit_mask": torch.ones(2, 4),
        "observer_days": torch.tensor(
            [[0.0, 8.0, 20.0, 42.0], [0.0, 11.0, 19.0, 35.0]]
        ),
        "flux_error_shape": torch.log(error),
        "visit_flux_scale": torch.tensor([[2.0, 3.0, 4.0, 5.0], [1.5, 2.5, 3.5, 4.5]]),
    }
    output = model(batch)

    assert output["joint_logits"].shape == (
        2,
        2,
        len(model.redshift_grid),
    )
    assert output["joint_support"].shape == output["joint_logits"].shape
    assert (
        output["reference_sequence_joint_logits"].shape == output["joint_logits"].shape
    )
    assert output["joint_logits"][output["joint_support"]].abs().max() <= 20.0
    assert torch.isfinite(output["joint_logits"]).all()
    assert "simulation_rest_phase_days" not in batch
    assert "clean_flux_target" not in batch

    model.train()
    assert model.roman_reference is not None
    fixed_reference = model.roman_reference.coadd_full_profiles.clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4)
    training_output = model(batch)
    supported = training_output["joint_support"]
    flat_support = supported.flatten(1)
    targets = flat_support.to(torch.int64).argmax(dim=1)
    loss = torch.nn.functional.cross_entropy(
        training_output["joint_logits"].flatten(1),
        targets,
    )
    assert loss < 50.0
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    reference_parameters = dict(model.roman_reference.named_parameters())
    reference_buffers = dict(model.roman_reference.named_buffers())
    assert "coadd_full_profiles" not in reference_parameters
    assert "coadd_full_profiles" in reference_buffers
    assert model.roman_reference.coadd_scale.grad is not None
    optimizer.step()
    assert torch.equal(
        model.roman_reference.coadd_full_profiles,
        fixed_reference,
    )
    with torch.no_grad():
        updated = model(batch)
    assert updated["joint_logits"][updated["joint_support"]].abs().max() <= 20.0
    if spectral_encoder == "shared_cnn":
        encoder_gradient = model.roman_reference.spectral_encoder.first.conv.weight.grad
        attention_gradient = model.roman_reference.time_attention[0].weight.grad
        assert encoder_gradient is not None
        assert attention_gradient is not None
        assert torch.isfinite(encoder_gradient).all()
        assert torch.isfinite(attention_gradient).all()
    elif spectral_encoder == "multiscale_attention":
        encoder = model.roman_reference.spectral_encoder
        assert isinstance(encoder, MaskAwareMultiscaleAttentionEncoder)
        encoder_gradient = encoder.branches[0].conv.weight.grad
        spectral_attention_gradient = encoder.attention.in_proj_weight.grad
        assert encoder_gradient is not None
        assert spectral_attention_gradient is not None
        assert torch.isfinite(encoder_gradient).all()
        assert torch.isfinite(spectral_attention_gradient).all()
        if sequence_combination == "continuous_time_attention":
            time_attention_gradient = model.roman_reference.time_attention[0].weight.grad
            assert time_attention_gradient is not None
            assert torch.isfinite(time_attention_gradient).all()
        else:
            temporal = model.roman_reference.temporal_transformer
            assert isinstance(temporal, CandidateTemporalTransformer)
            input_gradient = temporal.input_projection[0].weight.grad
            output_gradient = temporal.output_projection.weight.grad
            assert input_gradient is not None
            assert output_gradient is not None
            assert torch.isfinite(input_gradient).all()
            assert torch.isfinite(output_gradient).all()


def test_candidate_temporal_transformer_uses_time_and_ignores_unsupported_scores() -> None:
    torch.manual_seed(1_927)
    temporal = CandidateTemporalTransformer(
        fine_class_count=3,
        starting_phase_count=2,
        hidden_dim=16,
        attention_heads=4,
        layers=1,
        feedforward_multiplier=2,
        dropout=0.0,
        initial_correction_scale=0.20,
    ).eval()
    scores = torch.randn(2, 4, 5, 3, 2)
    support = torch.ones_like(scores, dtype=torch.bool)
    support[0, 3] = False
    rest_offset = torch.tensor(
        [
            [[0.0] * 5, [4.0] * 5, [12.0] * 5, [25.0] * 5],
            [[0.0] * 5, [7.0] * 5, [15.0] * 5, [31.0] * 5],
        ]
    )
    signal_to_noise = torch.tensor([[2.0, 3.0, 1.5, 4.0], [1.0, 2.0, 3.0, 4.0]])

    with torch.no_grad():
        original = temporal(scores, support, rest_offset, signal_to_noise)
        changed = scores.clone()
        changed[~support] = 1.0e6
        unsupported_changed = temporal(
            changed, support, rest_offset, signal_to_noise
        )
        reversed_time = temporal(
            scores, support, rest_offset.flip(dims=(1,)), signal_to_noise
        )
        constant = scores[:, :1].expand_as(scores).clone()
        constant_output = temporal(
            constant, support, rest_offset, signal_to_noise
        )
        constant_count = support.sum(dim=1)
        constant_baseline = (
            constant * support.to(constant.dtype)
        ).sum(dim=1) / constant_count.clamp_min(1)

    assert original.shape == (2, 5, 3, 2)
    assert torch.isfinite(original).all()
    assert torch.allclose(original, unsupported_changed, atol=1.0e-6)
    assert not torch.allclose(original, reversed_time)
    assert torch.allclose(constant_output, constant_baseline, atol=1.0e-6)

    training_scores = scores.clone()
    training_scores[0] = training_scores[0, :1].expand_as(training_scores[0])
    training_scores.requires_grad_(True)
    temporal.train()
    temporal(
        training_scores, support, rest_offset, signal_to_noise
    ).square().mean().backward()
    assert torch.isfinite(training_scores.grad).all()
    for parameter in temporal.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_candidate_temporal_transformer_uses_only_relative_flux_change() -> None:
    torch.manual_seed(8_441)
    temporal = CandidateTemporalTransformer(
        fine_class_count=3,
        starting_phase_count=2,
        hidden_dim=16,
        attention_heads=4,
        layers=1,
        feedforward_multiplier=2,
        dropout=0.0,
        initial_correction_scale=0.20,
        use_relative_flux=True,
    ).eval()
    scores = torch.ones(1, 4, 3, 3, 2)
    support = torch.ones_like(scores, dtype=torch.bool)
    rest_offset = torch.tensor(
        [[[0.0, 0.0, 0.0], [5.0, 4.0, 3.0], [12.0, 10.0, 8.0], [24.0, 20.0, 16.0]]]
    )
    signal_to_noise = torch.full((1, 4), 2.0)
    constant_flux = torch.ones(1, 4)
    evolving_flux = torch.tensor([[0.2, 0.8, 1.4, 0.6]])

    with torch.no_grad():
        baseline = temporal(
            scores,
            support,
            rest_offset,
            signal_to_noise,
            constant_flux,
        )
        evolving = temporal(
            scores,
            support,
            rest_offset,
            signal_to_noise,
            evolving_flux,
        )
        shifted = temporal(
            scores,
            support,
            rest_offset,
            signal_to_noise,
            evolving_flux + 13.0,
        )

    expected = scores.mean(dim=1)
    assert torch.allclose(baseline, expected, atol=1.0e-6)
    assert not torch.allclose(evolving, expected, atol=1.0e-7)
    assert torch.allclose(evolving, shifted, atol=1.0e-6)


def test_candidate_temporal_transformer_can_exclude_signal_to_noise() -> None:
    torch.manual_seed(9_241)
    temporal = CandidateTemporalTransformer(
        fine_class_count=3,
        starting_phase_count=2,
        hidden_dim=16,
        attention_heads=4,
        layers=1,
        feedforward_multiplier=2,
        dropout=0.0,
        initial_correction_scale=0.20,
        use_signal_to_noise=False,
        use_relative_flux=True,
    ).eval()
    scores = torch.randn(1, 4, 3, 3, 2)
    support = torch.ones_like(scores, dtype=torch.bool)
    rest_offset = torch.tensor(
        [[[0.0, 0.0, 0.0], [5.0, 4.0, 3.0], [12.0, 10.0, 8.0], [24.0, 20.0, 16.0]]]
    )
    relative_flux = torch.tensor([[0.2, 0.8, 1.4, 0.6]])

    with torch.no_grad():
        low_quality = temporal(
            scores,
            support,
            rest_offset,
            torch.full((1, 4), -100.0),
            relative_flux,
        )
        high_quality = temporal(
            scores,
            support,
            rest_offset,
            torch.full((1, 4), 100.0),
            relative_flux,
        )

    assert temporal.input_projection[0].in_features == 3 * 2 + 2 + 1
    assert torch.equal(low_quality, high_quality)


def test_multiscale_attention_ignores_unmeasured_flux_values() -> None:
    torch.manual_seed(982)
    encoder = MaskAwareMultiscaleAttentionEncoder(
        token_dim=16,
        attention_heads=4,
        dropout=0.0,
        pool_size=4,
        minimum_support=0.5,
    ).eval()
    flux = torch.randn(2, 64)
    mask = torch.ones_like(flux)
    mask[0, 16:24] = 0.0
    mask[1, 44:48] = 0.0
    changed = flux.clone()
    changed[mask == 0] = 1.0e6

    with torch.no_grad():
        first, first_support = encoder(flux, mask)
        second, second_support = encoder(changed, mask)

    assert first.shape == (2, 16, 16)
    assert torch.equal(first_support, second_support)
    assert torch.allclose(first, second, atol=1.0e-6)
    assert torch.isfinite(first).all()
    assert not first_support[0, 4:6].any()
    assert not first_support[1, 11]


def test_attention_reference_runs_without_reported_errors_or_visit_scales(
    tmp_path: Path,
) -> None:
    config = load_config(
        ROOT / "configs/nersc/ia_binary_20k_roman_reference_attention.yaml"
    )
    config["reference"]["bank_path"] = str(
        _test_bank(tmp_path / "reference.npz", rest_bins=64)
    )
    config["data"]["include_flux_error_channel"] = False
    config["model"]["redshift_bins"] = 31
    model = Strider(config).eval()
    generator = torch.Generator().manual_seed(338)
    wavelength_bins = int(config["observation"]["wavelength_bins"])
    flux = torch.randn(2, 4, wavelength_bins, generator=generator)
    batch = {
        "flux": flux,
        "wavelength_mask": torch.ones_like(flux),
        "visit_mask": torch.ones(2, 4),
        "observer_days": torch.tensor(
            [[0.0, 7.0, 19.0, 36.0], [0.0, 10.0, 22.0, 41.0]]
        ),
    }

    with torch.no_grad():
        output = model(batch)

    assert output["joint_logits"].shape == (2, 2, 31)
    assert torch.isfinite(output["joint_logits"]).all()
    assert torch.equal(output["coadd_used_reported_error"], torch.zeros(2))
    assert (output["coadded_valid_fraction"] > 0.0).all()


def test_attention_reference_retains_bins_with_soft_uncertainty_weighting(
    tmp_path: Path,
) -> None:
    config = load_config(
        ROOT / "configs/nersc/ia_binary_20k_roman_reference_attention.yaml"
    )
    config["reference"]["bank_path"] = str(
        _test_bank(tmp_path / "reference.npz", rest_bins=64)
    )
    config["reference"]["spectral_uncertainty_weighting"] = "inverse_variance"
    config["reference"]["edge_trim_fraction"] = 0.0
    config["reference"]["edge_taper_fraction"] = 0.0
    config["model"]["coadd_edge_trim_fraction"] = 0.0
    config["model"]["coadd_maximum_relative_error"] = None
    config["model"]["redshift_bins"] = 31
    model = Strider(config).eval()
    generator = torch.Generator().manual_seed(8_104)
    wavelength_bins = int(config["observation"]["wavelength_bins"])
    flux = torch.randn(2, 4, wavelength_bins, generator=generator)
    error = torch.ones_like(flux)
    error[..., :32] = 100.0
    batch = {
        "flux": flux,
        "wavelength_mask": torch.ones_like(flux),
        "visit_mask": torch.ones(2, 4),
        "observer_days": torch.tensor(
            [[0.0, 7.0, 19.0, 36.0], [0.0, 10.0, 22.0, 41.0]]
        ),
        "flux_error_shape": torch.log(error),
        "visit_flux_scale": torch.ones(2, 4),
    }

    with torch.no_grad():
        output = model(batch)

    assert torch.isfinite(output["joint_logits"]).all()
    assert torch.equal(output["coadd_used_reported_error"], torch.ones(2))
    assert torch.equal(output["coadded_valid_fraction"], torch.ones(2))


def test_soft_uncertainty_weighting_stabilizes_relative_flux_evolution(
    tmp_path: Path,
) -> None:
    config = load_config(ROOT / "configs/nersc/uncertainty_test.yaml")
    config["reference"]["bank_path"] = str(
        _test_bank(tmp_path / "reference.npz", rest_bins=64)
    )
    config["model"]["redshift_bins"] = 31
    model = Strider(config).eval()
    assert model.roman_reference is not None
    wavelength_bins = int(config["observation"]["wavelength_bins"])
    flux = torch.ones(1, 2, wavelength_bins)
    error = torch.ones_like(flux)
    flux[0, 0, wavelength_bins // 2] = 1_000.0
    error[0, 0, wavelength_bins // 2] = 1_000.0
    batch = {
        "flux": flux,
        "wavelength_mask": torch.ones_like(flux),
        "visit_mask": torch.ones(1, 2),
        "observer_days": torch.tensor([[0.0, 10.0]]),
        "flux_error_shape": torch.log(error),
        "visit_flux_scale": torch.ones(1, 2),
    }

    selected = model.roman_reference._select_sequence_visits(batch)

    assert torch.allclose(
        selected["relative_flux"][0, :2],
        torch.ones(2),
        atol=1.0e-4,
    )


def test_soft_uncertainty_weighting_prevents_encoder_contamination(
    tmp_path: Path,
) -> None:
    config = load_config(ROOT / "configs/nersc/uncertainty_test.yaml")
    config["reference"]["bank_path"] = str(
        _test_bank(tmp_path / "reference.npz", rest_bins=64)
    )
    config["model"]["redshift_bins"] = 31
    torch.manual_seed(991)
    model = Strider(config).eval()
    assert model.roman_reference is not None
    wavelength_bins = int(config["observation"]["wavelength_bins"])
    coordinate = torch.linspace(0.0, 2.0 * np.pi, wavelength_bins)
    reference = (2.0 + torch.sin(coordinate))[None, :]
    mask = torch.ones_like(reference)
    reference_error = torch.ones_like(reference)
    contaminated = reference.clone()
    contaminated_error = reference_error.clone()
    support_edge = slice(
        wavelength_bins // 2 - 16,
        wavelength_bins // 2 + 16,
    )
    contaminated[0, support_edge] = 1.0e8
    contaminated_error[0, support_edge] = 1.0e8
    reference_error[0, support_edge] = 1.0e8

    with torch.no_grad():
        reference_tokens, _ = model.roman_reference._encode_spectra(
            reference,
            mask,
            relative_inverse_variance(reference_error, mask),
        )
        contaminated_tokens, _ = model.roman_reference._encode_spectra(
            contaminated,
            mask,
            relative_inverse_variance(contaminated_error, mask),
        )

    similarity = torch.nn.functional.cosine_similarity(
        reference_tokens.flatten(),
        contaminated_tokens.flatten(),
        dim=0,
    )
    assert similarity > 0.995


def test_reference_cosine_taper_is_symmetric_and_keeps_interior_support(
    tmp_path: Path,
) -> None:
    config = load_config(
        ROOT / "configs/nersc/ia_binary_20k_roman_reference_attention.yaml"
    )
    config["reference"]["bank_path"] = str(
        _test_bank(tmp_path / "reference.npz", rest_bins=64)
    )
    config["reference"]["edge_trim_fraction"] = 0.0
    config["reference"]["edge_taper_fraction"] = 0.05
    config["model"]["coadd_edge_trim_fraction"] = 0.0
    config["model"]["coadd_maximum_relative_error"] = 3.0
    config["model"]["redshift_bins"] = 31

    model = Strider(config).eval()
    assert model.roman_reference is not None
    weight = model.roman_reference.observed_matching_weight
    mask = model.roman_reference.observed_matching_mask

    assert weight[0] == 0.0
    assert weight[-1] == 0.0
    assert torch.allclose(weight, weight.flip(0), atol=1.0e-7)
    assert (weight[64:-64] == 1.0).all()
    assert mask.sum() == mask.numel() - 2


def test_reference_cosine_taper_weights_flux_without_reshaping_it(
    tmp_path: Path,
) -> None:
    config = load_config(
        ROOT / "configs/nersc/ia_binary_20k_roman_reference_attention.yaml"
    )
    config["reference"]["bank_path"] = str(
        _test_bank(tmp_path / "reference.npz", rest_bins=64)
    )
    config["reference"]["edge_trim_fraction"] = 0.0
    config["reference"]["edge_taper_fraction"] = 0.05
    config["model"]["coadd_edge_trim_fraction"] = 0.0
    config["model"]["redshift_bins"] = 31
    model = Strider(config).eval()
    assert model.roman_reference is not None
    wavelength_bins = int(config["observation"]["wavelength_bins"])
    spectrum = torch.linspace(12.0, 1.0, wavelength_bins)
    flux = spectrum[None, None, :].expand(1, 4, -1).clone()
    error = torch.ones_like(flux)
    batch = {
        "flux": flux,
        "wavelength_mask": torch.ones_like(flux),
        "visit_mask": torch.ones(1, 4),
        "observer_days": torch.tensor([[0.0, 7.0, 19.0, 36.0]]),
        "flux_error_shape": torch.log(error),
        "visit_flux_scale": torch.ones(1, 4),
    }
    coadd, coadd_error, coadd_mask, _ = model._coadded_measurement(batch)
    captured: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def capture_continuum_inputs(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        captured.append(tuple(value.detach().clone() for value in inputs))

    handle = model.roman_reference.continuum_removal.register_forward_pre_hook(
        capture_continuum_inputs
    )
    with torch.no_grad():
        model.roman_reference(batch, coadd, coadd_mask, coadd_error)
    handle.remove()

    matching_flux, matching_mask, matching_weight = captured[0]
    taper = model.roman_reference.observed_matching_weight
    first_weighted_bin = 1
    assert torch.allclose(matching_flux, coadd)
    assert torch.allclose(matching_weight[0], taper)
    normalized = _scale_invariant_normalize(
        matching_flux,
        matching_mask,
        matching_weight,
    )
    artificially_tapered = _scale_invariant_normalize(
        matching_flux * taper[None, :],
        matching_mask,
        matching_weight,
    )
    assert normalized[0, first_weighted_bin] > 0.0
    assert artificially_tapered[0, first_weighted_bin] < 0.0


def test_reference_cosine_taper_ignores_exact_endpoint_flux(
    tmp_path: Path,
) -> None:
    config = load_config(
        ROOT / "configs/nersc/ia_binary_20k_roman_reference_attention.yaml"
    )
    config["reference"]["bank_path"] = str(
        _test_bank(tmp_path / "reference.npz", rest_bins=64)
    )
    config["reference"]["edge_trim_fraction"] = 0.0
    config["reference"]["edge_taper_fraction"] = 0.05
    config["model"]["coadd_edge_trim_fraction"] = 0.0
    config["model"]["redshift_bins"] = 31
    model = Strider(config).eval()
    generator = torch.Generator().manual_seed(2_609)
    wavelength_bins = int(config["observation"]["wavelength_bins"])
    flux = torch.randn(2, 4, wavelength_bins, generator=generator)
    error = 0.5 + torch.rand(2, 4, wavelength_bins, generator=generator)
    batch = {
        "flux": flux,
        "wavelength_mask": torch.ones_like(flux),
        "visit_mask": torch.ones(2, 4),
        "observer_days": torch.tensor(
            [[0.0, 7.0, 19.0, 36.0], [0.0, 10.0, 22.0, 41.0]]
        ),
        "flux_error_shape": torch.log(error),
        "visit_flux_scale": torch.ones(2, 4),
    }
    changed = dict(batch)
    changed_flux = flux.clone()
    changed_flux[..., 0] = 1.0e6
    changed_flux[..., -1] = -1.0e6
    changed["flux"] = changed_flux

    with torch.no_grad():
        reference = model(batch)
        endpoints_changed = model(changed)

    assert torch.equal(
        reference["joint_support"], endpoints_changed["joint_support"]
    )
    assert torch.allclose(
        reference["joint_logits"],
        endpoints_changed["joint_logits"],
        atol=1.0e-6,
    )
