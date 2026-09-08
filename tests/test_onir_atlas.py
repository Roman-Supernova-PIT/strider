from pathlib import Path

import numpy as np
import pytest
import torch

from strider.atlas.build import (
    _assert_geometric_profile_masks,
    _cluster_profiles,
    _profile_input_spectra,
    _random_anchor_wavelengths,
    align_to_rest_grid,
    extract_feature_windows,
)
from strider.atlas.catalog import feature_geometry, load_catalog, overlap_weights
from strider.config import load_config
from strider.data.dataset import log_wavelength_grid
from strider.model import Strider
from strider.model.encoded_onir import (
    common_support_cosine,
    interpolate_sequence,
    normalize_valid_bins,
)
from strider.training.trainer import _load_initial_checkpoint
from strider.training.losses import _coadd_reconstruction_loss


ROOT = Path(__file__).resolve().parents[1]


def _write_encoded_test_bank(
    path: Path,
    config: dict,
    missing_class_index: int | None = None,
) -> Path:
    """Write the smallest complete ONIR bank needed by structural model tests."""
    catalog = load_catalog(ROOT / "configs/onir_features.yaml")
    model_settings = config["model"]
    onir_settings = config["onir"]
    rest = log_wavelength_grid(
        model_settings["rest_wavelength_min"],
        model_settings["rest_wavelength_max"],
        model_settings["rest_wavelength_bins"],
    )
    centers, radii, window_mask = feature_geometry(
        rest,
        catalog,
        int(onir_settings["maximum_radius_bins"]),
        allow_radius_clipping=bool(onir_settings.get("allow_radius_clipping", False)),
    )
    weights = overlap_weights(
        centers,
        window_mask,
        len(rest),
    )
    classes = len(model_settings["classes"])
    features, window = window_mask.shape
    prototypes = 2
    rng = np.random.default_rng(901)
    prototype_profiles = rng.normal(
        size=(classes, features, prototypes, window)
    ).astype(np.float32)
    prototype_mask = np.broadcast_to(
        window_mask[None, :, None, :], prototype_profiles.shape
    ).copy()
    prototype_profiles *= prototype_mask
    prototype_support = np.full((classes, features, prototypes), 10, dtype=np.int64)
    support = prototype_support.sum(axis=-1)
    if missing_class_index is not None:
        support[missing_class_index] = 0
        prototype_support[missing_class_index] = 0
    mean_profiles = prototype_profiles.mean(axis=2)
    mean_mask = np.broadcast_to(window_mask[None, :, :], mean_profiles.shape).copy()
    np.savez_compressed(
        path,
        mean_profiles=mean_profiles,
        mean_profile_mask=mean_mask,
        medoid_profiles=prototype_profiles[:, :, 0],
        medoid_profile_mask=mean_mask,
        prototype_profiles=prototype_profiles,
        prototype_profile_mask=prototype_mask,
        prototype_support_counts=prototype_support,
        support_counts=support,
        class_names=np.asarray(model_settings["classes"]),
        feature_names=np.asarray(catalog.names),
        rest_wavelengths=catalog.rest_wavelengths,
        feature_radii=radii,
        window_mask=window_mask,
        overlap_weights=weights,
        rest_grid=rest,
    )
    return path


def test_catalog_geometry_and_overlap_weights() -> None:
    catalog = load_catalog(ROOT / "configs/onir_features.yaml")
    rest_grid = log_wavelength_grid(2500.0, 10000.0, 160)
    centers, radii, mask = feature_geometry(
        rest_grid, catalog, 7, allow_radius_clipping=True
    )
    weights = overlap_weights(centers, mask, len(rest_grid))
    assert len(centers) == 15
    assert np.all((radii >= 1) & (radii <= 7))
    assert mask.shape == (15, 15)
    assert np.all(weights > 0)
    assert np.isclose(weights.mean(), 1.0)


def test_shifted_feature_returns_to_named_rest_position() -> None:
    observed = log_wavelength_grid(7500.0, 20000.0, 256)
    rest = log_wavelength_grid(2500.0, 10000.0, 160)
    redshift = 1.0
    rest_feature = 4555.0
    observed_flux = np.exp(-0.5 * ((observed - rest_feature * (1 + redshift)) / 80.0) ** 2)
    restored, mask = align_to_rest_grid(
        observed, observed_flux.astype(np.float32), np.ones_like(observed), redshift, rest
    )
    recovered = float(rest[np.argmax(restored)])
    assert abs(recovered - rest_feature) < 80.0
    assert mask[np.argmax(restored)] == 1.0


def test_window_extraction_rejects_missing_coverage() -> None:
    rest = log_wavelength_grid(2500.0, 10000.0, 160)
    catalog = load_catalog(ROOT / "configs/onir_features.yaml")
    centers, _, window_mask = feature_geometry(
        rest, catalog, 7, allow_radius_clipping=True
    )
    flux = np.sin(np.linspace(0, 8, len(rest))).astype(np.float32)
    no_coverage = np.zeros_like(rest)
    _, _, usable = extract_feature_windows(
        flux, no_coverage, centers, window_mask, minimum_valid_fraction=0.8
    )
    assert not usable.any()


def test_profile_masks_must_match_feature_geometry() -> None:
    geometry = np.asarray([[1, 1, 0], [0, 1, 1]], dtype=bool)
    mean_masks = np.broadcast_to(geometry[None], (2, 2, 3)).copy()
    medoid_masks = mean_masks.copy()
    support = np.asarray([[4, 0], [3, 2]])
    prototype_masks = np.broadcast_to(
        geometry[None, :, None, :], (2, 2, 2, 3)
    ).copy()
    prototype_support = np.asarray([[[2, 2], [0, 0]], [[2, 1], [1, 1]]])

    _assert_geometric_profile_masks(
        mean_masks,
        medoid_masks,
        support,
        prototype_masks,
        prototype_support,
        geometry,
    )

    prototype_masks[1, 1, 0, 1] = False
    with pytest.raises(RuntimeError, match="prototype masks"):
        _assert_geometric_profile_masks(
            mean_masks,
            medoid_masks,
            support,
            prototype_masks,
            prototype_support,
            geometry,
        )


def test_random_anchors_preserve_named_wavelength_strata() -> None:
    rest = log_wavelength_grid(2500.0, 10000.0, 160)
    named = load_catalog(ROOT / "configs/onir_features.yaml").rest_wavelengths
    random = _random_anchor_wavelengths(rest, named, seed=19)
    midpoints = np.sqrt(named[:-1] * named[1:])
    assert np.all(random[1:] >= midpoints)
    assert np.all(random[:-1] < midpoints)
    assert np.all(np.diff(random) > 0)
    assert random[0] > 0.9 * named[0]
    assert random[-1] < 1.1 * named[-1]


def test_phase_neutral_profile_clusters_are_supported_and_normalized() -> None:
    rng = np.random.default_rng(11)
    active = np.ones(15, dtype=bool)
    left = rng.normal(0.0, 0.05, size=(20, 15))
    right = rng.normal(0.0, 0.05, size=(20, 15))
    left[:, 4:7] -= 1.0
    right[:, 9:12] -= 1.0
    values = np.stack([
        value / max(np.linalg.norm(value - value.mean()), 1e-8)
        for value in np.concatenate([left, right]) - np.concatenate([left, right]).mean(axis=1, keepdims=True)
    ]).astype(np.float32)
    profiles, support = _cluster_profiles(values, active, 3)
    assert support.sum() == len(values)
    # The synthetic sample has two real modes; an unused third slot remains
    # explicit rather than fabricating a supported profile.
    assert np.count_nonzero(support) == 2
    assert np.allclose(np.linalg.norm(profiles[support > 0], axis=1), 1.0, atol=1e-5)


def test_coadded_bank_input_matches_equal_weight_runtime_coadd() -> None:
    item = {
        "flux": torch.tensor(
            [[1.0, 2.0, 100.0], [3.0, 4.0, 6.0], [9.0, 9.0, 9.0]]
        ),
        "wavelength_mask": torch.tensor(
            [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
        ),
        "simulation_rest_phase_days": torch.tensor([-10.0, 20.0, 120.0]),
    }
    spectra = _profile_input_spectra(item, "coadded_flux", -20.0, 80.0)
    assert len(spectra) == 1
    flux, mask = spectra[0]
    assert np.allclose(flux, [2.0, 2.0, 6.0])
    assert np.array_equal(mask, [1.0, 1.0, 1.0])


def test_phase_neutral_onir_scores_do_not_read_observer_dates(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/experiments/onir_named_clean.yaml")
    catalog = load_catalog(ROOT / "configs/onir_features.yaml")
    rest = log_wavelength_grid(2500.0, 10000.0, 160)
    centers, radii, window_mask = feature_geometry(
        rest, catalog, 7, allow_radius_clipping=True
    )
    weights = overlap_weights(centers, window_mask, len(rest))
    rng = np.random.default_rng(7)
    profiles = rng.normal(size=(2, 15, 15)).astype(np.float32)
    bank_path = tmp_path / "onir_test_bank.npz"
    np.savez_compressed(
        bank_path,
        mean_profiles=profiles,
        medoid_profiles=profiles,
        support_counts=np.full((2, 15), 100, dtype=np.int64),
        class_names=np.asarray(["Ia", "other"]),
        feature_names=np.asarray(catalog.names),
        rest_wavelengths=catalog.rest_wavelengths,
        feature_radii=radii,
        window_mask=window_mask,
        overlap_weights=weights,
        rest_grid=rest,
    )
    config["onir"]["bank_path"] = str(bank_path)
    model = Strider(config).eval()
    flux = torch.from_numpy(rng.normal(size=(2, 3, 256)).astype(np.float32))
    batch = {
        "flux": flux,
        "wavelength_mask": torch.ones_like(flux),
        "visit_mask": torch.ones(2, 3),
        "observer_days": torch.tensor([[-20.0, 0.0, 20.0], [-8.0, 2.0, 33.0]]),
    }
    changed = dict(batch)
    changed["observer_days"] = torch.tensor([[100.0, 110.0, 400.0], [0.0, 1.0, 2.0]])
    first = model(batch)["joint_logits"]
    second = model(changed)["joint_logits"]
    assert torch.equal(first, second)


def test_fractional_token_gather_interpolates_and_requires_both_neighbors() -> None:
    sequence = torch.tensor([[[0.0], [2.0], [4.0], [6.0]]])
    support = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    values, measured = interpolate_sequence(
        sequence,
        support,
        torch.tensor([0, 2]),
        torch.tensor([1, 3]),
        torch.tensor([0.5, 0.5]),
        torch.tensor([1.0, 1.0]),
    )
    assert torch.allclose(values[0, 0], torch.tensor([1.0]))
    assert measured[0, 0] == 1.0
    assert torch.equal(values[0, 1], torch.zeros(1))
    assert measured[0, 1] == 0.0


def test_valid_bin_normalization_ignores_masked_values() -> None:
    flux = torch.tensor([[1.0, 2.0, 3.0, 1.0e6]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    normalized = normalize_valid_bins(flux, mask)
    assert normalized[0, 3] == 0.0
    assert torch.allclose(normalized[0, :3].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(normalized[0, :3].square().mean(), torch.tensor(1.0), atol=1e-5)


def test_valid_bin_normalization_removes_only_a_global_positive_scale() -> None:
    flux = torch.tensor([[1.0, 3.0, 2.0, -1.0]])
    mask = torch.ones_like(flux)

    reference = normalize_valid_bins(flux, mask)
    brighter = normalize_valid_bins(10.0 * flux, mask)

    assert torch.allclose(reference, brighter, atol=1e-6)


def test_cosine_uses_only_support_shared_by_measurement_and_profile() -> None:
    measured = torch.tensor([[[[1.0], [1.0], [0.0]]]])
    measured_mask = torch.tensor([[[1.0, 1.0, 0.0]]])
    profiles = torch.tensor([[[[1.0], [1.0], [100.0]]]])
    profile_mask = torch.ones(1, 1, 3)
    similarity = common_support_cosine(
        measured, measured_mask, profiles, profile_mask
    )
    assert torch.allclose(similarity, torch.ones_like(similarity))


def test_encoded_onir_is_date_independent_and_ignores_masked_flux(
    tmp_path: Path,
) -> None:
    config = load_config(ROOT / "configs/experiments/encoded_onir_named_clean.yaml")
    config["onir"]["bank_path"] = str(
        _write_encoded_test_bank(tmp_path / "encoded_onir.npz", config)
    )
    model = Strider(config).eval()
    rng = np.random.default_rng(29)
    flux = torch.from_numpy(rng.normal(size=(2, 3, 256)).astype(np.float32))
    mask = torch.ones_like(flux)
    mask[..., 20:40] = 0.0
    batch = {
        "flux": flux * mask,
        "wavelength_mask": mask,
        "visit_mask": torch.ones(2, 3),
        "observer_days": torch.tensor([[-20.0, 0.0, 20.0], [-8.0, 2.0, 33.0]]),
    }
    changed = dict(batch)
    changed["observer_days"] = torch.tensor([[100.0, 110.0, 400.0], [0.0, 1.0, 2.0]])
    changed["flux"] = batch["flux"].clone()
    changed["flux"][..., 20:40] = 1.0e6
    first = model(batch)["joint_logits"]
    second = model(changed)["joint_logits"]
    assert torch.equal(first, second)


def test_encoded_onir_keeps_spectral_and_temporal_results_separate(
    tmp_path: Path,
) -> None:
    config = load_config(ROOT / "configs/experiments/encoded_onir_named_clean.yaml")
    config["onir"]["bank_path"] = str(
        _write_encoded_test_bank(tmp_path / "encoded_onir.npz", config)
    )
    config["model"]["temporal_mode"] = "spectral_evolution"
    config["model"]["temporal_initial_scale"] = 1.0
    model = Strider(config).eval()
    flux = torch.randn(2, 3, 256)
    batch = {
        "flux": flux,
        "wavelength_mask": torch.ones_like(flux),
        "visit_mask": torch.ones(2, 3),
        "observer_days": torch.tensor([[-20.0, 0.0, 20.0], [-8.0, 2.0, 33.0]]),
    }
    changed = dict(batch)
    changed["observer_days"] = torch.tensor([[0.0, 1.0, 90.0], [0.0, 20.0, 21.0]])
    first = model(batch)
    second = model(changed)
    assert torch.equal(first["spectral_joint_logits"], second["spectral_joint_logits"])
    assert not torch.equal(
        first["raw_temporal_joint_logits"], second["raw_temporal_joint_logits"]
    )


def test_encoded_onir_rejects_a_declared_class_without_profiles(
    tmp_path: Path,
) -> None:
    config = load_config(ROOT / "configs/experiments/encoded_onir_named_clean.yaml")
    bank_path = _write_encoded_test_bank(
        tmp_path / "missing_other_class.npz",
        config,
        missing_class_index=1,
    )
    config["onir"]["bank_path"] = str(bank_path)
    try:
        Strider(config)
    except ValueError as error:
        assert "no usable prototype" in str(error)
        assert "other" in str(error)
    else:
        raise AssertionError("class with no ONIR profiles was accepted")


def test_encoded_onir_can_hold_clean_profiles_fixed(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/experiments/encoded_onir_named_clean.yaml")
    config["onir"]["bank_path"] = str(
        _write_encoded_test_bank(tmp_path / "fixed_profiles.npz", config)
    )
    config["onir"]["train_profiles"] = False

    model = Strider(config)

    assert model.onir.profiles.requires_grad is False
    assert model.onir.profile_drift().item() < 1.0e-7


def _factored_model(
    tmp_path: Path,
    temporal_scale: float = 0.0,
    full_spectrum_context: bool = False,
    dense_scan: bool = False,
    dense_detail: bool = False,
    dense_scan_view: str | None = None,
    dense_scale: float = 0.25,
    relative_brightness: bool = False,
    brightness_scale: float = 0.0,
    input_normalization: str | None = None,
    coadd_input: bool = False,
    coadd_reconstruction: bool = False,
) -> Strider:
    config = load_config(ROOT / "configs/experiments/encoded_onir_named_clean.yaml")
    config["onir"]["bank_path"] = str(
        _write_encoded_test_bank(tmp_path / "factored_onir.npz", config)
    )
    if input_normalization is not None:
        config["onir"]["input_normalization"] = input_normalization
    if coadd_input:
        config["data"]["include_flux_error_channel"] = True
    if coadd_reconstruction:
        config["data"]["include_clean_flux_target"] = True
    config["model"].update(
        {
            "architecture": "factored_onir",
            "temporal_mode": "spectral_evolution",
            "temporal_initial_scale": temporal_scale,
            "factored_attention_heads": 4,
            "factored_shape_initial_scale": 0.5,
            "full_spectrum_context": full_spectrum_context,
            "context_patch_size": 8,
            "context_attention_heads": 4,
            "context_attention_layers": 1,
            "context_initial_scale": 0.25,
            "dense_rest_frame_scan": dense_scan,
            "dense_continuum_detail": dense_detail,
            "dense_scan_initial_scale": dense_scale,
            "dense_initial_detail_weight": 0.5,
            "dense_continuum_sigma_bins": 8.0,
            "dense_scan_chunk_size": 8,
            "dense_scan_token_dim": 16,
            "dense_scan_patch_size": 8,
            "dense_scan_rest_bins": 64,
            "dense_scan_minimum_overlap": 0.2,
            "relative_brightness_evolution": relative_brightness,
            "relative_brightness_initial_scale": brightness_scale,
            "dense_scan_input_mode": (
                "inverse_variance_coadd" if coadd_input else "individual_visits"
            ),
            "coadd_maximum_relative_error": 3.0,
            "coadd_edge_trim_fraction": 0.05,
            "coadd_reconstruction": coadd_reconstruction,
        }
    )
    if dense_scan_view is not None:
        config["model"]["dense_scan_view"] = dense_scan_view
    return Strider(config).eval()


def test_coadd_dense_route_is_scale_invariant_and_exposes_reconstruction(
    tmp_path: Path,
) -> None:
    model = _factored_model(
        tmp_path,
        dense_scan=True,
        dense_detail=True,
        dense_scan_view="detail",
        input_normalization="per_visit",
        coadd_input=True,
        coadd_reconstruction=True,
    )
    generator = torch.Generator().manual_seed(811)
    physical_flux = torch.randn(2, 3, 256, generator=generator)
    physical_error = 0.5 + torch.rand(2, 3, 256, generator=generator)
    first_scale = torch.tensor([[2.0, 5.0, 7.0], [3.0, 6.0, 9.0]])
    second_scale = torch.tensor([[0.2, 50.0, 1.4], [30.0, 0.6, 4.5]])
    common = {
        "wavelength_mask": torch.ones_like(physical_flux),
        "visit_mask": torch.ones(2, 3),
        "observer_days": torch.tensor(
            [[0.0, 10.0, 20.0], [0.0, 8.0, 30.0]]
        ),
    }

    def represented(scale: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            **common,
            "flux": physical_flux / scale[..., None],
            "flux_error_shape": torch.log(
                physical_error / scale[..., None]
            ),
            "visit_flux_scale": scale,
        }

    with torch.no_grad():
        first = model(represented(first_scale))
        second = model(represented(second_scale))

    assert first["coadd_reconstruction"].shape == (2, 256)
    assert first["coadd_reconstruction_mask"].shape == (2, 256)
    assert torch.all(first["coadded_valid_fraction"] < 1.0)
    assert torch.all(first["coadded_valid_fraction"] > 0.85)
    assert torch.allclose(
        first["dense_scan_joint_logits"],
        second["dense_scan_joint_logits"],
        atol=2.0e-5,
    )
    assert torch.allclose(
        first["coadd_reconstruction"],
        second["coadd_reconstruction"],
        atol=2.0e-5,
    )

    training_batch = {
        **represented(first_scale),
        "clean_flux_target": physical_flux / first_scale[..., None],
        "has_source": torch.ones(2),
    }
    reconstruction_loss = _coadd_reconstruction_loss(
        first,
        training_batch,
        weight=0.05,
    )
    assert torch.isfinite(reconstruction_loss)
    assert reconstruction_loss > 0.0


def test_coadd_reconstruction_head_can_start_from_dense_checkpoint(
    tmp_path: Path,
) -> None:
    baseline = _factored_model(
        tmp_path,
        dense_scan=True,
        dense_detail=True,
        dense_scan_view="detail",
    )
    candidate = _factored_model(
        tmp_path,
        dense_scan=True,
        dense_detail=True,
        dense_scan_view="detail",
        coadd_input=True,
        coadd_reconstruction=True,
    )
    checkpoint_path = tmp_path / "dense_without_reconstruction.pt"
    torch.save(
        {
            "model_state": baseline.state_dict(),
            "classes": list(candidate.class_names),
            "redshift_grid": candidate.redshift_grid,
        },
        checkpoint_path,
    )
    config = load_config(ROOT / "configs/experiments/encoded_onir_named_clean.yaml")
    config["data"].update(
        {
            "include_flux_error_channel": True,
            "include_clean_flux_target": True,
        }
    )
    config["model"].update(
        {
            "architecture": "factored_onir",
            "temporal_mode": "spectral_evolution",
            "dense_rest_frame_scan": True,
            "dense_continuum_detail": True,
            "dense_scan_view": "detail",
            "dense_scan_input_mode": "inverse_variance_coadd",
            "coadd_reconstruction": True,
        }
    )

    _load_initial_checkpoint(candidate, config, str(checkpoint_path))


def test_factored_onir_detail_scan_exposes_only_the_detail_component(
    tmp_path: Path,
) -> None:
    model = _factored_model(
        tmp_path,
        dense_scan=True,
        dense_detail=True,
        dense_scan_view="detail",
    )
    flux = torch.randn(2, 3, 256)
    output = model(
        {
            "flux": flux,
            "wavelength_mask": torch.ones_like(flux),
            "visit_mask": torch.ones(2, 3),
            "observer_days": torch.tensor(
                [[0.0, 10.0, 20.0], [0.0, 8.0, 30.0]]
            ),
        }
    )

    assert "dense_detail_joint_logits" in output
    assert "dense_whole_joint_logits" not in output
    assert torch.equal(
        output["dense_scan_joint_logits"],
        output["dense_detail_joint_logits"],
    )


def test_factored_onir_exposes_separate_evidence_terms(tmp_path: Path) -> None:
    model = _factored_model(tmp_path)
    flux = torch.randn(2, 3, 256)
    batch = {
        "flux": flux,
        "wavelength_mask": torch.ones_like(flux),
        "visit_mask": torch.ones(2, 3),
        "observer_days": torch.tensor([[-20.0, 0.0, 20.0], [-8.0, 2.0, 33.0]]),
    }

    output = model(batch)

    redshift_count = len(model.redshift_grid)
    assert output["joint_logits"].shape == (2, 2, redshift_count)
    assert output["shape_feature_attention"].shape == (
        2,
        2,
        redshift_count,
        15,
    )
    assert output["temporal_feature_attention"].shape == (
        2,
        2,
        redshift_count,
        15,
    )
    assert torch.allclose(
        output["joint_logits"], output["spectral_joint_logits"], atol=1e-7
    )
    assert torch.allclose(
        output["spectral_joint_logits"],
        output["onir_joint_logits"] + output["shape_joint_logits"],
        atol=1e-7,
    )


def test_full_spectrum_context_adds_class_evidence_without_setting_redshift(
    tmp_path: Path,
) -> None:
    model = _factored_model(tmp_path, full_spectrum_context=True)
    flux = torch.randn(2, 3, 256)
    batch = {
        "flux": flux,
        "wavelength_mask": torch.ones_like(flux),
        "visit_mask": torch.ones(2, 3),
        "observer_days": torch.tensor([[0.0, 10.0, 20.0], [0.0, 8.0, 30.0]]),
    }

    output = model(batch)

    context = output["context_joint_logits"]
    assert context.shape == output["joint_logits"].shape
    assert torch.equal(context, context[..., :1].expand_as(context))
    assert torch.allclose(
        output["spectral_joint_logits"],
        output["onir_joint_logits"]
        + output["shape_joint_logits"]
        + context,
        atol=1e-7,
    )


def test_dense_scan_adds_redshift_dependent_whole_spectrum_evidence(
    tmp_path: Path,
) -> None:
    model = _factored_model(tmp_path, dense_scan=True)
    flux = torch.randn(2, 3, 256)
    batch = {
        "flux": flux,
        "wavelength_mask": torch.ones_like(flux),
        "visit_mask": torch.ones(2, 3),
        "observer_days": torch.tensor([[0.0, 10.0, 20.0], [0.0, 8.0, 30.0]]),
    }

    output = model(batch)

    assert output["dense_scan_joint_logits"].shape == output["joint_logits"].shape
    assert output["dense_scan_overlap_fraction"].shape == (
        2,
        len(model.redshift_grid),
    )
    assert output["dense_scan_overlap_fraction"].max() > 0.2
    assert output["dense_scan_joint_logits"].abs().sum() > 0.0
    assert torch.allclose(
        output["spectral_joint_logits"],
        output["onir_joint_logits"]
        + output["shape_joint_logits"]
        + output["dense_scan_joint_logits"],
        atol=1e-7,
    )


def test_dense_scan_can_start_from_the_current_factored_checkpoint(
    tmp_path: Path,
) -> None:
    baseline = _factored_model(tmp_path)
    dense = _factored_model(tmp_path, dense_scan=True, dense_scale=0.0)
    checkpoint_path = tmp_path / "baseline.pt"
    torch.save(
        {
            "model_state": baseline.state_dict(),
            "classes": list(dense.class_names),
            "redshift_grid": dense.redshift_grid,
        },
        checkpoint_path,
    )
    config = load_config(ROOT / "configs/experiments/encoded_onir_named_clean.yaml")
    config["model"].update(
        {
            "architecture": "factored_onir",
            "temporal_mode": "spectral_evolution",
            "dense_rest_frame_scan": True,
        }
    )

    _load_initial_checkpoint(dense, config, str(checkpoint_path))

    flux = torch.randn(2, 3, 256)
    batch = {
        "flux": flux,
        "wavelength_mask": torch.ones_like(flux),
        "visit_mask": torch.ones(2, 3),
        "observer_days": torch.tensor([[0.0, 10.0, 20.0], [0.0, 8.0, 30.0]]),
    }
    with torch.no_grad():
        baseline_output = baseline(batch)
        dense_output = dense(batch)

    assert torch.equal(baseline_output["joint_support"], dense_output["joint_support"])
    assert torch.allclose(
        baseline_output["joint_logits"],
        dense_output["joint_logits"],
        atol=1.0e-7,
    )


def test_dual_dense_scan_adds_whole_and_continuum_detail_evidence(
    tmp_path: Path,
) -> None:
    model = _factored_model(tmp_path, dense_scan=True, dense_detail=True)
    flux = torch.randn(2, 3, 256)
    batch = {
        "flux": flux,
        "wavelength_mask": torch.ones_like(flux),
        "visit_mask": torch.ones(2, 3),
        "observer_days": torch.tensor([[0.0, 10.0, 20.0], [0.0, 8.0, 30.0]]),
    }

    output = model(batch)

    assert output["dense_detail_joint_logits"].shape == output["joint_logits"].shape
    assert output["dense_scan_overlap_fraction"].shape == (
        2,
        len(model.redshift_grid),
    )
    assert output["dense_detail_joint_logits"].abs().sum() > 0.0
    assert torch.allclose(
        output["spectral_joint_logits"],
        output["onir_joint_logits"]
        + output["shape_joint_logits"]
        + output["dense_scan_joint_logits"],
        atol=1.0e-7,
    )
    assert torch.allclose(
        output["dense_scan_joint_logits"],
        output["dense_whole_contribution"]
        + output["dense_detail_contribution"],
        atol=1.0e-7,
    )


def test_dual_dense_scan_starts_at_the_baseline_prediction(tmp_path: Path) -> None:
    baseline = _factored_model(tmp_path)
    dual = _factored_model(
        tmp_path,
        dense_scan=True,
        dense_detail=True,
        dense_scale=0.0,
    )
    checkpoint_path = tmp_path / "dual_baseline.pt"
    torch.save(
        {
            "model_state": baseline.state_dict(),
            "classes": list(dual.class_names),
            "redshift_grid": dual.redshift_grid,
        },
        checkpoint_path,
    )
    config = load_config(ROOT / "configs/experiments/encoded_onir_named_clean.yaml")
    config["model"].update(
        {
            "architecture": "factored_onir",
            "temporal_mode": "spectral_evolution",
            "dense_rest_frame_scan": True,
            "dense_continuum_detail": True,
        }
    )
    _load_initial_checkpoint(dual, config, str(checkpoint_path))
    flux = torch.randn(2, 3, 256)
    batch = {
        "flux": flux,
        "wavelength_mask": torch.ones_like(flux),
        "visit_mask": torch.ones(2, 3),
        "observer_days": torch.tensor([[0.0, 10.0, 20.0], [0.0, 8.0, 30.0]]),
    }

    with torch.no_grad():
        baseline_output = baseline(batch)
        dual_output = dual(batch)

    assert torch.allclose(
        baseline_output["joint_logits"], dual_output["joint_logits"], atol=1.0e-7
    )


def test_initial_checkpoint_can_drop_the_flat_context_route(tmp_path: Path) -> None:
    baseline = _factored_model(tmp_path, full_spectrum_context=True)
    candidate = _factored_model(tmp_path, dense_scan=True, dense_detail=True)
    checkpoint_path = tmp_path / "context_baseline.pt"
    torch.save(
        {
            "model_state": baseline.state_dict(),
            "classes": list(candidate.class_names),
            "redshift_grid": candidate.redshift_grid,
        },
        checkpoint_path,
    )
    config = load_config(ROOT / "configs/experiments/encoded_onir_named_clean.yaml")
    config["model"].update(
        {
            "architecture": "factored_onir",
            "temporal_mode": "spectral_evolution",
            "full_spectrum_context": False,
            "dense_rest_frame_scan": True,
            "dense_continuum_detail": True,
        }
    )

    _load_initial_checkpoint(candidate, config, str(checkpoint_path))


def test_whole_detail_and_relative_brightness_remove_absolute_object_gain(
    tmp_path: Path,
) -> None:
    model = _factored_model(
        tmp_path,
        temporal_scale=0.5,
        dense_scan=True,
        dense_detail=True,
        relative_brightness=True,
        brightness_scale=0.5,
        input_normalization="per_visit",
    )
    flux = torch.randn(2, 3, 256)
    batch = {
        "flux": flux,
        "wavelength_mask": torch.ones_like(flux),
        "visit_mask": torch.ones(2, 3),
        "observer_days": torch.tensor([[0.0, 10.0, 20.0], [0.0, 8.0, 30.0]]),
    }
    brighter = {**batch, "flux": 25.0 * flux}

    with torch.no_grad():
        reference = model(batch)
        scaled = model(brighter)

    assert torch.allclose(
        reference["joint_logits"], scaled["joint_logits"], atol=2.0e-5
    )


def test_relative_brightness_can_change_only_temporal_evidence(tmp_path: Path) -> None:
    model = _factored_model(
        tmp_path,
        temporal_scale=0.5,
        relative_brightness=True,
        brightness_scale=0.5,
        input_normalization="per_visit",
    )
    base = torch.randn(1, 1, 256).repeat(1, 3, 1)
    changed = base * torch.tensor([1.0, 2.0, 0.5])[None, :, None]
    common = {
        "wavelength_mask": torch.ones_like(base),
        "visit_mask": torch.ones(1, 3),
        "observer_days": torch.tensor([[0.0, 10.0, 20.0]]),
    }

    with torch.no_grad():
        reference = model({**common, "flux": base})
        evolved = model({**common, "flux": changed})

    assert torch.allclose(
        reference["spectral_joint_logits"],
        evolved["spectral_joint_logits"],
        atol=2.0e-5,
    )
    assert not torch.allclose(
        reference["temporal_joint_logits"], evolved["temporal_joint_logits"]
    )


def test_factored_onir_dates_act_only_through_measured_change(tmp_path: Path) -> None:
    model = _factored_model(tmp_path, temporal_scale=1.0)
    model.eval()
    assert model.factored_evidence.time_pool.norm.elementwise_affine is False
    one_visit = torch.randn(1, 1, 256)
    flux = one_visit.expand(-1, 3, -1).clone()
    batch = {
        "flux": flux,
        "wavelength_mask": torch.ones_like(flux),
        "visit_mask": torch.ones(1, 3),
        "observer_days": torch.tensor([[0.0, 10.0, 40.0]]),
    }
    changed_dates = dict(batch)
    changed_dates["observer_days"] = torch.tensor([[0.0, 2.0, 100.0]])

    first = model(batch)
    second = model(changed_dates)

    assert torch.allclose(
        first["raw_temporal_joint_logits"],
        torch.zeros_like(first["raw_temporal_joint_logits"]),
        atol=1e-7,
    )
    assert torch.equal(first["joint_logits"], second["joint_logits"])


def test_factored_onir_uses_intervals_not_absolute_dates(tmp_path: Path) -> None:
    model = _factored_model(tmp_path, temporal_scale=1.0)
    flux = torch.randn(1, 3, 256)
    batch = {
        "flux": flux,
        "wavelength_mask": torch.ones_like(flux),
        "visit_mask": torch.ones(1, 3),
        "observer_days": torch.tensor([[0.0, 10.0, 40.0]]),
    }
    shifted = dict(batch)
    shifted["observer_days"] = batch["observer_days"] + 62000.0

    first = model(batch)
    second = model(shifted)

    assert torch.allclose(first["joint_logits"], second["joint_logits"], atol=1e-6)
    assert torch.allclose(
        first["raw_temporal_joint_logits"],
        second["raw_temporal_joint_logits"],
        atol=1e-6,
    )
