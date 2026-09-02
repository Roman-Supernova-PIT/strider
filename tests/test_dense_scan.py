from __future__ import annotations

import numpy as np
import torch

from strider.model.dense_scan import (
    DenseRestFrameEvidence,
    WholeDetailRestFrameEvidence,
)
from strider.model.spectral_tokens import (
    MaskAwareContinuumRemoval,
    object_normalized_flux_amplitude,
    relative_visit_amplitude,
    velocity_sigma_to_log_bins,
)


def _pattern(wavelength: np.ndarray) -> np.ndarray:
    position = np.log(wavelength / 2500.0) / np.log(4.0)
    angle = 2.0 * np.pi * (7.0 * position + 0.4 * np.sin(2.0 * np.pi * position))
    return np.stack((np.sin(angle), np.cos(angle)), axis=-1).astype(np.float32)


def _scan() -> tuple[DenseRestFrameEvidence, np.ndarray, np.ndarray]:
    observed = np.geomspace(7500.0, 18175.0, 128)
    rest = np.geomspace(2500.0, 10000.0, 96)
    redshift = np.asarray([0.4, 1.0, 2.0], dtype=np.float32)
    model = DenseRestFrameEvidence(
        observed_wavelength=observed,
        rest_wavelength=rest,
        redshift_grid=redshift,
        hidden_dim=2,
        token_dim=2,
        class_count=1,
        patch_size=1,
        rest_bins=len(rest),
        initial_scale=1.0,
        evidence_scale=10.0,
        redshift_chunk_size=2,
        minimum_overlap=0.2,
        overlap_exponent=0.0,
    ).eval()
    with torch.no_grad():
        model.projection.weight.copy_(torch.eye(2))
        model.templates[0].copy_(torch.from_numpy(_pattern(rest)))
    return model, observed, redshift


def test_dense_scan_recovers_a_shifted_whole_spectrum() -> None:
    model, observed, redshift = _scan()
    true_redshift = 1.0
    tokens = torch.from_numpy(_pattern(observed / (1.0 + true_redshift)))[
        None, None
    ]
    mask = torch.ones(1, 1, len(observed))

    output = model(tokens, mask, torch.ones(1, 1))

    best = int(output["raw_dense_scan_joint_logits"][0, 0].argmax())
    assert redshift[best] == true_redshift
    assert output["dense_scan_overlap_fraction"].shape == (1, len(redshift))
    assert output["dense_scan_support"].all()


def test_dense_scan_ignores_masked_tokens() -> None:
    model, observed, _ = _scan()
    tokens = torch.from_numpy(_pattern(observed / 2.0))[None, None]
    mask = torch.ones(1, 1, len(observed))
    mask[..., :32] = 0.0
    changed = tokens.clone()
    changed[..., :32, :] = 1.0e6

    first = model(tokens, mask, torch.ones(1, 1))
    second = model(changed, mask, torch.ones(1, 1))

    assert torch.equal(
        first["raw_dense_scan_joint_logits"],
        second["raw_dense_scan_joint_logits"],
    )


def test_dense_scan_adds_no_evidence_without_measurements() -> None:
    model, observed, redshift = _scan()
    output = model(
        torch.randn(2, 3, len(observed), 2),
        torch.zeros(2, 3, len(observed)),
        torch.zeros(2, 3),
    )

    assert torch.equal(
        output["dense_scan_joint_logits"],
        torch.zeros(2, 1, len(redshift)),
    )
    assert not output["dense_scan_support"].any()


def test_continuum_removal_is_mask_aware() -> None:
    removal = MaskAwareContinuumRemoval(sigma_bins=4.0)
    mask = torch.ones(1, 2, 64)
    mask[..., 20:30] = 0.0
    first = torch.ones_like(mask)
    second = first.clone()
    second[..., 20:30] = 1.0e6

    first_output = removal(first, mask)
    second_output = removal(second, mask)

    assert torch.allclose(first_output, torch.zeros_like(first_output), atol=1.0e-6)
    assert torch.equal(first_output, second_output)


def test_continuum_removal_keeps_narrow_structure() -> None:
    removal = MaskAwareContinuumRemoval(sigma_bins=12.0)
    position = torch.linspace(0.0, 1.0, 256)
    continuum = 2.0 + 0.5 * torch.sin(2.0 * torch.pi * position)
    narrow = torch.exp(-0.5 * ((position - 0.5) / 0.006).square())
    mask = torch.ones(1, 1, len(position))

    broad_only = removal(continuum[None, None], mask)
    with_line = removal((continuum + narrow)[None, None], mask)

    assert with_line[..., 128].item() > 0.5
    assert broad_only[..., 128].abs().item() < with_line[..., 128].item()


def test_whole_detail_scan_uses_one_shared_matcher() -> None:
    observed = np.geomspace(7500.0, 18175.0, 64)
    rest = np.geomspace(2500.0, 10000.0, 48)
    redshift = np.asarray([0.5, 1.0], dtype=np.float32)
    model = WholeDetailRestFrameEvidence(
        observed_wavelength=observed,
        rest_wavelength=rest,
        redshift_grid=redshift,
        hidden_dim=4,
        token_dim=4,
        class_count=2,
        patch_size=2,
        rest_bins=48,
        initial_scale=0.5,
        evidence_scale=10.0,
        redshift_chunk_size=2,
        minimum_overlap=0.2,
        overlap_exponent=1.0,
    )

    tokens = torch.randn(2, 3, len(observed), 4)
    mask = torch.ones(2, 3, len(observed))
    output = model(tokens, mask, tokens, mask, torch.ones(2, 3))

    assert output["dense_scan_joint_logits"].shape == (2, 2, 2)
    assert torch.allclose(
        output["dense_scan_joint_logits"],
        output["dense_whole_contribution"]
        + output["dense_detail_contribution"],
        atol=1.0e-7,
    )
    assert list(name for name, _ in model.named_modules()).count("matcher") == 1


def test_whole_detail_scan_can_preserve_whole_spectrum_evidence() -> None:
    observed = np.geomspace(7500.0, 18175.0, 64)
    rest = np.geomspace(2500.0, 10000.0, 48)
    model = WholeDetailRestFrameEvidence(
        observed_wavelength=observed,
        rest_wavelength=rest,
        redshift_grid=np.asarray([0.5, 1.0], dtype=np.float32),
        hidden_dim=4,
        token_dim=4,
        class_count=2,
        patch_size=2,
        rest_bins=48,
        initial_scale=0.5,
        evidence_scale=10.0,
        redshift_chunk_size=2,
        minimum_overlap=0.2,
        overlap_exponent=1.0,
        initial_detail_weight=0.25,
        minimum_whole_weight=0.5,
    )

    output = model(
        torch.randn(2, 3, len(observed), 4),
        torch.ones(2, 3, len(observed)),
        torch.randn(2, 3, len(observed), 4),
        torch.ones(2, 3, len(observed)),
        torch.ones(2, 3),
    )

    assert torch.all(output["dense_detail_weight"] <= 0.5)
    assert torch.all(1.0 - output["dense_detail_weight"] >= 0.5)
    assert torch.allclose(
        output["dense_scan_joint_logits"],
        output["dense_whole_contribution"]
        + output["dense_detail_contribution"],
        atol=1.0e-7,
    )


def test_detail_only_scan_uses_one_continuum_match() -> None:
    observed = np.geomspace(7500.0, 18175.0, 64)
    rest = np.geomspace(2500.0, 10000.0, 48)
    model = WholeDetailRestFrameEvidence(
        observed_wavelength=observed,
        rest_wavelength=rest,
        redshift_grid=np.asarray([0.5, 1.0], dtype=np.float32),
        hidden_dim=4,
        token_dim=4,
        class_count=2,
        patch_size=2,
        rest_bins=48,
        initial_scale=0.5,
        evidence_scale=10.0,
        redshift_chunk_size=2,
        minimum_overlap=0.2,
        overlap_exponent=1.0,
        scan_view="detail",
    )
    calls = 0
    raw_evidence = model.matcher.raw_evidence

    def counted_raw_evidence(*args: torch.Tensor) -> dict[str, torch.Tensor]:
        nonlocal calls
        calls += 1
        return raw_evidence(*args)

    model.matcher.raw_evidence = counted_raw_evidence
    tokens = torch.randn(2, 3, len(observed), 4)
    mask = torch.ones(2, 3, len(observed))
    output = model(tokens, mask, tokens, mask, torch.ones(2, 3))

    assert calls == 1
    assert "dense_whole_joint_logits" not in output
    assert torch.equal(
        output["dense_scan_joint_logits"],
        output["dense_detail_joint_logits"],
    )
    assert torch.equal(
        output["dense_scan_joint_logits"],
        output["dense_detail_contribution"],
    )
    assert torch.equal(output["dense_detail_weight"], torch.ones(2))


def test_relative_visit_amplitude_removes_one_object_gain() -> None:
    flux = torch.tensor(
        [[[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [0.5, 1.0, 0.5]]]
    )
    mask = torch.ones_like(flux)
    visits = torch.ones(1, 3)

    reference = relative_visit_amplitude(flux, mask, visits)
    brighter = relative_visit_amplitude(20.0 * flux, mask, visits)

    assert torch.allclose(reference, brighter, atol=1.0e-6)
    assert torch.allclose(reference.mean(dim=1), torch.zeros(1), atol=1.0e-6)
    assert reference[0, 1] > reference[0, 0] > reference[0, 2]


def test_object_normalized_flux_amplitude_recovers_raw_visit_evolution() -> None:
    raw_flux = torch.tensor(
        [[[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [0.5, 1.0, 0.5]]]
    )
    visit_scale = torch.tensor([[2.0, 8.0, 0.25]])
    background_scaled = raw_flux / visit_scale[..., None]
    mask = torch.ones_like(raw_flux)
    visits = torch.ones(1, 3)
    wavelength_weight = torch.tensor([1.0, 2.0, 1.0])

    reference = object_normalized_flux_amplitude(
        raw_flux,
        mask,
        visits,
        torch.ones_like(visit_scale),
        wavelength_weight,
    )
    recovered = object_normalized_flux_amplitude(
        background_scaled,
        mask,
        visits,
        visit_scale,
        wavelength_weight,
    )
    brighter = object_normalized_flux_amplitude(
        20.0 * background_scaled,
        mask,
        visits,
        visit_scale,
        wavelength_weight,
    )

    assert torch.allclose(reference, recovered, atol=1.0e-6)
    assert torch.allclose(reference, brighter, atol=1.0e-6)
    assert reference[0, 1] > reference[0, 0] > reference[0, 2]


def test_object_normalized_flux_amplitude_softly_downweights_noisy_bins() -> None:
    flux = torch.ones(1, 2, 4)
    flux[0, 0, -1] = 1_000.0
    mask = torch.ones_like(flux)
    visits = torch.ones(1, 2)
    visit_scale = torch.ones(1, 2)
    wavelength_weight = torch.ones(4)
    reliability = torch.ones_like(flux)
    reliability[0, 0, -1] = 1.0e-6

    unweighted = object_normalized_flux_amplitude(
        flux,
        mask,
        visits,
        visit_scale,
        wavelength_weight,
    )
    weighted = object_normalized_flux_amplitude(
        flux,
        mask,
        visits,
        visit_scale,
        wavelength_weight,
        reliability,
    )

    assert unweighted[0, 0] > 1.4
    assert unweighted[0, 1] < 0.01
    assert torch.allclose(weighted[0], torch.ones(2), atol=5.0e-4)


def test_continuum_velocity_width_is_grid_independent() -> None:
    coarse = velocity_sigma_to_log_bins(7500.0, 18175.0, 256, 6000.0)
    fine = velocity_sigma_to_log_bins(7500.0, 18175.0, 1024, 6000.0)

    assert 5.0 < coarse < 6.5
    assert 22.0 < fine < 24.0
    assert np.isclose(fine / coarse, 1023.0 / 255.0)
