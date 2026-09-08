import torch

from strider.model.spectral_context import FullSpectrumContext


def _context() -> FullSpectrumContext:
    return FullSpectrumContext(
        wavelength_bins=32,
        token_dim=16,
        class_count=3,
        patch_size=4,
        attention_heads=4,
        attention_layers=1,
        dropout=0.0,
        initial_scale=0.25,
        input_normalization="per_visit",
        use_visit_attention=True,
        minimum_support=0.5,
    ).eval()


def test_full_spectrum_context_ignores_masked_flux() -> None:
    model = _context()
    flux = torch.randn(2, 3, 32)
    mask = torch.ones_like(flux)
    mask[..., 8:16] = 0.0
    changed = flux.clone()
    changed[..., 8:16] = 1.0e6
    visit_mask = torch.ones(2, 3)

    first = model(flux * mask, mask, visit_mask)
    second = model(changed, mask, visit_mask)

    assert torch.equal(
        first["scaled_context_class_logits"],
        second["scaled_context_class_logits"],
    )


def test_full_spectrum_context_returns_no_evidence_without_measurements() -> None:
    model = _context()
    flux = torch.randn(2, 3, 32)
    mask = torch.zeros_like(flux)
    visit_mask = torch.zeros(2, 3)

    output = model(flux, mask, visit_mask)

    assert torch.equal(
        output["scaled_context_class_logits"],
        torch.zeros(2, 3),
    )


def test_full_spectrum_context_is_invariant_to_positive_flux_scale() -> None:
    model = _context()
    flux = torch.randn(2, 3, 32)
    mask = torch.ones_like(flux)
    visit_mask = torch.ones(2, 3)

    reference = model(flux, mask, visit_mask)["scaled_context_class_logits"]
    brighter = model(12.0 * flux, mask, visit_mask)["scaled_context_class_logits"]

    assert torch.allclose(reference, brighter, atol=2e-6, rtol=2e-6)


def test_full_spectrum_context_does_not_read_visit_order() -> None:
    model = _context()
    flux = torch.randn(2, 4, 32)
    mask = torch.ones_like(flux)
    visit_mask = torch.ones(2, 4)
    order = torch.tensor([2, 0, 3, 1])

    reference = model(flux, mask, visit_mask)["scaled_context_class_logits"]
    reordered = model(
        flux[:, order],
        mask[:, order],
        visit_mask[:, order],
    )["scaled_context_class_logits"]

    assert torch.allclose(reference, reordered, atol=2e-6, rtol=2e-6)


def test_full_spectrum_context_ignores_padded_visits() -> None:
    model = _context()
    flux = torch.randn(2, 3, 32)
    mask = torch.ones_like(flux)
    visit_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    changed = flux.clone()
    changed[0, 2] = 1.0e6

    reference = model(flux, mask, visit_mask)["scaled_context_class_logits"]
    padded = model(changed, mask, visit_mask)["scaled_context_class_logits"]

    assert torch.equal(reference[0], padded[0])
