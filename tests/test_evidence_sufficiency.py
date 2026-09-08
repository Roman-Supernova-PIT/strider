import torch

from strider.model.evidence_sufficiency import observation_summary_features


def test_evidence_sufficiency_features_retain_background_scaled_amplitude() -> None:
    batch = {
        "flux": torch.tensor([[[0.0, 1.0, 2.0, 1.0], [0.0, 1.5, 2.5, 1.0]]]),
        "wavelength_mask": torch.ones(1, 2, 4),
        "visit_mask": torch.ones(1, 2),
        "observer_days": torch.tensor([[0.0, 10.0]]),
    }
    stronger = dict(batch)
    stronger["flux"] = 5.0 * batch["flux"]

    reference = observation_summary_features(batch, visit_count_reference=2)
    scaled = observation_summary_features(stronger, visit_count_reference=2)

    # The shape matcher may standardize a visit, but this separate route must
    # retain the measured amplitude relative to the background scale.
    assert torch.all(scaled[:, 1:5] > reference[:, 1:5])
    assert torch.equal(scaled[:, 6:], reference[:, 6:])


def test_spectral_strength_features_do_not_read_observation_schedule() -> None:
    batch = {
        "flux": torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]),
        "wavelength_mask": torch.ones(1, 2, 2),
        "visit_mask": torch.ones(1, 2),
        "observer_days": torch.tensor([[0.0, 10.0]]),
    }
    shifted = dict(batch)
    shifted["observer_days"] = torch.tensor([[0.0, 80.0]])

    reference = observation_summary_features(
        batch,
        visit_count_reference=32,
        use_visit_count_and_span=False,
    )
    changed = observation_summary_features(
        shifted,
        visit_count_reference=32,
        use_visit_count_and_span=False,
    )

    assert reference.shape == (1, 6)
    assert torch.equal(reference, changed)
