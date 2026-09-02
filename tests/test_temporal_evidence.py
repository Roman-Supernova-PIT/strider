from pathlib import Path

import pytest
import torch

from strider.config import load_config
from strider.evaluation.evaluate import _transform_times
from strider.evaluation.time_controls import _dataset_time_reassignment
from strider.model import Strider
from strider.model.temporal import SpectralEvolutionEvidence, _bounded_change


ROOT = Path(__file__).resolve().parents[1]


def test_dates_cannot_create_evidence_without_spectral_change() -> None:
    module = SpectralEvolutionEvidence(8, 2, dropout=0.0, initial_scale=1.0).eval()
    spectral = torch.ones(3, 4, 5, 8)
    dates = torch.tensor(
        [[-20.0, -5.0, 8.0, 30.0], [-50.0, 0.0, 1.0, 90.0], [1.0, 2.0, 3.0, 4.0]]
    )
    mask = torch.ones(3, 4)
    redshift = torch.linspace(0.1, 2.5, 5)
    scaled, raw = module(spectral, dates, mask, redshift)
    assert torch.equal(raw, torch.zeros_like(raw))
    assert torch.equal(scaled, torch.zeros_like(scaled))


def test_small_spectral_changes_remain_small() -> None:
    change = torch.linspace(-1.0, 1.0, 8).reshape(1, 1, 1, 8)
    full = _bounded_change(change)
    small = _bounded_change(change * 1.0e-4)
    assert small.abs().max() < full.abs().max() * 2.0e-4


def test_zero_spectral_change_has_finite_gradient() -> None:
    change = torch.zeros(2, 3, 4, 8, requires_grad=True)
    _bounded_change(change).sum().backward()
    assert torch.isfinite(change.grad).all()


def test_one_visit_has_no_temporal_evidence() -> None:
    module = SpectralEvolutionEvidence(8, 2, dropout=0.0, initial_scale=1.0).eval()
    spectral = torch.randn(2, 1, 5, 8)
    scaled, raw = module(
        spectral,
        torch.zeros(2, 1),
        torch.ones(2, 1),
        torch.linspace(0.1, 2.5, 5),
    )
    assert torch.equal(raw, torch.zeros_like(raw))
    assert torch.equal(scaled, torch.zeros_like(scaled))


def test_new_model_starts_from_phase_neutral_spectral_result() -> None:
    config = load_config(ROOT / "configs/experiments/spectral_evolution.yaml")
    model = Strider(config).eval()
    flux = torch.randn(2, 3, config["observation"]["wavelength_bins"])
    batch = {
        "flux": flux,
        "wavelength_mask": torch.ones_like(flux),
        "visit_mask": torch.ones(2, 3),
        "observer_days": torch.tensor([[-20.0, 0.0, 25.0], [-8.0, 3.0, 40.0]]),
    }
    output = model(batch)
    assert torch.equal(output["joint_logits"], output["spectral_joint_logits"])
    assert torch.equal(output["temporal_joint_logits"], torch.zeros_like(output["joint_logits"]))


def test_spectral_result_is_independent_of_dates() -> None:
    config = load_config(ROOT / "configs/experiments/spectral_evolution.yaml")
    model = Strider(config).eval()
    flux = torch.randn(2, 3, config["observation"]["wavelength_bins"])
    batch = {
        "flux": flux,
        "wavelength_mask": torch.ones_like(flux),
        "visit_mask": torch.ones(2, 3),
        "observer_days": torch.tensor([[-20.0, 0.0, 25.0], [-8.0, 3.0, 40.0]]),
    }
    changed = dict(batch)
    changed["observer_days"] = torch.tensor([[0.0, 1.0, 400.0], [-100.0, 50.0, 51.0]])
    first = model(batch)
    second = model(changed)
    assert torch.equal(first["spectral_joint_logits"], second["spectral_joint_logits"])


def test_reassigned_dates_exchange_intervals_only_within_matched_groups() -> None:
    batch = {
        "observer_days": torch.tensor(
            [[0.0, 5.0, 20.0], [0.0, 12.0, 31.0], [0.0, 40.0, 0.0]]
        ),
        "visit_mask": torch.tensor(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]
        ),
        "redshift": torch.tensor([1.01, 1.08, 1.06]),
    }
    batch["snid"] = torch.tensor([101, 102, 103])
    reassignment = {
        101: batch["observer_days"][1, :3].clone(),
        102: batch["observer_days"][0, :3].clone(),
    }
    changed = _transform_times(
        batch,
        "reassigned_within_redshift",
        reassignment,
    )
    assert torch.equal(changed["observer_days"][0, :3], batch["observer_days"][1, :3])
    assert torch.equal(changed["observer_days"][1, :3], batch["observer_days"][0, :3])
    # An object without a dataset-wide match is left unchanged.
    assert torch.equal(changed["observer_days"][2], batch["observer_days"][2])


def test_reassigned_dates_require_a_dataset_wide_mapping() -> None:
    batch = {
        "observer_days": torch.tensor([[0.0, 5.0], [0.0, 12.0]]),
        "visit_mask": torch.ones(2, 2),
        "snid": torch.tensor([101, 102]),
    }
    with pytest.raises(ValueError, match="full dataset"):
        _transform_times(batch, "reassigned_within_redshift")


def test_dataset_time_reassignment_matches_redshift_and_visit_count() -> None:
    dataset = [
        {
            "snid": torch.tensor(101),
            "redshift": torch.tensor(1.01),
            "observer_days": torch.tensor([0.0, 5.0, 20.0]),
        },
        {
            "snid": torch.tensor(102),
            "redshift": torch.tensor(1.08),
            "observer_days": torch.tensor([0.0, 12.0, 31.0]),
        },
        {
            "snid": torch.tensor(103),
            "redshift": torch.tensor(1.06),
            "observer_days": torch.tensor([0.0, 40.0]),
        },
    ]
    reassignment = _dataset_time_reassignment(dataset)

    assert torch.equal(reassignment[101], dataset[1]["observer_days"])
    assert torch.equal(reassignment[102], dataset[0]["observer_days"])
    assert 103 not in reassignment


def test_reverse_control_reverses_spectra_but_keeps_observer_dates() -> None:
    flux = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    wavelength_mask = torch.ones_like(flux)
    wavelength_mask[0, 1, 0] = 0.0
    batch = {
        "flux": flux,
        "wavelength_mask": wavelength_mask,
        "observer_days": torch.tensor(
            [[0.0, 5.0, 20.0, 0.0], [0.0, 12.0, 31.0, 50.0]]
        ),
        "visit_mask": torch.tensor(
            [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
        ),
    }

    changed = _transform_times(batch, "reverse_within_object")

    assert torch.equal(changed["observer_days"], batch["observer_days"])
    assert torch.equal(changed["flux"][0, :3], torch.flip(flux[0, :3], dims=(0,)))
    assert torch.equal(
        changed["wavelength_mask"][0, :3],
        torch.flip(wavelength_mask[0, :3], dims=(0,)),
    )
    assert torch.equal(changed["flux"][0, 3], flux[0, 3])
