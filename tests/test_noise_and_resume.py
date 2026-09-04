"""Contracts for controlled native-bin noise and exact continuation."""

from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch

from strider.data.noise import (
    draw_controlled_background_flux,
    draw_controlled_background_measurement,
    repeatable_rng,
)
from strider.training.resume import (
    atomic_torch_save,
    capture_random_state,
    load_training_state,
    restore_random_state,
)
from strider.training.trainer import (
    _freeze_spectral_model,
    _load_initial_checkpoint,
    _set_training_mode,
)
from strider.config import load_config
from strider.model import Strider


ROOT = Path(__file__).resolve().parents[1]


def test_controlled_noise_is_repeatable_and_drawn_at_native_shape() -> None:
    wavelength = np.linspace(7500.0, 20000.0, 197, dtype=np.float32)
    clean = np.linspace(-0.2, 0.3, 197, dtype=np.float32)
    settings = {
        "background_shape_strength": 0.2,
        "background_scale_log_std": 0.1,
        "source_variance_fraction": 0.0,
    }
    first = draw_controlled_background_flux(
        wavelength, clean, 2.0, True, settings, repeatable_rng(7, 19, 2)
    )
    second = draw_controlled_background_flux(
        wavelength, clean, 2.0, True, settings, repeatable_rng(7, 19, 2)
    )
    changed = draw_controlled_background_flux(
        wavelength, clean, 2.0, True, settings, repeatable_rng(7, 19, 3)
    )
    assert first.shape == wavelength.shape
    assert np.array_equal(first, second)
    assert not np.array_equal(first, changed)


def test_zero_noise_scale_returns_only_the_requested_signal() -> None:
    wavelength = np.linspace(7500.0, 20000.0, 31, dtype=np.float32)
    clean = np.linspace(-0.2, 0.3, 31, dtype=np.float32)
    settings = {
        "background_shape_strength": 0.2,
        "background_scale_log_std": 0.1,
        "source_variance_fraction": 0.0,
    }
    source = draw_controlled_background_flux(
        wavelength,
        clean,
        2.0,
        True,
        settings,
        repeatable_rng(4, 7),
        noise_scale=0.0,
    )
    blank = draw_controlled_background_flux(
        wavelength,
        clean,
        2.0,
        False,
        settings,
        repeatable_rng(4, 7),
        noise_scale=0.0,
    )
    assert np.array_equal(source, clean)
    assert np.count_nonzero(blank) == 0


def test_controlled_measurement_returns_the_error_used_for_the_draw() -> None:
    wavelength = np.linspace(7500.0, 20000.0, 31, dtype=np.float32)
    clean = np.linspace(-0.2, 0.3, 31, dtype=np.float32)
    settings = {
        "background_shape_strength": 0.2,
        "background_scale_log_std": 0.1,
        "source_variance_fraction": 0.0,
    }
    flux, error = draw_controlled_background_measurement(
        wavelength,
        clean,
        2.0,
        True,
        settings,
        repeatable_rng(8, 3),
        noise_scale=1.5,
    )
    legacy_flux = draw_controlled_background_flux(
        wavelength,
        clean,
        2.0,
        True,
        settings,
        repeatable_rng(8, 3),
        noise_scale=1.5,
    )

    assert np.array_equal(flux, legacy_flux)
    assert error.shape == flux.shape
    assert np.isfinite(error).all()
    assert (error > 0.0).all()


def test_random_state_restores_the_next_training_draws() -> None:
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    state = capture_random_state()
    expected = (
        random.random(),
        np.random.normal(size=5),
        torch.rand(5),
        torch.randperm(13),
    )
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    restore_random_state(state)
    actual = (
        random.random(),
        np.random.normal(size=5),
        torch.rand(5),
        torch.randperm(13),
    )
    assert actual[0] == expected[0]
    assert np.array_equal(actual[1], expected[1])
    assert torch.equal(actual[2], expected[2])
    assert torch.equal(actual[3], expected[3])


def test_optimizer_continuation_repeats_the_next_update() -> None:
    torch.manual_seed(23)
    uninterrupted = torch.nn.Linear(4, 2)
    uninterrupted_optimizer = torch.optim.AdamW(uninterrupted.parameters(), lr=1e-2)
    _random_training_step(uninterrupted, uninterrupted_optimizer)
    saved_model = {
        name: value.detach().clone() for name, value in uninterrupted.state_dict().items()
    }
    saved_optimizer = copy.deepcopy(uninterrupted_optimizer.state_dict())
    saved_random = capture_random_state()
    _random_training_step(uninterrupted, uninterrupted_optimizer)

    resumed = torch.nn.Linear(4, 2)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-2)
    resumed.load_state_dict(saved_model)
    resumed_optimizer.load_state_dict(saved_optimizer)
    restore_random_state(saved_random)
    _random_training_step(resumed, resumed_optimizer)

    for expected, actual in zip(uninterrupted.parameters(), resumed.parameters()):
        assert torch.equal(expected, actual)


def test_training_state_is_loaded_on_cpu(monkeypatch, tmp_path: Path) -> None:
    received = {}

    def fake_load(path, *, map_location, weights_only):
        received.update(
            path=path,
            map_location=map_location,
            weights_only=weights_only,
        )
        return {"epoch": 4}

    monkeypatch.setattr(torch, "load", fake_load)
    path = tmp_path / "training_state.pt"
    assert load_training_state(path) == {"epoch": 4}
    assert received == {
        "path": path,
        "map_location": "cpu",
        "weights_only": False,
    }


def test_atomic_checkpoint_replaces_the_complete_file(tmp_path: Path) -> None:
    path = tmp_path / "training_state.pt"
    random_state = capture_random_state()
    assert random_state["numpy"]["keys"].dtype == torch.int64
    atomic_torch_save({"epoch": 8, "random": random_state}, path)
    loaded = load_training_state(path)
    assert loaded["epoch"] == 8
    assert loaded["random"]["torch_cpu"].device.type == "cpu"
    assert not path.with_suffix(".pt.tmp").exists()


def test_temporal_only_training_freezes_the_spectral_model() -> None:
    config = load_config(ROOT / "configs/experiments/spectral_evolution.yaml")
    model = Strider(config)
    _freeze_spectral_model(model)
    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    assert trainable
    assert all(name.startswith("temporal_evidence.") for name in trainable)


def test_temporal_only_training_keeps_the_spectral_model_fixed() -> None:
    config = load_config(ROOT / "configs/experiments/spectral_evolution.yaml")
    model = Strider(config)
    _freeze_spectral_model(model)
    _set_training_mode(model, True)
    assert model.temporal_evidence.training
    assert not model.spectral_encoder.training
    assert not model.visit_score.training
    assert not model.evidence_sufficiency.training


def test_spectral_checkpoint_can_initialize_a_temporal_run(tmp_path: Path) -> None:
    temporal_config = load_config(ROOT / "configs/experiments/spectral_evolution.yaml")
    spectral_config = copy.deepcopy(temporal_config)
    spectral_config["model"]["temporal_mode"] = "none"
    spectral_model = Strider(spectral_config)
    checkpoint = tmp_path / "spectral.pt"
    torch.save(
        {
            "model_state": spectral_model.state_dict(),
            "classes": list(spectral_config["model"]["classes"]),
            "redshift_grid": spectral_model.redshift_grid,
        },
        checkpoint,
    )
    temporal_model = Strider(temporal_config)
    _load_initial_checkpoint(temporal_model, temporal_config, str(checkpoint))
    for name, value in spectral_model.state_dict().items():
        assert torch.equal(value, temporal_model.state_dict()[name])


def _random_training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    inputs = torch.randn(7, 4)
    target = torch.randn(7, 2)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(inputs), target)
    loss.backward()
    optimizer.step()
