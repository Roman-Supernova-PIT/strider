from pathlib import Path

import numpy as np
import pytest

from strider import load_model
from strider.engine.model.redshift_scan import redshift_cell_widths
from strider.engine.posterior_summary import posterior_basin_candidates


def test_load_model_dispatches_model_package_directories(
    tmp_path: Path, monkeypatch
) -> None:
    package = tmp_path / "model-package"
    package.mkdir()
    sentinel = object()
    calls = []

    def fake_load(path: Path, *, device: str):
        calls.append((path, device))
        return sentinel

    monkeypatch.setattr("strider.engine.deployment.load_model_package", fake_load)

    assert load_model(package, device="cpu") is sentinel
    assert calls == [(package, "cpu")]


def test_current_redshift_summary_preserves_two_distinct_basins() -> None:
    grid = np.linspace(0.0, 1.0, 101)
    density = np.exp(-0.5 * ((grid - 0.25) / 0.035) ** 2)
    density += 0.65 * np.exp(-0.5 * ((grid - 0.78) / 0.045) ** 2)
    widths = redshift_cell_widths(grid)
    mass = density * widths
    mass /= mass.sum()

    candidates = posterior_basin_candidates(grid, mass, widths)

    assert len(candidates) == 2
    assert candidates[0]["peak_redshift"] == pytest.approx(0.25, abs=0.01)
    assert candidates[1]["peak_redshift"] == pytest.approx(0.78, abs=0.01)
