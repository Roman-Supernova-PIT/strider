from pathlib import Path

import numpy as np

from strider.current_io import ObservedSeriesInput
from strider.current_plotting import save_current_evidence_map


def test_current_evidence_map_writes_a_complete_figure(tmp_path: Path) -> None:
    wavelength = np.geomspace(7500.0, 18175.0, 64)
    coordinate = np.linspace(0.0, 3.0 * np.pi, len(wavelength))
    data = ObservedSeriesInput(
        wavelength=wavelength,
        flux=np.stack([np.sin(coordinate), 1.5 * np.sin(coordinate + 0.2)]),
        flux_error=np.full((2, len(wavelength)), 0.7),
        observer_time=np.asarray([62000.0, 62012.0]),
        metadata={"object": "roman-1"},
    )
    grid = np.asarray([0.1, 0.3, 0.5, 0.7, 0.9])
    joint = np.asarray(
        [
            [0.01, 0.04, 0.20, 0.40, 0.15],
            [0.03, 0.05, 0.05, 0.04, 0.03],
        ]
    )
    result = {
        "model": {"classes": ["Ia", "other"]},
        "classification": {
            "class": "Ia",
            "confidence": 0.8,
            "probabilities": {"Ia": 0.8, "other": 0.2},
        },
        "redshift": {
            "grid": grid.tolist(),
            "z_STRIDER": 0.7,
            "primary_basin": {
                "peak_redshift": 0.7,
                "lower_68": 0.6,
                "upper_68": 0.8,
            },
            "candidate_basins": [
                {"peak_redshift": 0.7, "mass": 0.8},
                {"peak_redshift": 0.3, "mass": 0.2},
            ],
        },
        "signal": {"grade": "high"},
        "joint_probability": (joint / joint.sum()).tolist(),
    }
    output = tmp_path / "evidence.png"

    save_current_evidence_map(result, data, output, object_id="roman-1")

    assert output.is_file()
    assert output.stat().st_size > 10_000
