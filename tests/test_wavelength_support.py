from pathlib import Path

import pandas as pd

from strider.data.template_support import template_support_policy
from strider.data.wavelength_support import wavelength_support_report


def _config(tmp_path: Path) -> dict:
    return {
        "_project_root": str(tmp_path),
        "data": {"prepared_dir": "prepared"},
        "observation": {
            "wavelength_min": 7500.0,
            "wavelength_max": 18175.0,
            "template_support_policy": "retain",
        },
    }


def _write_tables(tmp_path: Path, starts: list[float], ends: list[float]) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    count = len(starts)
    pd.DataFrame(
        {
            "object_index": range(count),
            "split": ["train"] * count,
            "class_name": ["Ia" if index < count // 2 else "other" for index in range(count)],
            "redshift": [0.2 if index < count // 2 else 1.8 for index in range(count)],
        }
    ).to_parquet(prepared / "objects.parquet", index=False)
    pd.DataFrame(
        {
            "object_index": range(count),
            "native_wavelength_min": starts,
            "native_wavelength_max": ends,
        }
    ).to_parquet(prepared / "observations.parquet", index=False)


def test_constant_template_support_has_no_label_association(tmp_path: Path) -> None:
    _write_tables(tmp_path, [7400.0] * 20, [18300.0] * 20)
    report = wavelength_support_report(_config(tmp_path))
    assert report["passed"]
    assert report["splits"]["train"]["support_signatures"] == 1


def test_template_endpoint_that_tracks_labels_fails(tmp_path: Path) -> None:
    starts = [7500.0] * 10 + [9000.0] * 10
    _write_tables(tmp_path, starts, [18175.0] * 20)
    report = wavelength_support_report(_config(tmp_path))
    assert not report["passed"]
    assert report["splits"]["train"]["redshift_explained_fraction"] == 1.0
    assert report["splits"]["train"]["ia_explained_fraction"] == 1.0


def test_previous_complete_setting_remains_readable() -> None:
    assert template_support_policy({"require_complete_wavelength_coverage": True}) == "complete"


def test_conflicting_template_support_settings_are_rejected() -> None:
    try:
        template_support_policy(
            {
                "template_support_policy": "retain",
                "require_complete_wavelength_coverage": True,
            }
        )
    except ValueError as error:
        assert "conflicts" in str(error)
    else:
        raise AssertionError("Conflicting template support settings were accepted")
