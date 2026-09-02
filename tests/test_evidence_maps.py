"""Readable output contracts for the STRIDER evidence figure."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from strider.evaluation.evidence_maps import (
    _manifest_indices,
    _representative_indices,
    _route_audit_rows,
    draw_evidence_map,
    draw_evidence_summary,
    evidence_grade,
)
from strider.evaluation.evidence_animation import (
    _animation_visit_counts,
    _basin_trajectory_summary,
    _basin_trajectory_row,
    _visit_prefix,
)


def test_evidence_map_shows_joint_region_and_marginal_results() -> None:
    wavelength = np.geomspace(7500.0, 20000.0, 12)
    item = {
        "flux": torch.randn(2, 12),
        "wavelength_mask": torch.ones(2, 12),
        "observer_days": torch.tensor([0.0, 18.0]),
        "snid": torch.tensor(71),
        "redshift": torch.tensor(1.2),
        "class_index": torch.tensor(0),
    }
    redshift = np.geomspace(1.05, 3.0, 7) - 1.0
    width = np.diff(np.r_[redshift[0], 0.5 * (redshift[:-1] + redshift[1:]), redshift[-1]])
    joint = np.arange(14, dtype=float).reshape(2, 7) + 1.0
    joint /= joint.sum()
    figure = plt.figure(figsize=(13, 13.5), constrained_layout=True)
    draw_evidence_map(
        figure,
        item=item,
        wavelength_angstrom=wavelength,
        class_names=["Ia", "other"],
        feature_names=["Ca", "Fe", "Mg"],
        redshift_grid=redshift,
        redshift_cell_width=width,
        joint_probability_mass=joint,
        feature_evidence=np.linspace(-0.4, 0.6, 21).reshape(3, 7),
        feature_support=np.asarray(
            [[True] * 7, [True] * 5 + [False, False], [False] * 7]
        ),
        predicted_class=1,
        predicted_redshift=1.1,
        evidence_sufficiency=0.72,
        split="calibration",
        view="original",
    )
    titles = {axis.get_title(loc="left") for axis in figure.axes}
    assert "Class–redshift evidence" in titles
    assert "Named-feature diagnostic for other (ONIR route)" in titles
    assert "Class probabilities" in titles
    assert any(
        title.startswith("Redshift posterior · primary 68%") for title in titles
    )
    assert "other P=0.73 · z_STRIDER=" in figure._suptitle.get_text()
    assert "posterior median=1.029 · MEDIUM" in figure._suptitle.get_text()
    assert "true: Ia, z=1.200 · primary basin mass" in figure._suptitle.get_text()
    assert "evidence score 0.72" in figure._suptitle.get_text()
    plt.close(figure)


def test_evidence_grade_uses_the_configured_boundaries() -> None:
    assert evidence_grade(0.86) == "HIGH"
    assert evidence_grade(0.72) == "MEDIUM"
    assert evidence_grade(0.49) == "LOW"
    assert evidence_grade(0.12) == "LIMITED"


def test_evidence_summary_is_concise_and_aligns_the_redshift_panels() -> None:
    wavelength = np.geomspace(7500.0, 20000.0, 12)
    item = {
        "flux": torch.randn(2, 12),
        "wavelength_mask": torch.ones(2, 12),
        "observer_days": torch.tensor([0.0, 18.0]),
        "epoch_observed_signal_to_noise": torch.tensor([1.2, 3.4]),
        "snid": torch.tensor(71),
        "redshift": torch.tensor(1.2),
        "class_index": torch.tensor(0),
    }
    redshift = np.geomspace(1.05, 3.0, 7) - 1.0
    width = np.diff(
        np.r_[redshift[0], 0.5 * (redshift[:-1] + redshift[1:]), redshift[-1]]
    )
    joint = np.arange(14, dtype=float).reshape(2, 7) + 1.0
    joint /= joint.sum()
    figure = plt.figure(figsize=(11.8, 7.6), constrained_layout=True)

    draw_evidence_summary(
        figure,
        item=item,
        wavelength_angstrom=wavelength,
        class_names=["Ia", "other"],
        redshift_grid=redshift,
        redshift_cell_width=width,
        joint_probability_mass=joint,
        predicted_class=1,
        predicted_redshift=1.1,
        evidence_sufficiency=0.72,
        split="test",
        view="original",
    )

    spectra_axis, joint_axis, redshift_axis, class_axis, solution_axis = figure.axes
    assert all(axis.get_title(loc="left") == "" for axis in figure.axes)
    assert joint_axis.get_shared_x_axes().joined(joint_axis, redshift_axis)
    assert joint_axis.get_shared_y_axes().joined(joint_axis, class_axis)
    assert redshift_axis.get_ylabel() == "posterior density"
    assert class_axis.get_xlabel() == r"$P(\mathrm{class})$"
    assert all(
        "relative evidence" not in axis.get_title(loc="left").lower()
        for axis in figure.axes
    )
    assert spectra_axis.get_legend() is None
    assert (
        spectra_axis.get_title(loc="right")
        == "Best observed spectrum — visit 2 of 2    day +18    S/N 3.40"
    )
    assert spectra_axis.get_xlim()[0] > wavelength[0]
    assert spectra_axis.get_xlim()[1] < wavelength[-1]
    solution_text = [text.get_text() for text in solution_axis.texts]
    assert "Redshift estimates" not in solution_text
    assert any(text.startswith("STRIDER  ") for text in solution_text)
    assert any(text.startswith("alternate  ") for text in solution_text)
    assert any(text.startswith("truth  ") for text in solution_text)
    assert figure._suptitle.get_text().startswith("Truth: Ia, z=1.200\nSTRIDER: other")
    assert any(text.get_text() == "SNID 71" for text in figure.texts)
    plt.close(figure)


def test_evidence_animation_summary_shows_the_newest_spectrum() -> None:
    wavelength = np.geomspace(7500.0, 20000.0, 12)
    item = {
        "flux": torch.randn(2, 12),
        "wavelength_mask": torch.ones(2, 12),
        "observer_days": torch.tensor([0.0, 18.0]),
        "epoch_observed_signal_to_noise": torch.tensor([9.0, 1.0]),
        "total_visit_count": torch.tensor(4),
        "snid": torch.tensor(71),
        "redshift": torch.tensor(1.2),
        "class_index": torch.tensor(0),
    }
    redshift = np.geomspace(1.05, 3.0, 7) - 1.0
    width = np.diff(
        np.r_[redshift[0], 0.5 * (redshift[:-1] + redshift[1:]), redshift[-1]]
    )
    joint = np.arange(14, dtype=float).reshape(2, 7) + 1.0
    joint /= joint.sum()
    figure = plt.figure(figsize=(11.8, 7.6), constrained_layout=True)

    draw_evidence_summary(
        figure,
        item=item,
        wavelength_angstrom=wavelength,
        class_names=["Ia", "other"],
        redshift_grid=redshift,
        redshift_cell_width=width,
        joint_probability_mass=joint,
        predicted_class=1,
        predicted_redshift=1.1,
        evidence_sufficiency=0.72,
        split="test",
        view="original",
        spectrum_selection="latest",
    )

    spectra_axis = figure.axes[0]
    assert (
        spectra_axis.get_title(loc="right")
        == "Evidence after 2 of 4 visits    new spectrum: visit 2    "
        "day +18    S/N 1.00"
    )
    plt.close(figure)


def test_evidence_animation_accumulates_visits_in_order() -> None:
    item = {
        "flux": torch.arange(20).reshape(5, 4),
        "wavelength_mask": torch.ones(5, 4),
        "observer_days": torch.arange(5),
        "visit_flux_scale": torch.arange(5, dtype=torch.float32) + 1.0,
        "simulation_rest_phase_days": torch.arange(5),
        "epoch_observed_signal_to_noise": torch.arange(5, dtype=torch.float32),
        "total_visit_count": torch.tensor(5),
        "snid": torch.tensor(7),
    }
    prefix = _visit_prefix(item, 3)

    assert prefix["flux"].shape == (3, 4)
    assert prefix["observer_days"].tolist() == [0, 1, 2]
    assert prefix["visit_flux_scale"].tolist() == [1.0, 2.0, 3.0]
    assert prefix["epoch_observed_signal_to_noise"].tolist() == [0.0, 1.0, 2.0]
    assert int(prefix["total_visit_count"]) == 5
    assert int(prefix["snid"]) == 7
    assert _animation_visit_counts(5, 24) == [1, 2, 3, 4, 5]
    assert len(_animation_visit_counts(50, 10)) == 10


def test_basin_trajectory_records_competing_modes_and_peak_motion() -> None:
    grid = np.linspace(0.0, 1.0, 101)
    width = np.full_like(grid, 0.01)
    first_density = np.exp(-0.5 * ((grid - 0.30) / 0.03) ** 2)
    first_density += 0.4 * np.exp(-0.5 * ((grid - 0.75) / 0.04) ** 2)
    second_density = np.exp(-0.5 * ((grid - 0.32) / 0.03) ** 2)
    first = {
        "joint": np.stack((0.8 * first_density, 0.2 * first_density)),
        "evidence_score": 0.4,
    }
    second = {
        "joint": np.stack((0.9 * second_density, 0.1 * second_density)),
        "evidence_score": 0.8,
    }

    first_row = _basin_trajectory_row(first, 1, grid, width)
    second_row = _basin_trajectory_row(
        second,
        4,
        grid,
        width,
        previous=first,
    )

    assert first_row["basin_count"] == 2
    assert first_row["secondary_to_primary_mass_ratio"] > 0.2
    assert np.isfinite(first_row["primary_log_peak_to_competitor_saddle_ratio"])
    assert not first_row["primary_mode_switch_from_previous"]
    assert second_row["basin_count"] == 1
    assert np.isclose(
        second_row["primary_peak_shift_from_previous"], 0.02, atol=1.0e-5
    )
    assert second_row["primary_class_index"] == 0

    third_density = np.exp(-0.5 * ((grid - 0.72) / 0.03) ** 2)
    third = {
        "joint": np.stack((0.9 * third_density, 0.1 * third_density)),
        "evidence_score": 0.7,
    }
    third_row = _basin_trajectory_row(
        third,
        8,
        grid,
        width,
        previous=second,
    )
    summary = _basin_trajectory_summary([first_row, second_row, third_row])

    assert third_row["primary_mode_switch_from_previous"]
    assert summary["primary_mode_switch_count"] == 1
    assert summary["late_primary_mode_switch"]
    assert summary["maximum_primary_peak_shift"] > 0.3


def test_representative_objects_prefer_distinct_classes() -> None:
    objects = pd.DataFrame(
        {
            "redshift": [1.01, 1.02, 1.03, 1.04],
            "class_index": [0, 0, 1, 2],
        }
    )
    selected = _representative_indices(objects, [1.0], 3)
    classes = [int(objects.loc[index, "class_index"]) for _, index in selected]
    assert classes == [0, 1, 2]


def test_representative_objects_can_focus_on_ia() -> None:
    objects = pd.DataFrame(
        {
            "redshift": [1.01, 1.02, 1.03, 1.04, 1.05],
            "class_index": [0, 0, 0, 1, 2],
        }
    )
    selected = _representative_indices(
        objects,
        [1.0],
        4,
        preferred_class_index=0,
        preferred_count=3,
    )

    classes = [int(objects.loc[index, "class_index"]) for _, index in selected]
    assert classes == [0, 0, 0, 1]


def test_evidence_map_manifest_preserves_requested_order_and_cohort() -> None:
    objects = pd.DataFrame(
        {
            "snid": [31, 17, 92],
            "redshift": [0.4, 1.1, 1.8],
            "class_index": [0, 0, 1],
        },
        index=[4, 8, 12],
    )
    manifest = pd.DataFrame(
        {
            "snid": [92, 31],
            "cohort": ["alias", "control"],
        }
    )

    assert _manifest_indices(objects, manifest) == [("alias", 12), ("control", 4)]


def test_route_audit_compares_the_same_redshifts_for_every_route() -> None:
    grid = np.asarray([0.0, 0.5, 1.0, 1.5])
    rows = _route_audit_rows(
        cohort="alias",
        item={
            "snid": torch.tensor(71),
            "redshift": torch.tensor(1.0),
            "class_index": torch.tensor(0),
        },
        class_name="Ia",
        predicted_class_probability=0.9,
        redshift_grid=grid,
        redshift_probability=np.asarray([0.1, 0.5, 0.3, 0.1]),
        redshift_cell_width=np.ones(4),
        peak_summary={
            "dominant_redshift": 0.5,
            "secondary_redshift": 1.0,
            "dominant_mass": 0.5,
            "secondary_mass": 0.3,
            "secondary_to_dominant_mass_ratio": 0.6,
            "distinct_peak_count": 2,
        },
        route_evidence={
            "Whole spectrum": np.asarray([0.0, 0.7, 1.2, 0.1]),
            "Temporal": np.asarray([0.0, 0.4, 0.2, 0.0]),
        },
    )

    assert [row["route"] for row in rows] == ["Whole spectrum", "Temporal"]
    assert np.isclose(rows[0]["true_minus_dominant"], 0.5)
    assert np.isclose(rows[1]["true_minus_dominant"], -0.2)
