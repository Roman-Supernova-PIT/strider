from types import SimpleNamespace

import pytest
import torch

from strider.evaluation.evaluate import (
    _candidate_route_logits,
    _selected_joint_logits,
)
from strider.evaluation.route_check import (
    _continuum_fraction_by_redshift,
    _dense_contribution,
    _onir_contribution,
)


def test_route_logits_are_built_in_order() -> None:
    outputs = {
        "onir_joint_logits": torch.full((2, 3, 4), 1.0),
        "shape_joint_logits": torch.full((2, 3, 4), 2.0),
        "dense_scan_joint_logits": torch.full((2, 3, 4), 3.0),
        "dense_whole_joint_logits": torch.full((2, 3, 4), 1.0),
        "dense_detail_joint_logits": torch.full((2, 3, 4), 4.0),
        "context_joint_logits": torch.full((2, 3, 4), 0.5),
        "temporal_joint_logits": torch.full((2, 3, 4), 0.25),
        "joint_support": torch.ones((2, 3, 4), dtype=torch.bool),
        "spectral_joint_logits": torch.full((2, 3, 4), 10.0),
        "joint_logits": torch.full((2, 3, 4), 10.0),
    }
    assert torch.equal(_selected_joint_logits(outputs, "onir"), outputs["onir_joint_logits"])
    assert torch.equal(
        _selected_joint_logits(outputs, "onir_shape"),
        outputs["onir_joint_logits"] + outputs["shape_joint_logits"],
    )
    assert torch.equal(
        _selected_joint_logits(outputs, "named_shape"),
        outputs["shape_joint_logits"],
    )
    assert torch.equal(
        _selected_joint_logits(outputs, "spectral"), outputs["spectral_joint_logits"]
    )
    assert torch.equal(
        _selected_joint_logits(outputs, "dense"), outputs["dense_scan_joint_logits"]
    )
    assert torch.equal(
        _selected_joint_logits(outputs, "dense_whole"),
        outputs["dense_whole_joint_logits"],
    )
    assert torch.equal(
        _selected_joint_logits(outputs, "dense_detail"),
        outputs["dense_detail_joint_logits"],
    )
    assert torch.equal(
        _selected_joint_logits(outputs, "without_dense"),
        outputs["spectral_joint_logits"] - outputs["dense_scan_joint_logits"],
    )
    expected_without_onir = (
        outputs["shape_joint_logits"]
        + outputs["context_joint_logits"]
        + outputs["dense_scan_joint_logits"]
        + outputs["temporal_joint_logits"]
    )
    assert torch.equal(
        _selected_joint_logits(outputs, "without_onir"),
        expected_without_onir,
    )
    assert torch.equal(
        _selected_joint_logits(outputs, "without_onir_spectral"),
        expected_without_onir - outputs["temporal_joint_logits"],
    )
    assert torch.equal(
        _selected_joint_logits(outputs, "without_onir_masked"),
        expected_without_onir,
    )
    assert torch.equal(
        _selected_joint_logits(outputs, "with_onir_masked"),
        expected_without_onir + outputs["onir_joint_logits"],
    )
    assert torch.equal(
        _selected_joint_logits(outputs, "global_spectrum"),
        outputs["dense_scan_joint_logits"] + outputs["context_joint_logits"],
    )
    assert torch.equal(_selected_joint_logits(outputs, "combined"), outputs["joint_logits"])


def test_route_logits_require_exposed_components() -> None:
    with pytest.raises(ValueError, match="onir_shape"):
        _selected_joint_logits({"joint_logits": torch.zeros(1, 2, 3)}, "onir_shape")
    with pytest.raises(ValueError, match="without_dense"):
        _selected_joint_logits(
            {"spectral_joint_logits": torch.zeros(1, 2, 3)},
            "without_dense",
        )


def test_reference_candidate_routes_are_reported_without_duplication() -> None:
    coadd = torch.full((1, 2, 3), 1.0)
    sequence = torch.full((1, 2, 3), 2.0)
    routes = _candidate_route_logits(
        {
            "reference_coadd_joint_logits": coadd,
            "reference_sequence_joint_logits": sequence,
            "spectral_joint_logits": coadd,
            "temporal_joint_logits": sequence,
        }
    )

    assert tuple(routes) == ("reference_coadd", "reference_sequence")
    assert routes["reference_coadd"] is coadd
    assert routes["reference_sequence"] is sequence
    with pytest.raises(ValueError, match="joint_support"):
        _selected_joint_logits(
            {"shape_joint_logits": torch.zeros(1, 2, 3)},
            "without_onir_masked",
        )


def test_without_onir_masked_preserves_the_onir_support_policy() -> None:
    support = torch.tensor([[[True, False], [False, True]]])
    outputs = {
        "shape_joint_logits": torch.ones(1, 2, 2),
        "dense_scan_joint_logits": torch.full((1, 2, 2), 2.0),
        "joint_support": support,
    }

    unmasked = _selected_joint_logits(outputs, "without_onir")
    masked = _selected_joint_logits(outputs, "without_onir_masked")

    assert torch.equal(unmasked, torch.full((1, 2, 2), 3.0))
    assert torch.equal(masked[support], torch.full((2,), 3.0))
    assert torch.equal(masked[~support], torch.full((2,), -1.0e4))


def test_dense_contribution_compares_spectral_with_without_dense() -> None:
    rows = [
        {
            "view": "generated",
            "route": "without_dense",
            "balanced_accuracy": 0.70,
            "Ia_f1": 0.75,
            "Ia_median_absolute_delta_z": 0.08,
        },
        {
            "view": "generated",
            "route": "spectral",
            "balanced_accuracy": 0.80,
            "Ia_f1": 0.85,
            "Ia_median_absolute_delta_z": 0.03,
        },
        {
            "view": "no_source",
            "route": "without_dense",
            "blank_redshift_lock": 0.08,
        },
        {
            "view": "no_source",
            "route": "spectral",
            "blank_redshift_lock": 0.09,
        },
    ]

    contribution = _dense_contribution(rows)
    assert contribution == [
        {
            "view": "generated",
            "delta_balanced_accuracy": pytest.approx(0.10),
            "delta_Ia_f1": pytest.approx(0.10),
            "delta_Ia_median_absolute_delta_z": pytest.approx(-0.05),
        },
        {
            "view": "no_source",
            "delta_blank_redshift_lock": pytest.approx(0.01),
        },
    ]


def test_onir_contribution_separates_score_mask_and_named_regions() -> None:
    source_rows = []
    for route, balanced, f1, delta_z, outlier in (
        ("with_onir_masked", 0.90, 0.88, 0.04, 0.20),
        ("without_onir_masked", 0.88, 0.86, 0.05, 0.22),
        ("without_onir", 0.89, 0.87, 0.045, 0.21),
        ("without_onir_spectral", 0.87, 0.85, 0.055, 0.23),
        ("global_spectrum", 0.84, 0.82, 0.07, 0.28),
    ):
        source_rows.append(
            {
                "view": "generated",
                "route": route,
                "balanced_accuracy": balanced,
                "Ia_f1": f1,
                "Ia_median_absolute_delta_z": delta_z,
                "Ia_outlier_fraction": outlier,
            }
        )

    contribution = _onir_contribution(source_rows)

    assert [row["comparison"] for row in contribution] == [
        "profile score",
        "support mask",
        "named-region shape",
        "temporal evolution",
    ]
    assert contribution[0]["delta_Ia_f1"] == pytest.approx(0.02)
    assert contribution[1]["delta_Ia_outlier_fraction"] == pytest.approx(0.01)
    assert contribution[2]["delta_Ia_median_absolute_delta_z"] == pytest.approx(-0.015)
    assert contribution[3]["delta_Ia_f1"] == pytest.approx(0.02)


def test_continuum_fraction_is_reported_across_redshift() -> None:
    model = SimpleNamespace(
        redshift_grid=torch.tensor([0.0, 1.0, 2.0, 3.0]),
        dense_scan=SimpleNamespace(
            detail_intercept=torch.tensor(0.0),
            detail_redshift_slope=torch.tensor(1.0),
            redshift_coordinate=torch.tensor([-1.0 / 3.0, 0.0, 1.0 / 3.0, 2.0 / 3.0]),
            maximum_detail_weight=0.5,
        ),
    )

    rows = _continuum_fraction_by_redshift(model)

    assert [row["redshift"] for row in rows] == [0.0, 1.0, 2.0, 3.0]
    assert rows[0]["continuum_fraction"] < rows[1]["continuum_fraction"]
    assert rows[1]["continuum_fraction"] == pytest.approx(0.25)
    assert all(row["continuum_fraction"] <= 0.5 for row in rows)
    assert rows[-1]["continuum_fraction"] > rows[1]["continuum_fraction"]
