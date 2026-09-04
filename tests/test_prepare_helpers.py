from pathlib import Path

import numpy as np
import pandas as pd

from strider.data.prepare import (
    _check_block_separation,
    _class_support_by_redshift,
    _objects_with_complete_template_support,
    _require_separate_output_directory,
    _stratified_sample,
    _template_support_exclusion_summary,
    source_pair_assignments,
)
from strider.data.records import SourcePair


def test_split_blocks_must_be_distinct() -> None:
    try:
        _check_block_separation({"train": {1, 2}, "test": {2}})
    except ValueError as error:
        assert "block 2" in str(error)
    else:
        raise AssertionError("Overlapping blocks should fail")


def test_stratified_sample_keeps_requested_size() -> None:
    frame = pd.DataFrame(
        {
            "class_index": [0] * 20 + [1] * 20,
            "redshift_group": ([0] * 10 + [1] * 10) * 2,
            "value": range(40),
        }
    )
    sampled = _stratified_sample(frame, maximum=20, seed=12, group_columns=["class_index", "redshift_group"])
    assert len(sampled) == 20
    assert sampled.groupby(["class_index", "redshift_group"]).size().eq(5).all()


def test_stratified_sample_can_keep_every_simulation_block() -> None:
    frame = pd.DataFrame(
        {
            "class_index": [0] * 20,
            "redshift_group": [0] * 20,
            "block": np.repeat(np.arange(1, 11), 2),
        }
    )

    sampled = _stratified_sample(
        frame,
        maximum=10,
        seed=19,
        group_columns=["class_index", "redshift_group", "block"],
    )

    assert set(sampled["block"]) == set(range(1, 11))


def test_separate_products_can_reuse_block_numbers(monkeypatch, tmp_path: Path) -> None:
    directories = {name: tmp_path / name for name in ("training", "validation")}
    for directory in directories.values():
        directory.mkdir()

    def fake_discover(path):
        product = Path(path).name
        return [
            SourcePair(
                block=1,
                model="SNIaMODEL00",
                head_path=f"{product}_HEAD.FITS.gz",
                spec_path=f"{product}_SPEC.FITS.gz",
            )
        ]

    monkeypatch.setattr("strider.data.prepare.discover_source_pairs", fake_discover)
    assignments, recorded = source_pair_assignments(
        {
            "source_products": {
                "flat_training": {
                    "source_dir": str(directories["training"]),
                    "split_blocks": {"train": [1]},
                },
                "flat_validation": {
                    "source_dir": str(directories["validation"]),
                    "split_blocks": {"selection": [1]},
                },
            }
        }
    )
    assert [(product, split, pair.block) for product, split, pair in assignments] == [
        ("flat_training", "train", 1),
        ("flat_validation", "selection", 1),
    ]
    assert set(recorded) == {"flat_training", "flat_validation"}


def test_prepared_data_cannot_be_written_inside_a_simulation(tmp_path: Path) -> None:
    source = tmp_path / "simulations" / "flat_training"
    source.mkdir(parents=True)
    products = {"flat_training": {"source_dir": str(source)}}

    try:
        _require_separate_output_directory(source / "prepared", products)
    except ValueError as error:
        assert "read-only simulation directory" in str(error)
    else:
        raise AssertionError("Prepared output inside a simulation directory was accepted")

    _require_separate_output_directory(tmp_path / "strider" / "data", products)


def test_class_support_reports_ia_only_intervals() -> None:
    objects = pd.DataFrame(
        {
            "split": ["train"] * 5,
            "class_name": ["Ia", "IIP", "Ia", "Ia", "IIn"],
            "redshift": [0.04, 0.06, 0.14, 0.24, 0.26],
        }
    )
    support = _class_support_by_redshift(
        objects,
        np.asarray([0.0, 0.3]),
        bin_width=0.1,
    )
    middle = support[support["redshift_min"].eq(0.1)].iloc[0]
    assert middle.n_ia == 1
    assert middle.n_non_ia == 0
    assert middle.ia_fraction == 1.0


def test_complete_template_support_requires_every_visit_to_span_requested_range() -> None:
    observations = pd.DataFrame(
        {
            "object_index": [0, 0, 1, 1],
            "native_wavelength_min": [7441.0, 7441.0, 7441.0, 7600.0],
            "native_wavelength_max": [18220.0, 18220.0, 18220.0, 18220.0],
        }
    )
    complete = _objects_with_complete_template_support(
        observations, 7500.0, 18175.0
    )
    assert complete == {0}


def test_template_support_exclusions_are_reported_by_split_and_class() -> None:
    excluded = pd.DataFrame(
        {
            "split": ["train", "train", "selection"],
            "class_name": ["Ia", "IIP", "Ia"],
            "redshift": [0.4, 1.2, 2.1],
        }
    )
    report = _template_support_exclusion_summary(excluded)
    assert report["objects"] == 3
    assert report["by_split_and_class"]["train"] == {"IIP": 1, "Ia": 1}
    assert report["redshift"]["selection"]["median"] == 2.1
