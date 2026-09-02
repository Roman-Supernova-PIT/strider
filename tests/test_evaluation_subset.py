from __future__ import annotations

import pandas as pd

from strider.evaluation.evaluate import _evaluation_summary_name
from strider.evaluation.subset import stratified_positions


def test_stratified_positions_are_repeatable_and_cover_each_group() -> None:
    objects = pd.DataFrame(
        {
            "class_index": [0] * 20 + [1] * 20,
            "redshift": [0.2] * 10 + [1.2] * 10 + [0.2] * 10 + [1.2] * 10,
        }
    )

    first = stratified_positions(objects, 12, 17, [0.0, 1.0, 2.0])
    second = stratified_positions(objects, 12, 17, [0.0, 1.0, 2.0])

    assert first == second
    selected = objects.iloc[first]
    groups = selected.groupby(
        ["class_index", pd.cut(selected["redshift"], [0.0, 1.0, 2.0])],
        observed=True,
    ).size()
    assert groups.to_list() == [3, 3, 3, 3]


def test_focused_evaluation_uses_a_distinct_summary_name() -> None:
    assert (
        _evaluation_summary_name(
            "calibration", split_overridden=False, selected_views=None
        )
        == "evaluation_summary.json"
    )
    assert (
        _evaluation_summary_name(
            "calibration",
            split_overridden=False,
            selected_views=["original"],
        )
        == "evaluation_summary_views_original.json"
    )
    assert (
        _evaluation_summary_name(
            "test",
            split_overridden=True,
            selected_views=["original", "generated"],
        )
        == "test_evaluation_summary_views_original_generated.json"
    )
