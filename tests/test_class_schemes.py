"""Class labels must preserve the class grouping declared by a run."""

import pandas as pd

from strider.data.classes import (
    GROUPED_7_CLASSES,
    GROUPED_8_CLASSES,
    HOURGLASS_15_CLASSES,
    class_name_for_source,
    class_names_for_scheme,
    fine_class_name_for_source,
    fine_to_output_class_indices,
    output_class_name_for_fine_class,
)
from strider.data.dataset import _apply_class_scheme


def test_hourglass_15_mapping_covers_direct_and_core_collapse_classes() -> None:
    assert class_names_for_scheme("hourglass_15") == HOURGLASS_15_CLASSES
    assert class_name_for_source(10, 0, "hourglass_15") == "Ia"
    assert class_name_for_source(11, 0, "hourglass_15") == "91bg"
    assert class_name_for_source(30, 701, "hourglass_15") == "IIP"
    assert class_name_for_source(30, 709, "hourglass_15") == "Ic-BL"
    assert class_name_for_source(50, 0, "hourglass_15") == "KN"


def test_fine_classes_marginalize_exactly_to_supported_outputs() -> None:
    binary = fine_to_output_class_indices(
        HOURGLASS_15_CLASSES, "normal_ia_binary"
    )
    grouped = fine_to_output_class_indices(HOURGLASS_15_CLASSES, "grouped_7")

    assert binary[HOURGLASS_15_CLASSES.index("Ia")] == 0
    assert set(binary[1:]) == {1}
    assert output_class_name_for_fine_class("IIP", "grouped_7") == "H-rich CC"
    assert (
        output_class_name_for_fine_class("Ic-BL", "grouped_7")
        == "stripped-envelope CC"
    )
    assert output_class_name_for_fine_class("PISN", "grouped_7") == "other"
    assert grouped[HOURGLASS_15_CLASSES.index("91bg")] == 1
    assert fine_class_name_for_source(30, 701) == "IIP"


def test_unknown_core_collapse_template_is_not_silently_relabelled() -> None:
    assert class_name_for_source(30, 999999, "hourglass_15") is None


def test_grouped_8_mapping_preserves_cosmology_relevant_families() -> None:
    assert class_names_for_scheme("grouped_8") == GROUPED_8_CLASSES
    assert class_name_for_source(10, 0, "grouped_8") == "Ia"
    assert class_name_for_source(11, 0, "grouped_8") == "91bg"
    assert class_name_for_source(12, 0, "grouped_8") == "Iax"
    assert class_name_for_source(30, 701, "grouped_8") == "H-rich CC"
    assert class_name_for_source(30, 709, "grouped_8") == "stripped-envelope CC"
    assert class_name_for_source(40, 0, "grouped_8") == "SLSN"
    assert class_name_for_source(59, 0, "grouped_8") == "PISN"
    assert class_name_for_source(42, 0, "grouped_8") == "other"
    assert class_name_for_source(45, 0, "grouped_8") == "other"
    assert class_name_for_source(50, 0, "grouped_8") == "other"
    assert class_name_for_source(30, 999999, "grouped_8") is None


def test_grouped_7_keeps_slsn_distinct_and_moves_pisn_to_other() -> None:
    assert class_names_for_scheme("grouped_7") == GROUPED_7_CLASSES
    assert class_name_for_source(10, 0, "grouped_7") == "Ia"
    assert class_name_for_source(11, 0, "grouped_7") == "91bg"
    assert class_name_for_source(12, 0, "grouped_7") == "Iax"
    assert class_name_for_source(30, 701, "grouped_7") == "H-rich CC"
    assert (
        class_name_for_source(30, 709, "grouped_7")
        == "stripped-envelope CC"
    )
    assert class_name_for_source(40, 0, "grouped_7") == "SLSN"
    assert class_name_for_source(59, 0, "grouped_7") == "other"
    assert class_name_for_source(42, 0, "grouped_7") == "other"
    assert class_name_for_source(45, 0, "grouped_7") == "other"
    assert class_name_for_source(50, 0, "grouped_7") == "other"
    assert class_name_for_source(30, 999999, "grouped_7") is None


def test_prepared_objects_can_be_reused_for_binary_classification() -> None:
    objects = pd.DataFrame(
        {
            "object_index": [4, 9, 12],
            "gentype": [10, 11, 30],
            "template_index": [0, 0, 701],
            "class_name": ["Ia", "91bg", "IIP"],
            "class_index": [0, 1, 3],
        }
    )
    config = {
        "data": {"class_scheme": "normal_ia_binary"},
        "model": {"classes": ["Ia", "other"]},
    }

    remapped = _apply_class_scheme(objects, config)

    assert remapped["object_index"].tolist() == [4, 9, 12]
    assert remapped["class_name"].tolist() == ["Ia", "other", "other"]
    assert remapped["class_index"].tolist() == [0, 1, 1]
