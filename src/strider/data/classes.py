"""Map SNANA source labels onto the supported STRIDER class schemes.

The full Hourglass scheme keeps normal Ia, peculiar Ia, seven core-collapse
subtypes and five exotic classes separate. Template identifiers are used only
to decode GENTYPE 30 core-collapse simulations during offline preparation.
"""

from __future__ import annotations


HOURGLASS_15_CLASSES = (
    "Ia",
    "91bg",
    "Iax",
    "IIP",
    "IIL",
    "IIb",
    "IIn",
    "Ib",
    "Ic",
    "Ic-BL",
    "SLSN",
    "TDE",
    "ILOT",
    "KN",
    "PISN",
)

GROUPED_8_CLASSES = (
    "Ia",
    "91bg",
    "Iax",
    "H-rich CC",
    "stripped-envelope CC",
    "SLSN",
    "PISN",
    "other",
)

GROUPED_7_CLASSES = (
    "Ia",
    "91bg",
    "Iax",
    "H-rich CC",
    "stripped-envelope CC",
    "SLSN",
    "other",
)

_DIRECT_HOURGLASS_CLASS = {
    10: "Ia",
    11: "91bg",
    12: "Iax",
    40: "SLSN",
    42: "TDE",
    45: "ILOT",
    50: "KN",
    59: "PISN",
}

_CORE_COLLAPSE_TEMPLATES = {
    "IIP": (701, 735, 744, 737, 739, 731, 752, 711, 708, 756, 757, 761, 766, 767, 764, 755),
    "IIL": (751, 745, 740, 725, 734, 719),
    "IIb": (743, 702, 724, 763, 760, 738, 758),
    "IIn": (748, 749, 750, 722, 732, 747, 730, 729, 759, 704, 765),
    "Ib": (707, 754, 728, 713, 703, 742, 741, 715, 727, 716, 721, 718),
    "Ic": (723, 762, 705, 746, 710, 712, 714),
    "Ic-BL": (709, 753, 726, 706, 736),
}
_TEMPLATE_TO_CLASS = {
    template: class_name
    for class_name, templates in _CORE_COLLAPSE_TEMPLATES.items()
    for template in templates
}


def class_names_for_scheme(name: str) -> tuple[str, ...]:
    if name == "normal_ia_binary":
        return ("Ia", "other")
    if name == "grouped_8":
        return GROUPED_8_CLASSES
    if name == "grouped_7":
        return GROUPED_7_CLASSES
    if name == "hourglass_15":
        return HOURGLASS_15_CLASSES
    raise ValueError(f"Unsupported class scheme: {name}")


def fine_class_name_for_source(gentype: int, template_index: int) -> str | None:
    """Decode one simulation source into the canonical physical class list."""
    gentype = int(gentype)
    if gentype == 30:
        return _TEMPLATE_TO_CLASS.get(int(template_index))
    return _DIRECT_HOURGLASS_CLASS.get(gentype)


def output_class_name_for_fine_class(fine_class: str, scheme: str) -> str:
    """Map a canonical physical class into one supported reporting scheme."""
    if fine_class not in HOURGLASS_15_CLASSES:
        raise ValueError(f"Unsupported physical class: {fine_class}")
    if scheme == "hourglass_15":
        return fine_class
    if scheme == "normal_ia_binary":
        return "Ia" if fine_class == "Ia" else "other"
    if scheme not in {"grouped_7", "grouped_8"}:
        raise ValueError(f"Unsupported class scheme: {scheme}")
    if fine_class in {"IIP", "IIL", "IIn"}:
        return "H-rich CC"
    if fine_class in {"IIb", "Ib", "Ic", "Ic-BL"}:
        return "stripped-envelope CC"
    if fine_class in {"TDE", "ILOT", "KN"} or (
        scheme == "grouped_7" and fine_class == "PISN"
    ):
        return "other"
    return fine_class


def fine_to_output_class_indices(
    fine_classes: tuple[str, ...], scheme: str
) -> tuple[int, ...]:
    """Return exact indices for marginalising fine-class model evidence."""
    output_names = class_names_for_scheme(scheme)
    name_to_index = {name: index for index, name in enumerate(output_names)}
    return tuple(
        name_to_index[output_class_name_for_fine_class(name, scheme)]
        for name in fine_classes
    )


def class_name_for_source(gentype: int, template_index: int, scheme: str) -> str | None:
    """Return the class name, or None when the source is unsupported."""
    gentype = int(gentype)
    if scheme == "normal_ia_binary":
        return "Ia" if gentype == 10 else "other"
    if scheme not in {"grouped_7", "grouped_8", "hourglass_15"}:
        raise ValueError(f"Unsupported class scheme: {scheme}")
    class_name = fine_class_name_for_source(gentype, template_index)
    if class_name is None:
        return None
    return output_class_name_for_fine_class(class_name, scheme)
