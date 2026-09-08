"""The visit-growth sweep must prefer calibrated source intervals safely."""

from __future__ import annotations

from strider.evaluation.evidence_growth import (
    _assess_sweep,
    _validated_exponents,
    _validated_visit_counts,
)


def test_sweep_selects_best_coverage_among_blank_safe_values() -> None:
    exponents = (0.0, 0.25, 0.5)
    visits = (1, 32)
    cases = {
        "alpha=0": _cases(coverage=(0.80, 0.92), lock=(0.08, 0.08), info=(0.3, 0.2)),
        "alpha=0.25": _cases(
            coverage=(0.72, 0.69), lock=(0.08, 0.09), info=(0.3, 0.3)
        ),
        "alpha=0.5": _cases(
            coverage=(0.66, 0.60), lock=(0.08, 0.16), info=(0.3, 0.5)
        ),
    }

    report = _assess_sweep(cases, exponents, visits)

    assert report["best_safe_exponent_by_coverage"] == 0.25
    assert report["rows"][0]["blank_gate_passed"]
    assert report["rows"][1]["blank_gate_passed"]
    assert not report["rows"][2]["blank_gate_passed"]


def test_sweep_arguments_are_bounded_and_ordered() -> None:
    assert _validated_exponents([0.0, 0.2]) == (0.0, 0.2)
    assert _validated_visit_counts([1, 4, 16]) == (1, 4, 16)

    for values in ([-0.1], [1.1], [0.2, 0.2]):
        try:
            _validated_exponents(values)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid evidence exponents were accepted")

    for values in ([0, 1], [4, 1], [1, 1]):
        try:
            _validated_visit_counts(values)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid visit counts were accepted")


def _cases(
    *,
    coverage: tuple[float, float],
    lock: tuple[float, float],
    info: tuple[float, float],
) -> dict[str, dict[str, float | int]]:
    result = {}
    for index, visits in enumerate((1, 32)):
        result[f"generated:{visits}"] = {
            "N": 100,
            "Ia_N": 50,
            "Ia_f1": 0.8,
            "Ia_median_absolute_delta_z": 0.05,
            "Ia_posterior_68_interval_coverage": coverage[index],
            "mean_evidence_sufficiency": 0.6,
        }
        result[f"no_source:{visits}"] = {
            "N": 100,
            "median_absolute_delta_z_to_simulation": 0.6,
            "fraction_within_delta_z_0_1_of_simulation": lock[index],
            "mean_evidence_sufficiency": 0.02,
            "mean_redshift_information_gain_nats": info[index],
        }
    return result
