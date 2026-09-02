#!/usr/bin/env python3
"""Summarize which STRIDER routes prefer the true or competing redshift peak."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def summarize(audit: pd.DataFrame) -> pd.DataFrame:
    required = {
        "cohort",
        "route",
        "true_minus_dominant",
        "secondary_minus_dominant",
    }
    missing = sorted(required - set(audit))
    if missing:
        raise ValueError(f"Route audit is missing columns: {missing}")
    grouped = audit.groupby(["cohort", "route"], sort=False)
    rows = []
    for (cohort, route), values in grouped:
        true_difference = values["true_minus_dominant"]
        secondary_difference = values["secondary_minus_dominant"].dropna()
        rows.append(
            {
                "cohort": cohort,
                "route": route,
                "n_objects": int(len(values)),
                "median_true_minus_dominant": float(true_difference.median()),
                "fraction_route_prefers_true": float((true_difference > 0.0).mean()),
                "median_secondary_minus_dominant": float(
                    secondary_difference.median()
                )
                if len(secondary_difference)
                else float("nan"),
                "fraction_route_prefers_secondary": float(
                    (secondary_difference > 0.0).mean()
                )
                if len(secondary_difference)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    arguments = _arguments()
    summary = summarize(pd.read_csv(arguments.input))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(arguments.output, index=False, float_format="%.6g")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(arguments.output)


if __name__ == "__main__":
    main()
