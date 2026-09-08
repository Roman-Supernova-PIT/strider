"""Check whether simulation-template endpoints reveal class or redshift.

The report reads prepared tables only. Flux values are never used, so any
association comes from the wavelength range supplied by the simulation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from strider.config import project_path


def wavelength_support_report(config: dict[str, Any]) -> dict[str, Any]:
    prepared = project_path(config, config["data"]["prepared_dir"])
    objects = pd.read_parquet(prepared / "objects.parquet")
    observations = pd.read_parquet(prepared / "observations.parquet")
    required = {"object_index", "native_wavelength_min", "native_wavelength_max"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(
            "Prepared observations lack wavelength-support fields: "
            + ", ".join(sorted(missing))
        )

    minimum = float(config["observation"]["wavelength_min"])
    maximum = float(config["observation"]["wavelength_max"])
    clipped = observations[list(required)].copy()
    clipped["start"] = clipped["native_wavelength_min"].clip(minimum, maximum)
    clipped["end"] = clipped["native_wavelength_max"].clip(minimum, maximum)
    clipped["fraction"] = (
        (clipped["end"] - clipped["start"]).clip(lower=0.0) / (maximum - minimum)
    )
    support = clipped.groupby("object_index").agg(
        start_min=("start", "min"),
        start_max=("start", "max"),
        end_min=("end", "min"),
        end_max=("end", "max"),
        fraction_min=("fraction", "min"),
        fraction_mean=("fraction", "mean"),
    )
    joined = objects.merge(support, left_on="object_index", right_index=True, how="left")
    joined["support_signature"] = _support_signatures(joined, minimum, maximum)

    # Test labels remain untouched until the model and calibration are fixed.
    checked = joined[joined["split"].isin(("train", "selection"))]
    split_reports = {
        str(split): _split_report(rows)
        for split, rows in checked.groupby("split", sort=False)
    }
    if not split_reports:
        raise ValueError("No train or selection objects are available for this check")
    passed = all(report["passed"] for report in split_reports.values())
    return {
        "prepared_dir": str(prepared),
        "observer_wavelength_range": [minimum, maximum],
        "policy": config["observation"].get(
            "template_support_policy",
            "complete"
            if config["observation"].get("require_complete_wavelength_coverage")
            else "retain",
        ),
        "thresholds": {
            "maximum_redshift_explained_fraction": 0.02,
            "maximum_ia_explained_fraction": 0.02,
        },
        "splits": split_reports,
        "passed": passed,
    }


def _support_signatures(
    values: pd.DataFrame, minimum: float, maximum: float
) -> pd.Series:
    columns = ("start_min", "start_max", "end_min", "end_max")
    scaled = []
    for name in columns:
        position = 32.0 * (values[name] - minimum) / (maximum - minimum)
        scaled.append(np.rint(position).clip(0, 32).astype(int).astype(str))
    for name in ("fraction_min", "fraction_mean"):
        scaled.append(np.rint(32.0 * values[name]).clip(0, 32).astype(int).astype(str))
    signature = scaled[0]
    for part in scaled[1:]:
        signature = signature.str.cat(part, sep=":")
    return signature


def _split_report(values: pd.DataFrame) -> dict[str, Any]:
    redshift_fraction = _explained_fraction(
        values["redshift"].to_numpy(dtype=float), values["support_signature"]
    )
    ia_fraction = _explained_fraction(
        values["class_name"].eq("Ia").to_numpy(dtype=float),
        values["support_signature"],
    )
    counts = values["support_signature"].value_counts()
    passed = redshift_fraction <= 0.02 and ia_fraction <= 0.02
    return {
        "objects": int(len(values)),
        "support_signatures": int(len(counts)),
        "largest_signature_fraction": float(counts.iloc[0] / len(values)),
        "minimum_supported_fraction": float(values["fraction_min"].min()),
        "redshift_explained_fraction": redshift_fraction,
        "ia_explained_fraction": ia_fraction,
        "passed": passed,
    }


def _explained_fraction(values: np.ndarray, groups: pd.Series) -> float:
    finite = np.isfinite(values)
    values = values[finite]
    groups = groups.iloc[np.flatnonzero(finite)].reset_index(drop=True)
    total = float(np.square(values - values.mean()).sum())
    if total <= 0.0:
        return 0.0
    frame = pd.DataFrame({"value": values, "group": groups})
    means = frame.groupby("group")["value"].transform("mean").to_numpy()
    between = float(np.square(means - values.mean()).sum())
    return between / total
