"""Build the local native-bin Sundial store and fixed object splits."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from strider.config import project_path

from .classes import class_name_for_source, class_names_for_scheme
from .records import OBJECT_COLUMNS, OBSERVATION_COLUMNS, SourcePair
from .snana import (
    discover_source_pairs,
    read_head_table,
    read_selected_spectra,
    read_spectrum_snids,
)
from .template_support import template_support_policy


def build_native_store(
    config: dict[str, Any],
    *,
    require_train_split: bool = True,
) -> dict[str, Any]:
    data_config = config["data"]
    output_dir = project_path(config, data_config["prepared_dir"])
    assignments, source_products = source_pair_assignments(data_config)
    _require_separate_output_directory(output_dir, source_products)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_order = tuple(dict.fromkeys(split for _, split, _ in assignments))
    if require_train_split and "train" not in split_order:
        raise ValueError("At least one source product must supply the train split")

    heads = []
    pair_lookup: dict[tuple[str, int, str], SourcePair] = {}
    for source_product, split, pair in assignments:
        frame = read_head_table(pair)
        spectrum_snids = read_spectrum_snids(pair)
        frame["has_spectrum"] = frame["snid"].isin(spectrum_snids)
        frame["split"] = split
        frame["source_product"] = source_product
        frame["source_id"] = source_product + ":" + frame["snid"].astype(str)
        heads.append(frame)
        pair_lookup[(source_product, pair.block, pair.model)] = pair
    catalog = pd.concat(heads, ignore_index=True)
    if catalog["source_id"].duplicated().any():
        duplicated = (
            catalog.loc[catalog["source_id"].duplicated(), "source_id"].head().tolist()
        )
        raise ValueError(
            "Object identifiers are not unique within their source product: "
            + ", ".join(duplicated)
        )

    catalog_rows_without_spectra = int((~catalog["has_spectrum"]).sum())
    catalog = catalog[catalog["has_spectrum"]].copy()
    class_scheme = str(data_config.get("class_scheme", "normal_ia_binary"))
    class_names = class_names_for_scheme(class_scheme)
    configured_classes = tuple(str(name) for name in config["model"]["classes"])
    if configured_classes != class_names:
        raise ValueError(
            f"model.classes {configured_classes} do not match {class_scheme}: {class_names}"
        )
    catalog["class_name"] = [
        class_name_for_source(row.gentype, row.template_index, class_scheme)
        for row in catalog.itertuples(index=False)
    ]
    unsupported_class_rows = int(catalog["class_name"].isna().sum())
    catalog = catalog[catalog["class_name"].notna()].copy()
    class_to_index = {name: index for index, name in enumerate(class_names)}
    catalog["class_index"] = catalog["class_name"].map(class_to_index).astype(np.int64)
    catalog["redshift_group"] = pd.cut(
        catalog["redshift"],
        bins=np.asarray(data_config["redshift_edges"], dtype=float),
        include_lowest=True,
        labels=False,
    ).fillna(-1).astype(int)

    seed = int(config["project"]["seed"])
    selected_parts = []
    for split_number, split in enumerate(split_order):
        split_frame = catalog[catalog["split"] == split].copy()
        maximum = int(data_config.get("max_objects", {}).get(split, 0))
        group_columns = ["class_index", "redshift_group"]
        if split == "train" and bool(data_config.get("training_sample_by_block", False)):
            # A short run should still see every simulation seed. Otherwise a
            # random subset can understate sensitivity to a particular draw.
            group_columns.append("block")
        selected_parts.append(
            _stratified_sample(
                split_frame,
                maximum,
                seed + 1009 * split_number,
                group_columns=group_columns,
            )
        )
    selected = pd.concat(selected_parts, ignore_index=True)
    empty_object_splits = [
        split for split in split_order if not bool((selected["split"] == split).any())
    ]
    if empty_object_splits:
        raise ValueError(
            "No usable objects remain in split(s): " + ", ".join(empty_object_splits)
        )
    selected = selected.sort_values(
        ["split", "source_product", "block", "model", "snid"]
    ).reset_index(drop=True)
    training_classes = set(selected.loc[selected["split"] == "train", "class_name"])
    missing_training_classes = [name for name in class_names if name not in training_classes]
    if (
        require_train_split
        and missing_training_classes
        and bool(data_config.get("require_all_training_classes", True))
    ):
        raise ValueError(
            "Training split lacks requested classes: "
            + ", ".join(missing_training_classes)
            + ". Revise split blocks or the declared class scheme."
        )
    minimum_class_counts = {
        str(split): int(count)
        for split, count in data_config.get("minimum_class_counts", {}).items()
    }
    _validate_class_counts(selected, class_names, minimum_class_counts)
    selected["object_index"] = np.arange(len(selected), dtype=np.int64)
    object_index = dict(
        zip(selected["source_id"].astype(str), selected["object_index"].astype(int))
    )

    h5_path = output_dir / "spectra.h5"
    observation_rows: list[dict[str, Any]] = []
    with h5py.File(h5_path, "w") as store:
        store.attrs["format_version"] = "strider-spectra-v2"
        store.attrs["description"] = "Ragged native-bin Sundial spectra; one row per detector bin."
        datasets = {
            name: store.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                dtype="f4",
                chunks=(16_384,),
                compression="gzip",
                compression_opts=2,
                shuffle=True,
            )
            for name in (
                "wavelength_min",
                "wavelength_max",
                "observed_flux",
                "flux_error",
                "clean_flux",
            )
        }
        total_bins = 0
        observation_index = 0
        for (source_product, block, model), group in selected.groupby(
            ["source_product", "block", "model"], sort=True
        ):
            pair = pair_lookup[(str(source_product), int(block), str(model))]
            snids = set(group["snid"].astype(int))
            pair_observations, pair_arrays = read_selected_spectra(pair, snids)
            pair_bin_count = len(pair_arrays["observed_flux"])
            for dataset_name, values in pair_arrays.items():
                dataset = datasets[dataset_name]
                dataset.resize(total_bins + pair_bin_count, axis=0)
                dataset[total_bins:] = values
            for observation in pair_observations:
                snid = int(observation["snid"])
                source_id = f"{source_product}:{snid}"
                current_object = object_index[source_id]
                first_bin = total_bins + int(observation["first_bin"])
                bin_count = int(observation["bin_count"])
                error_values = pair_arrays["flux_error"][
                    int(observation["first_bin"]): int(observation["first_bin"]) + bin_count
                ]
                native_min = float(
                    pair_arrays["wavelength_min"][int(observation["first_bin"])]
                )
                native_max = float(
                    pair_arrays["wavelength_max"][
                        int(observation["first_bin"]) + bin_count - 1
                    ]
                )
                positive_errors = error_values[np.isfinite(error_values) & (error_values > 0)]
                background_scale = (
                    float(np.quantile(positive_errors, 0.25)) if positive_errors.size else 1.0
                )
                row = {
                    "observation_index": observation_index,
                    "object_index": current_object,
                    "source_id": source_id,
                    "source_product": str(source_product),
                    "snid": snid,
                    "mjd": float(observation["mjd"]),
                    "exposure_seconds": float(observation["exposure_seconds"]),
                    "first_bin": first_bin,
                    "bin_count": bin_count,
                    "native_wavelength_min": native_min,
                    "native_wavelength_max": native_max,
                    "background_scale": background_scale,
                }
                observation_rows.append(row)
                observation_index += 1
            total_bins += pair_bin_count

    observations = pd.DataFrame(observation_rows, columns=OBSERVATION_COLUMNS)
    objects_with_spectra = set(observations["object_index"].astype(int))
    missing_observation_mask = ~selected["object_index"].isin(objects_with_spectra)
    selected_rows_without_spectra = int(missing_observation_mask.sum())
    selected = selected.loc[~missing_observation_mask].copy()

    observation_config = config["observation"]
    support_policy = template_support_policy(observation_config)
    excluded_for_support = 0
    support_exclusions: dict[str, Any] = {}
    if support_policy == "complete":
        complete_objects = _objects_with_complete_template_support(
            observations,
            float(observation_config["wavelength_min"]),
            float(observation_config["wavelength_max"]),
        )
        keep = selected["object_index"].isin(complete_objects)
        excluded_for_support = int((~keep).sum())
        support_exclusions = _template_support_exclusion_summary(selected.loc[~keep])
        selected = selected.loc[keep].copy()
        observations = observations[
            observations["object_index"].isin(complete_objects)
        ].copy()

    old_to_new = {
        int(old): int(new)
        for new, old in enumerate(selected["object_index"].astype(int).tolist())
    }
    selected["object_index"] = selected["object_index"].map(old_to_new).astype(np.int64)
    observations["object_index"] = observations["object_index"].map(old_to_new).astype(np.int64)
    observations = observations.sort_values(
        ["object_index", "mjd", "observation_index"]
    ).reset_index(drop=True)
    observations["observation_index"] = np.arange(len(observations), dtype=np.int64)
    grouped_observations = observations.groupby("object_index")["observation_index"].agg(
        first="min", count="size"
    )
    selected["first_observation"] = selected["object_index"].map(
        grouped_observations["first"]
    ).astype(np.int64)
    selected["observation_count"] = selected["object_index"].map(
        grouped_observations["count"]
    ).astype(np.int64)
    _validate_class_counts(selected, class_names, minimum_class_counts)
    _check_contiguous_observations(selected, observations)

    training_classes = set(selected.loc[selected["split"] == "train", "class_name"])
    missing_training_classes = (
        [name for name in class_names if name not in training_classes]
        if require_train_split
        else []
    )

    objects = selected[OBJECT_COLUMNS].copy()
    objects.to_parquet(output_dir / "objects.parquet", index=False)
    observations.to_parquet(output_dir / "observations.parquet", index=False)
    class_support = _class_support_by_redshift(
        objects,
        np.asarray(data_config["redshift_edges"], dtype=float),
        bin_width=0.1,
    )
    class_support.to_csv(output_dir / "class_support_by_redshift.csv", index=False)

    summary = {
        "format_version": "strider-spectra-v2",
        "preparation_mode": (
            "training_and_evaluation" if require_train_split else "external_test"
        ),
        "objects": int(len(objects)),
        "observations": int(len(observations)),
        "native_bins": int(observations["bin_count"].sum()),
        "stored_native_bins": int(total_bins),
        "catalog_rows_without_spectra": catalog_rows_without_spectra,
        "catalog_rows_with_unsupported_class": unsupported_class_rows,
        "missing_training_classes": missing_training_classes,
        "selected_rows_without_spectra": selected_rows_without_spectra,
        "template_support_policy": support_policy,
        "objects_excluded_incomplete_template_support": excluded_for_support,
        "incomplete_template_support": support_exclusions,
        "required_observer_wavelength_range": (
            [
                float(observation_config["wavelength_min"]),
                float(observation_config["wavelength_max"]),
            ]
            if support_policy == "complete"
            else None
        ),
        "splits": {
            split: _split_summary(objects[objects["split"] == split]) for split in split_order
        },
        "class_support": {
            split: _split_class_support_summary(
                objects[objects["split"] == split],
                class_support[class_support["split"] == split],
            )
            for split in split_order
        },
        "source_products": source_products,
        "files": {
            "objects": "objects.parquet",
            "observations": "observations.parquet",
            "spectra": "spectra.h5",
            "class_support": "class_support_by_redshift.csv",
        },
        "notes": [
            "SIM_FLAM is simulation-only clean flux for controlled generation and tests.",
            "FLAM is retained as the original SNANA observation.",
            "The stored background_scale is the lower-quartile positive FLAMERR for each visit.",
            "Per-bin FLAMERR does not determine wavelength support.",
            "Complete support excludes objects when a simulation template ends inside the requested observer-frame interval.",
            "Retained support is experimental until the wavelength-support-only check passes.",
            "estimated_peak_mjd is the SNANA initial light-curve estimate, not simulation truth.",
            "simulation_peak_mjd is retained for bank construction and evaluation only.",
        ],
    }
    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _objects_with_complete_template_support(
    observations: pd.DataFrame,
    wavelength_min: float,
    wavelength_max: float,
) -> set[int]:
    """Return objects whose every visit covers the requested observer interval."""
    complete_visit = (
        observations["native_wavelength_min"].le(wavelength_min)
        & observations["native_wavelength_max"].ge(wavelength_max)
    )
    complete_object = complete_visit.groupby(observations["object_index"]).all()
    return set(complete_object[complete_object].index.astype(int))


def _template_support_exclusion_summary(objects: pd.DataFrame) -> dict[str, Any]:
    if objects.empty:
        return {"objects": 0, "by_split_and_class": {}, "redshift": {}}
    grouped = objects.groupby(["split", "class_name"], observed=True).size()
    by_split_and_class: dict[str, dict[str, int]] = {}
    for (split, class_name), count in grouped.items():
        by_split_and_class.setdefault(str(split), {})[str(class_name)] = int(count)
    redshift = {
        str(split): {
            "minimum": float(group["redshift"].min()),
            "median": float(group["redshift"].median()),
            "maximum": float(group["redshift"].max()),
        }
        for split, group in objects.groupby("split", observed=True)
    }
    return {
        "objects": int(len(objects)),
        "by_split_and_class": by_split_and_class,
        "redshift": redshift,
    }


def _validate_class_counts(
    objects: pd.DataFrame,
    class_names: tuple[str, ...],
    minimum_class_counts: dict[str, int],
) -> None:
    for split, minimum in minimum_class_counts.items():
        if minimum < 0:
            raise ValueError("minimum_class_counts values cannot be negative")
        counts = (
            objects.loc[objects["split"] == split, "class_name"]
            .value_counts()
            .reindex(class_names, fill_value=0)
        )
        below = counts[counts < minimum]
        if len(below):
            detail = ", ".join(f"{name}={int(count)}" for name, count in below.items())
            raise ValueError(
                f"Split {split!r} has fewer than {minimum} objects for: {detail}"
            )


def source_pair_assignments(
    data_config: dict[str, Any],
) -> tuple[list[tuple[str, str, SourcePair]], dict[str, Any]]:
    """Assign complete SNANA files from named products to their data roles."""
    configured_products = data_config.get("source_products")
    if configured_products is None:
        configured_blocks = data_config.get("split_blocks")
        if configured_blocks is None:
            configured_blocks = {
                "train": data_config["train_blocks"],
                "validation": data_config["validation_blocks"],
                "test": data_config["test_blocks"],
            }
        configured_products = {
            "default": {
                "source_dir": data_config["source_dir"],
                "split_blocks": configured_blocks,
            }
        }
    if not isinstance(configured_products, dict) or not configured_products:
        raise ValueError("data.source_products must be a non-empty mapping")

    assignments: list[tuple[str, str, SourcePair]] = []
    recorded_products: dict[str, Any] = {}
    for product_name, settings in configured_products.items():
        if not isinstance(settings, dict):
            raise ValueError(f"Settings for source product {product_name!r} must be a mapping")
        source_dir = Path(str(settings["source_dir"])).expanduser().resolve()
        split_blocks = {
            str(split): set(int(block) for block in blocks)
            for split, blocks in settings["split_blocks"].items()
        }
        _check_block_separation(split_blocks)
        pairs = discover_source_pairs(source_dir)
        for split, blocks in split_blocks.items():
            chosen = [pair for pair in pairs if pair.block in blocks]
            if not chosen:
                raise ValueError(
                    f"Source product {product_name!r} has no HEAD/SPEC pairs for "
                    f"split {split!r}; requested blocks were {sorted(blocks)}"
                )
            assignments.extend((str(product_name), split, pair) for pair in chosen)
        recorded_products[str(product_name)] = {
            "source_dir": str(source_dir),
            "split_blocks": {
                split: sorted(blocks) for split, blocks in split_blocks.items()
            },
        }
    return assignments, recorded_products


def _require_separate_output_directory(
    output_dir: Path,
    source_products: dict[str, Any],
) -> None:
    """Refuse to create prepared files inside a simulation product."""
    output = output_dir.expanduser().resolve()
    for product_name, settings in source_products.items():
        source = Path(str(settings["source_dir"])).expanduser().resolve()
        if output == source or output.is_relative_to(source):
            raise ValueError(
                f"Prepared data for {product_name!r} cannot be written inside the "
                f"read-only simulation directory {source}; choose a scratch path"
            )


def _stratified_sample(
    frame: pd.DataFrame,
    maximum: int,
    seed: int,
    group_columns: list[str],
) -> pd.DataFrame:
    if maximum <= 0 or maximum >= len(frame):
        return frame.copy()
    groups = [(key, group) for key, group in frame.groupby(group_columns, observed=True) if len(group)]
    if maximum < len(groups):
        raise ValueError(
            f"max_objects={maximum} is smaller than the {len(groups)} populated "
            f"strata defined by {group_columns}; increase the limit so every stratum is retained"
        )
    base = maximum // len(groups)
    chosen = []
    leftovers = []
    for group_number, (_, group) in enumerate(groups):
        take = min(base, len(group))
        sampled = group.sample(n=take, random_state=seed + group_number)
        chosen.append(sampled)
        leftovers.append(group.drop(sampled.index))
    result = pd.concat(chosen) if chosen else frame.iloc[:0]
    remaining = maximum - len(result)
    if remaining > 0:
        pool = pd.concat(leftovers)
        result = pd.concat([result, pool.sample(n=remaining, random_state=seed + 7919)])
    return result.sample(frac=1.0, random_state=seed + 1543).reset_index(drop=True)


def _check_block_separation(split_blocks: dict[str, set[int]]) -> None:
    seen: dict[int, str] = {}
    for split, blocks in split_blocks.items():
        for block in blocks:
            if block in seen:
                raise ValueError(f"FITS block {block} belongs to both {seen[block]} and {split}")
            seen[block] = split


def _check_contiguous_observations(objects: pd.DataFrame, observations: pd.DataFrame) -> None:
    indices = observations["observation_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(indices, np.arange(len(observations), dtype=np.int64)):
        raise ValueError("Observation indices are not a contiguous zero-based sequence")
    grouped = observations.groupby("object_index", sort=False)["observation_index"].agg(
        first="min", last="max", count="size"
    )
    expected = objects.set_index("object_index")[["snid", "first_observation", "observation_count"]]
    checked = expected.join(grouped, how="left")
    valid = (
        checked["first"].eq(checked["first_observation"])
        & checked["count"].eq(checked["observation_count"])
        & (checked["last"] - checked["first"] + 1).eq(checked["count"])
    )
    if not bool(valid.all()):
        bad = checked.loc[~valid, "snid"].head().astype(str).tolist()
        raise ValueError("Observations are not contiguous for SNID(s): " + ", ".join(bad))


def _split_summary(frame: pd.DataFrame) -> dict[str, Any]:
    class_counts = Counter(frame["class_name"])
    return {
        "objects": int(len(frame)),
        "class_counts": {str(name): int(count) for name, count in sorted(class_counts.items())},
        "redshift_median": float(frame["redshift"].median()),
        "redshift_min": float(frame["redshift"].min()),
        "redshift_max": float(frame["redshift"].max()),
        "redshift_at_least_2": int((frame["redshift"] >= 2.0).sum()),
    }


def _class_support_by_redshift(
    objects: pd.DataFrame,
    redshift_range: np.ndarray,
    *,
    bin_width: float,
) -> pd.DataFrame:
    """Count Ia and non-Ia support without folding in a survey class prior."""
    lower = float(redshift_range[0])
    upper = float(redshift_range[-1])
    edges = np.arange(lower, upper + bin_width, bin_width, dtype=float)
    if edges[-1] <= upper:
        edges = np.append(edges, upper + bin_width)
    work = objects[["split", "class_name", "redshift"]].copy()
    work["redshift_bin"] = pd.cut(
        work["redshift"], edges, right=False, include_lowest=True
    )
    work["is_ia"] = work["class_name"].eq("Ia")
    rows = []
    for (split, interval), group in work.groupby(
        ["split", "redshift_bin"], observed=True
    ):
        n_ia = int(group["is_ia"].sum())
        n_total = len(group)
        rows.append(
            {
                "split": str(split),
                "redshift_min": float(interval.left),
                "redshift_max": float(interval.right),
                "n_objects": n_total,
                "n_ia": n_ia,
                "n_non_ia": n_total - n_ia,
                "ia_fraction": n_ia / n_total,
                "n_classes": int(group["class_name"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _split_class_support_summary(
    objects: pd.DataFrame,
    support: pd.DataFrame,
) -> dict[str, Any]:
    ia = objects[objects["class_name"] == "Ia"]
    non_ia = objects[objects["class_name"] != "Ia"]
    ia_only = support[(support["n_ia"] > 0) & (support["n_non_ia"] == 0)]
    return {
        "maximum_ia_redshift": float(ia["redshift"].max()) if len(ia) else None,
        "maximum_non_ia_redshift": (
            float(non_ia["redshift"].max()) if len(non_ia) else None
        ),
        "redshift_bins_with_ia_and_no_non_ia": [
            [float(row.redshift_min), float(row.redshift_max)]
            for row in ia_only.itertuples(index=False)
        ],
    }
