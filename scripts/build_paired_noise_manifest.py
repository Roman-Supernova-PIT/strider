#!/usr/bin/env python3
"""Write an exact Sundial cohort shared by legacy and current noise tests."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from strider.config import load_config
from strider.data.dataset import SundialDataset
from strider.evaluation.noise_check import _balanced_ia_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--objects-per-bin", type=int, default=100)
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="also write this many deterministic manifest shards",
    )
    cohort = parser.add_mutually_exclusive_group()
    cohort.add_argument(
        "--all-ia",
        action="store_true",
        help="write every Ia in the split instead of a redshift-balanced subset",
    )
    cohort.add_argument(
        "--all-classes",
        action="store_true",
        help="write the complete mixed-class split for purity and completeness",
    )
    parser.add_argument(
        "--redshift-edges",
        nargs="+",
        type=float,
        default=[0.0, 0.75, 1.25, 1.75, 2.25, 3.0],
    )
    args = parser.parse_args()
    if args.shard_count <= 0:
        raise ValueError("--shard-count must be positive")

    config = load_config(args.config)
    view = "reported_error_with_source"
    dataset = SundialDataset(config, args.split, view, training=False)
    if args.all_classes:
        indices = dataset.objects.index.astype(int).tolist()
    elif args.all_ia:
        indices = dataset.objects.index[
            dataset.objects["class_name"].eq("Ia")
        ].astype(int).tolist()
    else:
        indices = _balanced_ia_indices(
            config,
            args.split,
            view,
            args.redshift_edges,
            args.objects_per_bin,
        )
        if indices is None:
            raise RuntimeError("balanced selection unexpectedly returned no indices")
    cohort = dataset.objects.iloc[indices].copy()
    cohort = cohort.rename(columns={"redshift": "z_true"})
    cohort["paired_noise_key"] = cohort["snid"].astype(str)
    keep = [
        column
        for column in (
            "snid",
            "z_true",
            "class_name",
            "class_index",
            "source_product",
            "block",
            "model",
            "paired_noise_key",
        )
        if column in cohort
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected = cohort[keep].reset_index(drop=True)
    selected.to_csv(args.output, index=False)
    print(f"wrote {len(cohort):,} objects to {args.output}")
    if args.shard_count > 1:
        for shard_index in range(args.shard_count):
            shard = selected.iloc[shard_index :: args.shard_count]
            shard_path = args.output.with_name(
                f"{args.output.stem}_shard_{shard_index:02d}_of_"
                f"{args.shard_count:02d}{args.output.suffix}"
            )
            shard.to_csv(shard_path, index=False)
            print(f"wrote {len(shard):,} objects to {shard_path}")


if __name__ == "__main__":
    main()
