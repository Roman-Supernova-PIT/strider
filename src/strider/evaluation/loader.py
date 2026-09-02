"""Memory-aware inference loaders for variable-length spectral time series."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from strider.data.dataset import collate_objects
from strider.training.visit_batches import VisitCountBatchSampler


def inference_loader(
    dataset: Dataset,
    config: dict[str, Any],
    *,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> DataLoader:
    """Build an inference loader without padding every object to the longest one."""
    settings = config["training"]
    object_batch_size = int(batch_size or settings["batch_size"])
    workers = int(settings["num_workers"] if num_workers is None else num_workers)
    persistent = (
        bool(settings.get("persistent_workers", workers > 0)) if workers else False
    )
    options: dict[str, Any] = {}
    if workers:
        options["prefetch_factor"] = int(settings.get("prefetch_factor", 2))
    common = {
        "num_workers": workers,
        "collate_fn": collate_objects,
        "persistent_workers": persistent,
        "pin_memory": torch.cuda.is_available(),
        **options,
    }
    if bool(settings.get("batch_by_visit_count", False)):
        return DataLoader(
            dataset,
            batch_sampler=VisitCountBatchSampler(
                dataset,
                batch_size=object_batch_size,
                shuffle=False,
                maximum_visits_per_batch=settings.get(
                    "maximum_visits_per_batch"
                ),
                maximum_squared_visits_per_batch=settings.get(
                    "maximum_squared_visits_per_batch"
                ),
            ),
            **common,
        )
    return DataLoader(
        dataset,
        batch_size=object_batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
