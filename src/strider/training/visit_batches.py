"""Training batches with one independently assigned visit count per batch."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler, Subset

from strider.data.dataset import SundialDataset
from strider.data.noise import repeatable_rng


class VisitCountBatchSampler(Sampler[list[int]]):
    """Group examples by the visit count already assigned by the dataset."""

    def __init__(
        self,
        dataset: SundialDataset | Subset,
        batch_size: int,
        shuffle: bool = True,
        maximum_visits_per_batch: int | None = None,
        maximum_squared_visits_per_batch: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.maximum_visits_per_batch = _optional_positive_budget(
            maximum_visits_per_batch, "maximum_visits_per_batch"
        )
        self.maximum_squared_visits_per_batch = _optional_positive_budget(
            maximum_squared_visits_per_batch,
            "maximum_squared_visits_per_batch",
        )

    def __iter__(self) -> Iterator[list[int]]:
        groups = self._groups()
        rng = repeatable_rng(
            _dataset_attribute(self.dataset, "seed"),
            _dataset_attribute(self.dataset, "training_epoch"),
            "visit_count_batches",
        )
        batches: list[list[int]] = []
        for visit_count in sorted(groups):
            indices = np.asarray(groups[visit_count], dtype=np.int64)
            if self.shuffle:
                rng.shuffle(indices)
            batch_size = self._batch_size(visit_count)
            batches.extend(
                indices[start:start + batch_size].tolist()
                for start in range(0, len(indices), batch_size)
            )
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return sum(
            (len(indices) + self._batch_size(visit_count) - 1)
            // self._batch_size(visit_count)
            for visit_count, indices in self._groups().items()
        )

    def _batch_size(self, visit_count: int) -> int:
        return visit_limited_batch_size(
            self.batch_size,
            visit_count,
            self.maximum_visits_per_batch,
            self.maximum_squared_visits_per_batch,
        )

    def _groups(self) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = defaultdict(list)
        for item in range(len(self.dataset)):
            groups[_requested_visit_count(self.dataset, item)].append(item)
        return dict(groups)


def _optional_positive_budget(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def visit_limited_batch_size(
    object_batch_size: int,
    visit_count: int,
    maximum_visits_per_batch: int | None = None,
    maximum_squared_visits_per_batch: int | None = None,
) -> int:
    """Choose a practical microbatch size for one common visit count.

    At least one object is retained even when a single unusually long history
    exceeds a nominal budget. The GPU benchmark verifies that edge case.
    """
    size = int(object_batch_size)
    visits = int(visit_count)
    if size < 1 or visits < 1:
        raise ValueError("object_batch_size and visit_count must be positive")
    linear_budget = _optional_positive_budget(
        maximum_visits_per_batch, "maximum_visits_per_batch"
    )
    quadratic_budget = _optional_positive_budget(
        maximum_squared_visits_per_batch,
        "maximum_squared_visits_per_batch",
    )
    if linear_budget is not None:
        size = min(size, linear_budget // visits)
    if quadratic_budget is not None:
        size = min(size, quadratic_budget // (visits * visits))
    return max(1, size)


def _requested_visit_count(dataset: SundialDataset | Subset, item: int) -> int:
    if isinstance(dataset, Subset):
        return dataset.dataset.requested_visit_count(int(dataset.indices[item]))
    return dataset.requested_visit_count(item)


def _dataset_attribute(dataset: SundialDataset | Subset, name: str) -> int:
    source = dataset.dataset if isinstance(dataset, Subset) else dataset
    return int(getattr(source, name))
