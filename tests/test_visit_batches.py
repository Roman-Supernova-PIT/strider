from strider.training.visit_batches import VisitCountBatchSampler
from strider.training.visit_batches import visit_limited_batch_size


class _Dataset:
    seed = 17
    training_epoch = 3

    def __init__(self) -> None:
        self.counts = [4, 8, 4, 16, 8, 16, 4, 8, 16, 4]

    def __len__(self) -> int:
        return len(self.counts)

    def requested_visit_count(self, item: int) -> int:
        return self.counts[item]


def test_each_training_batch_has_one_requested_visit_count() -> None:
    dataset = _Dataset()
    sampler = VisitCountBatchSampler(dataset, batch_size=3)
    batches = list(sampler)
    assert sorted(item for batch in batches for item in batch) == list(range(len(dataset)))
    for batch in batches:
        assert len({dataset.requested_visit_count(item) for item in batch}) == 1


def test_visit_count_batches_are_repeatable_within_an_epoch() -> None:
    dataset = _Dataset()
    first = list(VisitCountBatchSampler(dataset, batch_size=3))
    second = list(VisitCountBatchSampler(dataset, batch_size=3))
    assert first == second
    dataset.training_epoch = 4
    changed = list(VisitCountBatchSampler(dataset, batch_size=3))
    assert changed != first


def test_long_sequences_receive_smaller_microbatches() -> None:
    dataset = _Dataset()
    sampler = VisitCountBatchSampler(
        dataset,
        batch_size=16,
        shuffle=False,
        maximum_visits_per_batch=64,
        maximum_squared_visits_per_batch=256,
    )
    batches = list(sampler)
    for batch in batches:
        visits = dataset.requested_visit_count(batch[0])
        assert len(batch) <= visit_limited_batch_size(16, visits, 64, 256)
        assert len(batch) * visits <= 64
        assert len(batch) * visits * visits <= 256


def test_visit_budget_never_reduces_a_microbatch_below_one() -> None:
    assert visit_limited_batch_size(16, 137, 512, 16384) == 1
