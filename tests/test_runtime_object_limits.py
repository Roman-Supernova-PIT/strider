import pandas as pd

from strider.data.dataset import _apply_runtime_object_limit


def test_zero_runtime_object_limit_uses_complete_split() -> None:
    objects = pd.DataFrame({"object_index": [0, 1, 2]})

    complete = _apply_runtime_object_limit(
        objects,
        split="test",
        runtime_limits={"test": 0},
        seed=7,
    )
    limited = _apply_runtime_object_limit(
        objects,
        split="test",
        runtime_limits={"test": 1},
        seed=7,
    )

    assert len(complete) == 3
    assert len(limited) == 1
