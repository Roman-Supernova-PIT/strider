import os
from pathlib import Path

import pytest

from strider.data.snana import discover_source_pairs, read_head_table


SOURCE_SETTING = os.environ.get("STRIDER_SUNDIAL_DIR")
SOURCE = Path(SOURCE_SETTING).expanduser() if SOURCE_SETTING else Path("__not_configured__")
pytestmark = pytest.mark.skipif(
    not SOURCE.is_dir(),
    reason="set STRIDER_SUNDIAL_DIR to run the local SNANA reader tests",
)


def test_discovers_complete_pairs() -> None:
    pairs = discover_source_pairs(SOURCE)
    assert pairs
    assert all(Path(pair.head_path).is_file() for pair in pairs)
    assert all(Path(pair.spec_path).is_file() for pair in pairs)
    assert len({(pair.block, pair.model) for pair in pairs}) == len(pairs)


def test_head_has_required_values() -> None:
    pair = discover_source_pairs(SOURCE)[0]
    frame = read_head_table(pair)
    assert len(frame) > 0
    assert frame["snid"].is_unique
    assert frame["redshift"].between(0, 4).all()
