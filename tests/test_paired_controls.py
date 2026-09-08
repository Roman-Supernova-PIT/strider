"""The larger paired evaluation must not alter the fitted-model config."""

from __future__ import annotations

import pytest

from strider.evaluation.paired_controls import _config_with_object_limit


def test_evaluation_object_limit_leaves_checkpoint_config_unchanged() -> None:
    config = {"data": {"runtime_object_limits": {"calibration": 300}}}

    evaluation_config = _config_with_object_limit(config, "calibration", 1600)

    assert config["data"]["runtime_object_limits"]["calibration"] == 300
    assert evaluation_config["data"]["runtime_object_limits"]["calibration"] == 1600


def test_evaluation_object_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _config_with_object_limit({"data": {}}, "calibration", 0)
