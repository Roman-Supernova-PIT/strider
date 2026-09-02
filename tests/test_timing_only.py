from __future__ import annotations

import torch

from strider.model.timing_only import TimingOnlyModel


def test_time_features_include_visit_gaps_and_ignore_padding() -> None:
    times = torch.tensor([[0.0, 2.0, 8.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

    features = TimingOnlyModel._time_features(times, mask)

    assert features.shape == (2, 8)
    assert torch.isfinite(features).all()
    assert torch.isclose(features[0, 4], torch.tensor(4.0 / 30.0))
    assert torch.isclose(features[0, 6], torch.tensor(2.0 / 30.0))
    assert torch.isclose(features[0, 7], torch.tensor(6.0 / 30.0))
    assert torch.equal(features[1, 4:], torch.zeros(4))
