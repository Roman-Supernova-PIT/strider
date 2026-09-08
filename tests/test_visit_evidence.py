"""Visit accumulation must have an explicit, testable count response."""

from __future__ import annotations

import torch

from strider.model.visit_evidence import combine_visit_scores


def test_zero_exponent_is_the_valid_visit_mean() -> None:
    scores = torch.tensor([[[[2.0]], [[4.0]], [[100.0]]]])
    valid = torch.tensor([[[[True]], [[True]], [[False]]]])
    combined = combine_visit_scores(scores, valid, exponent=0.0)
    assert torch.equal(combined, torch.tensor([[[3.0]]]))


def test_unit_exponent_is_the_valid_visit_sum() -> None:
    scores = torch.tensor([[[[2.0]], [[4.0]], [[100.0]]]])
    valid = torch.tensor([[[[True]], [[True]], [[False]]]])
    combined = combine_visit_scores(scores, valid, exponent=1.0)
    assert torch.equal(combined, torch.tensor([[[6.0]]]))


def test_invalid_exponent_is_rejected() -> None:
    scores = torch.ones(1, 1, 1, 1)
    valid = torch.ones_like(scores, dtype=torch.bool)
    try:
        combine_visit_scores(scores, valid, exponent=1.1)
    except ValueError as error:
        assert "[0, 1]" in str(error)
    else:
        raise AssertionError("invalid visit evidence exponent was accepted")
