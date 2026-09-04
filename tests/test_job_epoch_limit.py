"""Scheduler-job epoch limits must be strict and checkpoint-compatible."""

from types import SimpleNamespace

import pytest
import torch

from strider.training.trainer import _job_epoch_limit, _learned_scales


def test_job_epoch_limit_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STRIDER_MAX_EPOCHS_THIS_JOB", raising=False)
    assert _job_epoch_limit() is None


def test_job_epoch_limit_reads_positive_integer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STRIDER_MAX_EPOCHS_THIS_JOB", "2")
    assert _job_epoch_limit() == 2


@pytest.mark.parametrize("value", ["0", "-1", "two"])
def test_job_epoch_limit_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("STRIDER_MAX_EPOCHS_THIS_JOB", value)
    with pytest.raises(ValueError, match="positive integer"):
        _job_epoch_limit()


def test_learned_scales_reports_effective_continuum_fraction() -> None:
    model = SimpleNamespace(
        factored_evidence=SimpleNamespace(
            shape_scale=torch.tensor(0.0),
            temporal_scale=torch.tensor(0.0),
            brightness_scale=None,
        ),
        full_spectrum_context=None,
        dense_scan=SimpleNamespace(
            scale=torch.tensor(0.0),
            detail_intercept=torch.tensor(0.0),
            maximum_detail_weight=0.5,
        ),
        relative_amplitude_mode="none",
        phase_consistency=None,
    )

    assert _learned_scales(model)["dense_detail"] == pytest.approx(0.25)
