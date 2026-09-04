"""The from-scratch schedule must warm up before cosine decay."""

import torch

from strider.training.trainer import _learning_rate_scheduler


def test_warmup_then_cosine_schedule() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    scheduler = _learning_rate_scheduler(
        optimizer,
        {
            "epochs": 10,
            "learning_rate_schedule": "cosine",
            "warmup_epochs": 2,
            "minimum_learning_rate_fraction": 0.1,
        },
    )
    learning_rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(9):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])
    assert learning_rates[0] == 5e-4
    assert learning_rates[1] == 1e-3
    assert learning_rates[-1] == 1e-4
    assert all(
        left >= right for left, right in zip(learning_rates[1:], learning_rates[2:])
    )
