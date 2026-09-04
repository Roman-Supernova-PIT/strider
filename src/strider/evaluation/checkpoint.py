"""Load one trained model and verify that its configuration still matches."""

from __future__ import annotations

from typing import Any

import torch

from strider.config import project_path, resolved_config_sha256
from strider.model import Strider
from strider.training.device import choose_device


def load_trained_model(
    config: dict[str, Any],
) -> tuple[Strider, dict[str, Any], torch.device]:
    output_dir = project_path(config, config["project"]["output_dir"])
    checkpoint = torch.load(
        output_dir / "best_model.pt", map_location="cpu", weights_only=True
    )
    if checkpoint.get("config_sha256") != resolved_config_sha256(config):
        raise ValueError(
            "The checkpoint and resolved configuration do not match. "
            "Use the config.resolved.yaml stored with the run."
        )
    device = choose_device()
    model = Strider(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint, device
