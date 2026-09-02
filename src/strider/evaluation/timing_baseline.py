"""Train and evaluate the observation-schedule-only diagnostic."""

from __future__ import annotations

import json
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from strider.config import project_path
from strider.data.dataset import SundialDataset, collate_objects
from strider.model.timing_only import TimingOnlyModel
from strider.training.device import choose_device
from strider.training.losses import joint_targets

from .evaluate import _predict
from .metrics import metrics_by_redshift, source_metrics


def run_timing_baseline(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config["project"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = choose_device()
    settings = config["training"]
    selection_split = str(settings.get("selection_split", "validation"))
    evaluation_split = str(config["evaluation"].get("split", "test"))
    training_data = SundialDataset(
        config, "train", "generated", training=True, pair_no_source=False
    )
    training_data.no_source_fraction = 0.0
    selection_data = SundialDataset(
        config, selection_split, "generated", training=False, pair_no_source=False
    )
    loaders = {
        "train": _loader(training_data, settings, True),
        "selection": _loader(selection_data, settings, False),
    }
    model = TimingOnlyModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    best_state = None
    best_validation = float("inf")
    history = []
    epochs = int(settings.get("timing_baseline_epochs", settings["epochs"]))
    for epoch in range(epochs):
        training_data.set_epoch(epoch)
        values = {}
        for split, loader in loaders.items():
            is_training = split == "train"
            model.train(is_training)
            total = 0.0
            count = 0
            for batch in loader:
                batch = {name: value.to(device) for name, value in batch.items()}
                if is_training:
                    optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(is_training):
                    outputs = model(batch)
                    targets = joint_targets(
                        batch["class_index"], batch["redshift"], model.redshift_grid
                    )
                    loss = functional.cross_entropy(
                        outputs["joint_logits"].reshape(len(targets), -1), targets
                    )
                    if is_training:
                        loss.backward()
                        optimizer.step()
                total += float(loss.detach().cpu()) * len(batch["snid"])
                count += len(batch["snid"])
            values[split] = total / max(count, 1)
        history.append({"epoch": epoch + 1, **values})
        if values["selection"] < best_validation:
            best_validation = values["selection"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Timing-only training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    evaluation_data = SundialDataset(
        config, evaluation_split, "generated", training=False, pair_no_source=False
    )
    predictions = _predict(model, _loader(evaluation_data, settings, False), device)
    threshold = float(config["evaluation"]["outlier_delta_z"])
    report = {
        "device": str(device),
        "selection_split": selection_split,
        "evaluation_split": evaluation_split,
        "best_selection_loss": best_validation,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "evaluation": source_metrics(predictions, threshold),
        "redshift_groups": metrics_by_redshift(
            predictions, threshold, config["data"]["redshift_edges"]
        ),
        "history": history,
    }
    output_dir = project_path(config, config["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(
        output_dir / f"{evaluation_split}_timing_only_predictions.parquet", index=False
    )
    with (output_dir / "timing_only_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def _loader(dataset: SundialDataset, settings: dict[str, Any], shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=shuffle,
        num_workers=int(settings["num_workers"]),
        collate_fn=collate_objects,
        persistent_workers=False,
    )
