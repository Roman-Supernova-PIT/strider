"""Controlled check of the factored temporal branch.

Each object starts at a different phase and evolves over random rest-frame
intervals. No SNANA flux, errors, cadence, or class balance enters this example.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from strider.model.factored_attention import FactoredOnirEvidence
from strider.training.device import choose_device


def run_temporal_example(
    *,
    output: Path,
    epochs: int = 40,
    training_objects: int = 900,
    test_objects: int = 300,
    seed: int = 4,
    random_start_phase: bool = True,
    batch_size: int = 100,
    redshift_bins: int = 15,
    training_gap_min: int = 4,
    training_gap_max: int = 18,
    test_gap_min: int | None = None,
    test_gap_max: int | None = None,
    feature_noise_std: float = 0.03,
    intrinsic_variation_std: float = 0.0,
    minimum_training_visits: int = 5,
    binary: bool = False,
) -> dict[str, Any]:
    training_gap_range = _day_range(
        training_gap_min, training_gap_max, "training gap"
    )
    if (test_gap_min is None) != (test_gap_max is None):
        raise ValueError("test_gap_min and test_gap_max must be set together")
    test_gap_range = (
        None
        if test_gap_min is None
        else _day_range(test_gap_min, test_gap_max, "test gap")
    )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if redshift_bins < 2:
        raise ValueError("redshift_bins must be at least two")
    if not 1 <= minimum_training_visits <= 5:
        raise ValueError("minimum_training_visits must be between one and five")
    if feature_noise_std < 0 or intrinsic_variation_std < 0:
        raise ValueError("spectral variation values cannot be negative")
    torch.manual_seed(seed)
    device = choose_device()
    data = _make_sequences(
        training_objects + test_objects,
        seed,
        random_start_phase=random_start_phase,
        training_objects=training_objects,
        redshift_bins=redshift_bins,
        training_gap_range=training_gap_range,
        test_gap_range=test_gap_range,
        feature_noise_std=feature_noise_std,
        intrinsic_variation_std=intrinsic_variation_std,
        minimum_training_visits=minimum_training_visits,
        binary=binary,
    )
    class_count = 2 if binary else 3
    model = FactoredOnirEvidence(
        hidden_dim=16,
        class_count=class_count,
        attention_heads=4,
        dropout=0.0,
        shape_initial_scale=0.5,
        temporal_initial_scale=0.5,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3)
    training = torch.arange(training_objects)
    for _ in range(epochs):
        order = training[torch.randperm(training_objects)]
        for selected in order.split(batch_size):
            batch = _select(data, selected, device)
            output_values = model(
                batch["features"],
                batch["support"],
                batch["observer_days"],
                batch["visit_mask"],
                batch["redshift_grid"],
            )
            logits = output_values["shape_joint_logits"] + output_values[
                "temporal_joint_logits"
            ]
            target = batch["class_index"] * len(batch["redshift_grid"]) + batch[
                "redshift_index"
            ]
            loss = F.cross_entropy(logits.flatten(1), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    test = torch.arange(training_objects, training_objects + test_objects)
    report = {
        "device": str(device),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "training_objects": int(training_objects),
        "test_objects": int(test_objects),
        "redshift_bins": int(redshift_bins),
        "redshift_grid_step": float(data["redshift_grid"][1] - data["redshift_grid"][0]),
        "starting_phase_range_days": (
            [-25.0, 25.0] if random_start_phase else [0.0, 0.0]
        ),
        "training_rest_gap_range_days": list(training_gap_range),
        "test_rest_gap_range_days": (
            list(test_gap_range) if test_gap_range is not None else list(training_gap_range)
        ),
        "feature_noise_std": float(feature_noise_std),
        "intrinsic_variation_std": float(intrinsic_variation_std),
        "training_visit_count_range": [minimum_training_visits, 5],
        "class_names": ["Ia", "non-Ia"] if binary else ["class 0", "class 1", "class 2"],
        "non_ia_families": 14 if binary else 0,
        "cases": {},
    }
    with torch.no_grad():
        batch = _select(data, test, device)
        times = {
            "correct_dates": batch["observer_days"],
            "shifted_dates": batch["observer_days"] + 10_000.0,
            "reversed_dates": batch["observer_days"].flip(1),
            "reassigned_dates": batch["observer_days"].roll(1, dims=0),
        }
        for name, observer_days in times.items():
            report["cases"][name] = _score(model, batch, observer_days)
        identical = batch["features"][:, :1].expand_as(batch["features"])
        no_change = model(
            identical,
            batch["support"],
            batch["observer_days"],
            batch["visit_mask"],
            batch["redshift_grid"],
        )
        report["identical_visit_max_temporal_logit"] = float(
            no_change["raw_temporal_joint_logits"].abs().max().cpu()
        )
        correct = model(
            batch["features"],
            batch["support"],
            batch["observer_days"],
            batch["visit_mask"],
            batch["redshift_grid"],
        )
        shifted = model(
            batch["features"],
            batch["support"],
            batch["observer_days"] + 10_000.0,
            batch["visit_mask"],
            batch["redshift_grid"],
        )
        report["absolute_date_shift_max_logit_difference"] = float(
            (
                correct["temporal_joint_logits"]
                - shifted["temporal_joint_logits"]
            ).abs().max().cpu()
        )
        report["correct_date_branches"] = {
            "shape_only": _score_logits(
                correct["shape_joint_logits"], batch
            ),
            "temporal_only": _score_logits(
                correct["temporal_joint_logits"], batch
            ),
            "combined": _score_logits(
                correct["shape_joint_logits"] + correct["temporal_joint_logits"],
                batch,
            ),
        }
        report["visit_counts"] = {}
        for visit_count in (1, 2, 3, 5):
            visit_mask = batch["visit_mask"].clone()
            visit_mask[:, visit_count:] = False
            output_values = model(
                batch["features"],
                batch["support"],
                batch["observer_days"],
                visit_mask,
                batch["redshift_grid"],
            )
            report["visit_counts"][str(visit_count)] = _score_logits(
                output_values["shape_joint_logits"]
                + output_values["temporal_joint_logits"],
                batch,
            )
        report["learned_scales"] = {
            "shape": float(torch.tanh(model.shape_scale).cpu()),
            "temporal": float(torch.tanh(model.temporal_scale).cpu()),
        }

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _make_sequences(
    objects: int,
    seed: int,
    *,
    random_start_phase: bool,
    training_objects: int | None = None,
    redshift_bins: int = 15,
    training_gap_range: tuple[int, int] = (4, 18),
    test_gap_range: tuple[int, int] | None = None,
    feature_noise_std: float = 0.03,
    intrinsic_variation_std: float = 0.0,
    minimum_training_visits: int = 5,
    binary: bool = False,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    output_classes = 2 if binary else 3
    spectral_families = 15 if binary else output_classes
    redshifts, regions, dimensions, visits = redshift_bins, 6, 16, 5
    redshift_grid = torch.linspace(0.1, 2.5, redshifts)
    if binary:
        class_index = torch.randint(2, (objects,), generator=generator)
        non_ia_family = torch.randint(
            1, spectral_families, (objects,), generator=generator
        )
        family_index = torch.where(
            class_index == 0, torch.zeros_like(non_ia_family), non_ia_family
        )
    else:
        family_index = torch.randint(
            spectral_families, (objects,), generator=generator
        )
        class_index = family_index
    redshift_index = torch.randint(redshifts, (objects,), generator=generator)
    true_redshift = redshift_grid[redshift_index]
    training_minimum, training_maximum = training_gap_range
    gaps = torch.randint(
        training_minimum,
        training_maximum + 1,
        (objects, visits - 1),
        generator=generator,
    ).float()
    if test_gap_range is not None:
        if training_objects is None or not 0 < training_objects < objects:
            raise ValueError("training_objects must separate train and test sequences")
        minimum, maximum = test_gap_range
        gaps[training_objects:] = torch.randint(
            minimum,
            maximum + 1,
            (objects - training_objects, visits - 1),
            generator=generator,
        ).float()
    elapsed_days = torch.cat(
        [torch.zeros(objects, 1), gaps.cumsum(dim=1)], dim=1
    )
    if random_start_phase:
        starting_phase = -25.0 + 50.0 * torch.rand(
            objects, 1, generator=generator
        )
    else:
        starting_phase = torch.zeros(objects, 1)
    rest_phase = starting_phase + elapsed_days
    observer_days = elapsed_days * (1.0 + true_redshift[:, None])

    base = torch.randn(spectral_families, regions, dimensions, generator=generator) * 0.8
    slope = torch.randn(spectral_families, regions, dimensions, generator=generator) * 0.6
    curve = torch.randn(spectral_families, regions, dimensions, generator=generator) * 0.5
    frequency = torch.linspace(11.0, 27.0, spectral_families)
    phase = rest_phase / frequency[family_index, None]
    object_offset = intrinsic_variation_std * torch.randn(
        objects, regions, 1, dimensions, generator=generator
    )
    slope_scale = 1.0 + intrinsic_variation_std * torch.randn(
        objects, 1, 1, 1, generator=generator
    )
    curve_scale = 1.0 + intrinsic_variation_std * torch.randn(
        objects, 1, 1, 1, generator=generator
    )
    features = base[family_index, :, None, :] + object_offset
    features = features + slope_scale * rest_phase[:, None, :, None] * slope[
        family_index, :, None, :
    ] / 35.0
    features = features + curve_scale * torch.sin(
        phase[:, None, :, None]
    ) * curve[family_index, :, None, :]
    features = features.permute(0, 2, 1, 3)
    features = features + feature_noise_std * torch.randn(
        features.shape, generator=generator
    )
    features = features[:, :, None].expand(-1, -1, redshifts, -1, -1)
    visit_mask = torch.ones(objects, visits, dtype=torch.bool)
    if minimum_training_visits < visits:
        if training_objects is None or not 0 < training_objects < objects:
            raise ValueError("training_objects must separate train and test sequences")
        visit_count = torch.randint(
            minimum_training_visits,
            visits + 1,
            (training_objects,),
            generator=generator,
        )
        visit_mask[:training_objects] = (
            torch.arange(visits)[None, :] < visit_count[:, None]
        )
    return {
        "features": features,
        "support": torch.ones(objects, visits, redshifts, regions, dtype=torch.bool),
        "observer_days": observer_days,
        "visit_mask": visit_mask,
        "redshift_grid": redshift_grid,
        "class_index": class_index,
        "redshift_index": redshift_index,
        "true_redshift": true_redshift,
        "rest_phase": rest_phase,
    }


def _day_range(minimum: int, maximum: int, name: str) -> tuple[int, int]:
    minimum = int(minimum)
    maximum = int(maximum)
    if minimum <= 0 or maximum < minimum:
        raise ValueError(f"{name} must contain positive increasing days")
    return minimum, maximum


def _select(
    data: dict[str, torch.Tensor],
    selected: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    shared = {"redshift_grid"}
    return {
        name: (value if name in shared else value[selected]).to(device)
        for name, value in data.items()
    }


def _score(
    model: FactoredOnirEvidence,
    batch: dict[str, torch.Tensor],
    observer_days: torch.Tensor,
) -> dict[str, float]:
    output = model(
        batch["features"],
        batch["support"],
        observer_days,
        batch["visit_mask"],
        batch["redshift_grid"],
    )
    logits = output["shape_joint_logits"] + output["temporal_joint_logits"]
    return _score_logits(logits, batch)


def _score_logits(
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    flat_prediction = logits.flatten(1).argmax(dim=1)
    redshift_count = len(batch["redshift_grid"])
    predicted_class = flat_prediction // redshift_count
    predicted_redshift = batch["redshift_grid"][flat_prediction % redshift_count]
    delta = predicted_redshift - batch["true_redshift"]
    return {
        "class_accuracy": float(
            (predicted_class == batch["class_index"]).float().mean().cpu()
        ),
        "median_delta_z": float(delta.median().cpu()),
        "median_absolute_delta_z": float(delta.abs().median().cpu()),
    }
