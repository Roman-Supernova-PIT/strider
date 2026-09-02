"""Short forward/backward timing test for choosing a local batch size."""

from __future__ import annotations

import time
import os
import platform
import resource
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from strider.data.dataset import SundialDataset, collate_objects
from strider.model import Strider, measurement_inputs

from .device import choose_device
from .losses import training_loss
from .precision import autocast_context
from .visit_batches import VisitCountBatchSampler, visit_limited_batch_size


def benchmark(config: dict[str, Any], batch_sizes: list[int] | None = None) -> dict[str, Any]:
    device = choose_device()
    dataset = SundialDataset(config, "train", "generated", training=True)
    sizes = batch_sizes or [
        int(value)
        for value in config["training"].get("benchmark_batch_sizes", [4, 8, 12, 16, 24])
    ]
    results = []
    # Exercise the longest sequence in the prepared training data. The object
    # batch size remains the optimization target; visit budgets choose the GPU
    # microbatch used for this worst-case forward/backward pass.
    maximum_visits = max(dataset._observation_counts)
    dataset.max_visits = None
    dataset.training_visit_counts = (None,)
    longest_positions = np.argsort(np.asarray(dataset._observation_counts))[::-1]
    for batch_size in sizes:
        microbatch_size = visit_limited_batch_size(
            batch_size,
            maximum_visits,
            config["training"].get("maximum_visits_per_batch"),
            config["training"].get("maximum_squared_visits_per_batch"),
        )
        object_positions = longest_positions[:microbatch_size]
        item_indices = (
            2 * object_positions if dataset.pair_no_source else object_positions
        )
        items = [dataset[int(index)] for index in item_indices]
        batch = {name: value.to(device) for name, value in collate_objects(items).items()}
        visit_count = int(batch["flux"].shape[1])
        model = Strider(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        failed = False
        try:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            _training_step(model, batch, optimizer, config)
            _synchronize(device)
            start = time.perf_counter()
            repeats = 3
            for _ in range(repeats):
                _training_step(model, batch, optimizer, config)
            _synchronize(device)
            seconds = (time.perf_counter() - start) / repeats
            results.append(
                {
                    "batch_size": batch_size,
                    "microbatch_size": microbatch_size,
                    "visits_per_object": visit_count,
                    "seconds_per_step": seconds,
                    "objects_per_second": microbatch_size / seconds,
                    "peak_gpu_memory_gib": _peak_gpu_memory_gib(device),
                    "process_rss_gib": _process_rss_gib(),
                    "open_file_count": _open_file_count(),
                    "status": "passed",
                }
            )
        except RuntimeError as error:
            results.append(
                {
                    "batch_size": batch_size,
                    "microbatch_size": microbatch_size,
                    "visits_per_object": visit_count,
                    "status": "failed",
                    "reason": str(error)[:300],
                }
            )
            failed = True
        finally:
            del optimizer, model, batch, items
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()
        if failed:
            break
    configured_batch_size = int(config["training"]["batch_size"])
    configured_result = next(
        (result for result in results if result["batch_size"] == configured_batch_size),
        None,
    )
    configured_batch_passed = bool(
        configured_result is not None and configured_result["status"] == "passed"
    )
    return {
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in Strider(config).parameters()),
        "candidate_redshift_bins": int(config["model"]["redshift_bins"]),
        "wavelength_bins": int(config["observation"]["wavelength_bins"]),
        "configured_batch_size": configured_batch_size,
        "configured_batch_passed": configured_batch_passed,
        "model_step_results": results,
        "data_loading_results": _benchmark_data_loading(config),
    }


def _benchmark_data_loading(config: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    batch_size = int(config["training"]["batch_size"])
    settings = config["training"]
    worker_counts = [
        int(value) for value in settings.get("benchmark_worker_counts", [0, 4, 8, 12])
    ]
    maximum_batches = int(settings.get("benchmark_batches", 10))
    for trial, workers in enumerate(worker_counts):
        dataset = SundialDataset(config, "train", "generated", training=True)
        requested = maximum_batches * batch_size
        start_index = (trial * requested) % max(len(dataset), 1)
        indices = (start_index + torch.arange(min(requested, len(dataset)))) % len(dataset)
        trial_data = Subset(dataset, indices.tolist())
        options: dict[str, Any] = {}
        if workers:
            options["persistent_workers"] = True
            options["prefetch_factor"] = int(settings.get("prefetch_factor", 2))
        loader = DataLoader(
            trial_data,
            batch_sampler=VisitCountBatchSampler(
                trial_data,
                batch_size=batch_size,
                shuffle=False,
                maximum_visits_per_batch=settings.get(
                    "maximum_visits_per_batch"
                ),
                maximum_squared_visits_per_batch=settings.get(
                    "maximum_squared_visits_per_batch"
                ),
            ),
            num_workers=workers,
            collate_fn=collate_objects,
            pin_memory=torch.cuda.is_available(),
            **options,
        )
        start = time.perf_counter()
        object_count = 0
        for batch_number, batch in enumerate(loader):
            object_count += int(batch["flux"].shape[0])
            if batch_number + 1 >= maximum_batches:
                break
        seconds = time.perf_counter() - start
        results.append(
            {
                "workers": workers,
                "objects": object_count,
                "seconds": seconds,
                "objects_per_second": object_count / seconds,
                "estimated_seconds_per_training_epoch": len(dataset)
                / max(object_count / seconds, 1e-12),
                "process_rss_gib": _process_rss_gib(),
                "open_file_count": _open_file_count(),
            }
        )
    return results


def _training_step(
    model: Strider,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> None:
    optimizer.zero_grad(set_to_none=True)
    settings = config["training"]
    precision = str(settings.get("mixed_precision", "float32"))
    with autocast_context(model.redshift_grid.device, precision):
        output = model(measurement_inputs(batch))
        loss, _ = training_loss(
            output,
            batch,
            model.redshift_grid,
            model.redshift_cell_width,
            model.redshift_prior,
            float(settings["evidence_sufficiency_loss_weight"]),
            float(settings.get("no_source_redshift_loss_weight", 0.0)),
            float(settings.get("no_source_class_loss_weight", 0.0)),
            float(
                config.get(
                    "reference"
                    if str(config["model"].get("architecture", ""))
                    == "roman_reference"
                    else "onir",
                    {},
                ).get("drift_loss_weight", 0.0)
            ),
            phase_loss_weight=float(settings.get("phase_loss_weight", 0.0)),
            coadd_reconstruction_loss_weight=float(
                settings.get("coadd_reconstruction_loss_weight", 0.0)
            ),
            alias_ranking_loss_weight=float(
                settings.get("alias_ranking_loss_weight", 0.0)
            ),
            alias_ranking_minimum_delta_z=float(
                settings.get("alias_ranking_minimum_delta_z", 0.1)
            ),
            alias_ranking_margin=float(settings.get("alias_ranking_margin", 0.0)),
        )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), float(settings.get("max_gradient_norm", 1.0))
    )
    optimizer.step()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _peak_gpu_memory_gib(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / 1024**3)


def _process_rss_gib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux, including Perlmutter, reports KiB.
    bytes_used = value if platform.system() == "Darwin" else value * 1024.0
    return bytes_used / 1024**3


def _open_file_count() -> int | None:
    for directory in ("/proc/self/fd", "/dev/fd"):
        try:
            return len(os.listdir(directory))
        except OSError:
            continue
    return None
