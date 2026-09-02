"""Write the resolved configuration and execution environment for a run."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from strider.config import resolved_config_sha256, resolved_config_text


def write_run_record(config: dict[str, Any], output_dir: Path, device: torch.device) -> str:
    """Write deterministic configuration text and a concise environment record."""
    config_text = resolved_config_text(config)
    digest = resolved_config_sha256(config)
    config_path = output_dir / "config.resolved.yaml"
    if config_path.exists() and config_path.read_text(encoding="utf-8") != config_text:
        raise ValueError(f"Run directory contains a different resolved config: {output_dir}")
    config_path.write_text(config_text, encoding="utf-8")
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda": torch.version.cuda,
        "gpu": _gpu_record(device),
        "packages": _package_versions(),
        "config_sha256": digest,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm": {
            name: os.environ.get(name)
            for name in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NAME",
                "SLURM_JOB_QOS",
                "SLURM_JOB_ACCOUNT",
                "SLURM_CPUS_PER_TASK",
                "SLURM_JOB_NUM_NODES",
            )
        },
        "git": _git_state(Path(config["_project_root"])),
    }
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    return digest


def _package_versions() -> dict[str, str | None]:
    names = ("astropy", "h5py", "numpy", "pandas", "pyarrow", "PyYAML", "torch")
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _gpu_record(device: torch.device) -> dict[str, Any] | None:
    if device.type != "cuda":
        return None
    properties = torch.cuda.get_device_properties(device)
    return {
        "name": properties.name,
        "memory_gib": float(properties.total_memory / 1024**3),
        "compute_capability": [int(properties.major), int(properties.minor)],
    }


def _git_state(project_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        changed = subprocess.run(
            ["git", "status", "--short"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except FileNotFoundError:
        return {"available": False, "reason": "git executable not found"}
    except subprocess.CalledProcessError:
        return {"available": False, "reason": "project directory is not a Git repository"}
    return {
        "available": True,
        "commit": commit,
        "working_tree_changed": bool(changed),
        "changed_paths": changed,
    }
