"""Versioned preparation and provenance helpers for external Sundial tests."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from strider.config import resolved_config_sha256, resolved_config_text

from .prepare import build_native_store, source_pair_assignments


_DATASET_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def external_test_config(
    model_config: dict[str, Any],
    *,
    source_dir: str | Path,
    prepared_dir: str | Path,
    blocks: Iterable[int],
    dataset_tag: str,
) -> dict[str, Any]:
    """Build a test-only data configuration without changing model settings."""
    if not _DATASET_TAG.fullmatch(dataset_tag):
        raise ValueError(
            "dataset_tag must start with an alphanumeric character and contain "
            "only letters, digits, '.', '_', or '-'"
        )
    selected_blocks = sorted({int(block) for block in blocks})
    if not selected_blocks or any(block < 1 for block in selected_blocks):
        raise ValueError("External-test blocks must contain positive integers")
    source = Path(source_dir).expanduser().resolve()
    prepared = Path(prepared_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"External-test source directory does not exist: {source}")
    if prepared == source or prepared.is_relative_to(source):
        raise ValueError("External-test prepared data must be outside the source product")

    config = copy.deepcopy(model_config)
    config["project"]["name"] = f"{model_config['project']['name']}__{dataset_tag}"
    config["data"]["source_products"] = {
        dataset_tag: {
            "source_dir": str(source),
            "split_blocks": {"test": selected_blocks},
        }
    }
    config["data"]["prepared_dir"] = str(prepared)
    config["data"]["max_objects"] = {"test": 0}
    config["data"]["runtime_object_limits"] = {"test": 0}
    config["data"]["minimum_class_counts"] = {"test": 1}
    config["data"]["require_all_training_classes"] = False
    config["evaluation"]["split"] = "test"
    return config


def config_for_prepared_external_test(
    model_config: dict[str, Any],
    prepared_dir: str | Path,
) -> dict[str, Any]:
    """Point a copy of a checkpoint config at a versioned prepared test store."""
    config = copy.deepcopy(model_config)
    config["data"]["prepared_dir"] = str(Path(prepared_dir).expanduser().resolve())
    return config


def prepare_external_test(
    model_config: dict[str, Any],
    *,
    source_dir: str | Path,
    prepared_dir: str | Path,
    blocks: Iterable[int],
    dataset_tag: str,
) -> dict[str, Any]:
    """Prepare one external test product and freeze its source manifest."""
    config = external_test_config(
        model_config,
        source_dir=source_dir,
        prepared_dir=prepared_dir,
        blocks=blocks,
        dataset_tag=dataset_tag,
    )
    prepared = Path(config["data"]["prepared_dir"])
    summary = build_native_store(config, require_train_split=False)
    prepared.mkdir(parents=True, exist_ok=True)

    config_path = prepared / "external_test_config.resolved.yaml"
    config_path.write_text(resolved_config_text(config), encoding="utf-8")
    manifest = _source_manifest(config, dataset_tag)
    manifest_path = prepared / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    summary.update(
        {
            "dataset_tag": dataset_tag,
            "external_test_config_sha256": resolved_config_sha256(config),
            "external_test_config": str(config_path),
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": _sha256_file(manifest_path),
        }
    )
    summary_path = prepared / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def prepared_external_test_provenance(prepared_dir: str | Path) -> dict[str, str]:
    """Return immutable identifiers recorded with an external prepared store."""
    prepared = Path(prepared_dir).expanduser().resolve()
    summary_path = prepared / "dataset_summary.json"
    manifest_path = prepared / "source_manifest.json"
    missing = [str(path) for path in (summary_path, manifest_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Prepared external test is missing provenance file(s): " + ", ".join(missing)
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("preparation_mode") != "external_test":
        raise ValueError(f"Prepared data are not marked as an external test: {prepared}")
    dataset_tag = str(summary.get("dataset_tag", ""))
    if not _DATASET_TAG.fullmatch(dataset_tag):
        raise ValueError(f"Prepared external test has an invalid dataset tag: {dataset_tag!r}")
    manifest_digest = _sha256_file(manifest_path)
    recorded_manifest_digest = str(summary.get("source_manifest_sha256", ""))
    if manifest_digest != recorded_manifest_digest:
        raise ValueError(
            f"Prepared external-test source manifest does not match its summary: {prepared}"
        )
    return {
        "dataset_tag": dataset_tag,
        "prepared_dir": str(prepared),
        "dataset_summary_sha256": _sha256_file(summary_path),
        "source_manifest_sha256": manifest_digest,
    }


def _source_manifest(config: dict[str, Any], dataset_tag: str) -> dict[str, Any]:
    assignments, products = source_pair_assignments(config["data"])
    paths = sorted(
        {
            Path(path).resolve()
            for _, _, pair in assignments
            for path in (pair.head_path, pair.spec_path)
        }
    )
    files = []
    for path in paths:
        stat = path.stat()
        files.append(
            {
                "path": str(path),
                "bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": _sha256_file(path),
            }
        )
    return {
        "format_version": "strider-external-test-manifest-v1",
        "dataset_tag": dataset_tag,
        "source_products": products,
        "pairs": len(assignments),
        "files": files,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
