import json
from pathlib import Path

import pytest

from strider.data import external_test
from strider.evaluation.evaluate import evaluate


def _config(tmp_path: Path) -> dict:
    return {
        "_config_path": str(tmp_path / "model.yaml"),
        "_project_root": str(tmp_path),
        "project": {"name": "model", "seed": 7, "output_dir": str(tmp_path / "run")},
        "data": {
            "class_scheme": "normal_ia_binary",
            "source_products": {"old": {"source_dir": "/old", "split_blocks": {"test": [1]}}},
            "prepared_dir": str(tmp_path / "old_prepared"),
            "max_objects": {"train": 0, "test": 0},
            "runtime_object_limits": {"test": 0},
            "minimum_class_counts": {"train": 5, "test": 1},
            "require_all_training_classes": True,
        },
        "observation": {"wavelength_min": 7500.0, "wavelength_max": 18175.0},
        "model": {"classes": ["Ia", "other"]},
        "training": {},
        "evaluation": {"split": "test", "views": ["original"]},
        "onir": {},
    }


def test_external_test_config_replaces_sources_without_mutating_model_config(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    model_config = _config(tmp_path)

    result = external_test.external_test_config(
        model_config,
        source_dir=source,
        prepared_dir=tmp_path / "prepared",
        blocks=[10, 1, 1],
        dataset_tag="sundial_jun24",
    )

    assert list(result["data"]["source_products"]) == ["sundial_jun24"]
    assert result["data"]["source_products"]["sundial_jun24"]["split_blocks"] == {
        "test": [1, 10]
    }
    assert result["data"]["minimum_class_counts"] == {"test": 1}
    assert result["data"]["require_all_training_classes"] is False
    assert "old" in model_config["data"]["source_products"]


def test_external_test_tag_is_path_safe(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(ValueError, match="dataset_tag"):
        external_test.external_test_config(
            _config(tmp_path),
            source_dir=source,
            prepared_dir=tmp_path / "prepared",
            blocks=[1],
            dataset_tag="../../bad",
        )


def test_prepare_external_test_freezes_config_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    prepared = tmp_path / "prepared"

    def fake_build(config: dict, *, require_train_split: bool) -> dict:
        assert require_train_split is False
        Path(config["data"]["prepared_dir"]).mkdir(parents=True)
        return {"format_version": "test", "preparation_mode": "external_test"}

    monkeypatch.setattr(external_test, "build_native_store", fake_build)
    monkeypatch.setattr(
        external_test,
        "_source_manifest",
        lambda _config, tag: {
            "format_version": "test",
            "dataset_tag": tag,
            "pairs": 1,
            "files": [],
        },
    )

    summary = external_test.prepare_external_test(
        _config(tmp_path),
        source_dir=source,
        prepared_dir=prepared,
        blocks=[1],
        dataset_tag="sundial_jun24",
    )

    assert summary["dataset_tag"] == "sundial_jun24"
    assert (prepared / "external_test_config.resolved.yaml").is_file()
    assert (prepared / "source_manifest.json").is_file()
    assert json.loads((prepared / "dataset_summary.json").read_text())["dataset_tag"] == (
        "sundial_jun24"
    )
    provenance = external_test.prepared_external_test_provenance(prepared)
    assert provenance["dataset_tag"] == "sundial_jun24"


def test_external_evaluation_requires_both_prepared_and_output_directories(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="reserved"):
        evaluate(config, output_dir=tmp_path / "output")
    with pytest.raises(ValueError, match="requires"):
        evaluate(config, external_prepared_dir=tmp_path / "prepared")
