from pathlib import Path

import pooch
import pytest

from strider import registry


def test_model_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("STRIDER_MODEL_DIR", raising=False)
    assert registry.model_dir() == Path.home() / ".cache" / "strider"
    monkeypatch.setenv("STRIDER_MODEL_DIR", str(tmp_path))
    assert registry.model_path("m") == tmp_path / "m.tar.gz"
    monkeypatch.setenv("STRIDER_MODEL_DIR", "~/models")
    assert registry.model_dir() == Path.home() / "models"


def test_unknown_model(monkeypatch):
    monkeypatch.setattr(registry, "MODELS", {})
    with pytest.raises(ValueError, match="no model is released yet; use the test model"):
        registry.fetch()
    monkeypatch.setitem(registry.MODELS, "m", {"url": "", "sha256": ""})
    with pytest.raises(ValueError, match="unknown model 'no-such-model'; released models: m"):
        registry.fetch("no-such-model")


def test_fetch_checks_hash_and_caches(monkeypatch, tmp_path):
    monkeypatch.setenv("STRIDER_MODEL_DIR", str(tmp_path))
    monkeypatch.setitem(registry.MODELS, "m", {"url": "https://example.org/m.tar.gz", "sha256": "ab" * 32})
    calls = []
    monkeypatch.setattr(pooch, "retrieve", lambda url, **kw: calls.append((url, kw)) or str(kw["path"] / kw["fname"]))
    assert registry.fetch("m") == tmp_path / "m.tar.gz"
    assert calls == [
        ("https://example.org/m.tar.gz", {"known_hash": "sha256:" + "ab" * 32, "fname": "m.tar.gz", "path": tmp_path})
    ]


def test_fetch_refuses_an_entry_without_a_full_sha256(monkeypatch):
    monkeypatch.setitem(registry.MODELS, "m", {"url": "https://example.org/m.tar.gz", "sha256": ""})
    with pytest.raises(ValueError, match="no full sha256"):
        registry.fetch("m")
