from __future__ import annotations

import pytest

from strider.model_info import model_id, sha256, verify_model


def test_verify_model_returns_hash_of_existing_file(tmp_path):
    path = tmp_path / "any.pt"
    path.write_bytes(b"some bytes")
    assert verify_model(path) == sha256(path)


def test_verify_model_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_model(tmp_path / "does_not_exist.pt")


def test_model_id_derives_from_class_count():
    assert model_id(15) == "strider-15class"
    assert model_id(11) == "strider-11class"
