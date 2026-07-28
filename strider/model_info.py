"""Identity helpers for STRIDER checkpoints.

Records which model was run (id + SHA-256) rather than pinning one exact file, so any
published checkpoint loads. Structural validity is checked by ``load_model`` on read.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

MODEL_FILENAME = "strider.pt"
_CHECKPOINT_FAMILY_TAG = "strider"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_id(n_classes: int) -> str:
    return f"strider-{int(n_classes)}class"


def verify_model(path: str | Path) -> str:
    """Confirm the checkpoint exists and return its SHA-256 for provenance."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}")
    return sha256(path)
