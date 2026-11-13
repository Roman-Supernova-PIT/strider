"""Released STRIDER models, and the folder they are downloaded to."""

import os
from importlib.resources import files
from pathlib import Path

import pooch

__all__ = ["MODELS", "DEFAULT_MODEL", "TEST_MODEL", "model_dir", "model_path", "fetch"]

# one entry per released package; the package itself states its format
MODELS: dict[str, dict[str, str]] = {
    "strider-15class": {
        "url": "https://github.com/Roman-Supernova-PIT/strider/releases/download/0.1.0/strider-15class-1.0.tar.gz",
        "sha256": "24706ea842884dc7d5068ffcdd465db1fb95e87de8c390decaa48debbb016d1b",
    },
}
DEFAULT_MODEL: str | None = "strider-15class"
# the bundled test model, whose results are placeholders
TEST_MODEL = "test"


def model_dir() -> Path:
    """Return the folder models are downloaded to. This is ``$STRIDER_MODEL_DIR`` if set, otherwise
    ``~/.cache/strider``."""
    return Path(os.environ.get("STRIDER_MODEL_DIR") or Path.home() / ".cache" / "strider").expanduser()


def model_path(name: str) -> Path:
    """Return where the released model ``name`` is downloaded to."""
    return model_dir() / f"{name}.tar.gz"


def fetch(name: str | None = None) -> Path:
    """Return the local path of a released model package, downloading it if needed.

    Parameters
    ----------
    name : str, optional
        A model name, as ``strider models`` lists them, or ``"test"`` for the bundled test model. If left out, the
        default model.

    Returns
    -------
    pathlib.Path
        The downloaded package.
    """
    if name == TEST_MODEL:
        return Path(str(files("strider") / "tests" / "data" / "strider-synthetic-test-1.0.tar.gz"))
    if not MODELS:
        raise ValueError(
            'no model is released yet; use the test model, "test", whose results are placeholders, or '
            "give the path of a model package"
        )
    name = name or DEFAULT_MODEL
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; released models: {', '.join(MODELS)}")
    entry = MODELS[name]
    if len(entry.get("sha256", "")) != 64:
        raise ValueError(f"model {name!r} has no full sha256 in the registry, so it cannot be checked")
    path = model_path(name)
    return Path(pooch.retrieve(entry["url"], known_hash=f"sha256:{entry['sha256']}", fname=path.name, path=path.parent))
