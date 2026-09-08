"""Run STRIDER directly from this checkout without installing the package."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from strider.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
