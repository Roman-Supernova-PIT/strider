"""STRIDER inference.

Public entry points:

    from strider import load_model, STRIDER

"""

__version__ = "0.2.0"

from strider.model import (
    STRIDER,
    load_model,
)

__all__ = [
    "STRIDER",
    "load_model",
]
