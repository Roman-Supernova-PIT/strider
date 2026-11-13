"""STRIDER identifies transients from their Roman prism spectra."""

from .io import Classification, Transient, Visit, load_example
from .model import Model, load_model

__all__ = ["load_model", "load_example", "Model", "Visit", "Transient", "Classification"]

try:
    from ._version import version as __version__  # written by setuptools_scm on install
except ImportError:
    __version__ = "unknown"
