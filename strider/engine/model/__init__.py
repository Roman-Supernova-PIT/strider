"""STRIDER model components with a lazy public model import."""

__all__ = ["Strider3", "measurement_inputs"]


def __getattr__(name: str):
    if name == "Strider3":
        from .strider import Strider3

        return Strider3
    if name == "measurement_inputs":
        from .strider import measurement_inputs

        return measurement_inputs
    raise AttributeError(name)
