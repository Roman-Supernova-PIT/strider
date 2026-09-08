"""STRIDER model components with a lazy public model import."""

__all__ = ["Strider", "measurement_inputs"]


def __getattr__(name: str):
    if name == "Strider":
        from .strider import Strider

        return Strider
    if name == "measurement_inputs":
        from .strider import measurement_inputs

        return measurement_inputs
    raise AttributeError(name)
