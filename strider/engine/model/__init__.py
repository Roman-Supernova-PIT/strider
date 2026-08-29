"""STRIDER model components with a lazy public model import."""

__all__ = ["StriderModel", "measurement_inputs"]


def __getattr__(name: str):
    if name == "StriderModel":
        from .strider import StriderModel

        return StriderModel
    if name == "measurement_inputs":
        from .strider import measurement_inputs

        return measurement_inputs
    raise AttributeError(name)
