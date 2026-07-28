"""Minimal configuration needed to reconstruct the production model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import TypeVar


@dataclass
class InstrumentConfig:
    n_wave: int = 1024
    wave_min: float = 7500.0
    wave_max: float = 18000.0
    phase_norm: float = 30.0
    z_norm: float = 2.5
    max_epochs: int = 64


@dataclass
class BackboneConfig:
    n_channels: int = 2
    patch_size: int = 8
    d_model: int = 128
    n_heads: int = 4
    n_pairs: int = 3
    dropout: float = 0.1
    n_freqs: int = 16
    cnn_stem_channels: list[int] = field(default_factory=lambda: [64, 128, 128])
    cnn_stem_kernels: list[int] = field(default_factory=lambda: [7, 5, 5])

    # Architecture switches from the checkpoint; `validate` rejects any value
    # this release does not implement.
    stem_type: str = "v2"          # historical tag for the single-scale stem
    use_cnn_stem: bool = True
    phase_film_mode: str = "shallow"
    use_z_prior: bool = False
    use_spectral_alibi: bool = False
    use_absolute_pos_enc: bool = True

    @property
    def n_patches(self) -> int:
        return 1024 // self.patch_size


@dataclass
class SchemeConfig:
    name: str = "strider"
    n_classes: int = 0
    class_names: list[str] = field(default_factory=list)


@dataclass
class DataConfig:
    normalization: str = "per_object"


T = TypeVar("T")


def _read(config_type: type[T], values: dict | None) -> T:
    values = values or {}
    names = {item.name for item in fields(config_type)}
    return config_type(**{key: value for key, value in values.items() if key in names})


@dataclass
class STRIDERConfig:
    instrument: InstrumentConfig = field(default_factory=InstrumentConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    scheme: SchemeConfig = field(default_factory=SchemeConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def _apply_overrides(cls, values: dict) -> "STRIDERConfig":
        config = cls(
            instrument=_read(InstrumentConfig, values.get("instrument")),
            backbone=_read(BackboneConfig, values.get("backbone")),
            scheme=_read(SchemeConfig, values.get("scheme")),
            data=_read(DataConfig, values.get("data")),
        )
        return config.validate()

    def validate(self) -> "STRIDERConfig":
        if self.instrument.n_wave != 1024:
            raise ValueError("The production model requires 1,024 wavelength bins")
        if self.instrument.n_wave % self.backbone.patch_size:
            raise ValueError("patch_size must divide the wavelength grid")
        if self.backbone.d_model % self.backbone.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if len(self.backbone.cnn_stem_channels) != len(self.backbone.cnn_stem_kernels):
            raise ValueError("CNN stem channels and kernels must have the same length")
        for name, expected in (
            ("stem_type", "v2"),
            ("use_cnn_stem", True),
            ("phase_film_mode", "shallow"),
            ("use_z_prior", False),
            ("use_spectral_alibi", False),
            ("use_absolute_pos_enc", True),
        ):
            actual = getattr(self.backbone, name)
            if actual != expected:
                raise ValueError(
                    f"This STRIDER release implements one backbone: it needs "
                    f"{name}={expected!r}, but the checkpoint asks for {actual!r}"
                )
        if self.scheme.n_classes != len(self.scheme.class_names):
            raise ValueError("Class count does not match class names")
        return self

    def to_dict(self) -> dict:
        return asdict(self)
