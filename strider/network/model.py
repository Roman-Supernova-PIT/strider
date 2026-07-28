"""Rebuild the production STRIDER model from checkpoint metadata."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn as nn

from strider.network.backbone import SpectrotemporalBackbone
from strider.network.config import STRIDERConfig
from strider.network.combiner import CombinerConfig
from strider.network.detector import DetectorConfig, EvidenceDetector
from strider.network.template_bank import TemplateBank


@dataclass
class ModelConfig:
    n_z_bins: int = 500
    z_min: float = 0.0
    z_max: float = 3.0
    scan_dim: int = 8
    temperature: float = 10.0
    top_k: int = 3
    window_radius_patches: int = 7
    ncc_chunk_size: int = 256
    combiner_mode: str = "snr_sum"
    snr_clip: float = 1.0
    phase_neighbor_fallback: bool = True

    @classmethod
    def from_dict(cls, values: dict) -> "ModelConfig":
        """Read only inference fields from the embedded historical config."""
        names = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in names})


def _ivar_per_patch(
    ivar_weight: torch.Tensor | None,
    patch_size: int,
    n_patches: int,
) -> torch.Tensor | None:
    """Average per-bin inverse-variance weights down to one value per patch."""
    if ivar_weight is None:
        return None
    width = ivar_weight.shape[-1]
    expected = n_patches * patch_size
    if width != expected:
        raise ValueError(f"ivar_weight width {width} != {expected}")
    return ivar_weight.reshape(-1, ivar_weight.shape[1], n_patches, patch_size).mean(dim=-1)


class STRIDERModel(nn.Module):
    """Spectro-temporal backbone plus joint class-redshift evidence detector."""

    def __init__(self, backbone: SpectrotemporalBackbone, detector: EvidenceDetector):
        super().__init__()
        self.backbone = backbone
        self.detector = detector
        self.n_patches = backbone.n_patches
        self.patch_size = backbone.config.patch_size

    def forward(
        self,
        spectra: torch.Tensor,
        phases: torch.Tensor,
        epoch_mask: torch.Tensor,
        ivar_weight: torch.Tensor | None = None,
        wave_mask: torch.Tensor | None = None,
        wave_grid_frac: torch.Tensor | None = None,
        **_unused,
    ) -> dict:
        backbone_output = self.backbone(
            spectra,
            phases,
            wave_grid_frac=wave_grid_frac,
            wave_mask=wave_mask,
            epoch_mask=epoch_mask,
        )
        patch_weight = _ivar_per_patch(ivar_weight, self.patch_size, self.n_patches)
        output = self.detector(
            backbone_output["tokens"],
            phases,
            self.n_patches,
            epoch_mask=epoch_mask,
            ivar_per_patch_te=patch_weight,
        )
        output["cls_embedding"] = backbone_output["cls"]
        return output


def build_model(
    config: STRIDERConfig,
    model_config: ModelConfig,
    bank: TemplateBank,
) -> STRIDERModel:
    backbone = SpectrotemporalBackbone(config.backbone)
    detector_config = DetectorConfig(
        n_z_bins=model_config.n_z_bins,
        z_min=model_config.z_min,
        z_max=model_config.z_max,
        patch_size=config.backbone.patch_size,
        window_radius_patches=model_config.window_radius_patches,
        scan_dim=model_config.scan_dim,
        temperature=model_config.temperature,
        top_k=model_config.top_k,
        phase_neighbor_fallback=model_config.phase_neighbor_fallback,
        ncc_chunk_size=model_config.ncc_chunk_size,
        combiner=CombinerConfig(
            mode=model_config.combiner_mode,
            snr_clip=model_config.snr_clip,
        ),
    )
    detector = EvidenceDetector(detector_config, bank, d_model=config.backbone.d_model)
    return STRIDERModel(backbone, detector)
