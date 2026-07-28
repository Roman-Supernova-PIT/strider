"""Per-class spectral signatures.

Wraps the ONIR template bank as an `nn.Parameter`: the signatures are
initialized from the bank's physical prototypes and then refined during
training, so a trained checkpoint carries its own learned signatures.

Same-class phase-neighbor fallback fills the unsupported cells (Ia at
+60-80d, IIb at -20 to -15d) with signatures from the nearest populated
phase bin of the same class. Signatures are never borrowed across classes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from strider.network.template_bank import TemplateBank


@dataclass(frozen=True)
class SignatureConfig:
    window_size: int                          # detector window width (W)
    scan_dim: int                             # subspace dim for matching (S)
    phase_neighbor_fallback: bool = True      # fill unsupported cells from nearest phase


class SpectralSignatures(nn.Module):
    """Per-class signatures, bank-initialized.

    Storage shape is class, phase, feature, prototype, wavelength, subspace.

    Active mask buffer (`active_mask`) is the bank's `prototype_active` after
    same-class phase-neighbor fallback. Inactive cells hold NaN and contribute
    0 to the match surface.
    """

    def __init__(self, config: SignatureConfig, bank: TemplateBank):
        super().__init__()
        if config.window_size <= 0 or config.scan_dim <= 0:
            raise ValueError(
                f'window_size and scan_dim must be positive, got '
                f'({config.window_size}, {config.scan_dim})'
            )
        self.config = config
        self.n_classes = bank.n_classes
        self.n_phase_bins = bank.n_phase_bins
        self.n_features = bank.n_features
        self.k_max = bank.k_max
        self.bank_source_path = bank.source_path

        resampled = bank.resampled_to(config.window_size)  # (C, P, F, K, W) NaN-padded
        active_init = bank.prototype_active.copy()         # (C, P, F, K) bool

        if config.phase_neighbor_fallback:
            resampled, active_init = _apply_phase_neighbor_fallback(
                resampled, active_init,
            )

        signatures_init = _broadcast_and_normalize(
            resampled, scan_dim=config.scan_dim,
        )  # (C, P, F, K, W, S) float32, NaN at !active

        # Small per-cell noise breaks the rank-1 symmetry of the broadcast.
        active_t = torch.from_numpy(active_init)
        noise = torch.randn_like(signatures_init) * (0.01 / float(config.scan_dim) ** 0.5)
        signatures_init = torch.where(
            active_t.unsqueeze(-1).unsqueeze(-1),
            signatures_init + noise,
            signatures_init,
        )

        self.signatures = nn.Parameter(signatures_init.contiguous())
        # Carried so trained checkpoints load; inference does not read it.
        self.register_buffer('_init_signatures', signatures_init.clone(), persistent=True)
        self.register_buffer('active_mask', active_t.contiguous(), persistent=True)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (signatures, active_mask).

        Signatures retain NaN at inactive cells so the matcher's NaN-symmetric
        NCC contract holds without modification.
        """
        return self.signatures, self.active_mask


def _apply_phase_neighbor_fallback(
    windows: np.ndarray,  # (C, P, F, K, W) float32, NaN at inactive
    active: np.ndarray,   # (C, P, F, K) bool
) -> tuple[np.ndarray, np.ndarray]:
    """For each (c, f) cell where (c, p, f) has zero active K, copy the
    nearest p' (same c, same f) with at least one active K.

    Returns (filled_windows, filled_active). Cells that have NO populated
    phase bin for that (c, f) stay unsupported (rare; sim coverage failure).
    """
    C, P, F_, K, W = windows.shape
    filled_w = windows.copy()
    filled_a = active.copy()

    cell_has_support = active.any(axis=-1)  # (C, P, F)
    for c in range(C):
        for f in range(F_):
            populated = np.flatnonzero(cell_has_support[c, :, f])
            if populated.size == 0 or populated.size == P:
                continue  # all unsupported or all supported, nothing to fill
            for p in range(P):
                if cell_has_support[c, p, f]:
                    continue
                # Nearest populated phase bin (in index space, not days —
                # we don't have phase_bin_centers here but the bank's
                # phase grid is monotonic so index distance suffices)
                nearest = populated[np.abs(populated - p).argmin()]
                filled_w[c, p, f] = windows[c, nearest, f]
                filled_a[c, p, f] = active[c, nearest, f]
    return filled_w, filled_a


def _broadcast_and_normalize(
    resampled: np.ndarray,  # (C, P, F, K, W) NaN-padded
    scan_dim: int,
) -> torch.Tensor:           # (C, P, F, K, W, S) float32 NaN-padded
    """Broadcast bank prototypes across scan_dim with 1/sqrt(S) scaling.

    NaN locations are preserved.
    """
    t = torch.from_numpy(resampled).unsqueeze(-1)
    t = t.expand(-1, -1, -1, -1, -1, scan_dim).contiguous()
    return t / float(scan_dim) ** 0.5
