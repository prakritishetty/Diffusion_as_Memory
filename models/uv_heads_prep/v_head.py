"""Detail head.

The original was a bare `Linear(d, d)` applied per slot -- close to the identity, and it added
essentially nothing beyond slot pooling.

Two changes, the second of which matters more than anything else in this file:

1. Slots exchange information (a couple of transformer blocks) and get a positional embedding,
   which breaks the permutation symmetry so "slot 3" means something stable across examples.

2. **Output scale is controlled.** The forward process is
   `v_t = sqrt(abar)*v0 + sqrt(1-abar)*eps` with `eps ~ N(0, I)`; that only produces *graded*
   corruption when `v0` has roughly unit variance. If `std(v0) >> 1` the noise term is negligible
   until `abar` becomes tiny, so nothing degrades and then everything degrades at once -- the
   "cliff" that earlier runs blamed on the schedule and on latent geometry. Either normalise here
   (`normalize=True`) or divide by a measured `sigma_data` (`set_data_scale`), which is the
   Stable-Diffusion style fix. `tools/diagnose_v0_scale.py` reports which is needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.layers import PreNormBlock


class VHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_slots: int,
        n_blocks: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        normalize: bool = True,
    ):
        super().__init__()
        self.normalize = normalize
        self.slot_pos = nn.Parameter(torch.randn(1, num_slots, hidden_dim) * hidden_dim**-0.5)
        self.blocks = nn.ModuleList(
            [PreNormBlock(hidden_dim, n_heads, dropout=dropout) for _ in range(n_blocks)]
        )
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        # Fixed rescale applied after the head; 1.0 until measured on real data.
        self.register_buffer("data_scale", torch.ones(1), persistent=True)

    def set_data_scale(self, sigma: float) -> None:
        """Divide the latent by a measured std, as an alternative to LayerNorm."""
        self.data_scale.fill_(float(sigma))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = h + self.slot_pos
        for blk in self.blocks:
            x = blk(x)
        v0 = self.out(x)
        if self.normalize:
            v0 = self.norm(v0)
        return v0 / self.data_scale.clamp(min=1e-6)
