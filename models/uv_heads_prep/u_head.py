"""Gist head.

The original was `h.mean(dim=1)` followed by a single `Linear` -- so `u` was a linear function of
the average slot, with no capacity to select or compose, and it discarded the slot structure that
the pooling layer had just built. Combined with `v0` also being a single `Linear` on the same
tensor, `u` and `v0` were not architecturally distinct at all; any separation had to come from the
losses, and only the reconstruction loss existed.

Changes:
  * attention pooling with a learned query, so the head can *select* which slots carry the gist;
  * an MLP with an expansion, so it can compose rather than project;
  * L2 normalisation, which keeps InfoNCE well conditioned and stops an unbounded `u` from
    drifting in scale while it is being concatenated into AdaLN conditioning.
  * the pooling attention weights are returned as an interpretability signal.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        normalize: bool = True,
    ):
        super().__init__()
        self.normalize = normalize
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * hidden_dim**-0.5)
        self.attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, output_dim),
        )

    def forward(self, h: torch.Tensor, return_attention: bool = False):
        q = self.query.expand(h.size(0), -1, -1)
        pooled, attn = self.attn(q, h, h, need_weights=return_attention,
                                 average_attn_weights=True)
        u = self.mlp(pooled.squeeze(1))
        if self.normalize:
            u = F.normalize(u, dim=-1)
        return (u, attn) if return_attention else u
