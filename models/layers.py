"""Shared conditioned-transformer building blocks for the denoiser and G_psi.

Both modules previously had their own copy of an AdaLN block, both had cross-attention commented
out, and G_psi additionally defaulted `use_attn=False` -- so it was a per-slot pointwise MLP that
could not mix information across slots at all.

Conditioning here has two channels:
  * **modulation** -- AdaLN scale/shift/gate from `[t_emb ; u]`, which can rescale features
  * **content**    -- cross-attention onto cue tokens derived from `u` (and optionally a question
    embedding), which can actually inject the gist

The modulation-only design is too thin for cue-conditioned recall: the cue could scale features
but never supply information.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by an MLP."""

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.SiLU(), nn.Linear(4 * d, d))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.d // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) /
            max(half - 1, 1)
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.d % 2 == 1:
            emb = torch.cat([emb, torch.zeros(t.shape[0], 1, device=t.device)], dim=-1)
        return self.mlp(emb)


class AdaLNModulation(nn.Module):
    """Produces (scale, shift, gate) per residual branch, DiT-style.

    The gate is zero-initialised so every block starts as the identity: at initialisation G_psi
    returns its input unchanged and the denoiser is a no-op, which keeps early training stable.
    """

    def __init__(self, d: int, d_cond: int, n_branches: int = 3):
        super().__init__()
        self.n_branches = n_branches
        self.norms = nn.ModuleList(
            [nn.LayerNorm(d, elementwise_affine=False) for _ in range(n_branches)]
        )
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(d_cond, 3 * n_branches * d))
        nn.init.zeros_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)

    def forward(self, cond: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        params = self.proj(cond)  # [B, 3*n_branches*d]
        chunks = params.chunk(3 * self.n_branches, dim=-1)
        out = []
        for i in range(self.n_branches):
            scale, shift, gate = chunks[3 * i], chunks[3 * i + 1], chunks[3 * i + 2]
            out.append((scale.unsqueeze(1), shift.unsqueeze(1), gate.unsqueeze(1)))
        return out

    def apply_norm(self, i: int, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor):
        return self.norms[i](x) * (1 + scale) + shift


class ConditionedBlock(nn.Module):
    """Self-attention over slots, cross-attention onto cue tokens, then an FFN."""

    def __init__(
        self,
        d: int,
        n_heads: int,
        d_ff: int,
        d_cond: int,
        use_self_attn: bool = True,
        use_cross_attn: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_self_attn = use_self_attn
        self.use_cross_attn = use_cross_attn
        self.mod = AdaLNModulation(d, d_cond, n_branches=3)

        if use_self_attn:
            self.self_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        if use_cross_attn:
            self.cross_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,               # [B, L, d]
        cond: torch.Tensor,            # [B, d_cond]
        cue: torch.Tensor | None = None,  # [B, n_cue, d]
    ) -> torch.Tensor:
        (s1, b1, g1), (s2, b2, g2), (s3, b3, g3) = self.mod(cond)

        if self.use_self_attn:
            h = self.mod.apply_norm(0, x, s1, b1)
            attn, _ = self.self_attn(h, h, h, need_weights=False)
            x = x + g1 * self.dropout(attn)

        if self.use_cross_attn and cue is not None:
            h = self.mod.apply_norm(1, x, s2, b2)
            attn, _ = self.cross_attn(h, cue, cue, need_weights=False)
            x = x + g2 * self.dropout(attn)

        h = self.mod.apply_norm(2, x, s3, b3)
        return x + g3 * self.dropout(self.ffn(h))


class CueProjection(nn.Module):
    """Turn the gist vector (optionally plus a question embedding) into cross-attention tokens."""

    def __init__(self, u_dim: int, d: int, n_cue: int = 4, q_dim: int = 0):
        super().__init__()
        self.n_cue = n_cue
        self.d = d
        self.q_dim = q_dim
        self.proj = nn.Linear(u_dim + q_dim, n_cue * d)
        self.pos = nn.Parameter(torch.randn(1, n_cue, d) * d**-0.5)
        self.norm = nn.LayerNorm(d)

    def forward(self, u: torch.Tensor, q: torch.Tensor | None = None) -> torch.Tensor:
        if self.q_dim > 0:
            if q is None:
                q = u.new_zeros(u.shape[0], self.q_dim)
            u = torch.cat([u, q], dim=-1)
        tokens = self.proj(u).view(u.shape[0], self.n_cue, self.d)
        return self.norm(tokens + self.pos)


class GistAnchor(nn.Module):
    """Maps the gist `u` to a slot-shaped mean `m(u)`.

    The forward process diffuses `v0 - m(u)`, so `v_T ~= m(u) + noise`. The abstract end of the
    ladder is then anchored on the gist rather than on nothing.
    """

    def __init__(self, u_dim: int, d: int, n_slots: int):
        super().__init__()
        self.n_slots = n_slots
        self.d = d
        self.proj = nn.Sequential(
            nn.Linear(u_dim, d), nn.SiLU(), nn.Linear(d, n_slots * d)
        )
        self.scale = nn.Parameter(torch.zeros(1))  # starts at 0 => identical to plain diffusion

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        m = self.proj(u).view(u.shape[0], self.n_slots, self.d)
        return self.scale * m


class PreNormBlock(nn.Module):
    """Plain (unconditioned) transformer block, used inside the v-head."""

    def __init__(self, d: int, n_heads: int, d_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout(attn)
        return x + self.dropout(self.ffn(self.norm2(x)))
