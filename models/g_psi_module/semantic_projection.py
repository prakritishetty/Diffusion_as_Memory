"""G_psi -- the semantic projection module.

Its job is to map an off-manifold latent (somewhere along the reverse chain) back onto the
manifold of latents the decoder was trained to read.

Defects in the previous version:

* `use_attn` defaulted to `False`, so the module was a **per-slot pointwise MLP** -- it could not
  mix information across slots at all, which is a large hole for something doing geometry on a
  slot-structured latent.
* `u` entered only through AdaLN. That channel can rescale features but cannot inject content, so
  the "cue" could not actually supply the gist.
* `input_dim = 2*d` was computed and then discarded (`input_proj = Linear(d, d)`), and `v_hat_0`
  was accepted and never used. The intended two-input design never ran.
* No `t`-dependent output gate: zero-init made it the identity *at initialisation*, not *at small
  t*.
* No standalone loss -- all gradient arrived through a distant cross-entropy at the decoder.
  `L_proj` (see `training/losses.py`) fixes that separately.

Inputs are `(v_t, v_hat_0, t, u)`: `v_t` is where we are, `v_hat_0` is where the denoiser thinks
the clean latent is, and how far to trust the second is a `t`-dependent decision -- which is
exactly what AdaLN on `t` is for.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configs.base import GPsiConfig
from models.layers import ConditionedBlock, CueProjection, TimestepEmbedding


class SemanticProjectionModule(nn.Module):
    def __init__(
        self,
        d: int,
        n_slots: int,
        u_dim: int,
        config: GPsiConfig | None = None,
        q_dim: int = 0,
        no_use_u: bool = False,
        no_use_vt: bool = False,
    ):
        super().__init__()
        cfg = config or GPsiConfig()
        self.cfg = cfg
        self.d = d
        self.L = n_slots
        self.u_dim = u_dim
        self.no_use_u = no_use_u
        self.no_use_vt = no_use_vt
        self.use_vhat0 = cfg.use_vhat0_input

        self.t_emb = TimestepEmbedding(d)
        self.slot_pos = nn.Parameter(torch.randn(1, n_slots, d) * d**-0.5)
        self.null_u = nn.Parameter(torch.zeros(1, u_dim))

        in_dim = d * (2 if self.use_vhat0 else 1)
        self.in_proj = nn.Linear(in_dim, d)

        self.cue = (
            CueProjection(u_dim, d, cfg.n_cue_tokens, q_dim=q_dim)
            if (cfg.use_cross_attn and not no_use_u) else None
        )

        d_cond = d if no_use_u else (d + u_dim)
        self.blocks = nn.ModuleList([
            ConditionedBlock(
                d=d, n_heads=cfg.n_heads, d_ff=cfg.d_ff, d_cond=d_cond,
                use_self_attn=cfg.use_self_attn,
                use_cross_attn=cfg.use_cross_attn and not no_use_u,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.n_blocks)
        ])

        self.out_norm = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, d)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Timestep-dependent residual gate: ~identity near t=0, free to correct hard at large t.
        self.gate = nn.Sequential(nn.SiLU(), nn.Linear(d, 1))
        nn.init.zeros_(self.gate[1].weight)
        nn.init.zeros_(self.gate[1].bias)

    def forward(
        self,
        t: torch.Tensor,
        v_t: torch.Tensor | None = None,
        v_hat_0: torch.Tensor | None = None,
        u: torch.Tensor | None = None,
        q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if v_t is None and v_hat_0 is None:
            raise ValueError("G_psi needs at least one of v_t / v_hat_0")
        base = v_hat_0 if (self.no_use_vt or v_t is None) else v_t

        if self.use_vhat0:
            other = v_hat_0 if v_hat_0 is not None else base
            x_in = torch.cat([base, other], dim=-1)
        else:
            x_in = base

        t_emb = self.t_emb(t)
        if self.no_use_u:
            cond, cue = t_emb, None
        else:
            u_eff = self.null_u.expand(base.shape[0], -1) if u is None else u
            cond = torch.cat([t_emb, u_eff], dim=-1)
            cue = self.cue(u_eff, q) if self.cue is not None else None

        x = self.in_proj(x_in) + self.slot_pos
        for blk in self.blocks:
            x = blk(x, cond, cue)

        gate = torch.sigmoid(self.gate(t_emb)).unsqueeze(1)  # [B, 1, 1]
        return base + gate * self.out_proj(self.out_norm(x))
