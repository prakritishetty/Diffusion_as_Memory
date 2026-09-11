"""Denoiser N_theta.

Changes from the original:

* **x0-prediction by default.** The network outputs an estimate of the clean latent instead of
  the noise. This removes the `1/sqrt(alpha_bar)` division that made high-`t` readout diverge, and
  it is what every downstream consumer (G_psi, the decoder) actually wants. Note this is a
  *parameterisation*, not a one-step shortcut: the estimate is consumed by the sampler at each
  step of a multi-step chain, never decoded directly.
* **Cross-attention onto cue tokens** built from `u` (it was written and commented out). AdaLN
  alone can only rescale features; it cannot inject the gist.
* **Classifier-free guidance support** -- `u=None` runs the unconditional branch, so dropping `u`
  during training with probability `p` gives a cue-strength dial at inference and makes the
  shuffled-cue control quantitative.
* Vectorised schedule lookups instead of a per-element `.item()` loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configs.base import DenoiserConfig
from models.layers import ConditionedBlock, CueProjection, TimestepEmbedding


class Denoiser(nn.Module):
    def __init__(
        self,
        d: int,
        n_slots: int,
        u_dim: int,
        config: DenoiserConfig | None = None,
        q_dim: int = 0,
    ):
        super().__init__()
        cfg = config or DenoiserConfig()
        self.cfg = cfg
        self.d = d
        self.L = n_slots
        self.u_dim = u_dim

        self.t_emb = TimestepEmbedding(d)
        self.slot_pos = nn.Parameter(torch.randn(1, n_slots, d) * d**-0.5)
        self.null_u = nn.Parameter(torch.zeros(1, u_dim))  # unconditional token for CFG

        self.cue = (
            CueProjection(u_dim, d, cfg.n_cue_tokens, q_dim=q_dim)
            if cfg.use_cross_attn else None
        )

        d_cond = d + u_dim
        self.blocks = nn.ModuleList([
            ConditionedBlock(
                d=d, n_heads=cfg.n_heads, d_ff=cfg.d_ff, d_cond=d_cond,
                use_self_attn=True, use_cross_attn=cfg.use_cross_attn, dropout=cfg.dropout,
            )
            for _ in range(cfg.n_blocks)
        ])

        self.out_norm = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, d)

    def forward(
        self,
        v_t: torch.Tensor,                  # [B, L, d]
        t: torch.Tensor,                    # [B]
        u: torch.Tensor | None,             # [B, u_dim]; None runs the unconditional branch
        q: torch.Tensor | None = None,      # optional question embedding for the QA setting
    ) -> torch.Tensor:
        B = v_t.shape[0]
        u_eff = self.null_u.expand(B, -1) if u is None else u
        cond = torch.cat([self.t_emb(t), u_eff], dim=-1)
        cue = self.cue(u_eff, q) if self.cue is not None else None

        x = v_t + self.slot_pos
        for blk in self.blocks:
            x = blk(x, cond, cue)
        return self.out_proj(self.out_norm(x))


class EMA:
    """Exponential moving average of denoiser weights. Cheap and materially better samples."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self._backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def apply_to(self, model: nn.Module) -> None:
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()
                        if k in self.shadow}
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=False)

    def restore(self, model: nn.Module) -> None:
        if self._backup:
            model.load_state_dict({**model.state_dict(), **self._backup}, strict=False)
            self._backup = {}

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}
