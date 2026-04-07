import torch
import torch.nn as nn
import math
from typing import Optional
from models.denoiser_module.denoiser import TimestepEmbedding, AdaLN 
from models.g_psi_module.g_psi_config import G_psi_config


class SPMBlock(nn.Module):
    def __init__(self, d: int, d_cond: int, d_ff: int,
                 use_attn: bool = False, n_heads: int = 8):
        super().__init__()
        self.adaln = AdaLN(d, d_cond)
        self.ff = nn.Sequential(
            nn.Linear(d, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d),
        )

        self.use_attn = use_attn
        if use_attn:
            self.attn_norm = AdaLN(d, d_cond)
            self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
            self.cross_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        u: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Optional self-attention across L slots
        if self.use_attn:
            normed = self.attn_norm(x, t_emb)
            attn_out, _ = self.attn(normed, normed, normed)
            x = x + attn_out
            if u is not None:
                cross_attn_out, _ = self.cross_attn(x, u, u)
                x = x + cross_attn_out

        # Feed-forward with AdaLN
        x = x + self.ff(self.adaln(x, t_emb))
        return x

class SemanticProjectionModule(nn.Module):
    """
    Maps (v_t, v_hat_0, u, t)  to  v_tilde_0

    v_t    : [B, L, d]   noised detail latent
    v_hat_0: [B, L, d]   denoiser one-step estimate of v_0
    u      : [B, u_dim]      gist / subconscious code
    t      : [B]         integer diffusion timesteps

    Returns v_tilde_0: [B, L, d]  projected onto encoder manifold
    """
    def __init__(
        self,
        config: G_psi_config,
        no_use_u: bool = False,
        no_use_vt: bool = False,
    ):
        super().__init__()
        self.d = config.d
        self.n_blocks = config.n_blocks
        self.d_ff = config.d_ff
        self.use_attn = config.use_attn
        self.n_heads = config.n_heads
        self.no_use_u = no_use_u
        self.no_use_vt = no_use_vt

        self.t_emb = TimestepEmbedding(self.d)

        # AdaLN is conditioned only on timestep embedding in this module.
        d_cond = self.d

        # x = [v_t, v_hat_0] if using v_t; otherwise x = v_hat_0 only
        input_dim = self.d if self.no_use_vt else (2 * self.d)
        self.input_proj = nn.Linear(input_dim, self.d)

        self.blocks = nn.ModuleList([
            SPMBlock(self.d, d_cond, self.d_ff, use_attn=self.use_attn, n_heads=self.n_heads)
            for _ in range(self.n_blocks)
        ])

        #zero-init residual on v_hat_0 At init, SPM = identity (v_hat_0 = v_tilde_0); learns corrections over time
        self.out_proj = nn.Linear(self.d, self.d)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        v_hat_0: torch.Tensor,  # [B, L, d]
        t:       torch.Tensor,  # [B]  integer timesteps
        v_t:     Optional[torch.Tensor] = None,  # [B, L, d] or None when no_use_vt=True
        u:       Optional[torch.Tensor] = None,  # [B, L, d] or None when unavailable
    ) -> torch.Tensor:

        t_emb = self.t_emb(t)          # [B, d]
        # if self.no_use_u:
        #     c = t_emb
        # else:
        #     if u is None:
        #         raise ValueError("u is required when no_use_u=False")
        #     c = torch.cat([u, t_emb], dim=-1)

        if self.no_use_vt:
            x = v_hat_0
        else:
            if v_t is None:
                raise ValueError("v_t is required when no_use_vt=False")
            x = torch.cat([v_t, v_hat_0], dim=-1)
        x = self.input_proj(x)                       # [B, L, d]

        for block in self.blocks:
            x = block(x, t_emb, u)                          # [B, L, d]

        v_tilde_0 = v_hat_0 + self.out_proj(x)      # [B, L, d]

        return v_tilde_0
