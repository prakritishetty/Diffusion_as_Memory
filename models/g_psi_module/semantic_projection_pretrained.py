import torch
import torch.nn as nn
import math
from typing import Optional
from models.denoiser_module.denoiser import TimestepEmbedding, AdaLN 
from models.g_psi_module.g_psi_config import G_psi_config
from transformers import AutoModel


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

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Optional self-attention across L slots
        if self.use_attn:
            normed = self.attn_norm(x, c)
            attn_out, _ = self.attn(normed, normed, normed)
            x = x + attn_out

        # Feed-forward with AdaLN
        x = x + self.ff(self.adaln(x, c))
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
        self.u_dim = config.u_dim
        self.n_blocks = config.n_blocks
        self.d_ff = config.d_ff
        self.use_attn = config.use_attn
        self.n_heads = config.n_heads
        self.no_use_u = no_use_u
        self.no_use_vt = no_use_vt


        pretrained_name = "bert-base-uncased"
        base_model = AutoModel.from_pretrained(pretrained_name)
        self.transformer_backbone = base_model.encoder 
        self.hidden_dim = base_model.config.hidden_size

        self.t_emb = TimestepEmbedding(self.d)

        # c = [u, t_emb] if using u; otherwise c = t_emb only
        

        # x = [v_t, v_hat_0] if using v_t; otherwise x = v_hat_0 only
        input_dim = self.d if self.no_use_vt else (2 * self.d)
        # self.input_proj = nn.Linear(input_dim, self.d)
        self.input_proj = nn.Linear(self.d, self.hidden_dim)
        d_cond = self.d if self.no_use_u else (self.u_dim + self.d)

        self.cond_proj = nn.Linear(d_cond, self.hidden_dim)

        # self.blocks = nn.ModuleList([
        #     SPMBlock(self.d, d_cond, self.d_ff, use_attn=self.use_attn, n_heads=self.n_heads)
        #     for _ in range(self.n_blocks)
        # ])

        #zero-init residual on v_hat_0 At init, SPM = identity (v_hat_0 = v_tilde_0); learns corrections over time
        self.out_proj = nn.Linear(self.hidden_dim, self.d)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        t:       torch.Tensor,  # [B]  integer timesteps
        v_hat_0: Optional[torch.Tensor] = None,  # [B, L, d]
        v_t:     Optional[torch.Tensor] = None,  # [B, L, d] or None when no_use_vt=True
        u:       Optional[torch.Tensor] = None,  # [B, u_dim] or None when no_use_u=True
    ) -> torch.Tensor:

        t_emb = self.t_emb(t)          # [B, d]
        if self.no_use_u:
            c = t_emb
        else:
            if u is None:
                raise ValueError("u is required when no_use_u=False")
            c = torch.cat([u, t_emb], dim=-1)

        c_mapped = self.cond_proj(c).unsqueeze(1)


        if self.no_use_vt:
            x_ini = v_hat_0
        else:
            if v_t is None:
                raise ValueError("v_t is required when no_use_vt=False")
            x_ini = v_t
        x = self.input_proj(x_ini)                       # [B, L, d]

        # for block in self.blocks:
        #     x = block(x, c)                          # [B, L, d]

        x = x+c_mapped
        extended_attention_mask = None
        encoder_outputs = self.transformer_backbone(
            x,
            attention_mask=extended_attention_mask,
        )
        x = encoder_outputs.last_hidden_state
        v_tilde_0 = x_ini + self.out_proj(x)      # [B, L, d]

        return v_tilde_0
