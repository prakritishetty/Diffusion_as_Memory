import torch
import torch.nn as nn
from models.denoiser_module.denoiser import TimestepEmbedding, AdaLN, MultiHeadAttention


class TransformerBlock(nn.Module):
    """Single Transformer block with self-attention, cross-attention, and FFN."""
    
    def __init__(self, d: int, n_heads: int, d_ff: int, u_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        
        # d_cond = d (t_emb) + u_dim (raw u); if u_dim=0 reverts to t-only conditioning
        d_cond = d + u_dim
        self.adalan1 = AdaLN(d, d_cond)
        self.adalan2 = AdaLN(d, d_cond)
        
        # Self-attention
        self.self_attn = MultiHeadAttention(d, n_heads)
        
        # Cross-attention (x attends to u)
        # self.cross_attn = MultiHeadAttention(d, n_heads)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        t_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: main input (vt) [batch_size, L, d]
            u: semantic anchor [batch_size, L, d]
            t_emb: timestep embedding [batch_size, d]
        
        Returns:
            output: [batch_size, L, d]
        
        """
        # fuse timestep and u into a single conditioning vector [B, d + u_dim]
        c = torch.cat([t_emb, u], dim=-1)

        # a. AdaLN + b. Self-Attention
        x_normalized = self.adalan1(x, c)
        x = x + self.dropout(self.self_attn(x_normalized, x_normalized, x_normalized))
        
        # c. Cross-Attention (x attends to u, no AdaLN before it)
        # x = x + self.dropout(self.cross_attn(x, u, u))
        
        # d. AdaLN + e. FFN
        x_normalized = self.adalan2(x, c)
        x = x + self.dropout(self.ffn(x_normalized))
        
        return x


class LatentDiffusionModel(nn.Module):
    def __init__(self, d: int, num_slots: int, u_dim: int, hidden_dim: int):
        super().__init__()
        self.d = d
        self.num_slots = num_slots
        self.u_dim = u_dim
        self.hidden_dim = hidden_dim
        self.N_blocks = 4

        self.timestep_emb = TimestepEmbedding(d)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d=self.d,
                n_heads=4,
                d_ff=self.hidden_dim,
                u_dim=self.u_dim,
                dropout=0.1
            )
            for _ in range(self.N_blocks)
        ])

        self.output_norm = nn.LayerNorm(self.d)
        self.output_projection = nn.Linear(self.d, self.d)


    def forward(self, vt: torch.Tensor, u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        This takes in degraded latent, gist and timestep.
        It predicts the latent at previous timestep.
        """

        B, L, D = vt.shape
        t = t.to(vt.device)
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=vt.device)

        t_emb = self.timestep_emb(t)                 # [B, H]
        for block in self.transformer_blocks:
            h = block(vt, u, t_emb)                      # [B, L, D]
        v_prev_hat = self.output_projection(self.output_norm(h))              # [B, L, D]
        return v_prev_hat
