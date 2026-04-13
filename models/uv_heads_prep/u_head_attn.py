import torch
import torch.nn as nn


class UHead(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()

        # Learnable query that attends over slots to decide importance
        self.attn_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, h):
        # h: [B, num_slots, hidden_dim]  e.g. [B, 8, 512]
        B = h.size(0)
        q = self.attn_query.expand(B, -1, -1)   # [B, 1, hidden_dim]
        pooled, _ = self.attn(q, h, h)           # [B, 1, hidden_dim]
        pooled = self.norm(pooled.squeeze(1))     # [B, hidden_dim]
        return self.proj(pooled)                  # [B, output_dim]