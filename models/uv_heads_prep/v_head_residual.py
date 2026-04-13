import torch.nn as nn


class VHead(nn.Module):
    def __init__(self, hidden_dim, expansion=2, dropout=0.1):
        super().__init__()

        mid_dim = hidden_dim * expansion

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h):
        # h: [B, num_slots, hidden_dim]
        # Residual connection: each slot is refined but retains original info
        return self.norm(h + self.ffn(h))  # [B, num_slots, hidden_dim]