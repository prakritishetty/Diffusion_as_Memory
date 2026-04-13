import torch.nn as nn

class UHead(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, h):
        h_mean = h.mean(dim = 1)
        return self.proj(h_mean)