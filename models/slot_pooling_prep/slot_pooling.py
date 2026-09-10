import torch
import torch.nn as nn

class SlotPooling(nn.Module):
    def __init__(self, hidden_dim, num_slots, num_heads = 4):
        # prakriti note => 4 is a hyperparam
        super().__init__()

        self.num_slots = num_slots
        self.hidden_dim = hidden_dim

        self.slots = nn.Parameter(torch.randn(num_slots, hidden_dim))

        self.cross_attention = nn.MultiheadAttention(
            embed_dim = hidden_dim,
            num_heads = num_heads,
            batch_first = True
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, H, attention_mask = None):
        B = H.size(0)
        slot_queries = self.slots.unsqueeze(0).expand(B, -1, -1)
        # add batch dim to slots

        #ignore padding duirng attention
        if attention_mask is not None:
            key_padding_mask = attention_mask ==0
        else:
            key_padding_mask = None
        
        h, _ = self.cross_attention(
            query = slot_queries,
            key = H,
            value = H,
            key_padding_mask = key_padding_mask
        )

        h = self.norm(h)

        return h