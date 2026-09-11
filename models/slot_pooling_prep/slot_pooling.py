"""Slot pooling: variable-length hidden states -> a fixed number of slots.

Unchanged in spirit from the original. Two additions:
  * attention weights are returned, which is the free interpretability signal the slot-masking
    ablation needs (which tokens does each slot read?);
  * slots can be masked or replaced with the dataset-mean slot, for the same ablation. Zeroing a
    slot is off-distribution, so `mask_slots` substitutes a learned//supplied mean instead.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SlotPooling(nn.Module):
    def __init__(self, hidden_dim: int, num_slots: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim
        self.slots = nn.Parameter(torch.randn(num_slots, hidden_dim) * hidden_dim**-0.5)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        H: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        B = H.size(0)
        queries = self.slots.unsqueeze(0).expand(B, -1, -1)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        h, attn = self.cross_attention(
            query=queries, key=H, value=H,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention, average_attn_weights=True,
        )
        h = self.norm(h)
        return (h, attn) if return_attention else h


def mask_slots(
    slots: torch.Tensor,
    slot_ids: list[int] | torch.Tensor,
    replacement: torch.Tensor | None = None,
) -> torch.Tensor:
    """Replace the given slot indices with `replacement` (defaults to the batch-mean slot)."""
    out = slots.clone()
    idx = torch.as_tensor(slot_ids, device=slots.device, dtype=torch.long)
    if replacement is None:
        replacement = slots.mean(dim=(0, 1), keepdim=True).expand(slots.size(0), len(idx), -1)
    elif replacement.dim() == 1:
        replacement = replacement.view(1, 1, -1).expand(slots.size(0), len(idx), -1)
    out[:, idx, :] = replacement.to(out.dtype)
    return out
