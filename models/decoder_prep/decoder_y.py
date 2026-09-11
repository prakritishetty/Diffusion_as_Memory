"""Gist decoder: `u` -> `y`.

This is the module that makes "u holds the gist" a trained property rather than a hope. It was
previously written but commented out, on the worry that it would create an input-distribution
mismatch with the detail decoder.

It does not, because it is a *separate* model and it is used in P0 only -- it is dropped before
P1/P2 and never appears at inference. The detail decoder therefore only ever sees slot-structured
input.

`u` is expanded into a few learned query slots rather than a single tiled vector, so the decoder
has somewhere to put different parts of the gist.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

from models.hf_utils import load_seq2seq


class DecoderY(nn.Module):
    def __init__(
        self,
        u_dim: int,
        model_name: str = "t5-small",
        num_slots: int = 4,
    ):
        super().__init__()
        self.model = load_seq2seq(model_name)
        self.hidden_dim = int(self.model.config.d_model)
        self.num_slots = num_slots
        self.u_proj = nn.Linear(u_dim, num_slots * self.hidden_dim)
        self.slot_pos = nn.Parameter(
            torch.randn(1, num_slots, self.hidden_dim) * self.hidden_dim**-0.5
        )
        self.norm = nn.LayerNorm(self.hidden_dim)

    def _memory(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = u.size(0)
        mem = self.u_proj(u).view(B, self.num_slots, self.hidden_dim)
        mem = self.norm(mem + self.slot_pos)
        mask = torch.ones(B, self.num_slots, device=u.device, dtype=torch.long)
        return mem, mask

    def forward(self, u: torch.Tensor, labels: torch.Tensor):
        mem, mask = self._memory(u)
        out = self.model(
            encoder_outputs=BaseModelOutput(last_hidden_state=mem),
            attention_mask=mask,
            labels=labels,
            return_dict=True,
        )
        return out.loss, out.logits

    @torch.no_grad()
    def generate(self, u: torch.Tensor, max_new_tokens: int = 32, num_beams: int = 4, **kw):
        mem, mask = self._memory(u)
        return self.model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=mem),
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            **kw,
        )
