"""Text encoders.

Two encoders, because `u` and `v0` want different things:

* the **detail** encoder is trainable and feeds slot pooling -> `v0`, which must retain entities,
  numbers and dates well enough to reconstruct the source;
* the **gist** encoder is a frozen off-the-shelf retrieval model used only to supply an alignment
  target for `u`.

The gist encoder is what makes cue-conditioned recall cheap later on. Anchoring `u` near
`E(gist)` puts it in a space where questions already embed close to their topics, so at QA time
the cue is just `E(question)` -- one frozen forward pass -- instead of a separately learned
question-to-latent map. The same vectors double as an ANN index over stored memories.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hf_utils import load_encoder


class TextEncoder(nn.Module):
    """Trainable detail encoder: token ids -> hidden states."""

    def __init__(self, model_name: str = "google/flan-t5-base"):
        super().__init__()
        self.model_name = model_name
        self.encoder, self.hidden_dim_size = load_encoder(model_name)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return out.last_hidden_state

    def gradient_checkpointing_enable(self):
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()


class FrozenGistEncoder(nn.Module):
    """Frozen sentence encoder producing one vector per text, for the `u` alignment target.

    Always eval-mode and always `no_grad`; its outputs are cached to disk, so in practice this
    runs once over the dataset and never again.
    """

    def __init__(self, model_name: str = "intfloat/e5-base-v2", pooling: str = "mean"):
        super().__init__()
        self.model_name = model_name
        self.pooling = pooling
        self.encoder, self.hidden_dim_size = load_encoder(model_name)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):  # stays frozen regardless of the parent's mode
        return super().train(False)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        h = out.last_hidden_state
        if self.pooling == "cls":
            pooled = h[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return F.normalize(pooled, dim=-1)
