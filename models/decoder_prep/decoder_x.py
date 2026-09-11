"""Detail decoder: latent slots -> text.

Two changes.

**Level conditioning.** A learned level token (and optionally a length-bucket token) is prepended
to the cross-attention memory. Previously the decoder received a degraded latent and a short
target and had to work out for itself whether the shortness was intentional or the latent was
broken; that ambiguity is a direct cause of degenerate output at high `t`. Now "produce an
abstract sentence" is an instruction rather than an inference.

**Free-running generation for evaluation.** `forward` is teacher-forced, which is the correct
*training* signal but a badly inflated *evaluation* signal: previously validation decoded
`argmax(logits)` from a pass that had already been handed the gold prefix at every position.
`generate` is the only thing that should ever produce a reported number.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

from models.hf_utils import load_seq2seq


class DecoderX(nn.Module):
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        d_model: int | None = None,
        use_level_token: bool = True,
        use_length_token: bool = True,
        max_levels: int = 13,
        n_length_buckets: int = 8,
    ):
        super().__init__()
        self.model = load_seq2seq(model_name)
        self.d_model = int(d_model or self.model.config.d_model)
        self.use_level_token = use_level_token
        self.use_length_token = use_length_token

        if use_level_token:
            self.level_emb = nn.Embedding(max_levels + 1, self.d_model)
            nn.init.normal_(self.level_emb.weight, std=self.d_model**-0.5)
        if use_length_token:
            self.length_emb = nn.Embedding(n_length_buckets, self.d_model)
            nn.init.normal_(self.length_emb.weight, std=self.d_model**-0.5)
        self.n_length_buckets = n_length_buckets

    # -- memory assembly --------------------------------------------------------------------
    def build_memory(
        self,
        latents: torch.Tensor,                  # [B, L, d]
        attention_mask: torch.Tensor | None = None,
        level: torch.Tensor | None = None,      # [B] rung index
        length_bucket: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = latents.shape
        device = latents.device
        if attention_mask is None:
            attention_mask = torch.ones(B, L, device=device, dtype=torch.long)

        extra = []
        if self.use_level_token:
            lv = level if level is not None else torch.zeros(B, dtype=torch.long, device=device)
            extra.append(self.level_emb(lv.long().clamp(min=0)).unsqueeze(1))
        if self.use_length_token:
            lb = (length_bucket if length_bucket is not None
                  else torch.zeros(B, dtype=torch.long, device=device))
            extra.append(self.length_emb(lb.long().clamp(0, self.n_length_buckets - 1)).unsqueeze(1))

        if not extra:
            return latents, attention_mask
        prefix = torch.cat(extra, dim=1)
        memory = torch.cat([prefix, latents], dim=1)
        mask = torch.cat(
            [torch.ones(B, prefix.size(1), device=device, dtype=attention_mask.dtype),
             attention_mask], dim=1,
        )
        return memory, mask

    # -- training ----------------------------------------------------------------------------
    def forward(
        self,
        latents: torch.Tensor,
        attention_mask: torch.Tensor | None,
        labels: torch.Tensor,
        level: torch.Tensor | None = None,
        length_bucket: torch.Tensor | None = None,
    ):
        memory, mask = self.build_memory(latents, attention_mask, level, length_bucket)
        out = self.model(
            encoder_outputs=BaseModelOutput(last_hidden_state=memory),
            attention_mask=mask,
            labels=labels,
            return_dict=True,
        )
        return out.loss, out.logits

    # -- evaluation ---------------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        latents: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        level: torch.Tensor | None = None,
        length_bucket: torch.Tensor | None = None,
        max_new_tokens: int = 64,
        num_beams: int = 4,
        min_new_tokens: int = 3,
        no_repeat_ngram_size: int = 3,
        do_sample: bool = False,
        **kw,
    ) -> torch.Tensor:
        memory, mask = self.build_memory(latents, attention_mask, level, length_bucket)
        return self.model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=memory),
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=do_sample,
            **kw,
        )


def length_bucket_of(token_counts: torch.Tensor, n_buckets: int = 8, max_len: int = 128
                     ) -> torch.Tensor:
    """Map a target token count to a coarse bucket, so the decoder can be told how long to be."""
    edges = torch.linspace(0, max_len, n_buckets + 1, device=token_counts.device)[1:-1]
    return torch.bucketize(token_counts.float(), edges).clamp(0, n_buckets - 1)
