"""The P0 memory model: encoder -> slot pooling -> (u, v0) -> decoders.

Replaces `forgetting_model.py`. Structural differences:

* `u` and `v0` come from genuinely different heads (attention-pooled MLP vs. slot transformer),
  not two `Linear`s on the same tensor.
* the gist decoder and the alignment/orthogonality terms are live, so the gist/detail split is
  trained rather than hoped for.
* `G_psi` is not part of P0 any more -- P0 trains the autoencoder, P2 trains the projection.
  Previously P0 optionally ran `v0` through `G_psi`, which entangled the two phases.
* an explicit `encode` returning everything (slots, attention maps, u, v0) so ablations do not
  need to re-implement the forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from configs.base import ExperimentConfig
from models.decoder_prep.decoder_x import DecoderX, length_bucket_of
from models.decoder_prep.decoder_y import DecoderY
from models.encoder_prep.encoder import TextEncoder
from models.losses import alignment_loss, info_nce, orthogonality_loss
from models.slot_pooling_prep.slot_pooling import SlotPooling, mask_slots
from models.uv_heads_prep.u_head import UHead
from models.uv_heads_prep.v_head import VHead


@dataclass
class EncodeOutput:
    slots: torch.Tensor                    # [B, L, d]
    u: torch.Tensor                        # [B, u_dim]
    v0: torch.Tensor                       # [B, L, d]
    pool_attn: torch.Tensor | None = None  # [B, L, seq]
    u_attn: torch.Tensor | None = None     # [B, 1, L]


class MemoryModel(nn.Module):
    def __init__(self, cfg: ExperimentConfig, gist_encoder_dim: int | None = None):
        super().__init__()
        self.cfg = cfg
        m = cfg.model

        self.encoder = TextEncoder(m.encoder_name)
        d = self.encoder.hidden_dim_size
        self.d_model = d

        self.slot_pooling = SlotPooling(d, m.n_slots, m.slot_pool_heads)
        self.u_head = UHead(d, m.u_dim, m.u_head_heads, m.dropout)
        self.v_head = VHead(d, m.n_slots, m.v_head_blocks, m.v_head_heads,
                            m.dropout, m.v_head_normalize)
        self.decoder_x = DecoderX(
            m.decoder_name, d_model=None,
            use_level_token=m.use_level_token, use_length_token=m.use_length_token,
            max_levels=cfg.data.max_levels, n_length_buckets=m.n_length_buckets,
        )
        self.decoder_y = (
            DecoderY(m.u_dim, m.gist_decoder_name) if m.gist_decoder_name else None
        )
        self.align_proj = (
            nn.Linear(m.u_dim, gist_encoder_dim) if gist_encoder_dim else None
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # -- encoding ---------------------------------------------------------------------------
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = False,
        slot_mask_ids: list[int] | None = None,
    ) -> EncodeOutput:
        H = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if return_attention:
            slots, pool_attn = self.slot_pooling(H, attention_mask, return_attention=True)
        else:
            slots, pool_attn = self.slot_pooling(H, attention_mask), None

        if slot_mask_ids:
            slots = mask_slots(slots, slot_mask_ids)

        if return_attention:
            u, u_attn = self.u_head(slots, return_attention=True)
        else:
            u, u_attn = self.u_head(slots), None
        return EncodeOutput(slots=slots, u=u, v0=self.v_head(slots),
                            pool_attn=pool_attn, u_attn=u_attn)

    @torch.no_grad()
    def encode_latents(self, input_ids, attention_mask) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.encode(input_ids, attention_mask)
        return out.u, out.v0

    # -- training -----------------------------------------------------------------------------
    def forward(self, batch: dict) -> tuple[torch.Tensor, dict]:
        cfg, dev = self.cfg, self.device
        lw = cfg.loss
        get = lambda k: batch[k].to(dev)  # noqa: E731

        enc = self.encode(get("x_input_ids"), get("x_attention"))
        u, v0 = enc.u, enc.v0
        B = u.shape[0]

        labels_x = get("x_labels")
        level0 = torch.zeros(B, dtype=torch.long, device=dev)
        lb = length_bucket_of(
            (labels_x != -100).sum(1), cfg.model.n_length_buckets, cfg.data.max_seq_len
        )
        slot_mask = torch.ones(B, cfg.model.n_slots, device=dev, dtype=torch.long)
        loss_x, logits_x = self.decoder_x(v0, slot_mask, labels_x, level=level0, length_bucket=lb)

        parts = {"loss_x": loss_x.detach()}
        total = lw.lambda_x * loss_x

        if self.decoder_y is not None and lw.lambda_y > 0 and "y_labels" in batch:
            loss_y, _ = self.decoder_y(u, get("y_labels"))
            total = total + lw.lambda_y * loss_y
            parts["loss_y"] = loss_y.detach()

        if self.align_proj is not None and lw.lambda_align > 0 and "gist_embedding" in batch:
            loss_a = alignment_loss(self.align_proj(u), get("gist_embedding"))
            total = total + lw.lambda_align * loss_a
            parts["loss_align"] = loss_a.detach()

        if lw.lambda_nce > 0 and "xpos_input_ids" in batch:
            u_pos = self.encode(get("xpos_input_ids"), get("xpos_attention")).u
            loss_n = info_nce(u, u_pos, lw.nce_temperature, lw.nce_symmetric)
            total = total + lw.lambda_nce * loss_n
            parts["loss_nce"] = loss_n.detach()

        if lw.lambda_orth > 0:
            loss_o = orthogonality_loss(u, v0)
            total = total + lw.lambda_orth * loss_o
            parts["loss_orth"] = loss_o.detach()

        parts["loss_total"] = total.detach()
        parts["v0_std"] = v0.detach().std()
        parts["u_norm"] = u.detach().norm(dim=-1).mean()
        return total, {"logits_x": logits_x, "u": u, "v0": v0, **parts}

    # -- evaluation ----------------------------------------------------------------------------
    @torch.no_grad()
    def generate_from_latent(
        self,
        v: torch.Tensor,
        level: torch.Tensor | None = None,
        length_bucket: torch.Tensor | None = None,
        **kw,
    ) -> torch.Tensor:
        """Free-running generation -- the only path that should produce a reported number."""
        mask = torch.ones(v.shape[0], v.shape[1], device=v.device, dtype=torch.long)
        t = self.cfg.train
        kw.setdefault("max_new_tokens", t.eval_max_new_tokens)
        kw.setdefault("num_beams", t.eval_num_beams)
        kw.setdefault("no_repeat_ngram_size", t.eval_no_repeat_ngram)
        kw.setdefault("min_new_tokens", t.eval_min_new_tokens)
        return self.decoder_x.generate(v, mask, level=level, length_bucket=length_bucket, **kw)

    @torch.no_grad()
    def generate_gist(self, u: torch.Tensor, **kw) -> torch.Tensor:
        if self.decoder_y is None:
            raise RuntimeError("no gist decoder in this configuration")
        return self.decoder_y.generate(u, **kw)

    # -- phase plumbing -------------------------------------------------------------------------
    def freeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def drop_gist_decoder(self) -> None:
        """P0 only: `D_y` is discarded before P1/P2 so it never touches the recall graph."""
        self.decoder_y = None

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
