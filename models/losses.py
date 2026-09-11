"""Loss terms.

The P0 objective is
    L = lx*L_x + ly*L_y + la*L_align + ln*L_nce + lo*L_orth

`L_x` alone (the original) leaves `v0` doing all the work and `u` decorative. The other three are
what make the gist/detail split a trained property:

* `L_y`     -- `u` must be enough to write the gist.
* `L_align` -- `u` must land near a frozen retrieval embedding of the gist, which puts it in a
               space where a question embeds close to its topic (see `encoder.py`).
* `L_nce`   -- `u` must be invariant across the ladder but distinct across memories.
* `L_orth`  -- `u` must not duplicate `v0`.

`L_y` and `L_nce` pull against each other for memories with similar gists, so `lambda_nce` is
deliberately small.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce(
    u: torch.Tensor,
    u_pos: torch.Tensor,
    temperature: float = 0.07,
    symmetric: bool = True,
) -> torch.Tensor:
    """Instance discrimination between an anchor and its positive, other batch items negative.

    The previous version was one-directional and, more importantly, was fed `u_pos == u` because
    the dataloader set `xpos_input_ids = x_input_ids`. That made the objective "be unique", which
    any injective encoder satisfies without learning anything.
    """
    u = F.normalize(u, dim=-1)
    u_pos = F.normalize(u_pos, dim=-1)
    logits = u @ u_pos.t() / temperature
    labels = torch.arange(u.size(0), device=u.device)
    loss = F.cross_entropy(logits, labels)
    if symmetric:
        loss = 0.5 * (loss + F.cross_entropy(logits.t(), labels))
    return loss


def alignment_loss(u_proj: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - cosine between the projected gist code and a frozen sentence embedding of `y`."""
    return (1 - F.cosine_similarity(F.normalize(u_proj, dim=-1),
                                    F.normalize(target, dim=-1), dim=-1)).mean()


def orthogonality_loss(u: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
    """Squared cosine between `u` and mean-pooled `v0`, after projecting to a shared width.

    Stable and one line, unlike a gradient-reversal detail probe.
    """
    v_pool = v0.mean(dim=1)
    d = min(u.shape[-1], v_pool.shape[-1])
    cos = F.cosine_similarity(F.normalize(u[..., :d], dim=-1),
                              F.normalize(v_pool[..., :d], dim=-1), dim=-1)
    return (cos ** 2).mean()


def projection_loss(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """L_proj: regress G_psi's output onto the frozen encoder's latent for the target rung text.

    That target is by construction a point the decoder was trained to read, so this is literally
    "project onto the manifold" -- a dense objective in place of a distant cross-entropy.
    """
    return F.mse_loss(v_pred, v_target, reduction=reduction)


def diffusion_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-sample-weighted MSE (min-SNR-gamma weighting comes from `diffusion.loss_weight`)."""
    per = F.mse_loss(pred, target, reduction="none").flatten(1).mean(1)
    if weight is not None:
        per = per * weight.to(per.dtype).flatten()
    return per.mean()


def slot_type_loss(logits_per_slot: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Optional: force slot j to carry entity type c_j, so slot masking becomes causal.

    logits_per_slot: [B, n_slots, n_vocab_bins]; targets: multi-hot, same shape.
    """
    return F.binary_cross_entropy_with_logits(logits_per_slot, targets.to(logits_per_slot.dtype))
