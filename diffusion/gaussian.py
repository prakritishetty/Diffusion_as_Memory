"""Forward process and parameterisation conversions.

Two departures from the original:

* **Gist-anchored forward process.** We diffuse the *residual* `w0 = v0 - m(u)` rather than `v0`,
  so `v_T ≈ m(u) + noise` instead of pure noise. The most abstract end of the ladder then has a
  semantic floor (the gist) rather than a noise floor -- "details dissolve, gist remains" becomes
  structural. It is ordinary DDPM on a shifted variable, so nothing else changes.

* **x0-parameterisation.** Predicting `v0` directly removes the `1/sqrt(ᾱ_t)` division that made
  the original one-step readout diverge as `t -> T`. No single-shot readout exists any more;
  the per-step `v0` estimate here is the standard quantity every sampler consumes.
"""

from __future__ import annotations

from typing import Literal

import torch

from diffusion.schedule import NoiseSchedule
from utils.seed import randn_like_seeded

Parameterization = Literal["x0", "eps", "v"]


def _bcast(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return x.view(-1, *([1] * (ref.dim() - 1)))


def q_sample(
    v0: torch.Tensor,
    t: torch.Tensor,
    schedule: NoiseSchedule,
    anchor: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Corrupt a clean latent. Returns (v_t, eps)."""
    if noise is None:
        noise = randn_like_seeded(v0, seed)
    base = v0 if anchor is None else v0 - anchor
    sa = _bcast(schedule.sqrt_alpha_bar(t), v0)
    sm = _bcast(schedule.sqrt_one_minus_alpha_bar(t), v0)
    wt = sa * base + sm * noise
    return (wt if anchor is None else wt + anchor), noise


def to_v0(
    pred: torch.Tensor,
    v_t: torch.Tensor,
    t: torch.Tensor,
    schedule: NoiseSchedule,
    parameterization: Parameterization,
    anchor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert a model prediction into an estimate of the clean latent."""
    wt = v_t if anchor is None else v_t - anchor
    sa = _bcast(schedule.sqrt_alpha_bar(t), v_t)
    sm = _bcast(schedule.sqrt_one_minus_alpha_bar(t), v_t)

    if parameterization == "x0":
        w0 = pred
    elif parameterization == "eps":
        # The only place ᾱ appears in a denominator; kept for the eps-vs-x0 ablation.
        w0 = (wt - sm * pred) / sa.clamp(min=1e-4)
    elif parameterization == "v":
        w0 = sa * wt - sm * pred
    else:
        raise ValueError(parameterization)
    return w0 if anchor is None else w0 + anchor


def to_eps(
    pred: torch.Tensor,
    v_t: torch.Tensor,
    t: torch.Tensor,
    schedule: NoiseSchedule,
    parameterization: Parameterization,
    anchor: torch.Tensor | None = None,
) -> torch.Tensor:
    wt = v_t if anchor is None else v_t - anchor
    sa = _bcast(schedule.sqrt_alpha_bar(t), v_t)
    sm = _bcast(schedule.sqrt_one_minus_alpha_bar(t), v_t)

    if parameterization == "eps":
        return pred
    if parameterization == "x0":
        return (wt - sa * pred) / sm.clamp(min=1e-4)
    if parameterization == "v":
        return sm * wt + sa * pred
    raise ValueError(parameterization)


def target_for(
    parameterization: Parameterization,
    v0: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
    schedule: NoiseSchedule,
    anchor: torch.Tensor | None = None,
) -> torch.Tensor:
    """What the denoiser should regress onto, given the chosen parameterisation."""
    w0 = v0 if anchor is None else v0 - anchor
    if parameterization == "x0":
        return w0
    if parameterization == "eps":
        return noise
    if parameterization == "v":
        sa = _bcast(schedule.sqrt_alpha_bar(t), v0)
        sm = _bcast(schedule.sqrt_one_minus_alpha_bar(t), v0)
        return sa * noise - sm * w0
    raise ValueError(parameterization)


def loss_weight(
    parameterization: Parameterization,
    t: torch.Tensor,
    schedule: NoiseSchedule,
    min_snr_gamma: float = 5.0,
) -> torch.Tensor:
    """min-SNR-γ. For x0-prediction the weight is `min(snr, γ)`, not `min(snr, γ)/snr`."""
    if min_snr_gamma <= 0:
        return torch.ones_like(schedule.alpha_bar(t))
    snr = schedule.snr(t)
    clamped = torch.clamp(snr, max=min_snr_gamma)
    if parameterization == "x0":
        return clamped
    if parameterization == "eps":
        return clamped / snr.clamp(min=1e-8)
    return clamped / (snr + 1).clamp(min=1e-8)
