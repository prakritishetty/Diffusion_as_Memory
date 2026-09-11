"""DDIM sampler -- the single implementation used by training, evaluation, ablations and the
downstream QA harness.

Why it exists: previously P2 fed G_psi a latent produced by *one* reverse step (via
`step_by_step_estimate`, which returns `v_{t-1}` and was mislabelled `v_hat_0`), while inference
would run the full chain. G_psi was trained on an input distribution it never sees at test time.

Running the same strided chain in both places removes that by construction. K=20 strided steps on
a 19M-param denoiser over a [B, L, d] tensor is negligible next to one encoder-decoder step.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import torch

from diffusion.gaussian import Parameterization, to_eps, to_v0
from diffusion.schedule import NoiseSchedule
from utils.seed import generator_for

# (v_t, t, u) -> prediction in the configured parameterisation
DenoiseFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]


def strided_timesteps(t_start: int, t_end: int, n_steps: int) -> list[int]:
    """Descending timesteps from t_start down to t_end inclusive, at most n_steps+1 entries."""
    t_start, t_end = int(t_start), int(t_end)
    if t_start <= t_end:
        return [t_start]
    n_steps = max(1, int(n_steps))
    if t_start - t_end <= n_steps:
        return list(range(t_start, t_end - 1, -1))
    xs = torch.linspace(t_start, t_end, n_steps + 1).round().long().tolist()
    out: list[int] = []
    for x in xs:
        if not out or x < out[-1]:
            out.append(int(x))
    if out[-1] != t_end:
        out.append(t_end)
    return out


class DDIMSampler:
    def __init__(
        self,
        schedule: NoiseSchedule,
        n_steps: int = 50,
        eta: float = 0.0,
        parameterization: Parameterization = "x0",
    ):
        self.schedule = schedule
        self.n_steps = n_steps
        self.eta = eta
        self.parameterization = parameterization

    # ------------------------------------------------------------------------------------
    def _predict(
        self,
        denoise_fn: DenoiseFn,
        v_t: torch.Tensor,
        t_scalar: int,
        u: torch.Tensor | None,
        anchor: torch.Tensor | None,
        cfg_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.full((v_t.shape[0],), t_scalar, device=v_t.device, dtype=torch.long)
        pred = denoise_fn(v_t, t, u)
        if cfg_scale != 1.0 and u is not None:
            pred_uncond = denoise_fn(v_t, t, None)
            pred = pred_uncond + cfg_scale * (pred - pred_uncond)
        v0_hat = to_v0(pred, v_t, t, self.schedule, self.parameterization, anchor)
        eps_hat = to_eps(pred, v_t, t, self.schedule, self.parameterization, anchor)
        return v0_hat, eps_hat

    def step(
        self,
        v_t: torch.Tensor,
        v0_hat: torch.Tensor,
        eps_hat: torch.Tensor,
        t: int,
        t_prev: int,
        anchor: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """One DDIM update from t to t_prev, in residual space when anchored."""
        dev = v_t.device
        ab_t = self.schedule.alpha_bar(torch.tensor([t], device=dev))
        ab_prev = (
            self.schedule.alpha_bar(torch.tensor([t_prev], device=dev))
            if t_prev >= 1 else torch.ones(1, device=dev)
        )
        ab_t = ab_t.view(-1, *([1] * (v_t.dim() - 1)))
        ab_prev = ab_prev.view(-1, *([1] * (v_t.dim() - 1)))

        sigma = self.eta * torch.sqrt(
            ((1 - ab_prev) / (1 - ab_t).clamp(min=1e-8)) * (1 - ab_t / ab_prev.clamp(min=1e-8))
        ).clamp(min=0.0)

        w0 = v0_hat if anchor is None else v0_hat - anchor
        dir_term = torch.sqrt((1 - ab_prev - sigma**2).clamp(min=0.0)) * eps_hat
        w_prev = ab_prev.sqrt() * w0 + dir_term

        if self.eta > 0 and t_prev >= 1:
            noise = torch.randn(w_prev.shape, generator=generator,
                                device=w_prev.device, dtype=w_prev.dtype)
            w_prev = w_prev + sigma * noise
        return w_prev if anchor is None else w_prev + anchor

    # ------------------------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        denoise_fn: DenoiseFn,
        v_start: torch.Tensor,
        t_start: int,
        t_end: int = 1,
        u: torch.Tensor | None = None,
        anchor: torch.Tensor | None = None,
        cfg_scale: float = 1.0,
        seed: int | None = None,
        record_at: Iterable[int] | None = None,
        n_steps: int | None = None,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Run the reverse chain. Returns (v_at_t_end, {t: v_t} for requested checkpoints).

        Requested stops are merged into the timestep grid so the recorded latent sits exactly at
        the asked-for t rather than at whichever stride happened to land nearby.
        """
        record = {int(r) for r in (record_at or []) if t_end <= int(r) <= t_start}
        ts = strided_timesteps(t_start, t_end, n_steps or self.n_steps)
        ts = sorted(set(ts) | record, reverse=True)

        gen = generator_for(seed, v_start.device) if seed is not None else None
        out: dict[int, torch.Tensor] = {}

        v = v_start
        if ts[0] in record:
            out[ts[0]] = v.clone()

        for t, t_prev in zip(ts, ts[1:]):
            v0_hat, eps_hat = self._predict(denoise_fn, v, t, u, anchor, cfg_scale)
            v = self.step(v, v0_hat, eps_hat, t, t_prev, anchor, gen)
            if t_prev in record:
                out[t_prev] = v.clone()
        out[ts[-1]] = v.clone()
        return v, out

    @torch.no_grad()
    def ladder(
        self,
        denoise_fn: DenoiseFn,
        v_start: torch.Tensor,
        t_start: int,
        stops: Sequence[int],
        **kw,
    ) -> dict[int, torch.Tensor]:
        """Descend once, harvesting the latent at each requested stop -- the early-exit path."""
        stops = sorted({int(s) for s in stops}, reverse=True)
        _, rec = self.sample(denoise_fn, v_start, t_start, min(stops), record_at=stops, **kw)
        return {s: rec[s] for s in stops if s in rec}
