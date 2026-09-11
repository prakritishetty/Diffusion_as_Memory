"""Timestep <-> abstraction-rung mapping.

The original rule was `k = min(t // 100, n_i - 1)` with T=1000, which assumes 10 rungs. MSR
ladders average 4.88, so ~61% of sampled timesteps collapsed onto the most abstract rung and the
objective carried almost no gradient separating intermediate levels.

Three maps are available:

``uniform_t`` (default)
    Each rung owns an equal number of timesteps, normalised per example: with n=6 and T=1000,
    rung k owns t in [167k+1, 167(k+1)]. Measured on the cosine schedule this is well behaved --
    per-rung signal fraction sqrt(alpha_bar) runs 1.00 -> 0.96 -> 0.86 -> 0.70 -> 0.50 -> 0.26
    -> 0.00, so every rung sits at a visibly different noise level.

``logsnr``
    Equal share of the log-SNR axis. Principled in theory, but log-SNR has very long tails
    (+9.2 at t=1, -13.8 at t=T), so unclipped the top rung owns 5 timesteps out of 1000.
    `logsnr_clip` bounds the axis to a usable window. Kept as an ablation arm.

``bucket``
    The original `t // bucket_size` rule, retained so the supervision-collapse ablation can
    reproduce it exactly.

`sample_stratified` then guarantees uniform rung coverage *per batch* rather than relying on
`t ~ U{1,T}` to spread out, which it does not do at batch size 16.
"""

from __future__ import annotations

from typing import Literal

import torch

from diffusion.schedule import NoiseSchedule

LevelMapType = Literal["uniform_t", "logsnr", "bucket"]


class LevelMap:
    """Maps timesteps to rung indices for ladders of varying length."""

    def __init__(
        self,
        schedule: NoiseSchedule,
        kind: LevelMapType = "uniform_t",
        bucket_size: int = 100,
        logsnr_clip: float = 6.0,
    ):
        self.schedule = schedule
        self.kind = kind
        self.bucket_size = bucket_size
        self.logsnr_clip = logsnr_clip
        self.T = schedule.T

        t_all = torch.arange(1, self.T + 1)
        if kind == "logsnr":
            lam = schedule.log_snr(t_all).clamp(-logsnr_clip, logsnr_clip)
            hi, lo = lam[0], lam[-1]
            frac = (hi - lam) / (hi - lo).clamp(min=1e-8)
        elif kind in ("uniform_t", "bucket"):
            frac = (t_all.float() - 1) / max(self.T - 1, 1)
        else:
            raise ValueError(f"unknown level map {kind!r}")

        # Monotone non-decreasing and spanning [0, 1]; searchsorted below relies on this.
        self._frac = torch.clamp(frac.contiguous(), 0.0, 1.0)

    # -- device plumbing -------------------------------------------------------------------
    def frac_table(self, device) -> torch.Tensor:
        if self._frac.device != torch.device(device):
            self._frac = self._frac.to(device)
        return self._frac

    def frac(self, t: torch.Tensor) -> torch.Tensor:
        """Fraction of information destroyed at t, in [0, 1]."""
        table = self.frac_table(t.device)
        return table[t.long().clamp(1, self.T) - 1]

    # -- forward map -----------------------------------------------------------------------
    def level_index(self, t: torch.Tensor, n_levels: torch.Tensor) -> torch.Tensor:
        """t [B] (1-based), n_levels [B] or scalar -> rung index [B] in [0, n_levels-1]."""
        t = t.long()
        if not torch.is_tensor(n_levels):
            n_levels = torch.full_like(t, int(n_levels))
        n_levels = n_levels.to(t.device).long().clamp(min=1)

        if self.kind == "bucket":
            k = t // self.bucket_size
        else:
            k = (self.frac(t) * n_levels.float()).floor().long()
        return torch.minimum(k, n_levels - 1).clamp(min=0)

    # -- inverse map -----------------------------------------------------------------------
    def level_bounds(self, k: torch.Tensor, n_levels: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inclusive [t_lo, t_hi] whose timesteps map to rung k."""
        k = k.long()
        if not torch.is_tensor(n_levels):
            n_levels = torch.full_like(k, int(n_levels))
        n_levels = n_levels.to(k.device).long().clamp(min=1)

        if self.kind == "bucket":
            t_lo = (k * self.bucket_size).clamp(min=1)
            last = k >= (n_levels - 1)
            t_hi = torch.where(last,
                               torch.full_like(k, self.T),
                               ((k + 1) * self.bucket_size - 1))
            return t_lo.clamp(1, self.T), t_hi.clamp(1, self.T)

        table = self.frac_table(k.device)
        lo_f = k.float() / n_levels.float()
        hi_f = (k + 1).float() / n_levels.float()
        t_lo = torch.searchsorted(table, lo_f.contiguous(), right=False) + 1
        t_hi = torch.searchsorted(table, hi_f.contiguous(), right=False)  # exclusive -> inclusive
        t_lo = t_lo.clamp(1, self.T)
        t_hi = torch.maximum(t_hi.clamp(1, self.T), t_lo)
        # The top rung always owns the tail of the schedule.
        t_hi = torch.where(k >= (n_levels - 1), torch.full_like(t_hi, self.T), t_hi)
        return t_lo, t_hi

    def sample_stratified(
        self,
        n_levels: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw rung k uniformly, then t uniformly inside rung k. Returns (t, k)."""
        device = n_levels.device
        n = n_levels.long().clamp(min=1)
        uk = torch.rand(n.shape, device=device, generator=generator)
        k = (uk * n.float()).floor().long().clamp(max=n - 1)

        t_lo, t_hi = self.level_bounds(k, n)
        ut = torch.rand(n.shape, device=device, generator=generator)
        t = t_lo + (ut * (t_hi - t_lo + 1).float()).floor().long()
        return t.clamp(1, self.T), k

    def sample_uniform_t(
        self,
        n_levels: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The original behaviour: t ~ U{1,T}, rung derived from t."""
        device = n_levels.device
        t = torch.randint(1, self.T + 1, n_levels.shape, device=device, generator=generator)
        return t, self.level_index(t, n_levels)

    def sample(self, n_levels, stratified: bool = True, generator=None):
        fn = self.sample_stratified if stratified else self.sample_uniform_t
        return fn(n_levels, generator)

    # -- inference helpers ------------------------------------------------------------------
    def t_for_level(self, k: int, n_levels: int, device=None, where: str = "mid") -> int:
        """Representative timestep for a requested rung -- used to pick recall depth."""
        kk = torch.tensor([k], device=device)
        nn_ = torch.tensor([n_levels], device=device)
        lo, hi = self.level_bounds(kk, nn_)
        if where == "lo":
            return int(lo.item())
        if where == "hi":
            return int(hi.item())
        return int(((lo + hi) // 2).item())

    def t_for_budget(self, budget: float, device=None) -> int:
        """budget in [0,1]: 0 = full detail (t=1), 1 = maximally abstract (t=T)."""
        table = self.frac_table(device or "cpu")
        val = torch.tensor([float(min(max(budget, 0.0), 1.0))], device=table.device)
        idx = torch.searchsorted(table, val.contiguous(), right=False)
        return int(idx.clamp(0, self.T - 1).item()) + 1

    def coverage_report(self, n_levels_list: list[int]) -> dict:
        """Fraction of timesteps landing on each rung -- the table that showed the collapse."""
        out: dict = {}
        for n in sorted(set(n_levels_list)):
            t = torch.arange(1, self.T + 1)
            k = self.level_index(t, torch.full_like(t, n))
            counts = torch.bincount(k, minlength=n).float() / self.T
            out[n] = {
                "per_level_fraction": [round(c, 4) for c in counts.tolist()],
                "fraction_on_final_level": round(counts[-1].item(), 4),
                "max_minus_min": round((counts.max() - counts.min()).item(), 4),
            }
        return out
