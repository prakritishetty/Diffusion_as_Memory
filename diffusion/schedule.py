"""Noise schedules.

Differences from the original `NoiseSchedule`:
  * buffers live on the model device instead of being read back with `.item()` per batch element
  * exposes `log_snr`, which is the axis the timestep -> abstraction-rung map is defined on
  * `sigmoid` schedule added as an ablation against `cosine`
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

ScheduleType = Literal["cosine", "linear", "sigmoid"]
_EPS = 1e-6


def _cosine_alpha_bar(T: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(T + 1, dtype=torch.float64)
    ac = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = torch.clip(1 - (ac[1:] / ac[:-1]), 1e-4, 0.9999)
    return torch.cumprod(1 - betas, dim=0)


def _linear_alpha_bar(T: int) -> torch.Tensor:
    betas = torch.linspace(1e-4, 0.02, T, dtype=torch.float64)
    return torch.cumprod(1 - betas, dim=0)


def _sigmoid_alpha_bar(T: int, start: float = -3.0, end: float = 3.0, tau: float = 1.0
                       ) -> torch.Tensor:
    """Roughly linear in log-SNR, so equal spacing in t is closer to equal information loss."""
    t = torch.linspace(0, 1, T + 1, dtype=torch.float64)
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    ac = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    ac = (ac / ac[0]).clamp(1e-9, 1.0)
    betas = torch.clip(1 - (ac[1:] / ac[:-1]), 1e-4, 0.9999)
    return torch.cumprod(1 - betas, dim=0)


class NoiseSchedule(nn.Module):
    """Holds ᾱ for t = 1..T. Index with 1-based timesteps throughout."""

    def __init__(self, T: int = 1000, schedule_type: ScheduleType = "cosine"):
        super().__init__()
        self.T = int(T)
        self.schedule_type = schedule_type

        if schedule_type == "cosine":
            ab = _cosine_alpha_bar(self.T)
        elif schedule_type == "linear":
            ab = _linear_alpha_bar(self.T)
        elif schedule_type == "sigmoid":
            ab = _sigmoid_alpha_bar(self.T)
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type!r}")

        ab = ab.clamp(_EPS, 1 - _EPS).float()
        self.register_buffer("alpha_bar_table", ab, persistent=False)
        self.register_buffer("log_snr_table", torch.log(ab / (1 - ab)), persistent=False)

    # -- lookups (vectorised, on-device) -------------------------------------------------
    def _gather(self, table: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.to(device=table.device, dtype=torch.long).clamp(1, self.T)
        return table[t - 1]

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        return self._gather(self.alpha_bar_table, t)

    def sqrt_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        return self.alpha_bar(t).sqrt()

    def sqrt_one_minus_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        return (1 - self.alpha_bar(t)).sqrt()

    def log_snr(self, t: torch.Tensor) -> torch.Tensor:
        return self._gather(self.log_snr_table, t)

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        return self.log_snr(t).exp()

    def min_snr_weight(self, t: torch.Tensor, gamma: float = 5.0) -> torch.Tensor:
        """min-SNR-γ loss weighting (Hang et al. 2023); flat when gamma <= 0."""
        if gamma <= 0:
            return torch.ones_like(self.alpha_bar(t))
        snr = self.snr(t)
        return torch.clamp(snr, max=gamma) / snr.clamp(min=_EPS)

    # -- convenience ---------------------------------------------------------------------
    def timesteps(self, device=None) -> torch.Tensor:
        return torch.arange(1, self.T + 1, device=device or self.alpha_bar_table.device)

    def extra_repr(self) -> str:
        return f"T={self.T}, schedule={self.schedule_type}"
