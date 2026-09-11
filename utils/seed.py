"""Seeding. Reverse-process noise was previously undrawn from any controlled generator, so
trajectories were not reproducible and monotonicity was measured on a fresh noise draw each run.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generator_for(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def trajectory_seed(base_seed: int, example_id: str | int, draw: int = 0) -> int:
    """Deterministic per-(example, draw) seed so a trajectory is reproducible anywhere."""
    h = hashlib.sha256(f"{example_id}|{draw}".encode()).digest()
    return (int(base_seed) * 1_000_003 + int.from_bytes(h[:4], "little")) % (2**31)


def randn_like_seeded(x: torch.Tensor, seed: int | None, domain: str = "noise") -> torch.Tensor:
    """Reproducible Gaussian noise shaped like `x`.

    The seed is domain-separated: calling `set_seed(s)` and then `q_sample(..., seed=s)` would
    otherwise draw the *same* stream, making the injected noise identical to data drawn straight
    after seeding.
    """
    if seed is None:
        return torch.randn_like(x)
    g = generator_for(_mix(seed, domain), device=x.device)
    return torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype)


def _mix(seed: int, domain: str) -> int:
    h = hashlib.sha256(f"{domain}:{int(seed)}".encode()).digest()
    return int.from_bytes(h[:4], "little") % (2**31)
