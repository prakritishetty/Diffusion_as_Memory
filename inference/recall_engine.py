"""RecallEngine -- the one place the forgetting/recall trajectory is implemented.

Replaces the five near-duplicate scripts that used to live in `scripts/inference/`. Evaluation,
ablations and the downstream QA harness all call this, so there is exactly one definition of what
"recall at budget b" means.

Two modes:
  * `forget(v0, t)`  -- push a stored latent forward along the schedule (elapsed time)
  * `recall(v_t, u)` -- run the reverse chain with the gist as cue, reading out at chosen stops

Everything is seeded per example, so a trajectory is reproducible across runs and machines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from configs.base import ExperimentConfig
from models.decoder_prep.decoder_x import length_bucket_of
from utils.seed import trajectory_seed


@dataclass
class Rung:
    budget: float
    t: int
    level: int
    text: str
    latent: torch.Tensor | None = field(default=None, repr=False)


class RecallEngine:
    def __init__(
        self,
        model,                 # MemoryModel
        diffusion,             # LatentDiffusion
        g_psi=None,            # SemanticProjectionModule (P2); None decodes v_t directly
        tokenizer=None,
        cfg: ExperimentConfig | None = None,
        device: torch.device | str = "cpu",
    ):
        self.model = model.eval().to(device)
        self.diffusion = diffusion.eval().to(device)
        self.g_psi = g_psi.eval().to(device) if g_psi is not None else None
        self.tokenizer = tokenizer
        self.cfg = cfg or model.cfg
        self.device = torch.device(device)
        self.level_map = diffusion.level_map
        self.T = diffusion.schedule.T

    # -- encoding -------------------------------------------------------------------------------
    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            list(texts), padding="max_length", truncation=True,
            max_length=self.cfg.data.max_seq_len, return_tensors="pt",
        )
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        out = self.model.encode(ids, mask)
        return out.u, out.v0

    # -- forgetting -----------------------------------------------------------------------------
    @torch.no_grad()
    def forget(self, v0: torch.Tensor, t: int, u: torch.Tensor | None = None,
               seed: int | None = None) -> torch.Tensor:
        t_vec = torch.full((v0.shape[0],), int(t), device=v0.device, dtype=torch.long)
        v_t, _ = self.diffusion.forget(v0, t_vec, u=u, seed=seed)
        return v_t

    # -- readout ---------------------------------------------------------------------------------
    @torch.no_grad()
    def project(self, v_t: torch.Tensor, t: int, u: torch.Tensor | None,
                q: torch.Tensor | None = None) -> torch.Tensor:
        """Map a latent on the trajectory onto the decoder's manifold."""
        if self.g_psi is None:
            return v_t
        t_vec = torch.full((v_t.shape[0],), int(t), device=v_t.device, dtype=torch.long)
        pred = self.diffusion.denoiser(v_t, t_vec, u, q)
        from diffusion.gaussian import to_v0
        v_hat_0 = to_v0(pred, v_t, t_vec, self.diffusion.schedule,
                        self.diffusion.param, self.diffusion.anchor(u))
        return self.g_psi(t=t_vec, v_t=v_t, v_hat_0=v_hat_0, u=u, q=q)

    @torch.no_grad()
    def decode(self, v: torch.Tensor, level: int, target_len: int | None = None) -> list[str]:
        B = v.shape[0]
        lvl = torch.full((B,), int(level), device=v.device, dtype=torch.long)
        lb = None
        if target_len is not None:
            lb = length_bucket_of(
                torch.full((B,), float(target_len), device=v.device),
                self.cfg.model.n_length_buckets, self.cfg.data.max_seq_len,
            )
        ids = self.model.generate_from_latent(v, level=lvl, length_bucket=lb)
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)

    # -- the early-exit ladder ----------------------------------------------------------------------
    @torch.no_grad()
    def ladder(
        self,
        texts: Sequence[str],
        budgets: Sequence[float] | None = None,
        n_levels: int | None = None,
        example_ids: Sequence[str] | None = None,
        cfg_scale: float | None = None,
        shuffle_cue: bool = False,
        draw: int = 0,
    ) -> list[list[Rung]]:
        """Encode, forget to the top of the schedule, then descend once reading out at each stop.

        One descent produces every rung, which is what makes early exit cheaper than regenerating
        per budget.
        """
        u, v0 = self.encode_texts(texts)
        if shuffle_cue:  # the control that tests whether recall is genuinely cue-conditioned
            u = u[torch.randperm(u.shape[0], device=u.device)]

        n_levels = n_levels or 6
        if budgets is None:
            budgets = [k / (n_levels - 1) for k in range(n_levels - 1, -1, -1)]
        budgets = sorted(budgets, reverse=True)

        stops = [self.level_map.t_for_budget(b, self.device) for b in budgets]
        seed = trajectory_seed(
            self.cfg.seed, (example_ids[0] if example_ids else "batch"), draw
        )

        t_top = max(stops)
        v_top = self.forget(v0, t_top, u=u, seed=seed)
        _, rec = self.diffusion.recall(
            v_top, t_top, min(stops), u, cfg_scale=cfg_scale, seed=seed, record_at=stops
        )

        per_example: list[list[Rung]] = [[] for _ in range(len(texts))]
        for budget, t in zip(budgets, stops):
            v_t = rec[t]
            level = int(self.level_map.level_index(
                torch.tensor([t], device=self.device),
                torch.tensor([n_levels], device=self.device),
            ).item())
            v_proj = self.project(v_t, t, u)
            for i, txt in enumerate(self.decode(v_proj, level)):
                per_example[i].append(
                    Rung(budget=budget, t=t, level=level, text=txt,
                         latent=v_proj[i].detach().cpu())
                )
        for rungs in per_example:
            rungs.sort(key=lambda r: r.t)  # least abstract first
        return per_example

    # -- diagnostics -------------------------------------------------------------------------------
    @torch.no_grad()
    def forgetting_ladder(self, texts: Sequence[str], budgets: Sequence[float],
                          n_levels: int = 6, draw: int = 0) -> list[list[Rung]]:
        """Decode the *forward* trajectory: what the memory looks like as time passes, with no
        denoising. Isolates decoder robustness from the denoiser's contribution."""
        u, v0 = self.encode_texts(texts)
        seed = trajectory_seed(self.cfg.seed, "forget", draw)
        out: list[list[Rung]] = [[] for _ in texts]
        for b in sorted(budgets):
            t = self.level_map.t_for_budget(b, self.device)
            v_t = self.forget(v0, t, u=u, seed=seed)
            level = int(self.level_map.level_index(
                torch.tensor([t], device=self.device),
                torch.tensor([n_levels], device=self.device)).item())
            for i, txt in enumerate(self.decode(v_t, level)):
                out[i].append(Rung(budget=b, t=t, level=level, text=txt))
        return out

    @torch.no_grad()
    def oracle_ladder(self, level_texts: Sequence[Sequence[str]]) -> list[list[Rung]]:
        """Encode each *gold* rung and decode it straight back.

        Upper bound on what the decoder can do, so any remaining gap is attributable to the
        diffusion path rather than to the reader.
        """
        out: list[list[Rung]] = []
        n = max(len(ls) for ls in level_texts)
        flat = [(i, k, s) for i, ls in enumerate(level_texts) for k, s in enumerate(ls)]
        _, v0 = self.encode_texts([s for _, _, s in flat])
        decoded: dict[tuple[int, int], str] = {}
        for start in range(0, len(flat), 32):
            chunk = flat[start:start + 32]
            v = v0[start:start + 32]
            for (i, k, _), txt in zip(chunk, self.decode(v, level=chunk[0][1])):
                decoded[(i, k)] = txt
        for i, ls in enumerate(level_texts):
            out.append([
                Rung(budget=k / max(n - 1, 1), t=0, level=k, text=decoded[(i, k)])
                for k in range(len(ls))
            ])
        return out
