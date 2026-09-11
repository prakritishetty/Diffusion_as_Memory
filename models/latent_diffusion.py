"""Ties the denoiser, schedule, gist anchor and sampler into one object.

Everything that touches the diffusion trajectory goes through here, so training, evaluation,
ablations and the downstream QA harness cannot drift apart -- which is the class of bug that
produced the one-step/multi-step mismatch in the first place.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configs.base import ExperimentConfig
from diffusion.gaussian import loss_weight, q_sample, target_for, to_v0
from diffusion.level_map import LevelMap
from diffusion.sampler import DDIMSampler
from diffusion.schedule import NoiseSchedule
from models.denoiser_module.denoiser import Denoiser
from models.layers import GistAnchor
from models.losses import diffusion_loss


class LatentDiffusion(nn.Module):
    def __init__(self, cfg: ExperimentConfig, d_model: int, q_dim: int = 0):
        super().__init__()
        self.cfg = cfg
        dc = cfg.diffusion
        self.param = dc.parameterization

        self.schedule = NoiseSchedule(dc.T, dc.schedule)
        self.level_map = LevelMap(self.schedule, dc.level_map, dc.bucket_size, dc.logsnr_clip)
        self.denoiser = Denoiser(d_model, cfg.model.n_slots, cfg.model.u_dim,
                                 cfg.denoiser, q_dim=q_dim)
        self.anchor_net = (
            GistAnchor(cfg.model.u_dim, d_model, cfg.model.n_slots) if dc.gist_anchored else None
        )
        self.train_sampler = DDIMSampler(self.schedule, dc.ddim_steps_train, dc.ddim_eta, self.param)
        self.eval_sampler = DDIMSampler(self.schedule, dc.ddim_steps_eval, dc.ddim_eta, self.param)

    # -- anchor -------------------------------------------------------------------------------
    def anchor(self, u: torch.Tensor | None) -> torch.Tensor | None:
        if self.anchor_net is None or u is None:
            return None
        return self.anchor_net(u)

    # -- forward process -----------------------------------------------------------------------
    def forget(self, v0, t, u=None, seed=None, noise=None):
        """Corrupt a stored latent to timestep t. Simulates elapsed time."""
        return q_sample(v0, t, self.schedule, anchor=self.anchor(u), noise=noise, seed=seed)

    # -- training -------------------------------------------------------------------------------
    def denoise_fn(self, u: torch.Tensor | None, q: torch.Tensor | None = None):
        return lambda v_t, t, u_override=None: self.denoiser(
            v_t, t, u if u_override is None else u_override, q
        )

    def training_loss(self, v0, u, t, seed=None) -> tuple[torch.Tensor, dict]:
        dc = self.cfg.diffusion
        anchor = self.anchor(u)
        v_t, noise = q_sample(v0, t, self.schedule, anchor=anchor, seed=seed)

        u_in = u
        if self.training and dc.cfg_dropout_prob > 0:
            drop = torch.rand(u.shape[0], device=u.device) < dc.cfg_dropout_prob
            u_in = torch.where(drop.unsqueeze(-1), self.denoiser.null_u.expand_as(u), u)

        pred = self.denoiser(v_t, t, u_in)
        target = target_for(self.param, v0, noise, t, self.schedule, anchor=anchor)
        w = loss_weight(self.param, t, self.schedule, dc.min_snr_gamma)
        loss = diffusion_loss(pred, target, w)

        with torch.no_grad():
            v0_hat = to_v0(pred, v_t, t, self.schedule, self.param, anchor)
            stats = {"loss_denoise": loss.detach(),
                     "v0_mse": (v0_hat - v0).pow(2).mean().detach()}
        return loss, stats

    # -- reverse process --------------------------------------------------------------------------
    @torch.no_grad()
    def recall(
        self,
        v_start: torch.Tensor,
        t_start: int,
        t_end: int,
        u: torch.Tensor | None,
        q: torch.Tensor | None = None,
        cfg_scale: float | None = None,
        seed: int | None = None,
        record_at=None,
        training_steps: bool = False,
    ):
        """Run the reverse chain. `training_steps=True` uses the training step count so the
        latent handed to G_psi during P2 comes from the same procedure as at inference."""
        sampler = self.train_sampler if training_steps else self.eval_sampler
        return sampler.sample(
            self.denoise_fn(u, q), v_start, t_start, t_end,
            u=u, anchor=self.anchor(u),
            cfg_scale=self.cfg.diffusion.cfg_scale if cfg_scale is None else cfg_scale,
            seed=seed, record_at=record_at,
        )

    @torch.no_grad()
    def noisy_state_for_training(self, v0, u, t, seed=None, from_T: bool = True):
        """Produce the `(v_t, v_hat_0)` pair that G_psi consumes in P2.

        Runs the same strided DDIM chain used at inference rather than a single reverse step.
        The original passed `step_by_step_estimate` (which returns `v_{t-1}`) as `v_hat_0`, so
        G_psi trained on an input distribution it never sees at test time.

        Timesteps differ per example, so the chain descends once to `min(t)` while recording at
        every distinct t in the batch, then each row is gathered from its own stop.
        """
        anchor = self.anchor(u)
        T = self.schedule.T

        if from_T:
            stops = sorted({int(v) for v in t.tolist()}, reverse=True)
            v_T, _ = q_sample(v0, torch.full_like(t, T), self.schedule, anchor=anchor, seed=seed)
            _, rec = self.recall(v_T, T, stops[-1], u, seed=seed,
                                 record_at=stops, training_steps=True)
            v_t = torch.stack([rec[int(t[b])][b] for b in range(v0.shape[0])])
        else:
            v_t, _ = q_sample(v0, t, self.schedule, anchor=anchor, seed=seed)

        pred = self.denoiser(v_t, t, u)
        v_hat_0 = to_v0(pred, v_t, t, self.schedule, self.param, anchor)
        return v_t, v_hat_0
