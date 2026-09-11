"""Single source of truth for every hyperparameter.

Config drift across `train_denoiser_prakriti.py` (L=8, u_dim=128), `train_phase2_config.py`
(L=16, u_dim=512) and `DenoiserConfig` (L=16, u_dim=512) previously made checkpoints silently
incompatible across phases. Everything now derives from `ExperimentConfig`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Literal


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
@dataclass
class DataConfig:
    data_dir: str = "data/final"
    train_file: str = "train.json"
    val_file: str = "validate.json"
    test_file: str = "test.json"

    # 64 truncated ~1/3 of MSR examples (mean 36.8 words, p95 56, max 135).
    max_seq_len: int = 128
    max_gist_len: int = 32

    # Ladder = [x] + xt, so level 0 is the verbatim source. Without this the model can
    # never be asked to reproduce full detail.
    include_source_as_level0: bool = True
    max_levels: int = 13

    # Contrastive positives.
    xplus_source: Literal["ladder", "paraphrase", "dropout", "identity"] = "ladder"
    xplus_dropout_prob: float = 0.15
    ladder_positive_max_level: int = 3  # sample rungs 0..k as "same memory" positives

    ladder_variant: Literal["llm", "control"] = "llm"


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
@dataclass
class ModelConfig:
    encoder_name: str = "google/flan-t5-base"
    decoder_name: str = "google/flan-t5-base"
    gist_decoder_name: str | None = "t5-small"          # P0 only, dropped at P1/P2
    gist_encoder_name: str | None = "intfloat/e5-base-v2"  # frozen, anchors u in a retrieval space

    d_model: int = 768        # overwritten from the encoder config at build time
    n_slots: int = 16
    u_dim: int = 256

    slot_pool_heads: int = 8
    u_head_heads: int = 8
    v_head_blocks: int = 2
    v_head_heads: int = 8
    # v0 must be ~unit variance or `sqrt(abar)*v0` swamps `sqrt(1-abar)*eps` until very late,
    # which produces the "cliff" degradation instead of graded forgetting.
    v_head_normalize: bool = True

    # Decoder is told which rung to produce instead of inferring it from a degraded latent.
    use_level_token: bool = True
    use_length_token: bool = True
    n_length_buckets: int = 8

    dropout: float = 0.1


@dataclass
class DenoiserConfig:
    n_blocks: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    n_cue_tokens: int = 4        # u -> cue tokens for cross-attention
    use_cross_attn: bool = True  # was commented out in the original
    ema_decay: float = 0.999


@dataclass
class GPsiConfig:
    n_blocks: int = 6            # was 3
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    n_cue_tokens: int = 4
    use_self_attn: bool = True   # was False -> module was a per-slot pointwise MLP
    use_cross_attn: bool = True
    use_vhat0_input: bool = True # concat [v_t, v_hat_0]; the original declared it and dropped it


# --------------------------------------------------------------------------------------
# Diffusion
# --------------------------------------------------------------------------------------
@dataclass
class DiffusionConfig:
    T: int = 1000
    schedule: Literal["cosine", "linear", "sigmoid"] = "cosine"

    # x0-prediction removes the 1/sqrt(abar) blow-up that made high-t readout unusable.
    parameterization: Literal["x0", "eps", "v"] = "x0"
    min_snr_gamma: float = 5.0

    # Diffuse w0 = v0 - m(u) so v_T ~= gist + noise rather than pure noise.
    gist_anchored: bool = True

    # Rung index from the timestep. "uniform_t" gives every rung an equal number of timesteps
    # (measured: per-rung sqrt(abar) 1.00/0.96/0.86/0.70/0.50/0.26/0.00, all distinguishable).
    # "logsnr" and "bucket" exist for the supervision-map ablation.
    level_map: Literal["uniform_t", "logsnr", "bucket"] = "uniform_t"
    logsnr_clip: float = 6.0
    bucket_size: int = 100  # only for level_map="bucket" (the original, kept for the ablation)
    stratified_t: bool = True

    # Same sampler, same K, in training and evaluation.
    ddim_steps_train: int = 20
    ddim_steps_eval: int = 50
    ddim_eta: float = 0.0

    cfg_dropout_prob: float = 0.1  # drop u during P1 to enable classifier-free guidance
    cfg_scale: float = 1.0


# --------------------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------------------
@dataclass
class LossConfig:
    lambda_x: float = 1.0      # decode v0 -> x
    lambda_y: float = 0.5      # decode u  -> y   (gist supervision)
    lambda_align: float = 0.5  # cos(proj(u), E_frozen(y))
    lambda_nce: float = 0.1    # small: fights lambda_y for memories with similar gists
    lambda_orth: float = 0.1   # (cos(u, mean_L v0))^2
    nce_temperature: float = 0.07
    nce_symmetric: bool = True

    # P2: dense latent target vs. distant cross-entropy through the decoder.
    lambda_proj: float = 1.0
    lambda_ce: float = 1.0


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
@dataclass
class TrainConfig:
    batch_size: int = 16
    epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 1e-5
    scheduler: Literal["cosine", "linear", "none"] = "cosine"
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    val_interval: int = 5
    num_workers: int = 0
    amp: bool = True
    freeze_decoder_in_p2: bool = False
    eval_max_new_tokens: int = 64
    eval_num_beams: int = 4
    eval_no_repeat_ngram: int = 3
    eval_min_new_tokens: int = 3


@dataclass
class CacheConfig:
    enabled: bool = True
    root: str = "cache"
    # A stale cache silently trains on the wrong latents, so keys carry a code version.
    code_version: str = "v3"
    n_trajectory_seeds: int = 4


# --------------------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    name: str = "default"
    seed: int = 0
    device: str = "auto"
    out_dir: str = "runs"

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    denoiser: DenoiserConfig = field(default_factory=DenoiserConfig)
    gpsi: GPsiConfig = field(default_factory=GPsiConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    wandb_project: str = "diffusion-as-memory"
    wandb_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def hash(self, *sections: str) -> str:
        """Stable short hash over selected sections (default: all), for cache keys."""
        d = self.to_dict()
        payload = {k: d[k] for k in sections} if sections else d
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        kw: dict[str, Any] = {}
        for f_ in dataclasses.fields(cls):
            if f_.name not in d:
                continue
            if dataclasses.is_dataclass(f_.type) or isinstance(d[f_.name], dict):
                sub = {
                    "data": DataConfig, "model": ModelConfig, "denoiser": DenoiserConfig,
                    "gpsi": GPsiConfig, "diffusion": DiffusionConfig, "loss": LossConfig,
                    "train": TrainConfig, "cache": CacheConfig,
                }.get(f_.name)
                kw[f_.name] = sub(**d[f_.name]) if sub else d[f_.name]
            else:
                kw[f_.name] = d[f_.name]
        return cls(**kw)

    def override(self, **dotted: Any) -> "ExperimentConfig":
        """`cfg.override(**{"model.n_slots": 32, "train.epochs": 5})` -> a new config."""
        d = self.to_dict()
        for key, val in dotted.items():
            parts = key.split(".")
            node = d
            for p in parts[:-1]:
                node = node[p]
            if parts[-1] not in node:
                raise KeyError(f"unknown config key: {key}")
            node[parts[-1]] = val
        return ExperimentConfig.from_dict(d)
