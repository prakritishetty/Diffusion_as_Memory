"""Named configs. `tiny` is what makes the whole pipeline testable on a laptop CPU."""

from __future__ import annotations

from configs.base import ExperimentConfig


def tiny() -> ExperimentConfig:
    """~1M params, runs P0->P1->P2 end to end on CPU in about a minute. No network access."""
    return ExperimentConfig(
        name="tiny",
        device="cpu",
        data=dict_to(dict(max_seq_len=32, max_gist_len=16, max_levels=6)),
        model=_model(
            encoder_name="tiny://d=64,vocab=128",
            decoder_name="tiny://d=64,vocab=128",
            gist_decoder_name="tiny://d=64,vocab=128",
            gist_encoder_name=None,
            d_model=64, n_slots=4, u_dim=32,
            slot_pool_heads=2, u_head_heads=2, v_head_blocks=1, v_head_heads=2,
        ),
        denoiser=_denoiser(n_blocks=2, n_heads=2, d_ff=128, n_cue_tokens=2),
        gpsi=_gpsi(n_blocks=2, n_heads=2, d_ff=128, n_cue_tokens=2),
        diffusion=_diffusion(T=50, ddim_steps_train=4, ddim_steps_eval=6),
        train=_train(batch_size=4, epochs=2, val_interval=1, amp=False,
                     eval_max_new_tokens=12, eval_num_beams=1),
        cache=_cache(enabled=False),
    )


def dev() -> ExperimentConfig:
    """Real models, tiny schedule. One GPU-minute smoke test before a long run."""
    cfg = ExperimentConfig(name="dev")
    return cfg.override(**{
        "train.epochs": 2, "train.batch_size": 8, "train.val_interval": 1,
        "diffusion.ddim_steps_train": 8, "diffusion.ddim_steps_eval": 10,
        "train.eval_num_beams": 1, "train.eval_max_new_tokens": 32,
    })


def small() -> ExperimentConfig:
    """t5-small everywhere: matches the original stack, for like-for-like comparison."""
    cfg = ExperimentConfig(name="small")
    return cfg.override(**{
        "model.encoder_name": "t5-small",
        "model.decoder_name": "t5-small",
        "model.d_model": 512,
    })


def full() -> ExperimentConfig:
    """flan-t5-base, the main configuration."""
    return ExperimentConfig(name="full")


def legacy() -> ExperimentConfig:
    """Reproduces the original setup, for the level-map / supervision-collapse ablation."""
    cfg = small()
    return cfg.override(**{
        "name": "legacy",
        "data.max_seq_len": 64,
        "data.include_source_as_level0": False,
        "data.xplus_source": "identity",
        "diffusion.level_map": "bucket",
        "diffusion.stratified_t": False,
        "diffusion.parameterization": "eps",
        "diffusion.gist_anchored": False,
        "model.use_level_token": False,
        "model.use_length_token": False,
        "model.v_head_normalize": False,
        "gpsi.use_self_attn": False,
        "gpsi.use_cross_attn": False,
        "gpsi.n_blocks": 3,
        "denoiser.use_cross_attn": False,
        "loss.lambda_y": 0.0,
        "loss.lambda_align": 0.0,
        "loss.lambda_orth": 0.0,
        "loss.lambda_proj": 0.0,
    })


PRESETS = {"tiny": tiny, "dev": dev, "small": small, "full": full, "legacy": legacy}


def get(name: str) -> ExperimentConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name]()


# -- small helpers so the preset bodies stay readable ------------------------------------
from configs.base import (  # noqa: E402
    CacheConfig, DataConfig, DenoiserConfig, DiffusionConfig, GPsiConfig, ModelConfig,
    TrainConfig,
)


def dict_to(d):  return DataConfig(**d)
def _model(**k): return ModelConfig(**k)
def _denoiser(**k): return DenoiserConfig(**k)
def _gpsi(**k):  return GPsiConfig(**k)
def _diffusion(**k): return DiffusionConfig(**k)
def _train(**k): return TrainConfig(**k)
def _cache(**k): return CacheConfig(**k)
