"""Shared training plumbing: builders, checkpoints, schedulers, logging.

Checkpoints embed the full `ExperimentConfig`, which is what stops the phases drifting apart --
previously P1 ran with L=8/u_dim=128 while P2 assumed L=16/u_dim=512 and the mismatch was silent.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from configs.base import ExperimentConfig
from dataloader.memory_dataset import MemoryLadderDataset, collate
from utils.cache import CacheStore
from utils.seed import set_seed

log = logging.getLogger(__name__)


def resolve_device(cfg: ExperimentConfig) -> torch.device:
    if cfg.device != "auto":
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_tokenizer(cfg: ExperimentConfig):
    from models.hf_utils import is_tiny
    if is_tiny(cfg.model.encoder_name):
        from tests.stubs import StubTokenizer
        return StubTokenizer()
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(cfg.model.encoder_name)


def build_cache(cfg: ExperimentConfig) -> CacheStore:
    return CacheStore(cfg.cache.root, cfg.cache.code_version, cfg.cache.enabled)


def build_loaders(
    cfg: ExperimentConfig,
    tokenizer,
    cache: CacheStore | None = None,
    splits: tuple[str, ...] = ("train", "val"),
) -> dict[str, DataLoader]:
    files = {"train": cfg.data.train_file, "val": cfg.data.val_file, "test": cfg.data.test_file}
    out: dict[str, DataLoader] = {}
    for split in splits:
        ds = MemoryLadderDataset(
            os.path.join(cfg.data.data_dir, files[split]), tokenizer, cfg, cache, split
        )
        out[split] = DataLoader(
            ds, batch_size=cfg.train.batch_size, shuffle=(split == "train"),
            num_workers=cfg.train.num_workers, collate_fn=collate, drop_last=(split == "train"),
        )
    return out


def build_scheduler(optimizer, cfg: ExperimentConfig, total_steps: int):
    name = cfg.train.scheduler
    if name == "none" or total_steps <= 0:
        return None
    from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
    warmup = max(0, min(int(total_steps * cfg.train.warmup_ratio), total_steps))
    fn = get_cosine_schedule_with_warmup if name == "cosine" else get_linear_schedule_with_warmup
    return fn(optimizer, num_warmup_steps=warmup, num_training_steps=total_steps)


# -- checkpoints -------------------------------------------------------------------------------
def save_checkpoint(path: str, cfg: ExperimentConfig, epoch: int, metrics: dict, **modules):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload: dict[str, Any] = {
        "config": cfg.to_dict(), "epoch": epoch, "metrics": metrics,
        "torch_version": torch.__version__,
    }
    for name, mod in modules.items():
        if mod is None:
            continue
        payload[f"{name}_state_dict"] = (
            mod.state_dict() if hasattr(mod, "state_dict") else mod
        )
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    log.info("saved checkpoint %s (epoch %d)", path, epoch)


def load_checkpoint(path: str, map_location="cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if "config" not in ckpt:
        raise ValueError(f"{path} has no embedded config; refusing to guess the architecture")
    return ckpt


def config_from_checkpoint(path: str) -> ExperimentConfig:
    return ExperimentConfig.from_dict(load_checkpoint(path)["config"])


def assert_compatible(a: ExperimentConfig, b: ExperimentConfig) -> None:
    """Fail loudly when two phases disagree on the latent geometry."""
    for key in ("n_slots", "u_dim", "encoder_name", "decoder_name"):
        av, bv = getattr(a.model, key), getattr(b.model, key)
        if av != bv:
            raise ValueError(f"config mismatch on model.{key}: {av!r} vs {bv!r}")
    for key in ("T", "schedule", "parameterization", "gist_anchored", "level_map"):
        av, bv = getattr(a.diffusion, key), getattr(b.diffusion, key)
        if av != bv:
            raise ValueError(f"config mismatch on diffusion.{key}: {av!r} vs {bv!r}")


# -- logging -----------------------------------------------------------------------------------
class RunLogger:
    """Thin wrapper so wandb is optional and jsonl is always written."""

    def __init__(self, cfg: ExperimentConfig, run_dir: str, phase: str):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.path = os.path.join(run_dir, f"{phase}_metrics.jsonl")
        self.wandb = None
        if cfg.wandb_enabled:
            try:
                import wandb
                self.wandb = wandb
                wandb.init(project=cfg.wandb_project, name=f"{cfg.name}-{phase}",
                           config=cfg.to_dict())
            except Exception as exc:  # never let logging kill a run
                log.warning("wandb disabled: %s", exc)
        cfg.save(os.path.join(run_dir, "config.json"))

    def log(self, metrics: dict, step: int | None = None) -> None:
        row = {k: (float(v) if torch.is_tensor(v) else v) for k, v in metrics.items()}
        if step is not None:
            row["step"] = step
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        if self.wandb:
            self.wandb.log(row, step=step)

    def finish(self) -> None:
        if self.wandb:
            self.wandb.finish()


@dataclass
class ETATracker:
    total_epochs: int
    _times: list[float] = None
    _start: float | None = None

    def __post_init__(self):
        self._times = []

    def start_epoch(self):
        self._start = time.time()

    def end_epoch(self) -> tuple[float, str]:
        elapsed = time.time() - (self._start or time.time())
        self._times.append(elapsed)
        remaining = self.total_epochs - len(self._times)
        eta = (sum(self._times) / len(self._times)) * max(remaining, 0)
        h, rem = divmod(int(eta), 3600)
        m, s = divmod(rem, 60)
        return elapsed, f"{h:02d}:{m:02d}:{s:02d}"


def prepare_run(cfg: ExperimentConfig, phase: str) -> tuple[torch.device, str, RunLogger]:
    set_seed(cfg.seed)
    device = resolve_device(cfg)
    run_dir = os.path.join(cfg.out_dir, cfg.name, phase)
    os.makedirs(run_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return device, run_dir, RunLogger(cfg, run_dir, phase)


def autocast_ctx(cfg: ExperimentConfig, device: torch.device):
    from contextlib import nullcontext
    if cfg.train.amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()
