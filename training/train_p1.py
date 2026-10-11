"""P1 -- train the denoiser on frozen latents.

Trains: the denoiser (plus the gist-anchor projection, when enabled). Everything from P0 is
frozen, so this runs on the `(u, v0)` tensors P0 cached rather than re-encoding text every step --
the original re-ran a frozen encoder on every batch, which was the single largest waste in the
pipeline.

Also here: x0-parameterisation, min-SNR-gamma weighting, classifier-free-guidance dropout on `u`,
and an EMA of the weights.

    python -m training.train_p1 --p0-checkpoint runs/full/p0/best_model.pt
"""

from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from configs.base import ExperimentConfig
from models.denoiser_module.denoiser import EMA
from models.latent_diffusion import LatentDiffusion
from training.common import (
    ETATracker, apply_set_overrides, build_cache, build_loaders, build_scheduler,
    build_tokenizer, config_from_checkpoint, load_checkpoint, prepare_run,
    restore_training_state, save_checkpoint,
)


def latent_loaders(cfg: ExperimentConfig, p0_ckpt: str) -> dict[str, DataLoader]:
    """Load `(u, v0)` for each split, computing them from the P0 checkpoint on a cache miss.

    Written so it behaves identically with caching on or off -- the cache is a speed-up, never a
    correctness dependency.
    """
    from utils.cache import file_hash

    cache = build_cache(cfg)
    ck = file_hash(p0_ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state: dict = {}

    def _encode_all() -> dict[str, dict[str, torch.Tensor]]:
        if "latents" in state:
            return state["latents"]
        from models.memory_model import MemoryModel
        from training.train_p0 import cache_latents

        tok = build_tokenizer(cfg)
        loaders = build_loaders(cfg, tok, cache, ("train", "val"))
        model = MemoryModel(cfg).to(device)
        model.load_state_dict(load_checkpoint(p0_ckpt, device)["model_state_dict"], strict=False)
        state["latents"] = cache_latents(model, loaders, cfg, device, cache, p0_ckpt)
        del model
        return state["latents"]

    out: dict[str, DataLoader] = {}
    for split in ("train", "val"):
        d = (cache.get_or_compute("p0_latents", [split, ck], lambda s=split: _encode_all()[s])
             if cache.enabled else _encode_all()[split])
        ds = TensorDataset(d["u"], d["v0"])
        out[split] = DataLoader(ds, batch_size=cfg.train.batch_size,
                                shuffle=(split == "train"), drop_last=(split == "train"))
    return out


def run_epoch(diffusion, loader, optimizer, scheduler, ema, cfg, device, train: bool):
    diffusion.train(train)
    totals: dict[str, float] = {}
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for u, v0 in tqdm(loader, desc="P1 " + ("train" if train else "val"), leave=False):
            u, v0 = u.to(device), v0.to(device)
            t = torch.randint(1, diffusion.schedule.T + 1, (u.shape[0],), device=device)
            loss, stats = diffusion.training_loss(v0, u, t)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.train.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(diffusion.parameters(),
                                                   cfg.train.max_grad_norm)
                optimizer.step()
                if scheduler:
                    scheduler.step()
                ema.update(diffusion.denoiser)
            for k, v in stats.items():
                totals[k] = totals.get(k, 0.0) + float(v)
    n = max(len(loader), 1)
    prefix = "train/" if train else "val/"
    return {prefix + k: v / n for k, v in totals.items()}


@torch.no_grad()
def multistep_error_by_t(diffusion, loader, device, probes=(50, 250, 500, 750, 1000)):
    """Latent error at several depths, using the multi-step chain only.

    The original reported one-step error, which diverges as t -> T because it divides by
    sqrt(alpha_bar). There is no one-step readout any more.
    """
    diffusion.eval()
    u, v0 = next(iter(loader))
    u, v0 = u.to(device), v0.to(device)
    T = diffusion.schedule.T
    out: dict[str, float] = {}
    for t in probes:
        t = min(int(t), T)
        v_t, _ = diffusion.forget(v0, torch.full((u.shape[0],), t, device=device), u=u, seed=0)
        v_rec, _ = diffusion.recall(v_t, t, 1, u, seed=0)
        out[f"val/multistep_mse_t{t}"] = float((v_rec - v0).pow(2).mean())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="P1: train the denoiser")
    ap.add_argument("--p0-checkpoint", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="config override, e.g. --set diffusion.gist_anchored=false")
    ap.add_argument("--resume", default=None, help="path to a last.pt to continue from")
    args = ap.parse_args(argv)

    cfg = config_from_checkpoint(args.p0_checkpoint)
    ov = {}
    if args.name:   ov["name"] = args.name
    if args.epochs: ov["train.epochs"] = args.epochs
    if args.wandb:  ov["wandb_enabled"] = True
    cfg = cfg.override(**ov) if ov else cfg
    cfg = apply_set_overrides(cfg, args.set)

    device, run_dir, logger = prepare_run(cfg, "p1")
    loaders = latent_loaders(cfg, args.p0_checkpoint)

    d_model = loaders["train"].dataset.tensors[1].shape[-1]
    diffusion = LatentDiffusion(cfg, d_model).to(device)
    n_params = sum(p.numel() for p in diffusion.parameters() if p.requires_grad)
    print(f"P1 denoiser params: {n_params/1e6:.1f}M  d={d_model}  param={cfg.diffusion.parameterization}")

    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = build_scheduler(optimizer, cfg, len(loaders["train"]) * cfg.train.epochs)
    ema = EMA(diffusion.denoiser, cfg.denoiser.ema_decay)

    best = float("inf")
    best_path = os.path.join(run_dir, "best_model.pt")
    last_path = os.path.join(run_dir, "last.pt")
    start_epoch = 0
    if args.resume:
        start_epoch, best = restore_training_state(
            args.resume, device, optimizer=optimizer, scheduler=scheduler,
            diffusion=diffusion, ema=ema,
        )
    eta = ETATracker(cfg.train.epochs)

    for epoch in range(start_epoch, cfg.train.epochs):
        eta.start_epoch()
        metrics = run_epoch(diffusion, loaders["train"], optimizer, scheduler, ema, cfg,
                            device, train=True)

        if (epoch + 1) % cfg.train.val_interval == 0 or epoch == cfg.train.epochs - 1:
            ema.apply_to(diffusion.denoiser)
            metrics.update(run_epoch(diffusion, loaders["val"], None, None, ema, cfg,
                                     device, train=False))
            metrics.update(multistep_error_by_t(diffusion, loaders["val"], device))
            score = metrics.get("val/loss_denoise", float("inf"))
            if score < best:
                best = score
                save_checkpoint(best_path, cfg, epoch, metrics, best=best,
                                diffusion=diffusion, ema=ema.state_dict())
            ema.restore(diffusion.denoiser)

        save_checkpoint(last_path, cfg, epoch, metrics, best=best, diffusion=diffusion,
                        ema=ema.state_dict(), optimizer=optimizer, scheduler=scheduler)

        elapsed, eta_str = eta.end_epoch()
        metrics.update({"epoch": epoch + 1, "epoch_time_s": elapsed, "eta": eta_str})
        logger.log(metrics, step=epoch + 1)
        print(f"epoch {epoch+1}/{cfg.train.epochs} "
              f"loss={metrics.get('train/loss_denoise', 0):.4f} eta={eta_str}")

    logger.finish()
    print(f"P1 done. best val loss {best:.4f}. checkpoint: {best_path}")
    return best_path


if __name__ == "__main__":
    main()
