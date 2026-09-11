"""P0 -- train the memory autoencoder.

Trains: encoder, slot pooling, u-head, v-head, detail decoder, gist decoder, alignment projection.
Objective: `lx*L_x + ly*L_y + la*L_align + ln*L_nce + lo*L_orth` (see `models/losses.py`).

At the end it measures `std(v0)` on the training set and caches `(u, v0)` so P1 never has to run
the frozen encoder again.

    python -m training.train_p0 --preset full --name p0-main
"""

from __future__ import annotations

import argparse
import os

import torch
from tqdm import tqdm

from configs import get as get_preset
from configs.base import ExperimentConfig
from models.memory_model import MemoryModel
from training.common import (
    ETATracker, autocast_ctx, build_cache, build_loaders, build_scheduler, build_tokenizer,
    prepare_run, save_checkpoint,
)
from utils.cache import CacheStore


def _maybe_gist_encoder(cfg: ExperimentConfig, device):
    if not cfg.model.gist_encoder_name or cfg.loss.lambda_align <= 0:
        return None
    from models.encoder_prep.encoder import FrozenGistEncoder
    return FrozenGistEncoder(cfg.model.gist_encoder_name).to(device)


def precompute_gist_embeddings(cfg, loaders, device, cache: CacheStore):
    """Encode every `y` once with the frozen retrieval encoder and attach to the datasets."""
    enc = _maybe_gist_encoder(cfg, device)
    if enc is None:
        return None
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model.gist_encoder_name)

    for split, loader in loaders.items():
        ds = loader.dataset

        def _compute(ds=ds):
            vecs = []
            for i in range(0, len(ds), 64):
                texts = [ds.examples[j].y for j in range(i, min(i + 64, len(ds)))]
                b = tok(texts, padding=True, truncation=True, max_length=cfg.data.max_gist_len,
                        return_tensors="pt").to(device)
                vecs.append(enc(b["input_ids"], b["attention_mask"]).cpu())
            return torch.cat(vecs)

        emb = cache.get_or_compute(
            "gist_embeddings", [split, cfg.model.gist_encoder_name, len(ds)], _compute
        )
        ds.attach_gist_embeddings(emb)
    return enc.hidden_dim_size


def train_epoch(model, loader, optimizer, scheduler, cfg, device):
    model.train()
    totals: dict[str, float] = {}
    for batch in tqdm(loader, desc="P0 train", leave=False):
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(cfg, device):
            loss, parts = model(batch)
        loss.backward()
        if cfg.train.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
        optimizer.step()
        if scheduler:
            scheduler.step()
        for k, v in parts.items():
            if torch.is_tensor(v) and v.ndim == 0:
                totals[k] = totals.get(k, 0.0) + float(v)
    n = max(len(loader), 1)
    return {f"train/{k}": v / n for k, v in totals.items()}


@torch.no_grad()
def validate(model, loader, cfg, device, tokenizer, n_samples: int = 8):
    """Loss is teacher-forced; reported text is free-running. Never mix the two."""
    model.eval()
    totals: dict[str, float] = {}
    samples: list[dict] = []
    for i, batch in enumerate(tqdm(loader, desc="P0 val", leave=False)):
        loss, parts = model(batch)
        for k, v in parts.items():
            if torch.is_tensor(v) and v.ndim == 0:
                totals[k] = totals.get(k, 0.0) + float(v)
        if len(samples) < n_samples:
            v0 = model.encode(batch["x_input_ids"].to(device),
                              batch["x_attention"].to(device)).v0
            ids = model.generate_from_latent(
                v0, level=torch.zeros(v0.shape[0], dtype=torch.long, device=device)
            )
            preds = tokenizer.batch_decode(ids, skip_special_tokens=True)
            for j, p in enumerate(preds):
                if len(samples) < n_samples:
                    samples.append({"source": batch["x_text"][j], "reconstruction": p})
    n = max(len(loader), 1)
    return {f"val/{k}": v / n for k, v in totals.items()}, samples


@torch.no_grad()
def cache_latents(model, loaders, cfg, device, cache: CacheStore, ckpt_path: str):
    """Freeze-and-store `(u, v0)` so P1 trains on tensors instead of re-encoding every step."""
    from utils.cache import file_hash
    model.eval()
    ck = file_hash(ckpt_path)
    out = {}
    for split, loader in loaders.items():
        def _compute(loader=loader):
            us, vs = [], []
            for batch in tqdm(loader, desc=f"cache {split} latents", leave=False):
                u, v0 = model.encode_latents(batch["x_input_ids"].to(device),
                                            batch["x_attention"].to(device))
                us.append(u.cpu())
                vs.append(v0.cpu())
            return {"u": torch.cat(us), "v0": torch.cat(vs)}
        out[split] = cache.get_or_compute("p0_latents", [split, ck], _compute)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="P0: train the memory autoencoder")
    ap.add_argument("--preset", default="full")
    ap.add_argument("--name", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)

    cfg = get_preset(args.preset)
    ov = {}
    if args.name:        ov["name"] = args.name
    if args.epochs:      ov["train.epochs"] = args.epochs
    if args.batch_size:  ov["train.batch_size"] = args.batch_size
    if args.data_dir:    ov["data.data_dir"] = args.data_dir
    if args.wandb:       ov["wandb_enabled"] = True
    if args.no_cache:    ov["cache.enabled"] = False
    cfg = cfg.override(**ov) if ov else cfg

    device, run_dir, logger = prepare_run(cfg, "p0")
    tokenizer = build_tokenizer(cfg)
    cache = build_cache(cfg)
    loaders = build_loaders(cfg, tokenizer, cache, ("train", "val"))

    gist_dim = precompute_gist_embeddings(cfg, loaders, device, cache)
    model = MemoryModel(cfg, gist_encoder_dim=gist_dim).to(device)
    print(f"P0 trainable params: {model.trainable_parameter_count()/1e6:.1f}M  d={model.d_model}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = build_scheduler(optimizer, cfg, len(loaders["train"]) * cfg.train.epochs)

    best = float("inf")
    best_path = os.path.join(run_dir, "best_model.pt")
    eta = ETATracker(cfg.train.epochs)

    for epoch in range(cfg.train.epochs):
        eta.start_epoch()
        metrics = train_epoch(model, loaders["train"], optimizer, scheduler, cfg, device)

        if (epoch + 1) % cfg.train.val_interval == 0 or epoch == cfg.train.epochs - 1:
            vmetrics, samples = validate(model, loaders["val"], cfg, device, tokenizer)
            metrics.update(vmetrics)
            score = vmetrics.get("val/loss_total", float("inf"))
            if score < best:
                best = score
                save_checkpoint(best_path, cfg, epoch, metrics, model=model)
            import json
            with open(os.path.join(run_dir, f"samples_epoch{epoch+1}.json"), "w",
                      encoding="utf-8") as f:
                json.dump(samples, f, indent=2)

        elapsed, eta_str = eta.end_epoch()
        metrics.update({"epoch": epoch + 1, "epoch_time_s": elapsed, "eta": eta_str})
        logger.log(metrics, step=epoch + 1)
        print(f"epoch {epoch+1}/{cfg.train.epochs} "
              f"loss={metrics.get('train/loss_total', 0):.4f} "
              f"v0_std={metrics.get('train/v0_std', 0):.3f} eta={eta_str}")

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device,
                                         weights_only=False)["model_state_dict"])
        cache_latents(model, loaders, cfg, device, cache, best_path)
    logger.finish()
    print(f"P0 done. best val loss {best:.4f}. checkpoint: {best_path}")
    return best_path


if __name__ == "__main__":
    main()
