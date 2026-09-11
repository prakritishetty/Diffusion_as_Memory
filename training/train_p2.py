"""P2 -- train G_psi (and optionally fine-tune the decoder).

Objective: `lambda_proj * L_proj + lambda_ce * L_CE`.

`L_proj` is the change that matters. G_psi previously had no standalone loss -- all gradient
arrived through a distant cross-entropy at the decoder output, which is a weak, high-variance
signal for a module doing geometry. Here it regresses onto the frozen encoder's latent for the
*ground-truth rung text*, which is by construction a point the decoder was trained to read. That
is literally "project onto the manifold", and it lets the decoder stay frozen
(`train.freeze_decoder_in_p2`), which in turn removes any P0/P2 input-distribution mismatch.

The latent handed to G_psi comes from the same strided DDIM chain used at inference.

    python -m training.train_p2 --p0-checkpoint ... --p1-checkpoint ...
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from tqdm import tqdm

from configs.base import ExperimentConfig
from dataloader.memory_dataset import labels_for_levels
from models.decoder_prep.decoder_x import length_bucket_of
from models.g_psi_module.semantic_projection import SemanticProjectionModule
from models.latent_diffusion import LatentDiffusion
from models.losses import projection_loss
from models.memory_model import MemoryModel
from training.common import (
    ETATracker, assert_compatible, autocast_ctx, build_cache, build_loaders, build_scheduler,
    build_tokenizer, config_from_checkpoint, load_checkpoint, prepare_run, save_checkpoint,
)
from utils.cache import CacheStore, file_hash


@torch.no_grad()
def rung_latent_table(model, loaders, cfg, device, cache: CacheStore, p0_ckpt: str):
    """Encode every ladder rung once with the frozen P0 encoder -- the `L_proj` targets.

    ~1 GB in fp16 for the full corpus, computed once and reused by every P2 run.
    """
    ck = file_hash(p0_ckpt)
    tables = {}
    for split, loader in loaders.items():
        ds = loader.dataset

        def _compute(ds=ds):
            n, L = len(ds), cfg.data.max_levels
            out = torch.zeros(n, L, cfg.model.n_slots, model.d_model, dtype=torch.float16)
            for i in range(0, n, 16):
                sl = slice(i, min(i + 16, n))
                ids = ds.enc["level_input_ids"][sl].to(device)
                mask = ds.enc["level_attention"][sl].to(device)
                b, lv, s = ids.shape
                _, v0 = model.encode_latents(ids.view(b * lv, s), mask.view(b * lv, s))
                out[sl] = v0.view(b, lv, cfg.model.n_slots, -1).half().cpu()
            return out

        tables[split] = cache.get_or_compute("rung_latents", [split, ck], _compute)
    return tables


def _sample_t_and_level(diffusion, n_levels, cfg, device):
    return diffusion.level_map.sample(
        n_levels.to(device), stratified=cfg.diffusion.stratified_t
    )


def run_epoch(model, diffusion, g_psi, loader, rung_table, optimizer, scheduler, cfg, device,
              train: bool):
    g_psi.train(train)
    model.decoder_x.train(train and not cfg.train.freeze_decoder_in_p2)
    totals: dict[str, float] = {}
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for batch in tqdm(loader, desc="P2 " + ("train" if train else "val"), leave=False):
            ids = batch["x_input_ids"].to(device)
            mask = batch["x_attention"].to(device)
            n_levels = batch["n_levels"].to(device)

            with torch.no_grad():
                u, v0 = model.encode_latents(ids, mask)
            t, k = _sample_t_and_level(diffusion, n_levels, cfg, device)

            with torch.no_grad():
                v_t, v_hat_0 = diffusion.noisy_state_for_training(v0, u, t, seed=cfg.seed)

            with autocast_ctx(cfg, device):
                v_tilde = g_psi(t=t, v_t=v_t, v_hat_0=v_hat_0, u=u)

                loss = v_tilde.new_zeros(())
                if cfg.loss.lambda_proj > 0:
                    idx = batch["idx"]
                    target = rung_table[idx, k.cpu()].to(device=device, dtype=v_tilde.dtype)
                    lp = projection_loss(v_tilde, target)
                    loss = loss + cfg.loss.lambda_proj * lp
                    totals["loss_proj"] = totals.get("loss_proj", 0.0) + float(lp)

                if cfg.loss.lambda_ce > 0:
                    labels = labels_for_levels(
                        batch["level_input_ids"].to(device), k, loader.dataset.pad_id
                    )
                    lb = length_bucket_of((labels != -100).sum(1),
                                          cfg.model.n_length_buckets, cfg.data.max_seq_len)
                    slot_mask = torch.ones(v_tilde.shape[0], cfg.model.n_slots,
                                           device=device, dtype=torch.long)
                    lce, _ = model.decoder_x(v_tilde, slot_mask, labels, level=k, length_bucket=lb)
                    loss = loss + cfg.loss.lambda_ce * lce
                    totals["loss_ce"] = totals.get("loss_ce", 0.0) + float(lce)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                params = list(g_psi.parameters())
                if not cfg.train.freeze_decoder_in_p2:
                    params += list(model.decoder_x.parameters())
                if cfg.train.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, cfg.train.max_grad_norm)
                optimizer.step()
                if scheduler:
                    scheduler.step()
            totals["loss_total"] = totals.get("loss_total", 0.0) + float(loss)

    n = max(len(loader), 1)
    prefix = "train/" if train else "val/"
    return {prefix + k: v / n for k, v in totals.items()}


@torch.no_grad()
def sample_ladders(model, diffusion, g_psi, loader, cfg, device, tokenizer, n: int = 8):
    """Free-running ladders for qualitative inspection. Never teacher-forced."""
    from inference.recall_engine import RecallEngine
    engine = RecallEngine(model, diffusion, g_psi, tokenizer, cfg, device)
    batch = next(iter(loader))
    texts = batch["x_text"][:n]
    ladders = engine.ladder(texts, n_levels=int(batch["n_levels"][:n].max()),
                            example_ids=batch["uid"][:n])
    gold = [loader.dataset.level_texts(int(i)) for i in batch["idx"][:n]]
    return [
        {"source": s, "gold_ladder": g,
         "predicted_ladder": [{"budget": round(r.budget, 3), "t": r.t, "text": r.text}
                              for r in rungs]}
        for s, g, rungs in zip(texts, gold, ladders)
    ]


def main(argv=None):
    ap = argparse.ArgumentParser(description="P2: train G_psi (+ optional decoder fine-tune)")
    ap.add_argument("--p0-checkpoint", required=True)
    ap.add_argument("--p1-checkpoint", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--freeze-decoder", action="store_true")
    ap.add_argument("--lambda-proj", type=float, default=None)
    ap.add_argument("--lambda-ce", type=float, default=None)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args(argv)

    cfg = config_from_checkpoint(args.p0_checkpoint)
    assert_compatible(cfg, config_from_checkpoint(args.p1_checkpoint))

    ov = {}
    if args.name:            ov["name"] = args.name
    if args.epochs:          ov["train.epochs"] = args.epochs
    if args.freeze_decoder:  ov["train.freeze_decoder_in_p2"] = True
    if args.lambda_proj is not None: ov["loss.lambda_proj"] = args.lambda_proj
    if args.lambda_ce is not None:   ov["loss.lambda_ce"] = args.lambda_ce
    if args.wandb:           ov["wandb_enabled"] = True
    cfg = cfg.override(**ov) if ov else cfg

    device, run_dir, logger = prepare_run(cfg, "p2")
    tokenizer = build_tokenizer(cfg)
    cache = build_cache(cfg)
    loaders = build_loaders(cfg, tokenizer, cache, ("train", "val"))

    model = MemoryModel(cfg).to(device)
    model.load_state_dict(load_checkpoint(args.p0_checkpoint, device)["model_state_dict"],
                          strict=False)
    model.drop_gist_decoder()
    model.freeze_all()

    diffusion = LatentDiffusion(cfg, model.d_model).to(device)
    diffusion.load_state_dict(load_checkpoint(args.p1_checkpoint, device)["diffusion_state_dict"])
    for p in diffusion.parameters():
        p.requires_grad = False
    diffusion.eval()

    g_psi = SemanticProjectionModule(model.d_model, cfg.model.n_slots, cfg.model.u_dim,
                                     cfg.gpsi).to(device)

    params = list(g_psi.parameters())
    if not cfg.train.freeze_decoder_in_p2:
        for p in model.decoder_x.parameters():
            p.requires_grad = True
        params += list(model.decoder_x.parameters())
    print(f"P2 trainable params: {sum(p.numel() for p in params)/1e6:.1f}M "
          f"(decoder {'frozen' if cfg.train.freeze_decoder_in_p2 else 'fine-tuned'})")

    rung = rung_latent_table(model, loaders, cfg, device, cache, args.p0_checkpoint)

    optimizer = torch.optim.AdamW(params, lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = build_scheduler(optimizer, cfg, len(loaders["train"]) * cfg.train.epochs)

    best = float("inf")
    best_path = os.path.join(run_dir, "best_model.pt")
    eta = ETATracker(cfg.train.epochs)

    for epoch in range(cfg.train.epochs):
        eta.start_epoch()
        metrics = run_epoch(model, diffusion, g_psi, loaders["train"], rung["train"],
                            optimizer, scheduler, cfg, device, train=True)

        if (epoch + 1) % cfg.train.val_interval == 0 or epoch == cfg.train.epochs - 1:
            metrics.update(run_epoch(model, diffusion, g_psi, loaders["val"], rung["val"],
                                     None, None, cfg, device, train=False))
            score = metrics.get("val/loss_total", float("inf"))
            if score < best:
                best = score
                save_checkpoint(best_path, cfg, epoch, metrics,
                                g_psi=g_psi, decoder=model.decoder_x)
            with open(os.path.join(run_dir, f"ladders_epoch{epoch+1}.json"), "w",
                      encoding="utf-8") as f:
                json.dump(sample_ladders(model, diffusion, g_psi, loaders["val"], cfg,
                                         device, tokenizer), f, indent=2)

        elapsed, eta_str = eta.end_epoch()
        metrics.update({"epoch": epoch + 1, "epoch_time_s": elapsed, "eta": eta_str})
        logger.log(metrics, step=epoch + 1)
        print(f"epoch {epoch+1}/{cfg.train.epochs} "
              f"loss={metrics.get('train/loss_total', 0):.4f} eta={eta_str}")

    logger.finish()
    print(f"P2 done. best val loss {best:.4f}. checkpoint: {best_path}")
    return best_path


if __name__ == "__main__":
    main()
