"""
W&B sweep entrypoint for Phase 3 (P2): train G_psi + fine-tune decoder.

This script is intended to be used as the `program` in a W&B sweep config.
Each invocation runs a single trial and reads tunable hyperparameters from
`wandb.config`.

Example:
    wandb sweep scripts/training/wandb_sweep_phase2.yaml
    wandb agent <entity>/<project>/<sweep_id>
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers import T5Tokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataloader.dataloader_augmentated import MSRAugmentedDataset
from models.denoiser_module.denoiser import NoiseSchedule, forward_diffusion, one_step_estimate
from train_phase2 import (
    evaluate_best_epoch,
    load_denoiser,
)
from train_phase2_config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    L_SLOTS,
    NOISE_SCHEDULE,
    T_DIFFUSION,
    U_DIM,
    VAL_INTERVAL,
    WEIGHT_DECAY,
    XT_BUCKET_SIZE,
)
from utils.training_utils import ETATracker, build_p0_model, log_sample_outputs, save_checkpoint, select_xt_labels


def _get_run_hparams(wandb_config):
    """Return trial hyperparameters with config defaults."""
    return {
        "epochs": int(wandb_config.get("epochs", EPOCHS)),
        "batch_size": int(wandb_config.get("batch_size", BATCH_SIZE)),
        "learning_rate": float(wandb_config.get("learning_rate", LEARNING_RATE)),
        "weight_decay": float(wandb_config.get("weight_decay", WEIGHT_DECAY)),
        "val_interval": int(wandb_config.get("val_interval", VAL_INTERVAL)),
        "optimizer": str(wandb_config.get("optimizer", "adamw")).lower(),
        "scheduler": str(wandb_config.get("scheduler", "none")).lower(),
        "warmup_ratio": float(wandb_config.get("warmup_ratio", 0.1)),
        "max_grad_norm": float(wandb_config.get("max_grad_norm", 1.0)),
        "xt_bucket_size": int(wandb_config.get("xt_bucket_size", XT_BUCKET_SIZE)),
    }


def _resolve_label_source(args, wandb_config):
    """Allow sweep config to override CLI label_source."""
    value = str(wandb_config.get("label_source", args.label_source)).lower()
    if value not in {"xt", "x"}:
        raise ValueError(f"Invalid label_source={value}; expected 'xt' or 'x'")
    return value


def _select_labels(batch, t, device, label_source, xt_bucket_size):
    if label_source == "x":
        labels = batch["x_input_ids"].to(device)
        label_index = torch.full((t.shape[0],), -1, dtype=torch.long, device=device)
        return labels, label_index
    return select_xt_labels(batch, t, device, xt_bucket_size)


def _build_optimizer(params, cfg):
    """Build optimizer from sweep config."""
    if cfg["optimizer"] == "adam":
        return torch.optim.Adam(
            params,
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
        )
    if cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(
            params,
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
        )
    raise ValueError(f"Unsupported optimizer: {cfg['optimizer']}")


def _build_scheduler(optimizer, cfg, total_steps):
    """Build optional LR scheduler from sweep config."""
    scheduler_name = cfg["scheduler"]
    if scheduler_name == "none":
        return None

    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    warmup_steps = max(0, min(warmup_steps, total_steps))

    if scheduler_name == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    if scheduler_name == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def train_epoch_sweep(
    p0_model,
    denoiser,
    noise_schedule,
    dataloader,
    optimizer,
    scheduler,
    device,
    label_source,
    xt_bucket_size,
    max_grad_norm,
):
    """Train one epoch with configurable gradient clipping and scheduler."""
    p0_model.g_psi.train()
    p0_model.decoder_x.train()

    total_loss = 0.0

    for batch in dataloader:
        optimizer.zero_grad()

        batch_size = batch["x_input_ids"].shape[0]

        with torch.no_grad():
            u, v0 = p0_model.encode_latents(batch)
        u = u.detach()
        v0 = v0.detach()

        t = torch.randint(1, T_DIFFUSION + 1, (batch_size,), device=device)
        vt, _ = forward_diffusion(v0, t, noise_schedule)
        with torch.no_grad():
            eps_hat = denoiser(vt, t, u)
            v_hat_0 = one_step_estimate(vt, eps_hat, t, noise_schedule)

        labels_noisy, _ = _select_labels(
            batch=batch,
            t=t,
            device=device,
            label_source=label_source,
            xt_bucket_size=xt_bucket_size,
        )
        vt_tilde = p0_model.g_psi(v_hat_0=v_hat_0, t=t, v_t=vt, u=u)
        slot_mask = torch.ones(batch_size, L_SLOTS, device=device)
        loss_recon, _ = p0_model.decoder_x(vt_tilde, slot_mask, labels_noisy)

        loss = loss_recon
        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                list(p0_model.g_psi.parameters()) + list(p0_model.decoder_x.parameters()),
                max_norm=max_grad_norm,
            )

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


@torch.no_grad()
def validate_epoch_sweep(
    p0_model,
    denoiser,
    noise_schedule,
    dataloader,
    device,
    label_source,
    xt_bucket_size,
):
    """Validation loop that mirrors training label selection for sweeps."""
    p0_model.g_psi.eval()
    p0_model.decoder_x.eval()

    total_loss = 0.0
    sample_outputs = []

    for batch in dataloader:
        batch_size = batch["x_input_ids"].shape[0]

        u, v0 = p0_model.encode_latents(batch)

        t = torch.randint(1, T_DIFFUSION + 1, (batch_size,), device=device)
        vt, _ = forward_diffusion(v0, t, noise_schedule)
        eps_hat = denoiser(vt, t, u)
        v_hat_0 = one_step_estimate(vt, eps_hat, t, noise_schedule)

        labels_noisy, label_index = _select_labels(
            batch=batch,
            t=t,
            device=device,
            label_source=label_source,
            xt_bucket_size=xt_bucket_size,
        )

        vt_tilde = p0_model.g_psi(v_hat_0=v_hat_0, t=t, v_t=vt, u=u)
        slot_mask = torch.ones(batch_size, L_SLOTS, device=device)
        loss_recon, logits_noisy = p0_model.decoder_x(vt_tilde, slot_mask, labels_noisy)

        total_loss += loss_recon.item()
        sample_outputs.append((batch, logits_noisy, t, label_index))

    return total_loss / max(1, len(dataloader)), sample_outputs


def _resolve_run_dirs(output_dir_base, checkpoint_dir_base, run_id):
    """Create per-run directories to avoid sweep trial collisions."""
    output_dir = os.path.join(output_dir_base, run_id)
    checkpoint_dir = os.path.join(checkpoint_dir_base, run_id)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return output_dir, checkpoint_dir


def _log_sample_table(wandb, sample_outputs, tokenizer, label_source, step):
    if not sample_outputs:
        return

    batch0, logits_n, t0, xi0 = sample_outputs[0]
    pred_n = tokenizer.batch_decode(torch.argmax(logits_n, dim=-1), skip_special_tokens=True)
    orig = tokenizer.batch_decode(batch0["x_input_ids"], skip_special_tokens=True)

    if label_source == "x":
        bsz = batch0["x_input_ids"].shape[0]
        xt_tgt = tokenizer.batch_decode(batch0["x_input_ids"], skip_special_tokens=True)
    else:
        bsz = batch0["xt_input_ids"].shape[0]
        xt_tgt = tokenizer.batch_decode(
            batch0["xt_input_ids"][torch.arange(bsz), xi0.cpu()],
            skip_special_tokens=True,
        )

    table = wandb.Table(columns=["original", "xt_target", "recon_noisy", "t", "xt_idx"])
    for o, xt, pn, ti, xi in zip(orig, xt_tgt, pred_n, t0, xi0):
        table.add_data(o, xt, pn, ti.item(), xi.item())

    wandb.log({"val/samples": table}, step=step)


def run_trial(args):
    import wandb

    wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    cfg = _get_run_hparams(wandb.config)
    label_source = _resolve_label_source(args, wandb.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    print("Loading P0 checkpoint...", flush=True)
    p0_model = build_p0_model(device, L_SLOTS, U_DIM)
    p0_ckpt = torch.load(args.p0_checkpoint, map_location=device)
    p0_model.load_state_dict(p0_ckpt["model_state_dict"])
    print(f"  loaded {args.p0_checkpoint} (epoch {p0_ckpt.get('epoch', '?')})", flush=True)

    print("Loading denoiser checkpoint...", flush=True)
    denoiser = load_denoiser(args.denoiser_checkpoint, device)

    for param in p0_model.parameters():
        param.requires_grad = False
    for param in p0_model.decoder_x.parameters():
        param.requires_grad = True
    for param in p0_model.g_psi.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in p0_model.g_psi.parameters()) + sum(
        p.numel() for p in p0_model.decoder_x.parameters()
    )

    run_id = wandb.run.id
    output_dir, checkpoint_dir = _resolve_run_dirs(args.output_dir, args.checkpoint_dir, run_id)

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    train_dataset = MSRAugmentedDataset(os.path.join(args.data_dir, "train.json"), tokenizer)
    val_dataset = MSRAugmentedDataset(os.path.join(args.data_dir, "validate.json"), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False)

    trainable_param_list = list(p0_model.g_psi.parameters()) + list(p0_model.decoder_x.parameters())
    optimizer = _build_optimizer(trainable_param_list, cfg)

    total_steps = max(1, len(train_loader) * cfg["epochs"])
    scheduler = _build_scheduler(optimizer, cfg, total_steps)

    noise_schedule = NoiseSchedule(T=T_DIFFUSION, schedule_type=NOISE_SCHEDULE)

    wandb.config.update(
        {
            "phase": "P2",
            "trainable_params": trainable_params,
            "t_diffusion": T_DIFFUSION,
            "noise_schedule": NOISE_SCHEDULE,
            "label_source": label_source,
            "output_dir": output_dir,
            "checkpoint_dir": checkpoint_dir,
        },
        allow_val_change=True,
    )

    print("-" * 60, flush=True)
    print("STARTING PHASE 2 SWEEP TRIAL", flush=True)
    print(
        f"epochs={cfg['epochs']} batch={cfg['batch_size']} lr={cfg['learning_rate']} wd={cfg['weight_decay']}",
        flush=True,
    )
    print(
        f"optimizer={cfg['optimizer']} scheduler={cfg['scheduler']} warmup_ratio={cfg['warmup_ratio']} max_grad_norm={cfg['max_grad_norm']}",
        flush=True,
    )
    print(f"label_source={label_source} xt_bucket_size={cfg['xt_bucket_size']}", flush=True)
    print("-" * 60, flush=True)

    best_val_loss = float("inf")
    eta_tracker = ETATracker(total_epochs=cfg["epochs"])

    for epoch in range(cfg["epochs"]):
        eta_tracker.start_epoch()

        train_loss = train_epoch_sweep(
            p0_model=p0_model,
            denoiser=denoiser,
            noise_schedule=noise_schedule,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            label_source=label_source,
            xt_bucket_size=cfg["xt_bucket_size"],
            max_grad_norm=cfg["max_grad_norm"],
        )

        epoch_elapsed, eta_seconds, eta_str = eta_tracker.end_epoch()
        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log(
            {
                "train/loss": train_loss,
                "train/lr": current_lr,
                "time/epoch_s": epoch_elapsed,
                "time/eta_s": eta_seconds,
            },
            step=epoch + 1,
        )

        if (epoch + 1) % cfg["val_interval"] == 0 or (epoch + 1) == cfg["epochs"]:
            val_loss, sample_outputs = validate_epoch_sweep(
                p0_model=p0_model,
                denoiser=denoiser,
                noise_schedule=noise_schedule,
                dataloader=val_loader,
                device=device,
                label_source=label_source,
                xt_bucket_size=cfg["xt_bucket_size"],
            )

            print(
                f"Epoch {epoch + 1} | Train: {train_loss:.4f} | Val: {val_loss:.4f}",
                flush=True,
            )

            wandb.log({"val/loss": val_loss}, step=epoch + 1)

            if sample_outputs:
                log_sample_outputs(
                    sample_outputs,
                    tokenizer,
                    epoch,
                    output_dir,
                    label_source=label_source,
                )
                _log_sample_table(
                    wandb=wandb,
                    sample_outputs=sample_outputs,
                    tokenizer=tokenizer,
                    label_source=label_source,
                    step=epoch + 1,
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    p0_model.g_psi,
                    p0_model.decoder_x,
                    optimizer,
                    epoch + 1,
                    train_loss,
                    val_loss,
                    os.path.join(checkpoint_dir, "best_model.pt"),
                )

                best_eval_metrics = {}
                if sample_outputs:
                    best_eval_metrics = evaluate_best_epoch(sample_outputs, tokenizer)
                    if best_eval_metrics:
                        metrics_path = os.path.join(output_dir, f"epoch_{epoch + 1}_best_eval_metrics.json")
                        with open(metrics_path, "w", encoding="utf-8") as f:
                            json.dump(best_eval_metrics, f, indent=2)
                        print(f"Saved best-epoch eval metrics to {metrics_path}", flush=True)

                wandb.run.summary["best_val_loss"] = best_val_loss
                wandb.run.summary["best_epoch"] = epoch + 1
                for k, v in best_eval_metrics.items():
                    wandb.run.summary[f"best_{k}"] = v
        else:
            print(f"Epoch {epoch + 1} | Train: {train_loss:.4f} | ETA: {eta_str}", flush=True)

    save_checkpoint(
        p0_model.g_psi,
        p0_model.decoder_x,
        optimizer,
        cfg["epochs"],
        train_loss,
        best_val_loss,
        os.path.join(checkpoint_dir, "final_model.pt"),
    )

    print(f"Training complete. Best val loss: {best_val_loss:.4f}", flush=True)
    wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="W&B sweep trial runner for Phase 2 training")
    parser.add_argument(
        "--p0-checkpoint",
        type=str,
        required=True,
        help="Path to Phase 0 checkpoint with model_state_dict",
    )
    parser.add_argument(
        "--denoiser-checkpoint",
        type=str,
        required=True,
        help="Path to denoiser checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(ROOT, "data", "final"),
        help="Directory containing train.json and validate.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/p2/sweeps",
        help="Base directory to write per-run outputs",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints/p2/sweeps",
        help="Base directory to write per-run checkpoints",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="diffusion-as-memory",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional run name prefix (W&B may override when sweeping)",
    )
    parser.add_argument(
        "--label-source",
        type=str,
        choices=["xt", "x"],
        default="xt",
        help="Decoder labels: 'xt' (recommended) or 'x' ablation",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_trial(args)


if __name__ == "__main__":
    main()
