"""
W&B sweep entrypoint for Phase 0 (P0): train ForgettingModel on augmented data.

This script mirrors scripts/training/training_dl_augmented.py but exposes key
hyperparameters to wandb sweeps:
- num_slots (SlotPooling)
- u_dim (UHead output_dim + SemanticProjection u_dim)
- epochs
- learning_rate
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataloader.dataloader_augmentated import MSRAugmentedDataset
from models.decoder_prep.decoder_x import DecoderX
from models.encoder_prep.encoder import TextEncoder
from models.forgetting_model import ForgettingModel
from models.g_psi_module.g_psi_config import G_psi_config
from models.g_psi_module.semantic_projection import SemanticProjectionModule
from models.slot_pooling_prep.slot_pooling import SlotPooling
from models.uv_heads_prep.u_head import UHead
from models.uv_heads_prep.v_head import VHead
from utils.training_utils import ETATracker


def _resolve_run_dirs(output_dir_base, checkpoint_dir_base, run_id):
    """Create per-run directories to avoid sweep trial collisions."""
    output_dir = os.path.join(output_dir_base, run_id)
    checkpoint_dir = os.path.join(checkpoint_dir_base, run_id)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return output_dir, checkpoint_dir


def _get_run_hparams(wandb_config, args):
    """Return trial hyperparameters with config/CLI defaults."""
    return {
        "num_slots": int(wandb_config.get("num_slots", args.num_slots)),
        "u_dim": int(wandb_config.get("u_dim", args.u_dim)),
        "epochs": int(wandb_config.get("epochs", args.epochs)),
        "batch_size": int(wandb_config.get("batch_size", args.batch_size)),
        "learning_rate": float(wandb_config.get("learning_rate", args.learning_rate)),
        "val_interval": int(wandb_config.get("val_interval", args.val_interval)),
    }


def _make_gpsi_config(u_dim):
    """Build a lightweight config object for SemanticProjectionModule."""

    class SweepGpsiConfig:
        pass

    cfg = SweepGpsiConfig()
    cfg.d = G_psi_config.d
    cfg.u_dim = u_dim
    cfg.n_blocks = G_psi_config.n_blocks
    cfg.d_ff = G_psi_config.d_ff
    cfg.use_attn = G_psi_config.use_attn
    cfg.n_heads = G_psi_config.n_heads
    return cfg


def build_p0_model(device, num_slots, u_dim):
    """Build ForgettingModel with sweepable num_slots and u_dim."""
    encoder = TextEncoder()
    slot_pool = SlotPooling(hidden_dim=encoder.hidden_dim_size, num_slots=num_slots)
    u_head = UHead(hidden_dim=encoder.hidden_dim_size, output_dim=u_dim)
    v_head = VHead(hidden_dim=encoder.hidden_dim_size)
    decoder_x = DecoderX()
    g_psi = SemanticProjectionModule(config=_make_gpsi_config(u_dim), no_use_vt=True)

    model = ForgettingModel(
        encoder=encoder,
        slot_pooling=slot_pool,
        u_head=u_head,
        v_head=v_head,
        decoder_x=decoder_x,
        g_psi=g_psi,
    )
    model.to(device)
    return model


def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0.0
    total_loss_nce = 0.0
    total_loss_x = 0.0

    for batch in dataloader:
        optimizer.zero_grad()
        loss, _, loss_nce, loss_x = model(batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_loss_nce += loss_nce.item()
        total_loss_x += loss_x.item()

    denom = max(1, len(dataloader))
    return total_loss / denom, total_loss_nce / denom, total_loss_x / denom


@torch.no_grad()
def validate_epoch(model, dataloader):
    model.eval()
    total_loss = 0.0
    total_loss_nce = 0.0
    total_loss_x = 0.0
    sample_outputs = []

    for batch in dataloader:
        loss, logits_x, loss_nce, loss_x = model(batch)
        total_loss += loss.item()
        total_loss_nce += loss_nce.item()
        total_loss_x += loss_x.item()
        sample_outputs.append((batch, logits_x))

    denom = max(1, len(dataloader))
    return total_loss / denom, total_loss_nce / denom, total_loss_x / denom, sample_outputs


def log_sample_outputs(sample_outputs, tokenizer, epoch, output_dir):
    """Decode and save validation predictions."""
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for batch, logits_x in sample_outputs:
        pred_ids_x = torch.argmax(logits_x, dim=-1)
        decoded_x = tokenizer.batch_decode(pred_ids_x, skip_special_tokens=True)
        original_x = tokenizer.batch_decode(batch["x_input_ids"], skip_special_tokens=True)

        for original, pred in zip(original_x, decoded_x):
            results.append({
                "original_x": original,
                "v0": pred,
            })

    out_path = os.path.join(output_dir, f"epoch_{epoch + 1}_samples.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, path):
    """Save a P0 checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        path,
    )


def _log_sample_table(wandb, sample_outputs, tokenizer, step):
    if not sample_outputs:
        return

    batch0, logits_x0 = sample_outputs[0]
    pred_ids_x = torch.argmax(logits_x0, dim=-1)
    dec_x = tokenizer.batch_decode(pred_ids_x, skip_special_tokens=True)
    orig_x = tokenizer.batch_decode(batch0["x_input_ids"], skip_special_tokens=True)

    table = wandb.Table(columns=["original_x", "v0_pred"])
    for ox, px in zip(orig_x, dec_x):
        table.add_data(ox, px)

    wandb.log({"val/samples": table}, step=step)


def run_trial(args):
    import wandb

    wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    cfg = _get_run_hparams(wandb.config, args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    run_id = wandb.run.id
    output_dir, checkpoint_dir = _resolve_run_dirs(args.output_dir, args.checkpoint_dir, run_id)

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    train_dataset = MSRAugmentedDataset(os.path.join(args.data_dir, "train.json"), tokenizer)
    val_dataset = MSRAugmentedDataset(os.path.join(args.data_dir, "validate.json"), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False)

    model = build_p0_model(device=device, num_slots=cfg["num_slots"], u_dim=cfg["u_dim"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    wandb.config.update(
        {
            "phase": "P0",
            "num_slots": cfg["num_slots"],
            "u_dim": cfg["u_dim"],
            "epochs": cfg["epochs"],
            "batch_size": cfg["batch_size"],
            "learning_rate": cfg["learning_rate"],
            "val_interval": cfg["val_interval"],
            "output_dir": output_dir,
            "checkpoint_dir": checkpoint_dir,
        },
        allow_val_change=True,
    )

    print("-" * 60, flush=True)
    print("STARTING P0 SWEEP TRIAL", flush=True)
    print(
        f"num_slots={cfg['num_slots']} u_dim={cfg['u_dim']} epochs={cfg['epochs']} ",
        f"batch={cfg['batch_size']} lr={cfg['learning_rate']}",
        flush=True,
    )
    print("-" * 60, flush=True)

    best_val_loss = float("inf")
    eta_tracker = ETATracker(total_epochs=cfg["epochs"])

    for epoch in range(cfg["epochs"]):
        eta_tracker.start_epoch()
        train_loss, train_loss_nce, train_loss_x = train_epoch(model, train_loader, optimizer)
        epoch_elapsed, eta_seconds, eta_str = eta_tracker.end_epoch()

        wandb.log(
            {
                "train/loss": train_loss,
                "train/loss_nce": train_loss_nce,
                "train/loss_x": train_loss_x,
                "time/epoch_s": epoch_elapsed,
                "time/eta_s": eta_seconds,
            },
            step=epoch + 1,
        )

        if (epoch + 1) % cfg["val_interval"] == 0 or (epoch + 1) == cfg["epochs"]:
            val_loss, val_loss_nce, val_loss_x, sample_outputs = validate_epoch(model, val_loader)

            print(
                f"Epoch {epoch + 1} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | ETA: {eta_str}",
                flush=True,
            )

            wandb.log(
                {
                    "val/loss": val_loss,
                    "val/loss_nce": val_loss_nce,
                    "val/loss_x": val_loss_x,
                },
                step=epoch + 1,
            )

            if sample_outputs:
                log_sample_outputs(sample_outputs, tokenizer, epoch, output_dir)
                _log_sample_table(wandb, sample_outputs, tokenizer, epoch + 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model,
                    optimizer,
                    epoch + 1,
                    train_loss,
                    val_loss,
                    os.path.join(checkpoint_dir, "best_model.pt"),
                )
                wandb.run.summary["best_val_loss"] = best_val_loss
                wandb.run.summary["best_epoch"] = epoch + 1
        else:
            print(f"Epoch {epoch + 1} | Train: {train_loss:.4f} | ETA: {eta_str}", flush=True)

    save_checkpoint(
        model,
        optimizer,
        cfg["epochs"],
        train_loss,
        best_val_loss,
        os.path.join(checkpoint_dir, "final_model.pt"),
    )

    print(f"Training complete. Best val loss: {best_val_loss:.4f}", flush=True)
    wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="W&B sweep trial runner for P0 augmented training")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(ROOT, "data", "final"),
        help="Directory containing train.json and validate.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/p0/sweeps",
        help="Base directory to write per-run outputs",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints/p0/sweeps",
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
        help="Optional run name (W&B sweep may override)",
    )
    parser.add_argument("--num-slots", type=int, default=8, help="Default SlotPooling num_slots")
    parser.add_argument("--u-dim", type=int, default=128, help="Default UHead output_dim and g_psi u_dim")
    parser.add_argument("--epochs", type=int, default=200, help="Default epochs")
    parser.add_argument("--batch-size", type=int, default=10, help="Default batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Default learning rate")
    parser.add_argument("--val-interval", type=int, default=10, help="Validate every N epochs")
    return parser.parse_args()


def main():
    args = parse_args()
    run_trial(args)


if __name__ == "__main__":
    main()
