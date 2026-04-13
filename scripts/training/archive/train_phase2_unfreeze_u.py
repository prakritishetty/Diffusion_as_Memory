"""
Phase 3 (P2) training: Train G_psi (Semantic Projection Module) + fine-tune Decoder.

Freezes: E (encoder), P (slot pooling), U (u-head), V (v-head), N (denoiser)
Trains:  G_psi (new semantic projection module), D (decoder_x, fine-tuned)
Label selection from xt (progressive degradations):
  index = min(t // bucket_size, len(xt) - 1)
  e.g. T=1000, bucket=100: t=50→xt[0], t=150→xt[1], t=950→xt[9] or last

Usage:
    python train_phase2.py \\
        --p0-checkpoint ./checkpoints/p0/2loss/best_model.pt \\
        --wandb-run-name p2-gpsi-run
"""

import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
import json
import os
import argparse
import sys
from typing import Dict, List
from bert_score import score as bert_score_fn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.training_utils import build_p0_model, select_xt_labels, log_sample_outputs, save_checkpoint, ETATracker
from tqdm import tqdm
from dataloader.dataloader_augmentated import MSRAugmentedDataset
from models.encoder_prep.encoder import TextEncoder
from models.slot_pooling_prep.slot_pooling import SlotPooling
from models.uv_heads_prep.u_head import UHead
from models.uv_heads_prep.v_head import VHead
from models.decoder_prep.decoder_x import DecoderX
from models.forgetting_model import ForgettingModel
from models.g_psi_module.semantic_projection import SemanticProjectionModule
from models.denoiser_module.config import DenoiserConfig
from models.denoiser_module.denoiser import Denoiser, NoiseSchedule, forward_diffusion, one_step_estimate
from models.g_psi_module.g_psi_config import G_psi_config
from evaluation.run_uni_eval import evaluate_factual_consistency_return
from train_phase2_config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    VAL_INTERVAL,
    GPSI_N_BLOCKS,
    GPSI_N_HEADS,
    GPSI_D_FF,
    GPSI_DROPOUT,
    T_DIFFUSION,
    NOISE_SCHEDULE,
    XT_BUCKET_SIZE,
    L_SLOTS,
    D_MODEL,
    U_DIM,
)


def _flatten_val_predictions(sample_outputs, tokenizer):
    """Convert validation outputs into plain text lists for metric eval."""
    src_list: List[str] = []
    pred_list: List[str] = []

    for batch, logits_noisy, _, _ in sample_outputs:
        pred_noisy = tokenizer.batch_decode(
            torch.argmax(logits_noisy, dim=-1), skip_special_tokens=True
        )
        original = tokenizer.batch_decode(
            batch["x_input_ids"], skip_special_tokens=True
        )
        src_list.extend(original)
        pred_list.extend(pred_noisy)

    return src_list, pred_list


def evaluate_best_epoch(sample_outputs, tokenizer) -> Dict[str, float]:
    """Compute BERTScore and UniEval(factual consistency) on validation predictions."""
    src_list, pred_list = _flatten_val_predictions(sample_outputs, tokenizer)
    metrics: Dict[str, float] = {}

    # BERTScore
    model_type_used = "roberta-base"
    p, r, f1 = bert_score_fn(
    pred_list,
    src_list,
    lang="en",
    verbose=False,
    model_type=model_type_used,
    use_fast_tokenizer=True,
    )
    metrics["bertscore/precision_mean"] = p.mean().item()
    metrics["bertscore/recall_mean"] = r.mean().item()
    metrics["bertscore/f1_mean"] = f1.mean().item()
    metrics["bertscore/model_used"] = model_type_used

    # UniEval (factual consistency)
    try:
        unieval_results = evaluate_factual_consistency_return(
            src_list=src_list,
            output_list=pred_list
        )
        metrics["unieval/consistency_mean"] = unieval_results["mean_consistency"]
    except Exception as exc:
        print(f"UniEval failed: {exc}", flush=True)

    return metrics


def load_denoiser(denoiser_checkpoint, device):
    """Load a trained denoiser checkpoint and freeze it for Phase 3."""
    checkpoint = torch.load(denoiser_checkpoint, map_location=device)

    config = DenoiserConfig()
    if "config" in checkpoint:
        saved_cfg = checkpoint["config"]
        config.L = saved_cfg.get("L", config.L)
        config.d = saved_cfg.get("d", config.d)
        config.u_dim = saved_cfg.get("u_dim", config.u_dim)
        config.T = saved_cfg.get("T", config.T)
        config.N_blocks = saved_cfg.get("N_blocks", config.N_blocks)
        config.n_heads = saved_cfg.get("n_heads", config.n_heads)
        config.d_ff = saved_cfg.get("d_ff", config.d_ff)
        config.schedule = saved_cfg.get("schedule", config.schedule)

    denoiser = Denoiser(config).to(device)
    denoiser.load_state_dict(checkpoint["model_state_dict"])

    denoiser.eval()
    for param in denoiser.parameters():
        param.requires_grad = False

    print(f"  Loaded denoiser from {denoiser_checkpoint} (epoch {checkpoint.get('epoch', '?')})")
    return denoiser


def unfreeze_half_module_params(module) -> Dict[str, int]:
    """Unfreeze approximately half of module parameters by element count."""
    params = list(module.parameters())
    total = sum(p.numel() for p in params)
    target = total // 2

    unfrozen = 0
    for p in params:
        if unfrozen < target:
            p.requires_grad = True
            unfrozen += p.numel()
        else:
            p.requires_grad = False

    return {
        "total": total,
        "target_half": target,
        "unfrozen": unfrozen,
    }


def train_epoch(
    p0_model,
    denoiser,
    noise_schedule,
    dataloader,
    optimizer,
    device,
    train_u_head=False,
):
    """Run one training epoch. Returns (total_loss) averages."""
    p0_model.g_psi.train()
    p0_model.decoder_x.train()

    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()

        batch_size = batch["x_input_ids"].shape[0]

        # Keep gradient flow through u_head only when it is partially unfrozen.
        if train_u_head:
            u, v0 = p0_model.encode_latents(batch)
        else:
            with torch.no_grad():
                u, v0 = p0_model.encode_latents(batch)
            u = u.detach()    # [B, 128]
        v0 = v0.detach()  # [B, 8, 512]


        # Noisy reconstruction loss (labels = degraded xt)
        t = torch.randint(1, T_DIFFUSION + 1, (batch_size,), device=device)
        vt, _ = forward_diffusion(v0, t, noise_schedule)
        with torch.no_grad():
            eps_hat = denoiser(vt, t, u)
            v_hat_0 = one_step_estimate(vt, eps_hat, t, noise_schedule)

        labels_noisy, _ = select_xt_labels(batch, t, device, XT_BUCKET_SIZE)   # [B, seq_len]
        vt_tilde = p0_model.g_psi(v_hat_0=v_hat_0, t=t, v_t=vt, u=u)    # [B, L, d]
        slot_mask = torch.ones(batch_size, L_SLOTS, device=device)
        loss_recon, logits = p0_model.decoder_x(vt_tilde, slot_mask, labels_noisy)


        # Total loss
        loss = loss_recon
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in p0_model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        optimizer.step()

        total_loss += loss.item()

    n = len(dataloader)
    return total_loss / n


@torch.no_grad()
def validate_epoch(p0_model, denoiser, noise_schedule, dataloader, device):
    """Run one validation epoch. Returns (total_loss, sample_outputs)."""
    p0_model.g_psi.eval()
    p0_model.decoder_x.eval()

    total_loss = 0
    sample_outputs = []

    for batch in tqdm(dataloader, desc="Validating"):
        batch_size = batch["x_input_ids"].shape[0]

        u, v0 = p0_model.encode_latents(batch)


        # Noisy (labels = degraded xt)
        t = torch.randint(1, T_DIFFUSION + 1, (batch_size,), device=device)
        vt, _ = forward_diffusion(v0, t, noise_schedule)
        eps_hat = denoiser(vt, t, u)
        v_hat_0 = one_step_estimate(vt, eps_hat, t, noise_schedule)
        labels_noisy, xt_index = select_xt_labels(batch, t, device, XT_BUCKET_SIZE)   # [B, seq_len]
        # Noisy reconstruction
        vt_tilde = p0_model.g_psi(v_hat_0=v_hat_0, t=t, v_t=vt, u=u)
        slot_mask = torch.ones(batch_size, L_SLOTS, device=device)
        loss_recon, logits_noisy = p0_model.decoder_x(vt_tilde, slot_mask, labels_noisy)

        loss = loss_recon
        total_loss += loss.item()

        sample_outputs.append((batch, logits_noisy, t, xt_index))

    n = len(dataloader)
    return total_loss / n, sample_outputs


def main():
    print(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Phase 3 (P2): Train G_psi + fine-tune Decoder"
    )
    parser.add_argument(
        "--p0-checkpoint", type=str, required=True,
        help="Path to Phase 0 best_model.pt checkpoint",
    )
    parser.add_argument(
        "--denoiser-checkpoint", type=str, required=True,
        help="Path to Phase 1 denoiser best_model.pt checkpoint",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./output/p2/temp",
        help="Directory to write sample output JSONs",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="./checkpoints/p2/temp",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(ROOT, "data", "final"),
        help="Directory containing train.json and validate.json",
    )
    parser.add_argument(
        "--wandb-project", type=str, default="diffusion-as-memory",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-run-name", type=str, required=True,
        help="W&B run name",
    )
    parser.add_argument(
        "--wandb-off", action="store_true",
        help="Disable W&B logging entirely",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    # Load P0 model
    print("\nLoading P0 checkpoint...")
    p0_model = build_p0_model(device, L_SLOTS, U_DIM)
    checkpoint = torch.load(args.p0_checkpoint, map_location=device)
    p0_model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded from {args.p0_checkpoint} (epoch {checkpoint.get('epoch', '?')})")

    print("\nLoading P1 denoiser checkpoint...")
    denoiser = load_denoiser(args.denoiser_checkpoint, device)

    # Freeze everything in P0 model
    for param in p0_model.parameters():
        param.requires_grad = False

    # Unfreeze decoder for fine-tuning
    for param in p0_model.decoder_x.parameters():
        param.requires_grad = True
    
    # Unfreeze G_psi 
    for param in p0_model.g_psi.parameters():
        param.requires_grad = True

    # Unfreeze half of u_head parameters for P2 training.
    u_head_unfreeze_info = unfreeze_half_module_params(p0_model.u_head)
    train_u_head = any(p.requires_grad for p in p0_model.u_head.parameters())
    print(
        "  u_head partial unfreeze: "
        f"{u_head_unfreeze_info['unfrozen']}/{u_head_unfreeze_info['total']} params",
        flush=True,
    )

    trainable_params = sum(
        p.numel() for p in p0_model.parameters() if p.requires_grad
    )

    # Noise schedule (alphas)
    noise_schedule = NoiseSchedule(T=T_DIFFUSION, schedule_type=NOISE_SCHEDULE)

    # Data 
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    train_dataset = MSRAugmentedDataset(
        os.path.join(args.data_dir, "train.json"), tokenizer
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = MSRAugmentedDataset(
        os.path.join(args.data_dir, "validate.json"), tokenizer
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Optimizer (decoder_x + G_psi + unfrozen half of u_head)
    optimizer = torch.optim.Adam(
        [p for p in p0_model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
    )

    output_dir = args.output_dir
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    use_wandb = not args.wandb_off
    if use_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "phase": "P2",
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "gpsi_n_blocks": GPSI_N_BLOCKS,
                "gpsi_n_heads": GPSI_N_HEADS,
                "gpsi_d_ff": GPSI_D_FF,
                "T_diffusion": T_DIFFUSION,
                "noise_schedule": NOISE_SCHEDULE,
                "trainable_params": trainable_params,
            },
        )

    print(f"\n{'-'*60}")
    print("STARTING PHASE 3 (P2) TRAINING")
    print(f"  Epochs={EPOCHS}  Batch={BATCH_SIZE}  LR={LEARNING_RATE}")
    print(f"  G_psi blocks={GPSI_N_BLOCKS}")
    print(f"  T={T_DIFFUSION}  Schedule={NOISE_SCHEDULE}")
    print(f"{'-'*60}\n")

    best_val_loss = float("inf")
    eta_tracker = ETATracker(total_epochs=EPOCHS)

    for epoch in range(EPOCHS):
        eta_tracker.start_epoch()

        train_loss = train_epoch(
            p0_model,
            denoiser,
            noise_schedule,
            train_loader,
            optimizer,
            device,
            train_u_head=train_u_head,
        )

        epoch_elapsed, eta_seconds, eta_str = eta_tracker.end_epoch()

        # Log train metrics
        if use_wandb:
            wandb.log(
                {
                    "train/loss": train_loss,
                    **eta_tracker.wandb_metrics(epoch_elapsed, eta_seconds),
                },
                step=epoch + 1,
            )

        # Validate every VAL_INTERVAL epochs and on the last epoch
        if (epoch + 1) % VAL_INTERVAL == 0 or (epoch + 1) == EPOCHS:
            val_loss, sample_outputs = validate_epoch(
                p0_model, denoiser, noise_schedule, val_loader, device,
            )

            print(
                f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f} ",
                flush=True,
            )
            print("-" * 30, flush=True)

            if use_wandb:
                wandb.log(
                    {
                        "val/loss": val_loss,
                    },
                    step=epoch + 1,
                )

            if sample_outputs:
                log_sample_outputs(sample_outputs, tokenizer, epoch, output_dir)

                # Log a sample table to W&B (first batch only)
                if use_wandb:
                    batch0, logits_n, t0, xi0 = sample_outputs[0]
                    pred_n = tokenizer.batch_decode(
                        torch.argmax(logits_n, dim=-1), skip_special_tokens=True
                    )
                    orig = tokenizer.batch_decode(
                        batch0["x_input_ids"], skip_special_tokens=True
                    )
                    B0 = batch0["xt_input_ids"].shape[0]
                    xt_tgt = tokenizer.batch_decode(
                        batch0["xt_input_ids"][torch.arange(B0), xi0.cpu()],
                        skip_special_tokens=True,
                    )
                    table = wandb.Table(
                        columns=["original", "xt_target", "recon_noisy", "t", "xt_idx"]
                    )
                    for o, xt, pn, ti, xi in zip(orig, xt_tgt, pred_n, t0, xi0):
                        table.add_data(o, xt, pn, ti.item(), xi.item())
                    wandb.log({"val/samples": table}, step=epoch + 1)

            # Save best model
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
                print(
                    f"  New best model saved (val_loss={val_loss:.4f})", flush=True
                )

                best_eval_metrics = {}
                if sample_outputs:
                    best_eval_metrics = evaluate_best_epoch(
                        sample_outputs=sample_outputs,
                        tokenizer=tokenizer,
                    )
                    if best_eval_metrics:
                        metrics_path = os.path.join(
                            output_dir, f"epoch_{epoch + 1}_best_eval_metrics.json"
                        )
                        with open(metrics_path, "w") as f:
                            json.dump(best_eval_metrics, f, indent=2)
                        print(f"  Saved best-epoch eval metrics to {metrics_path}", flush=True)

                if use_wandb:
                    wandb.run.summary["best_val_loss"] = best_val_loss
                    wandb.run.summary["best_epoch"] = epoch + 1
                    if best_eval_metrics:
                        for k, v in best_eval_metrics.items():
                            if k.startswith("bertscore/"):
                                wandb.run.summary[f"best_{k}"] = v
                            if k.startswith("unieval/"):
                                wandb.run.summary[f"best_{k}"] = v
        else:
            print(
                f"Epoch {epoch+1} | Train: {train_loss:.4f} | ETA: {eta_str}",
                flush=True,
            )

    # Save final checkpoint
    save_checkpoint(
        p0_model.g_psi,
        p0_model.decoder_x,
        optimizer,
        EPOCHS,
        train_loss,
        best_val_loss,
        os.path.join(checkpoint_dir, "final_model.pt"),
    )
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
