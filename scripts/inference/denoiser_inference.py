"""
Inference script for the trained Denoiser.

Usage:
    python denoiser_inference.py \
        --latents ../data/latents/2.0/val_latents.pt \
        --checkpoint ./checkpoints/p1/best_model.pt \
        --wandb-run-name p1-inference
"""

import argparse
import json
import os
import sys
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.denoiser_module.config import DenoiserConfig
from models.denoiser_module.denoiser import Denoiser, NoiseSchedule, forward_diffusion, one_step_estimate
from models.denoiser_module.trainer import LatentDataset

EVAL_TIMESTEPS = [50, 100, 250, 500, 750, 1000]  # timesteps to evaluate one-step denoising
MAX_SAMPLES = 20      # samples to save in output JSON
BATCH_SIZE = 32


def load_denoiser(checkpoint_path, device):
    """Load a trained denoiser from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct config from saved values if available
    config = DenoiserConfig()
    if "config" in checkpoint:
        saved_cfg = checkpoint["config"]
        config.L = saved_cfg.get("L", config.L)
        config.d = saved_cfg.get("d", config.d)
        config.T = saved_cfg.get("T", config.T)
        config.N_blocks = saved_cfg.get("N_blocks", config.N_blocks)
        config.n_heads = saved_cfg.get("n_heads", config.n_heads)
        config.d_ff = saved_cfg.get("d_ff", config.d_ff)
        config.schedule = saved_cfg.get("schedule", config.schedule)

    denoiser = Denoiser(config).to(device)
    denoiser.load_state_dict(checkpoint["model_state_dict"])
    denoiser.eval()

    metadata = {
        "epoch": checkpoint.get("epoch"),
    }

    print(f"Loaded denoiser from {checkpoint_path} (epoch {metadata['epoch']})")
    print(f"  Config: L={config.L}, d={config.d}, T={config.T}, blocks={config.N_blocks}")
    return denoiser, config, metadata


@torch.no_grad()
def evaluate_one_step(denoiser, noise_schedule, dataloader, eval_timesteps, device):
    """
    Evaluate one-step denoising at specific timesteps.

    For each timestep t:
      1. Add noise: vt = forward_diffusion(v0, t)
      2. Predict noise: eps_hat = denoiser(vt, t, u)
      3. Recover: v0_hat = one_step_estimate(vt, eps_hat, t)
      4. Compute MSE(v0, v0_hat)

        Returns:
            - avg_results: dict of {t: avg_mse}
            - mse_values_by_t: dict of {t: [mse_sample_0, mse_sample_1, ...]}
    """
    results = {t: {"total_mse": 0.0, "count": 0, "mse_values": []} for t in eval_timesteps}

    for v0_batch, u_batch in tqdm(dataloader, desc="One-step eval"):
        v0_batch = v0_batch.to(device)
        u_batch = u_batch.to(device)
        B = v0_batch.shape[0]

        for t_val in eval_timesteps:
            t = torch.full((B,), t_val, device=device, dtype=torch.long)

            vt, eps = forward_diffusion(v0_batch, t, noise_schedule)
            eps_hat = denoiser(vt, t, u_batch)
            v0_hat = one_step_estimate(vt, eps_hat, t, noise_schedule)

            per_sample_mse = ((v0_hat - v0_batch) ** 2).view(B, -1).mean(dim=1)
            mse_sum = per_sample_mse.sum().item()
            results[t_val]["total_mse"] += mse_sum
            results[t_val]["count"] += B
            results[t_val]["mse_values"].extend(per_sample_mse.detach().cpu().tolist())

    avg_results = {}
    mse_values_by_t = {}
    for t_val in eval_timesteps:
        count = results[t_val]["count"]
        avg_results[t_val] = results[t_val]["total_mse"] / count if count > 0 else 0.0
        mse_values_by_t[t_val] = results[t_val]["mse_values"]

    return avg_results, mse_values_by_t


def log_mse_lines_to_wandb(wandb_mod, mse_values_by_t, eval_timesteps):
    """Log per-sample MSE values as standard line-series metrics."""
    if not mse_values_by_t:
        return

    max_len = max((len(v) for v in mse_values_by_t.values()), default=0)
    for sample_idx in range(max_len):
        row_metrics = {"one_step/sample_idx": sample_idx}
        for t_val in eval_timesteps:
            values = mse_values_by_t.get(t_val, [])
            if sample_idx < len(values):
                row_metrics[f"one_step/mse_sample_t{t_val}"] = float(values[sample_idx])
        wandb_mod.log(row_metrics, step=sample_idx + 1)



@torch.no_grad()
def collect_samples(denoiser, noise_schedule, config, dataloader, eval_timesteps, max_samples, device):
    """
    Collect sample latents for qualitative inspection / downstream decoding.

    For each sample, saves:
      - v0 (ground truth)
      - v0_hat at each eval timestep (one-step)
      - u (semantic anchor)
    """
    samples = []

    for v0_batch, u_batch in dataloader:
        v0_batch = v0_batch.to(device)
        u_batch = u_batch.to(device)
        B = v0_batch.shape[0]

        remaining = max_samples - len(samples)
        if remaining <= 0:
            break
        n = min(B, remaining)

        for i in range(n):
            sample = {
                "v0": v0_batch[i].cpu(),
                "u": u_batch[i].cpu(),
                "one_step": {},
            }

            # One-step at each timestep
            for t_val in eval_timesteps:
                t = torch.full((1,), t_val, device=device, dtype=torch.long)
                v0_i = v0_batch[i:i+1]
                u_i = u_batch[i:i+1]

                vt, eps = forward_diffusion(v0_i, t, noise_schedule)
                eps_hat = denoiser(vt, t, u_i)
                v0_hat = one_step_estimate(vt, eps_hat, t, noise_schedule)

                sample["one_step"][t_val] = {
                    "v0_hat": v0_hat[0].cpu(),
                    "mse": ((v0_hat[0] - v0_batch[i]) ** 2).mean().item(),
                }

            samples.append(sample)

    return samples


def main():
    parser = argparse.ArgumentParser(description="Inference script for Denoiser (Phase 1)")
    parser.add_argument("--latents", type=str, required=True,
                        help="Path to latent .pt file (val or test)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained denoiser checkpoint")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--eval-timesteps", type=int, nargs="+", default=EVAL_TIMESTEPS,
                        help="Timesteps for one-step evaluation")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES,
                        help="Max samples to collect for output")
    parser.add_argument("--output-dir", type=str, default="./output/p1/inference",
                        help="Directory to save results")
    parser.add_argument("--wandb-project", type=str, default="diffusion-as-memory")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-off", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load denoiser
    denoiser, config, checkpoint_meta = load_denoiser(args.checkpoint, device)
    noise_schedule = NoiseSchedule(config.T, config.schedule)

    # Load dataset
    dataset = LatentDataset(args.latents, config.L, config.d)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Loaded {len(dataset)} samples, {len(dataloader)} batches")

    # wandb
    use_wandb = not args.wandb_off
    wandb_mod = None
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={
                    "checkpoint": args.checkpoint,
                    "latents": args.latents,
                    "batch_size": args.batch_size,
                    "eval_timesteps": args.eval_timesteps,
                    "device": str(device),
                    "checkpoint_epoch": checkpoint_meta.get("epoch"),
                    "L": config.L,
                    "d": config.d,
                    "T": config.T,
                },
            )
            wandb_mod = wandb
        except Exception as e:
            print(f"wandb: unavailable ({e}); continuing without wandb")
            use_wandb = False

    os.makedirs(args.output_dir, exist_ok=True)

    print("ONE-STEP DENOISING EVALUATION")
    print(f"Timesteps: {args.eval_timesteps}")

    one_step_results, mse_values_by_t = evaluate_one_step(
        denoiser, noise_schedule, dataloader, args.eval_timesteps, device
    )

    print(f"\n{'Timestep':>10} | {'MSE':>12}")
    print("-" * 27)
    for t_val in sorted(one_step_results.keys()):
        mse = one_step_results[t_val]
        print(f"{t_val:>10} | {mse:>12.6f}")

    if use_wandb and wandb_mod is not None:
        for t_val, mse in one_step_results.items():
            wandb_mod.log({f"one_step/mse_t{t_val}": mse})
        log_mse_lines_to_wandb(wandb_mod, mse_values_by_t, args.eval_timesteps)

    print(f"\n{'='*60}")
    print(f"Collecting {args.max_samples} samples...")

    samples = collect_samples(
        denoiser, noise_schedule, config, dataloader,
        args.eval_timesteps, args.max_samples, device
    )

    # Save results JSON (metrics only, no tensors)
    results_json = {
        "checkpoint": args.checkpoint,
        "latents": args.latents,
        "num_samples": len(dataset),
        "one_step_mse": {str(k): v for k, v in one_step_results.items()},
        "one_step_mse_values": {str(k): v for k, v in mse_values_by_t.items()},
        "samples": [
            {
                "one_step": {
                    str(t): {"mse": s["one_step"][t]["mse"]}
                    for t in s["one_step"]
                }
            }
            for s in samples
        ],
    }

    json_path = os.path.join(args.output_dir, "denoiser_results.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Results saved to {json_path}")

    # wandb summary + table
    if use_wandb and wandb_mod is not None:
        wandb_mod.run.summary["num_samples"] = len(dataset)

        # Log sample table
        columns = ["sample_idx"] + [f"mse_t{t}" for t in args.eval_timesteps]
        table = wandb_mod.Table(columns=columns)
        for idx, s in enumerate(samples):
            row = [idx]
            row += [s["one_step"][t]["mse"] for t in args.eval_timesteps]
            table.add_data(*row)
        wandb_mod.log({"inference/sample_mse": table})

        wandb_mod.finish()

if __name__ == "__main__":
    main()
