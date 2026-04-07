#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|h100|gh200"
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --job-name=denoiser-train
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=80GB
#SBATCH --output=sbatch_scripts/slurm_output/p1/denoiser-train-%j.out
#SBATCH --error=sbatch_scripts/slurm_output/p1/denoiser-train-%j.err

module load conda/latest
conda activate /work/pi_dagarwal_umass_edu/project_3/bdevarangadi/.conda/envs/diffusion


echo "DENOISER TRAINING"

# Create checkpoint directory
mkdir -p checkpoints

cd /work/pi_dagarwal_umass_edu/project_3/bdevarangadi/Diffusion_as_Memory

# Run training with default config
echo ""
echo "Starting training..."
echo "Train latents: ../data/latents/u_dim/train_latents.pt"
echo "Val latents: ../data/latents/u_dim/val_latents.pt"
echo ""


python scripts/training/train_on_latents.py \
    --train-latents data/latents/u_dim/train_latents.pt \
    --val-latents data/latents/u_dim/val_latents.pt \
    --checkpoint-dir checkpoints/p1/u_dim \
    --wandb-project diffusion-as-memory \
    --wandb-run-name p1-training-run_$(date +%Y%m%d_%H%M%S)

echo "Training complete!"
