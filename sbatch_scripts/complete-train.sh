#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="gh200|h100|a100|l40s|a40|rtx8000"
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --job-name=complete-train
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=80GB
#SBATCH --output=sbatch_scripts/slurm_output/etoe/complete-train-%j.out
#SBATCH --error=sbatch_scripts/slurm_output/etoe/complete-train-%j.err

nvidia-smi

module load conda/latest
conda activate /work/pi_dagarwal_umass_edu/project_3/bdevarangadi/.conda/envs/diffusion
echo "Starting training p0..."
set -e
python3 /work/pi_dagarwal_umass_edu/project_3/bdevarangadi/Diffusion_as_Memory/scripts/training/training_dl_augmented.py \
 --latents-dir ./data/latents/rev-diff-dim-inc \
 --checkpoint-dir ./checkpoints/p0/rev-diff-dim-inc \
 --output-dir ./output/p0/rev-diff-dim-inc \
 --wandb-project diffusion-as-memory \
 --wandb-run-name p0-training-run_$(date +%Y%m%d_%H%M%S)
 echo "Training complete!"

################################
echo "Starting training denoiser..."
python /work/pi_dagarwal_umass_edu/project_3/bdevarangadi/Diffusion_as_Memory/scripts/training/train_on_latents.py \
    --train-latents data/latents/rev-diff-dim-inc/train_latents.pt \
    --val-latents data/latents/rev-diff-dim-inc/val_latents.pt \
    --checkpoint-dir checkpoints/p1/rev-diff-dim-inc \
    --wandb-project diffusion-as-memory \
    --wandb-run-name p1-training-run_$(date +%Y%m%d_%H%M%S)
echo "Training complete!"

################################
echo "start training phase 2..."
python /work/pi_dagarwal_umass_edu/project_3/bdevarangadi/Diffusion_as_Memory/scripts/training/train_phase2.py \
    --p0-checkpoint ./checkpoints/p0/rev-diff-dim-inc/best_model.pt \
    --denoiser-checkpoint ./checkpoints/p1/rev-diff-dim-inc/best_model.pt \
    --checkpoint-dir ./checkpoints/p2/rev-diff-dim-inc \
    --output-dir ./output/p2/rev-diff-dim-inc \
    --data-dir ./data/final \
    --wandb-project diffusion-as-memory \
    --wandb-run-name p2-gpsi-run_$(date +%Y%m%d_%H%M%S)
echo "Training complete!"