#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --job-name=denoiser-train
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=90GB
#SBATCH --output=slurm_output/denoiser-train-%j.out
#SBATCH --error=slurm_output/denoiser-train-%j.err

module load conda/latest
# conda activate /work/pi_dagarwal_umass_edu/project_3/bdevarangadi/.conda/envs/diffusion
conda activate ./myvenv

echo "DENOISER TRAINING"

# Create checkpoint directory
# mkdir -p checkpoints

# cd /work/pi_dagarwal_umass_edu/project_3/bdevarangadi/Diffusion_as_Memory

# Run training with default config
echo ""
echo "Starting training..."
echo "Train latents: ../data/latents/temp/train_latents.pt"
echo "Val latents: ../data/latents/temp/val_latents.pt"
echo ""


python ../scripts/training/train_denoiser_decoder.py \
    --p0-checkpoint ../checkpoints/p0/train_29Mar_prak/best_model.pt \
    --train-dataset ../data/final/train.json \
    --val-dataset ../data/final/validate.json \
    --wandb-project diffusion-as-memory \
    --wandb-run-name p12-training-run_$(date +%Y%m%d_%H%M%S)


echo "Training complete!"
