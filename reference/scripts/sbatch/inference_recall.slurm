#!/bin/bash
#SBATCH --job-name=privasis_inference
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint="gh200|h100|a100|a40|l40s|rtx8000"
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --mem=50G
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err

# Load environment
module load conda/latest
conda activate latent-lang-diff

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting Inference (Forgetting & Recall Trajectory)..."
python inference_forget_recall.py \
    --latent_dir saved_latent_models/privasis_autoencoder \
    --diffusion_dir saved_diff_models/controlled_privasis_diff \
    --dataset_name privasis_abstraction \
    --num_samples 10 \
    --sampling_timesteps 50
echo "Done."
