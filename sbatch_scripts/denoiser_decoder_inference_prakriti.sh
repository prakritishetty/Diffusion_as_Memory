#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=04:00:00             
#SBATCH --job-name=p1-inference
#SBATCH --cpus-per-task=4
#SBATCH --mem=90GB
#SBATCH --output=slurm_output/p1-inference-%j.out
#SBATCH --error=slurm_output/p1-inference-%j.err

# 1. Environment Setup
module load conda/latest
conda activate ./myvenv


# Ensure output directory exists for the JSON
mkdir -p output/p1/inference_5thapr_prak

echo "RUNNING INFERENCE: Multi-step Reverse Diffusion"
echo "Date: $(date)"


python ../scripts/inference/denoiser_decoder_inference_multistep.py \
    --p0-checkpoint ../checkpoints/p0/train_29Mar_prak/best_model.pt \
    --denoiser-checkpoint ../checkpoints/p1/train_5thapr_prak/best_denoiser_model.pt \
    --decoder-gpsi-checkpoint ../checkpoints/p1/train_5thapr_prak/best_decoder_gpsi_model.pt \
    --dataset ../data/final/test.json \
    --wandb-project diffusion-as-memory \
    --wandb-run-name p12-inference-run_$(date +%Y%m%d_%H%M%S)

echo "Inference complete at $(date)!"