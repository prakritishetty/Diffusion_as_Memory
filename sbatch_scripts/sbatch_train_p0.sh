#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=l40s:1
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --job-name=p0-train
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=80GB
#SBATCH --output=sbatch_scripts/slurm_output/p0/p0-train-%j.out
#SBATCH --error=sbatch_scripts/slurm_output/p0/p0-train-%j.err

module load conda/latest
conda activate /work/pi_dagarwal_umass_edu/project_3/bdevarangadi/.conda/envs/diffusion
python3 /work/pi_dagarwal_umass_edu/project_3/bdevarangadi/Diffusion_as_Memory/scripts/training/training_dl_augmented.py \
 --latents-dir ./data/latents/u_dim \
 --checkpoint-dir ./checkpoints/p0/u_dim \
 --output-dir ./output/p0/u_dim \
 --wandb-project diffusion-as-memory \
 --wandb-run-name p0-training-run_$(date +%Y%m%d_%H%M%S)
