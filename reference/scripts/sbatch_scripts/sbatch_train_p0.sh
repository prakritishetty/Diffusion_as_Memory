#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --constraint="a16|gh200|h100|a100|l40s|a40|rtx8000"
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --job-name=p0-train
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=90GB
#SBATCH --output=slurm_output/p0/p0-train-%j.out
#SBATCH --error=slurm_output/p0/p0-train-%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"


module load conda/latest
conda activate ./myvenv
python3 ../scripts/training/training_dl_augmented.py \
 --latents-dir ../data/latents/rev-diff \
 --checkpoint-dir ../checkpoints/p0/train_10thapr_prak/rev-diff \
 --output-dir ../output/p0/train_10thapr_prak/rev-diff \
 --wandb-project diffusion-as-memory \
 --wandb-run-name p0-training-run_$(date +%Y%m%d_%H%M%S)

echo "End time: $(date)"