#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --job-name=p0-train
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=90GB
#SBATCH -o ./slurm_output/slurm-%j.out  
#SBATCH -e ./slurm_output/slurm-%j-error.out


echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"

module load conda/latest
conda activate ./myvenv
python3 ../scripts/training/training_dl_augmented.py \
 --latents-dir ../data/latents/temp \
 --checkpoint-dir ../checkpoints/p0/train_29Mar_prak \
 --output-dir ../output/p0/train_29Mar_prak \
 --wandb-project diffusion-as-memory \
 --wandb-run-name p0-training-run_$(date +%Y%m%d_%H%M%S)

echo "End time: $(date)"