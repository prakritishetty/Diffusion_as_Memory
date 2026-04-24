#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="gh200|h100|a100|l40s|a40|rtx8000"
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --job-name=complete-train
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=80GB
#SBATCH --output=sbatch_scripts/slurm_output/p2/complete-train-%j.out
#SBATCH --error=sbatch_scripts/slurm_output/p2/complete-train-%j.err

nvidia-smi

module load conda/latest
conda activate /work/pi_dagarwal_umass_edu/project_3/bdevarangadi/.conda/envs/diffusion
echo "Starting training p0..."
set -e
# python3 /work/pi_dagarwal_umass_edu/project_3/bdevarangadi/Diffusion_as_Memory/scripts/training/training_dl_augmented.py \
#  --latents-dir ./data/latents/rev-diff \
#  --checkpoint-dir ./checkpoints/p0/rev-diff \
#  --output-dir ./output/p0/rev-diff \
#  --wandb-project diffusion-as-memory \
#  --wandb-run-name p0-training-run_$(date +%Y%m%d_%H%M%S)
#  echo "Training complete!"

################################
# echo "Starting training denoiser..."
# python /work/pi_dagarwal_umass_edu/project_3/bdevarangadi/Diffusion_as_Memory/scripts/training/train_on_latents.py \
#     --train-latents data/latents/rev-diff/train_latents.pt \
#     --val-latents data/latents/rev-diff/val_latents.pt \
#     --checkpoint-dir checkpoints/p1/rev-diff \
#     --wandb-project diffusion-as-memory \
#     --wandb-run-name p1-training-run_$(date +%Y%m%d_%H%M%S)
# echo "Training complete!"

################################
echo "start training phase 2..."
python ../scripts/training/train_phase2.py \
    --p0-checkpoint /project/pi_dagarwal_umass_edu/project_3/checkpoints_apr20_prak_gpsipretrain/p0/train_19thapr_prak/best_model.pt \
    --denoiser-checkpoint /project/pi_dagarwal_umass_edu/project_3/checkpoints_apr_19/p1/rev-diff-dim-inc/best_model.pt \
    --checkpoint-dir /project/pi_dagarwal_umass_edu/project_3/checkpoints_apr20_prak_gpsipretrain/p2/train_20thapr_prak/rev-diff \
    --output-dir ../output/p2/output_20thapr_prak \
    --data-dir ../data/final \
    --wandb-project diffusion-as-memory \
    --wandb-run-name p2-gpsi-run_$(date +%Y%m%d_%H%M%S)
echo "Training complete!"