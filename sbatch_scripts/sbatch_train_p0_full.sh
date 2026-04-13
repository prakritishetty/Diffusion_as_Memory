#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=l40s:1
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --job-name=p0-full-ffn
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=80GB
#SBATCH --output=sbatch_scripts/slurm_output/p0/p0-full-%j.out
#SBATCH --error=sbatch_scripts/slurm_output/p0/p0-full-%j.err

module load conda/latest
source /work/pi_dagarwal_umass_edu/project_3/kghodasara_umass_edu/Diffusion_as_Memory/.venv/bin/activate

export WANDB_ENTITY="balachandradevarangadi-umass-amherst"

echo "=== P0 FULL TRAINING: FFN UHead (3001 samples, 500 epochs) ==="
which python3
echo ""
nvidia-smi

python3 /work/pi_dagarwal_umass_edu/project_3/kghodasara_umass_edu/Diffusion_as_Memory/scripts/training/training_dl_augmented.py \
    --latents-dir /project/pi_dagarwal_umass_edu/project_3/kghodasara_umass_edu/latents/full_ffn \
    --checkpoint-dir /project/pi_dagarwal_umass_edu/project_3/kghodasara_umass_edu/checkpoints/p0/full_ffn \
    --output-dir /project/pi_dagarwal_umass_edu/project_3/kghodasara_umass_edu/output/p0/full_ffn \
    --wandb-project diffusion-as-memory \
    --wandb-run-name p0-full-ffn-uhead_$(date +%Y%m%d_%H%M%S)

echo "End time: $(date)"