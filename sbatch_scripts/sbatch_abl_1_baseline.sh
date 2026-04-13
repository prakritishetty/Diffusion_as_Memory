#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=l40s:1
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --job-name=p0-abl-baseline
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20GB
#SBATCH --output=sbatch_scripts/slurm_output/p0/p0-abl-baseline-%j.out
#SBATCH --error=sbatch_scripts/slurm_output/p0/p0-abl-baseline-%j.err

module load conda/latest
source /work/pi_dagarwal_umass_edu/project_3/kghodasara_umass_edu/Diffusion_as_Memory/.venv/bin/activate

cd /work/pi_dagarwal_umass_edu/project_3/kghodasara_umass_edu/Diffusion_as_Memory

# Variant 1: Original UHead (mean + linear) + Original VHead (linear)
cp models/uv_heads_prep/u_head_mean.py models/uv_heads_prep/u_head.py
cp models/uv_heads_prep/v_head_original.py models/uv_heads_prep/v_head.py

echo "=== P0 ABLATION: 200 samples: Baseline (original UHead + original VHead) ==="
echo "u_head variant:"; head -3 models/uv_heads_prep/u_head.py
echo "v_head variant:"; head -3 models/uv_heads_prep/v_head.py
echo ""
nvidia-smi

export WANDB_ENTITY="balachandradevarangadi-umass-amherst"

python3 /work/pi_dagarwal_umass_edu/project_3/kghodasara_umass_edu/Diffusion_as_Memory/scripts/training/training_dl_augmented.py \
    --latents-dir /project/pi_dagarwal_umass_edu/project_3/kghodasara_umass_edu/Diffusion_as_Memory/data/latents/ablation_baseline \
    --checkpoint-dir /project/pi_dagarwal_umass_edu/project_3/kghodasara_umass_edu/Diffusion_as_Memory/checkpoints/p0/ablation_baseline \
    --output-dir /project/pi_dagarwal_umass_edu/project_3/kghodasara_umass_edu/Diffusion_as_Memory/output/p0/ablation_baseline \
    --wandb-project diffusion-as-memory \
    --wandb-run-name p0-abl-baseline_$(date +%Y%m%d_%H%M%S)

echo "End time: $(date)"