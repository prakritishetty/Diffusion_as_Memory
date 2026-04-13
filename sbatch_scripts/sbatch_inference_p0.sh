#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --job-name=p0-infer
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=80GB
#SBATCH --output=sbatch_scripts/slurm_output/p0/p0-infer-%j.out
#SBATCH --error=sbatch_scripts/slurm_output/p0/p0-infer-%j.err

module load conda/latest
conda activate /work/pi_dagarwal_umass_edu/project_3/bdevarangadi/.conda/envs/diffusion
python3 /work/pi_dagarwal_umass_edu/project_3/bdevarangadi/Diffusion_as_Memory/scripts/inference/forgetting_model_inference.py \
  --model-path checkpoints/p0/mod_g_psi/best_model.pt \
  --data-path data/final/test.json \
  --batch-size 16 \
  --wandb-project diffusion-as-memory \
  --wandb-run-name p0-inference-run_$(date +%Y%m%d_%H%M%S) \
  --output-json output/p0/inference/temp.json \
  --latents-output data/latents/inference/temp.pt
 
