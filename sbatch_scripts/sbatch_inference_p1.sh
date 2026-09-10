#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --job-name=p1-infer
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=80GB
#SBATCH --output=sbatch_scripts/slurm_output/p1/p1-infer-%j.out
#SBATCH --error=sbatch_scripts/slurm_output/p1/p1-infer-%j.err

module load conda/latest
conda activate /work/pi_dagarwal_umass_edu/project_3/bdevarangadi/.conda/envs/diffusion
python3 /work/pi_dagarwal_umass_edu/project_3/bdevarangadi/Diffusion_as_Memory/scripts/inference/denoiser_inference.py \
      --latents ./data/latents/inference/test_latents_p0_output.pt \
      --checkpoint ./checkpoints/p1/mod_g_psi/best_model.pt \
      --output-dir ./output/p1/inference/temp \
      --wandb-run-name p1-inference-run_$(date +%Y%m%d_%H%M%S) \
      --wandb-project diffusion-as-memory 
 
