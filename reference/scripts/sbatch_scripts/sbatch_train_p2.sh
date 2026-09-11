#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --job-name=p2-train
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=100GB
#SBATCH --output=sbatch_scripts/slurm_output/p2/p2-train-%j.out
#SBATCH --error=sbatch_scripts/slurm_output/p2/p2-train-%j.err

module load conda/latest
conda activate /work/pi_dagarwal_umass_edu/project_3/bdevarangadi/.conda/envs/diffusion

echo "PHASE 2 (P2): G_psi Semantic Projection + Decoder Fine-tuning"
echo ""
nvidia-smi
python /work/pi_dagarwal_umass_edu/project_3/bdevarangadi/Diffusion_as_Memory/scripts/training/train_phase2.py \
    --p0-checkpoint ./checkpoints/p0/mod_g_psi/best_model.pt \
    --denoiser-checkpoint ./checkpoints/p1/epoch_inc/best_model.pt \
    --checkpoint-dir ./checkpoints/p2/grnd_label_x_nw_config \
    --output-dir ./output/p2/grnd_label_x_nw_config \
    --data-dir ./data/final \
    --label-source x \
    --wandb-project diffusion-as-memory \
    --wandb-run-name p2-gpsi-run_$(date +%Y%m%d_%H%M%S)
