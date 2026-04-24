#!/bin/bash
#SBATCH --partition=gpu-preempt
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


python3 ../scripts/training/training_dl_augmented.py \
 --latents-dir /project/pi_dagarwal_umass_edu/project_3/latents_apr20_prak_gpsipretrain \
 --checkpoint-dir /project/pi_dagarwal_umass_edu/project_3/checkpoints_apr20_prak_gpsipretrain/p0/train_19thapr_prak \
 --output-dir ../output/p0/train_20thapr_prak_gpsipretrain \
 --wandb-project diffusion-as-memory \
 --wandb-run-name p0-training-run-prak-gpsipretrain-decoderbetter_$(date +%Y%m%d_%H%M%S)

echo "End time: $(date)"