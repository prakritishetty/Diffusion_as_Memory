#!/bin/bash
#SBATCH --job-name=privasis_train
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint="gh200|h100|a100|a40|l40s|rtx8000"
#SBATCH --time=04:00:00
#SBATCH --mem=50G
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Load environment
module load conda/latest
conda activate latent-lang-diff

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting Diffusion Training..."
python ../../train_text_diffusion.py \
    --dataset_name privasis_abstraction \
    --output_dir ../../saved_diff_models/controlled_privasis_diff \
    --num_train_steps 60000 \
    --train_batch_size 16 \
    --max_seq_len 64 \
    --decoding_loss \
    --normalize_latent \
    --decoding_loss_every 5 \
    --decoding_loss_weight 1.0
echo "Done."
