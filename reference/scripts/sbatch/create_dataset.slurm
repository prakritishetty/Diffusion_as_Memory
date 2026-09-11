#!/bin/bash
#SBATCH --job-name=privasis_data_gen
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint="gh200|h100|a100|a40|l40s|rtx8000"
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --mem=50G
#SBATCH --output=logs/data_gen_%j.out
#SBATCH --error=logs/data_gen_%j.err

# Load environment
module load conda/latest
conda activate latent-lang-diff

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting Dataset Generation..."
python dataset_utils/create_privacy_dataset.py
echo "Done."
