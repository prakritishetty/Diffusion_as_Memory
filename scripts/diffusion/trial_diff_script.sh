#!/bin/bash
#SBATCH --job-name=privasis_train
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint="gh200|h100|a100|a40|l40s|rtx8000"
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --mem=50G
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

python ../../train_text_diffusion.py \
    --dataset_name json \
    --train_file ../../dataset_utils/privasis_gpt4o_abstraction_1.json \
    --val_file ../../dataset_utils/val_privasis_gpt4o.json \
    --enc_dec_model facebook/bart-base \
    --latent_model_path ../../saved_latent_models_outputs/baseline_privasis_ae \
    --tx_dim 512 \
    --tx_depth 6 \
    --train_batch_size 16 \
    --num_train_steps 100 \
    --learning_rate 5e-5 \
    --wandb_name privacy_forgetting_joint_training \
    --output_dir ../../saved_diffusion_outputs/privacy_diff \
    --save_dir saved_diff_models/privacy_diff \
    --objective pred_v \
    --loss_type l2 \
    --save_and_sample_every 5