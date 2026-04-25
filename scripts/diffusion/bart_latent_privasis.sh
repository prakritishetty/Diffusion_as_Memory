#!/bin/bash

LATENT_MODEL_PATH="saved_latent_models/privasis_autoencoder"

python train_text_diffusion.py \
    --dataset_name privasis \
    --enc_dec_model facebook/bart-base \
    --latent_model_path $LATENT_MODEL_PATH \
    --tx_dim 512 \
    --tx_depth 6 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --num_train_steps 100000 \
    --learning_rate 1e-4 \
    --wandb_name baseline_privasis_diffusion \
    --output_dir saved_diffusion_outputs/baseline_privasis_diff \
    --save_dir saved_diff_models/baseline_privasis_diff \
    --objective pred_v \
    --loss_type l2
