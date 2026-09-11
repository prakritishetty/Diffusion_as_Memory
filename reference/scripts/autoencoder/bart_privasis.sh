#!/bin/bash
# ============================================================
# FAST MODE: --no_validation skips beam-search eval entirely.
# --save_every 5000 saves a checkpoint every 5k steps.
# --max_train_samples 50000 uses only 50k examples (~5% of 1M)
#   for a quick sanity-check run. Remove it for the full run.
# ============================================================

python train_latent_model.py \
    --dataset_name privasis \
    --enc_dec_model facebook/bart-base \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --num_encoder_latents 32 \
    --num_decoder_latents 32 \
    --dim_ae 256 \
    --num_layers 2 \
    --learning_rate 5e-5 \
    --num_train_steps 100000 \
    --eval_every 1000 \
    --no_validation \
    --save_every 5000 \
    --max_train_samples 50000 \
    --wandb_name baseline_privasis_autoencoder \
    --output_dir saved_latent_models_outputs/baseline_privasis_ae \
    --save_dir saved_latent_models/privasis_autoencoder
