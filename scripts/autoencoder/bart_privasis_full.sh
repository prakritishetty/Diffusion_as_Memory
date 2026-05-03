#!/bin/bash
# ============================================================
# FULL RUN: Uses the entire ~1M Privasis-Zero corpus.
# Remove --max_train_samples and run this once you've verified
# the pipeline works with the 50k sanity-check script.
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
    --wandb_name baseline_privasis_autoencoder_full \
    --output_dir saved_latent_models_outputs/baseline_privasis_ae_full \
    --save_dir saved_latent_models/privasis_autoencoder_full
