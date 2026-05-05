#!/bin/bash
# ============================================================
# EVAL-ONLY: Loads a trained autoencoder checkpoint and runs
# a single validation pass (capped to 50 batches for speed).
# Run this AFTER bart_privasis.sh finishes.
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
    --max_val_batches 50 \
    --wandb_name baseline_privasis_autoencoder_eval \
    --output_dir saved_latent_models_outputs/baseline_privasis_ae \
    --save_dir saved_latent_models/privasis_autoencoder \
    --resume_dir saved_latent_models_outputs/baseline_privasis_ae \
    --eval
