#!/bin/bash

# 1. Generate the Privacy Abstraction Dataset
python dataset_utils/create_privacy_dataset.py

# 2. Train the Controlled Diffusion Model
# Use --resume_training --resume_dir saved_diff_models/controlled_privasis_diff if resuming
python train_text_diffusion.py \
    --dataset_name privasis_abstraction \
    --output_dir saved_diff_models/controlled_privasis_diff \
    --num_train_steps 60000 \
    --train_batch_size 16 \
    --max_seq_len 64 \
    --decoding_loss \
    --decoding_loss_every 100 \
    --decoding_loss_weight 1.0

# 3. Run Inference and Recall Visualization (Table in W&B)
python inference_forget_recall.py \
    --diffusion_dir saved_diff_models/controlled_privasis_diff \
    --dataset_name privasis_abstraction \
    --split test \
    --num_samples 10 \
    --sampling_timesteps 100
