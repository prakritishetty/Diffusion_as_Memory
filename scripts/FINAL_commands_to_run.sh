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
    

python train_text_diffusion.py \
    --dataset_name privasis \
    --enc_dec_model facebook/bart-base \
    --latent_model_path saved_latent_models_outputs/baseline_privasis_ae \
    --tx_dim 512 \
    --tx_depth 6 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --num_train_steps 100000 \
    --learning_rate 1e-4 \
    --no_validation \
    --max_train_samples 50000 \
    --wandb_name baseline_privasis_diffusion \
    --output_dir saved_diffusion_outputs/baseline_privasis_diff \
    --save_dir saved_diff_models/baseline_privasis_diff \
    --objective pred_v \
    --loss_type l2


python inference_forget_recall.py --latent_dir saved_latent_models_outputs/baseline_privasis_ae --diffusion_dir saved_diffusion_outputs/baseline_privasis_diff --num_samples 5 --sampling_timesteps 100
