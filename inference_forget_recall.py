import argparse
import os
import json
import torch
import wandb
from tqdm import tqdm
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoConfig
from transformers.modeling_outputs import BaseModelOutput
from dataset_utils.text_dataset import get_dataset, get_dataloader
from latent_models.latent_utils import get_latent_model
from diffusion.text_denoising_diffusion import GaussianDiffusion
from model.diffusion_transformer import DiffusionTransformer

def load_models(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Latent Model (Autoencoder)
    with open(os.path.join(args.latent_dir, 'args.json'), 'rt') as f:
        latent_args = json.load(f)
    
    # Mocking argparse Namespace for get_latent_model
    class LatentArgs:
        def __init__(self, d): self.__dict__.update(d)
    
    l_args = LatentArgs(latent_args)
    bart_model, tokenizer, _ = get_latent_model(l_args)
    data = torch.load(os.path.join(args.latent_dir, 'model.pt'), map_location=device)
    bart_model.load_state_dict(data['model'])
    bart_model.to(device)
    bart_model.eval()

    # 3. Load Diffusion Model
    with open(os.path.join(args.diffusion_dir, 'args.json'), 'rt') as f:
        diff_args = json.load(f)
    
    latent_dim = latent_args['dim_ae']
    # Determine lm_dim like in train_text_diffusion.py
    if 'large' in diff_args.get('enc_dec_model', ''): lm_dim = 1024
    elif 'xl' in diff_args.get('enc_dec_model', ''): lm_dim = 2048
    else: lm_dim = 768

    model = DiffusionTransformer(
        tx_dim = diff_args['tx_dim'],
        tx_depth = diff_args['tx_depth'],
        heads = diff_args['tx_dim'] // 64,
        latent_dim = latent_dim,
        max_seq_len = diff_args['max_seq_len'],
        self_condition = diff_args.get('self_condition', False),
        scale_shift = diff_args.get('scale_shift', False),
        class_conditional = diff_args.get('class_conditional', False),
        num_classes = diff_args.get('num_classes', 0),
        seq2seq_context_dim = lm_dim,
    ).to(device)

    diffusion = GaussianDiffusion(
        model,
        max_seq_len = diff_args['max_seq_len'],
        sampling_timesteps = args.sampling_timesteps,
        sampler = args.sampler,
        train_schedule = diff_args.get('train_schedule', 'cosine'),
        sampling_schedule = args.sampling_schedule or diff_args.get('sampling_schedule') or diff_args.get('train_schedule', 'cosine'),
        objective = diff_args.get('objective', 'pred_noise'),
        scale = diff_args.get('scale', 1.0),
    ).to(device)

    # DYNAMIC ATTACHMENT: The training script attaches the encoder to the diffusion object
    diffusion.context_encoder = bart_model.get_encoder()

    diff_data = torch.load(os.path.join(args.diffusion_dir, 'model.pt'), map_location=device)
    diffusion.load_state_dict(diff_data['model'])
    diffusion.eval()

    # Store whether we need to unnormalize in a property for later use
    diffusion.should_unnormalize = diff_args.get('normalize_latent', False)

    return bart_model, diffusion, tokenizer, device

def decode_latent(bart_model, diffusion, tokenizer, latents):
    # latents: (batch, seq_len, dim)
    with torch.no_grad():
        if getattr(diffusion, 'should_unnormalize', False):
            latents = diffusion.unnormalize_latent(latents)
            
        decoder_input = bart_model.get_decoder_input(latents)
        encoder_output = BaseModelOutput(last_hidden_state=decoder_input)
        # Using beam search for better quality
        sample_ids = bart_model.generate(
            encoder_outputs=encoder_output, 
            max_length=128, 
            num_beams=4, 
            early_stopping=True
        )
        texts = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for g in sample_ids]
    return texts

def simulate_forgetting_recall(args, bart_model, diffusion, tokenizer, device):
    from evaluation.custom_metrics import CustomEvaluator
    evaluator = CustomEvaluator(device=device)

    # Load dataset
    dataset = get_dataset(args.dataset_name)
    samples = dataset[args.split].select(range(args.num_samples))
    
    dataloader = get_dataloader(
        args, 
        samples, 
        bart_model.config, 
        tokenizer, 
        diffusion.max_seq_len,
        shuffle=False
    )
    
    # We test at intervals matching the 11 levels
    # T=0, 0.1, 0.2, ..., 1.0
    test_ts = [round(i * 0.1, 1) for i in range(11)]

    # Unified Table for comparison
    columns = ["Original", "Process"] + [f"L{i} (T={t})" for i, t in enumerate(test_ts)]
    comparison_table = wandb.Table(columns=columns)

    for batch in dataloader:
        # batch['input_ids'] is [B, 11, L]
        # batch['attention_mask'] is [B, 11, L]
        all_input_ids = batch['input_ids'].to(device)
        all_attention_mask = batch['attention_mask'].to(device)
        
        B = all_input_ids.shape[0]
        original_texts = [tokenizer.decode(g, skip_special_tokens=True) for g in all_input_ids[:, 0, :]]

        with torch.no_grad():
            # Get all latents (L0...L10)
            # To avoid 11 passes, we can flatten the batch
            flat_input_ids = all_input_ids.view(-1, all_input_ids.shape[-1])
            flat_mask = all_attention_mask.view(-1, all_attention_mask.shape[-1])
            encoder_outputs = bart_model.get_encoder()(flat_input_ids, attention_mask=flat_mask)
            all_latents = bart_model.get_diffusion_latent(encoder_outputs, flat_mask)
            all_latents = all_latents.view(B, 11, -1, all_latents.shape[-1]) # [B, 11, S, D]
            
            if getattr(diffusion, 'should_unnormalize', False):
                all_latents = diffusion.normalize_latent(all_latents)

            x0 = all_latents[:, 0, :, :] # L0 for noising

            # Prepare rows
            for i in range(B):
                # Row 1: Ideal Levels (Ground Truth)
                ideal_texts = [tokenizer.decode(g, skip_special_tokens=True) for g in all_input_ids[i]]
                comparison_table.add_row(original_texts[i], "💎 Ideal Abstractions", *ideal_texts)

                # Prepare results for Forgetting and Recall
                forget_row = []
                recall_row = []

                for idx, t_val in enumerate(test_ts):
                    t = torch.full((1,), t_val, device=device)
                    alpha = diffusion.train_schedule(t).view(1, 1, 1)
                    
                    noise = torch.randn_like(x0[i:i+1])
                    xt = alpha.sqrt() * x0[i:i+1] + (1 - alpha).sqrt() * noise
                    
                    # 1. Forgetting (Noisy)
                    forget_txt = decode_latent(bart_model, diffusion, tokenizer, xt)[0]
                    f_score = evaluator.compute_bert_score_single(original_texts[i], forget_txt)
                    forget_row.append(f"{forget_txt}\n(BS: {f_score:.2f})")
                    
                    # 2. Recall (Denoised)
                    z_t = xt
                    steps = int(t_val * diffusion.sampling_timesteps)
                    if steps > 0:
                        times = torch.linspace(t_val, 0., steps + 1, device=device)
                        time_pairs = torch.stack((times[:-1], times[1:]), dim=0).unbind(dim=-1)
                        latent_mask = torch.ones((1, z_t.shape[1]), dtype=torch.bool, device=device)
                        
                        for time, time_next in time_pairs:
                            t_batch = torch.full((1,), time, device=device)
                            model_output = diffusion.diffusion_model_predictions(z_t, latent_mask, t_batch, sampling=True)
                            alpha_t = diffusion.sampling_schedule(torch.full((1,), time, device=device)).view(1, 1, 1)
                            alpha_next = diffusion.sampling_schedule(torch.full((1,), time_next, device=device)).view(1, 1, 1)
                            z_t = model_output.pred_x_start * alpha_next.sqrt() + model_output.pred_noise * (1 - alpha_next).sqrt()

                    recall_txt = decode_latent(bart_model, diffusion, tokenizer, z_t)[0]
                    r_score = evaluator.compute_bert_score_single(original_texts[i], recall_txt)
                    recall_row.append(f"{recall_txt}\n(BS: {r_score:.2f})")

                comparison_table.add_row(original_texts[i], "📉 Forgetting (Noisy)", *forget_row)
                comparison_table.add_row(original_texts[i], "🚀 Recall (Denoised)", *recall_row)

    wandb.log({"PII_Controlled_Forget_Recall": comparison_table})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dir", type=str, default="saved_latent_models/privasis_autoencoder")
    parser.add_argument("--diffusion_dir", type=str, default="saved_diff_models/controlled_privasis_diff")
    parser.add_argument("--dataset_name", type=str, default="privasis_abstraction")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--sampling_timesteps", type=int, default=50) # Reduced for speed
    parser.add_argument("--sampler", type=str, default="ddim")
    parser.add_argument("--sampling_schedule", type=str, default="cosine")
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--train_batch_size", type=int, default=1) # One by one for detailed table
    
    args = parser.parse_args()

    wandb.init(project="text_denoising_diffusion", name=f"controlled_forget_recall_{args.num_samples}")

    bart_model, diffusion, tokenizer, device = load_models(args)
    
    # Sync args with training args
    with open(os.path.join(args.diffusion_dir, 'args.json'), 'rt') as f:
        diff_args = json.load(f)
    args.enc_dec_model = diff_args.get('enc_dec_model', 'facebook/bart-base')
    args.max_seq_len = diff_args.get('max_seq_len', 64)
    
    simulate_forgetting_recall(args, bart_model, diffusion, tokenizer, device)
    
    wandb.finish()
