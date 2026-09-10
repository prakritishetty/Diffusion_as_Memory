import argparse
import torch
import os
import sys
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from tqdm import tqdm
import json

from utils.inference_utils import load_denoiser_from_checkpoint, load_p0_model_from_checkpoint
from models.denoiser_module.denoiser import NoiseSchedule, forward_diffusion, one_step_estimate
from dataloader.dataloader_augmentated import MSRAugmentedDataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

L_SLOTS = 8
U_DIM = 128
EVAL_TIMESTEPS = [50, 100, 250, 500, 750, 1000]


def run_inference(p0_model, denoiser_model, noise_schedule, dataloader, tokenizer, device):
    """
    1. Get u and v0 latent
    2. Run forward diffusion to get v_t
    3. Run reverse diffusion loop to get v0 estimate
    4. Decode v0 estimate to get recon_noisy
    """
    
    results = []
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Running inference"), start=1):
        u, v0 = p0_model.encode_latents(batch)
        u = u.detach()
        v0 = v0.detach()
        B = v0.shape[0]
        
        # for t_value in EVAL_TIMESTEPS:
        t_value = 500
        t = torch.full((B,), t_value, device=device, dtype=torch.long)
        vt, eps = forward_diffusion(v0, t, noise_schedule)
        eps_hat = denoiser_model(vt, t, u)
        v0_hat = one_step_estimate(vt, eps_hat, t, noise_schedule)
        
        v0_hat_projected = p0_model.g_psi(v_hat_0=v0_hat, t=t)
        generate_ids = p0_model.decode_latents(v0_hat_projected, attention_mask=torch.ones((B, L_SLOTS), device=device))
        decoded_texts = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
        original_texts = batch["x_text"]
        
        for sample_idx in range(B):
            results.append({
                "batch_idx": batch_idx,
                "sample_idx": sample_idx,
                "timestep": t_value,
                "original_text": original_texts[sample_idx],
                "decoded_text": decoded_texts[sample_idx],
            })
            
    output_path = "/project/pi_dagarwal_umass_edu/project_3/issinha/output/inference_results_phase2_full.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results

def main():
    """
    1. Load model P0 and denoiser from checkpoints
    2. Load test dataset (input memory text)
    3. Run inference loop, saving outputs for all batches
    
    Inference loop (do not use true v0):
    1. Get u from P0's u_head (input is memory text)
    2. Start from pure noiser OR run forward diffusion to get v_t
    3. Run reverse diffusion loop:
        a. 
    
    What are we evalauting: the reconstruction of original memory
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0-checkpoint", type=str, required=False)
    parser.add_argument("--decoder-gpsi-checkpoint", type=str, required=False)
    parser.add_argument("--denoiser-checkpoint", type=str, required=False)
    parser.add_argument("--dataset", type=str, required=False)
    args = parser.parse_args()
    
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    dataset_path = "/project/pi_dagarwal_umass_edu/project_3/issinha/Diffusion_as_Memory/data/final/test.json"
    dataset = MSRAugmentedDataset(dataset_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
    print(f"Loaded {len(dataset)} samples, {len(dataloader)} batches")
    
    p0_model_path = "/project/pi_dagarwal_umass_edu/project_3/issinha/checkpoints/p0/mod_g_psi/best_model.pt"
    denoiser_path = "/project/pi_dagarwal_umass_edu/project_3/issinha/checkpoints/p1/mod_g_psi/best_model.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p0_model, _ = load_p0_model_from_checkpoint(p0_model_path, device, L_SLOTS, U_DIM)
    print(f"Loaded P0 model from {p0_model_path}")
    denoiser_model, config, _ = load_denoiser_from_checkpoint(denoiser_path, device)
    print(f"Loaded denoiser model from {denoiser_path}")
    noise_schedule = NoiseSchedule(T=config.T, schedule_type=config.schedule)
    
    run_inference(p0_model, denoiser_model, noise_schedule, dataloader, tokenizer, device)
    

if __name__ == "__main__":
    main()