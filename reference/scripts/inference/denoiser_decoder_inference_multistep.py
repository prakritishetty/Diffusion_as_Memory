import argparse
import torch
import os
import sys
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from tqdm import tqdm
import json
import wandb

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    
from utils.inference_utils import load_denoiser_from_checkpoint, load_p0_model_with_gpsi_decoder_from_checkpoint
from models.denoiser_module.denoiser import NoiseSchedule, forward_diffusion, one_step_estimate, step_by_step_estimate
from dataloader.dataloader_augmentated import MSRAugmentedDataset



L_SLOTS = 8
U_DIM = 128
EVAL_TIMESTEPS = [50, 100, 250, 500, 750, 1000]


def run_inference(p0_model, denoiser_model, noise_schedule, dataloader, tokenizer, device, use_wandb=False):
    """
    1. Get u and v0 latent
    2. Run forward diffusion to get v_t
    3. Run reverse diffusion loop to get v0 estimate
    4. Decode v0 estimate to get recon_noisy
    """
    p0_model.eval()
    denoiser_model.eval()

    results = []
    wandb_data = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Running inference"), start=1):
            u, v0 = p0_model.encode_latents(batch)
            u = u.detach()
            v0 = v0.detach()
            B = v0.shape[0]

            # for t_value in EVAL_TIMESTEPS:
            t_value = random.randint(50, 1000)
            t = torch.full((B,), t_value, device=device, dtype=torch.long)
            vt, eps = forward_diffusion(v0, t, noise_schedule)
            original_texts = batch["x_text"]

            curr_v = vt
            for t_val in reversed(range(1, t_value + 1)):
                t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
                eps_hat = denoiser_model(curr_v, t_batch, u)
                curr_v = step_by_step_estimate(curr_v, eps_hat, t_batch, noise_schedule)

                if t_val in [500, 450, 400, 350, 300, 250, 200, 150, 100, 50, 1]:
                    v0_hat_current_guess = one_step_estimate(curr_v, eps_hat, t_batch, noise_schedule)

                    v0_hat_projected = p0_model.g_psi(
                        v_hat_0=v0_hat_current_guess, 
                        v_t=curr_v, 
                        t=t_batch, 
                        u=u
                    )
                    slot_mask = torch.ones((B, L_SLOTS), device=device)
                    ids = p0_model.decode_latents(v0_hat_projected, slot_mask)
                    decoded = tokenizer.batch_decode(ids, skip_special_tokens=True)

                    for i in range(B):
                        results.append({
                            "batch_idx": batch_idx,
                            "sample_idx": i,
                            "timestep": t_val,
                            "original_text": original_texts[i],
                            "decoded_text": decoded[i],
                            "is_final": (t_val == 1)
                            })
                        wandb_data.append([batch_idx, i, t_val, original_texts[i], decoded[i]])

            # if batch_idx >= 2: break 

    if use_wandb:
        columns = ["Batch", "Sample_ID", "Timestep", "Original", "Decoded"]
        inference_table = wandb.Table(data=wandb_data, columns=columns)
        wandb.log({"inference/progression_table": inference_table})

    # Save results showing the progression
    output_path = "../output/p1/inference_5thapr_prak/inference_progression.json"
    with open(output_path, "w",encoding="utf-8") as f:
        json.dump(results, f, indent=2,ensure_ascii=False)


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
    parser.add_argument("--wandb-project", type=str, default="diffusion-as-memory")
    parser.add_argument("--wandb-run-name", type=str, required=False)
    args = parser.parse_args()
    
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    dataset_path = "../data/final/test.json"
    dataset = MSRAugmentedDataset(dataset_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    print(f"Loaded {len(dataset)} samples, {len(dataloader)} batches")
    
    p0_model_path = "../checkpoints/p0/train_29Mar_prak/best_model.pt"
    denoiser_path = "../checkpoints/p1/train_5thapr_prak/best_denoiser_model.pt"
    gpsi_decoder_path = "../checkpoints/p1/train_5thapr_prak/best_decoder_gpsi_model.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p0_model, _ = load_p0_model_with_gpsi_decoder_from_checkpoint(p0_model_path, gpsi_decoder_path, device, L_SLOTS, U_DIM)
    print(f"Loaded P0 model from {p0_model_path}")
    denoiser_model, config, _ = load_denoiser_from_checkpoint(denoiser_path, device)
    print(f"Loaded denoiser model from {denoiser_path}")
    noise_schedule = NoiseSchedule(T=config.T, schedule_type=config.schedule)
    
    use_wandb = args.wandb_run_name is not None
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config={"mode": "inference"})

    run_inference(p0_model, denoiser_model, noise_schedule, dataloader, tokenizer, device, use_wandb=use_wandb)

    if use_wandb:
        wandb.finish()
    

if __name__ == "__main__":
    main()