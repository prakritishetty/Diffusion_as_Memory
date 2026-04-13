import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import T5Tokenizer
import json
import os
import numpy as np
import argparse
import wandb

import sys

# Ensure project root is on sys.path when running this script from a subdirectory
# so top-level packages like `dataloader` and `models` can be imported.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.training_utils import ETATracker
from tqdm import tqdm
from dataloader.dataloader_augmentated import MSRAugmentedDataset
from models.encoder_prep.encoder import TextEncoder
from models.slot_pooling_prep.slot_pooling import SlotPooling
from models.uv_heads_prep.u_head import UHead
from models.uv_heads_prep.v_head import VHead
from models.decoder_prep.decoder_x import DecoderX
from models.forgetting_model import ForgettingModel
from models.g_psi_module.semantic_projection import SemanticProjectionModule
from models.g_psi_module.g_psi_config import G_psi_config


def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    total_loss_nce = 0
    total_loss_x = 0
    for batch in dataloader:
        optimizer.zero_grad()
        loss, _, loss_nce, loss_x = model(batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_loss_nce += loss_nce.item()
        total_loss_x += loss_x.item()
    return total_loss / len(dataloader), total_loss_nce / len(dataloader), total_loss_x / len(dataloader)


@torch.no_grad()
def validate_epoch(model, dataloader):
    model.eval()
    total_loss = 0
    total_loss_nce = 0
    total_loss_x = 0
    sample_outputs = []

    # Iterate over validation batches and collect all sample outputs
    for i, batch in enumerate(dataloader):
        loss, logits_x, loss_nce, loss_x = model(batch)
        total_loss += loss.item()
        total_loss_nce += loss_nce.item()
        total_loss_x += loss_x.item()
        # store tuple for this batch
        sample_outputs.append((batch, logits_x))

    avg_loss = total_loss / len(dataloader)
    avg_loss_nce = total_loss_nce / len(dataloader)
    avg_loss_x = total_loss_x / len(dataloader)
    return avg_loss, avg_loss_nce, avg_loss_x, sample_outputs


def log_sample_outputs(sample_outputs, tokenizer, epoch, output_dir):
    """Decode and save predictions for all collected validation batches.

    Args:
        sample_outputs: list of (batch, logits_x) tuples for each val batch
        tokenizer: tokenizer for decoding ids
        epoch: current epoch index (0-based)
        output_dir: directory to write JSON file
    """
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for batch, logits_x in sample_outputs:
        pred_ids_x = torch.argmax(logits_x, dim=-1)

        # Decode predictions and originals (tokenizer.batch_decode handles lists/tensors)
        decoded_x = tokenizer.batch_decode(pred_ids_x, skip_special_tokens=True)
        # decoded_y = tokenizer.batch_decode(pred_ids_y, skip_special_tokens=True)

        original_x = tokenizer.batch_decode(batch["x_input_ids"], skip_special_tokens=True)

        for i in range(len(decoded_x)):
            results.append({
                "original_x": original_x[i],
                "v0": decoded_x[i]
            })

    # Write a single file containing all decoded samples from the validation set
    out_path = os.path.join(output_dir, f"epoch_{epoch+1}_samples.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

# Utility function to save model checkpoints
def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, path):
    """Save model checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }, path)


@torch.no_grad()
def extract_and_save_latents(model, dataloader, tokenizer, save_path):
    """
    Run frozen model on data to extract u and v0 latents by running data through trained model.
    Saves a .pt file with all latents for use in denoiser.
    """
    model.eval()

    all_u = []
    all_v0 = []
    all_texts = []

    pbar = tqdm(dataloader, desc="Extracting latents", leave=False)
    for batch in pbar:
        u, v0 = model.encode_latents(batch)
        all_u.append(u.cpu())
        all_v0.append(v0.cpu())

        texts = tokenizer.batch_decode(batch["x_input_ids"], skip_special_tokens=True)
        all_texts.extend(texts)

    # Concatenate all batches
    all_u = torch.cat(all_u, dim=0)    # [N, 128]
    all_v0 = torch.cat(all_v0, dim=0)  # [N, 8, 512]

    torch.save({
        "u": all_u,
        "v0": all_v0,
        "texts": all_texts,
    }, save_path)

    print(f"Saved latents to {save_path}")
    print(f"  u shape:  {all_u.shape}")
    print(f"  v0 shape: {all_v0.shape}")
    print(f"  samples:  {len(all_texts)}")


def main():
    print(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="./output/p0/temp",
                        help="Directory to write sample output JSONs")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/p0/temp",
                        help="Directory to save checkpoints")
    parser.add_argument("--latents-dir", type=str, required=True,
                        help="Directory to save extracted latents (.pt files)")
    parser.add_argument("--wandb-project", type=str, default="diffusion-as-memory",
                        help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, required=True,
                        help="W&B run name (defaults to auto-generated)")
    parser.add_argument("--wandb-off", action="store_true",
                        help="Disable W&B logging entirely")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    # Hardcodded datasets without x+ generation
    # train_dataset = MSRDataset("./data/final/train.json")
    # train_loader = DataLoader(train_dataset, batch_size=10, shuffle = True)
    # val_dataset = MSRDataset("./data/final/validate.json")
    # val_loader = DataLoader(val_dataset, batch_size=10, shuffle = True)

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    train_dataset = MSRAugmentedDataset("./data/final/train_subset_200.json", tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle = True)
    val_dataset = MSRAugmentedDataset("./data/final/validate_subset_50.json", tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle = True)

    encoder = TextEncoder()
    slot_pool = SlotPooling(hidden_dim = encoder.hidden_dim_size, num_slots = 8)
    u_head = UHead(hidden_dim = encoder.hidden_dim_size, output_dim = 128)
    v_head = VHead(hidden_dim = encoder.hidden_dim_size)
    decoder_x = DecoderX()
    g_psi = SemanticProjectionModule(config=G_psi_config,no_use_u=True,no_use_vt=True)
    # decoder_y = DecoderY(hidden_dim = encoder.hidden_dim_size, u_dim = 128, num_slots = 8)

    model = ForgettingModel(
        encoder = encoder,
        slot_pooling = slot_pool,
        u_head = u_head,
        v_head = v_head,
        decoder_x = decoder_x,
        g_psi = g_psi,
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    epochs = 500

    # Validate every n epochs
    val_interval = 10

    output_dir = args.output_dir
    checkpoint_dir = args.checkpoint_dir
    latents_dir = args.latents_dir

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(latents_dir, exist_ok=True)

    use_wandb = not args.wandb_off
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "epochs": epochs,
                "batch_size": 10,
                "learning_rate": 1e-4,
                "val_interval": val_interval,
                "output_dir": output_dir,
                "checkpoint_dir": checkpoint_dir,
                "latents_dir": latents_dir,
            },
        )
        # Log gradients and parameter histograms every epoch
        wandb.watch(model, log="all", log_freq=len(train_loader))

    best_val_loss = float("inf")
    eta_tracker = ETATracker(total_epochs=epochs)

    for epoch in range(epochs):
        eta_tracker.start_epoch()
        train_loss, train_loss_nce, train_loss_x = train_epoch(model, train_loader, optimizer)

        # Compute epoch timing and ETA
        epoch_elapsed, eta_seconds, eta_str = eta_tracker.end_epoch()

        # Log train loss to W&B every epoch
        if use_wandb:
            wandb.log({
                "train/loss": train_loss,
                "train/loss_nce": train_loss_nce,
                "train/loss_x": train_loss_x,
                **eta_tracker.wandb_metrics(epoch_elapsed, eta_seconds),
            }, step=epoch + 1)

        # Validate every val_interval epochs and on the last epoch
        if (epoch + 1) % val_interval == 0 or (epoch + 1) == epochs:
            val_loss, val_loss_nce, val_loss_x, sample_outputs = validate_epoch(model, val_loader)

            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | ETA: {eta_str}", flush=True)
            print("-" * 30, flush=True)

            if use_wandb:
                wandb.log({
                    "val/loss": val_loss,
                    "val/loss_nce": val_loss_nce,
                    "val/loss_x": val_loss_x,
                }, step=epoch + 1)

            if sample_outputs:
                log_sample_outputs(sample_outputs, tokenizer, epoch, output_dir)

                # Log a sample table to W&B first batch only
                if use_wandb:
                    batch0, logits_x0 = sample_outputs[0]
                    pred_ids_x = torch.argmax(logits_x0, dim=-1)
                    dec_x = tokenizer.batch_decode(pred_ids_x, skip_special_tokens=True)
                    orig_x = tokenizer.batch_decode(batch0["x_input_ids"], skip_special_tokens=True)
                    table = wandb.Table(columns=["original_x", "v0_pred"])
                    for ox, px in zip(orig_x, dec_x):
                        table.add_data(ox, px)
                    wandb.log({"val/samples": table}, step=epoch + 1)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch+1, train_loss, val_loss,
                                os.path.join(checkpoint_dir, "best_model.pt"))
                print(f"  -> New best model saved (val_loss={val_loss:.4f})", flush=True)
                if use_wandb:
                    wandb.run.summary["best_val_loss"] = best_val_loss
                    wandb.run.summary["best_epoch"] = epoch + 1
        else:
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | ETA: {eta_str}", flush=True)

    # Save final checkpoint
    save_checkpoint(model, optimizer, epochs, train_loss, best_val_loss,
                    os.path.join(checkpoint_dir, f"final_model.pt"))
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    if use_wandb:
        wandb.finish()

    print("\nExtracting latents from frozen model")

    # Load best checkpoint
    checkpoint = torch.load(os.path.join(checkpoint_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Extract latents from training data
    train_loader_noshuffle = DataLoader(train_dataset, batch_size=10, shuffle=False)
    extract_and_save_latents(
        model,
        train_loader_noshuffle,
        tokenizer,
        os.path.join(latents_dir, "train_latents.pt")
    )

    # Extract latents from validation data
    val_loader_noshuffle = DataLoader(val_dataset, batch_size=10, shuffle=False)
    extract_and_save_latents(
        model,
        val_loader_noshuffle,
        tokenizer,
        os.path.join(latents_dir, "val_latents.pt")
    )

if __name__ == "__main__":
    main()
