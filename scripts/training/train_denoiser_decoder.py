import argparse
import os
import sys
import torch
import torch.nn as nn
from transformers import T5Tokenizer
from torch.utils.data import DataLoader
import wandb

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.denoiser_module.config import DenoiserConfig
from utils.training_utils import build_p0_model, select_xt_labels, log_sample_outputs, save_denoiser_checkpoint, save_decoder_gpsi_checkpoint, ETATracker
from dataloader.dataloader_augmentated import MSRAugmentedDataset
from models.denoiser_module.denoiser import forward_diffusion, one_step_estimate, Denoiser, NoiseSchedule

# Training Hyperparameters
BATCH_SIZE = 10
NUM_EPOCHS = 500
LEARNING_RATE = 5e-5
L_SLOTS = 8
D_MODEL = 512
U_DIM = 128
T_DIFFUSION = 1000
XT_BUCKET_SIZE = T_DIFFUSION // 10
VAL_INTERVAL = 10


def set_trainable_params(p0_model):
    # Freeze everything in P0 model
    for param in p0_model.parameters():
        param.requires_grad = False

    # Unfreeze decoder for fine-tuning
    for param in p0_model.decoder_x.parameters():
        param.requires_grad = True
    
    # Unfreeze G_psi 
    for param in p0_model.g_psi.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in p0_model.g_psi.parameters()) + sum(
        p.numel() for p in p0_model.decoder_x.parameters()
    )
    
    return trainable_params


def train_one_epoch(p0_model, denoiser, noise_scheduler, train_loader, optimizer, device):
    """
    Train one epoch
    
    :param p0_model: the loaded P0 model (with frozen encoder and slot pooling)
    :param denoiser: the untrained Denoiser model
    :param noise_scheduler: the noise scheduler
    :param train_loader: the training data loader
    :param optimizer: the optimizer
    :param device: the device to run the training on
    """
    p0_model.g_psi.train()
    p0_model.decoder_x.train()
    denoiser.train()
    
    criterion = nn.MSELoss()
    total_loss = 0.0

    for batch in train_loader:
        optimizer.zero_grad()
        batch_size = batch["x_input_ids"].shape[0]
        
        with torch.no_grad():
            u, v0 = p0_model.encode_latents(batch)
        u = u.detach()
        v0 = v0.detach()
        
        # run diffusion process and compute denoiser loss
        t = torch.randint(1, denoiser.T + 1, (batch_size,), device=device)
        vt, eps = forward_diffusion(v0, t, noise_scheduler)
        eps_pred = denoiser(vt, t, u)
        loss_denoiser = criterion(eps_pred, eps)
        
        # get the denoised v0 estimate
        v0_denoised = one_step_estimate(vt, eps_pred, t, noise_scheduler)
        
        # run decoder and g_psi, compute their loss
        v0_denoised_projected = p0_model.g_psi(v_hat_0=v0_denoised, t=t, v_t=vt, u=u)
        # why need slot mask!!
        slot_mask = torch.ones(batch_size, denoiser.L, device=device)
        labels, _ = select_xt_labels(batch, t, device, XT_BUCKET_SIZE)
        loss_recon, logits = p0_model.decoder_x(v0_denoised_projected, slot_mask, labels)
        
        # maybe include some lambda
        loss = loss_denoiser + loss_recon
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(p0_model.g_psi. parameters()) + list(p0_model.decoder_x.parameters()),
            max_norm=1.0
        )
        optimizer.step()
        total_loss += loss.item()
        
    n = len(train_loader)
    return total_loss / n
    
@torch.no_grad()
def validate_epoch(p0_model, denoiser, noise_scheduler, val_loader, device):
    p0_model.g_psi.eval()
    p0_model.decoder_x.eval()
    denoiser.eval()
    
    criterion = nn.MSELoss()
    total_loss = 0.0
    sample_outputs = []
    
    for batch in val_loader:
        batch_size = batch["x_input_ids"].shape[0]
        
        with torch.no_grad():
            u, v0 = p0_model.encode_latents(batch)
            u = u.detach()
            v0 = v0.detach()
            
            t = torch.randint(1, denoiser.T + 1, (batch_size,), device=device)
            vt, eps = forward_diffusion(v0, t, noise_scheduler)
            eps_pred = denoiser(vt, t, u)
            loss_denoiser = criterion(eps_pred, eps)
            
            v0_denoised = one_step_estimate(vt, eps_pred, t, noise_scheduler)
            v0_denoised_projected = p0_model.g_psi(v_hat_0=v0_denoised, t=t, v_t=vt, u=u)
            slot_mask = torch.ones(batch_size, denoiser.L, device=device)
            labels, xt_index = select_xt_labels(batch, t, device, XT_BUCKET_SIZE)
            loss_recon, logits = p0_model.decoder_x(v0_denoised_projected, slot_mask, labels)
            
            loss = loss_denoiser + loss_recon
            total_loss += loss.item()
            
            sample_outputs.append((batch, logits, t, xt_index))
    
    n = len(val_loader)
    return total_loss / n, sample_outputs


def train(
    train_loader, 
    val_loader, 
    tokenizer, 
    p0_model, 
    denoiser, 
    noise_scheduler, 
    optimizer, 
    device, 
    num_epochs, 
    use_wandb, 
    output_dir,
    checkpoint_dir,
):
    best_val_loss = float("inf")
    eta_tracker = ETATracker(total_epochs=NUM_EPOCHS)
    print("Starting training loop...")
    for epoch in range(num_epochs):
        eta_tracker.start_epoch()
        train_loss = train_one_epoch(p0_model, denoiser, noise_scheduler, train_loader, optimizer, device)
        epoch_elapsed, eta_seconds, eta_str = eta_tracker.end_epoch()
        if use_wandb:
            wandb.log(
                {
                    "train/loss": train_loss,
                    **eta_tracker.wandb_metrics(epoch_elapsed, eta_seconds),
                },
                step=epoch + 1,
            )
        
        if (epoch + 1) % VAL_INTERVAL == 0 or (epoch + 1) == NUM_EPOCHS:
            val_loss, sample_outputs = validate_epoch(p0_model, denoiser, noise_scheduler, val_loader, device)
            print(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f}")
            if use_wandb:
                wandb.log(
                    {
                        "val/loss": val_loss,
                    },
                    step=epoch + 1,
                )
            
            if sample_outputs:
                log_sample_outputs(sample_outputs, tokenizer, epoch, output_dir)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_decoder_gpsi_checkpoint(
                    p0_model.g_psi,
                    p0_model.decoder_x,
                    optimizer,
                    epoch + 1,
                    train_loss,
                    val_loss,
                    os.path.join(checkpoint_dir, "best_decoder_gpsi_model.pt"),
                )
                save_denoiser_checkpoint(
                    denoiser,
                    optimizer,
                    epoch + 1,
                    train_loss,
                    val_loss,
                    os.path.join(checkpoint_dir, "best_denoiser_model.pt"),
                )
                print(
                    f"  New best model saved (val_loss={val_loss:.4f})", flush=True
                )
                
        
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")
        
        
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save({
        #         "model_state_dict": {
        #             "g_psi": p0_model.g_psi.state_dict(),
        #             "decoder_x": p0_model.decoder_x.state_dict(),
        #             "denoiser": denoiser.state_dict(),
        #         },
        #         "epoch": epoch + 1,
        #         "train_loss": train_loss,
        #         "val_loss": val_loss,
        #     }, "best_denoiser_decoder.pt")
        #     print(f"  New best model saved with val loss {val_loss:.4f}")
    


def main():
    """
    Train both denoiser and the decoder+g_psi from p0.
    1. Load P0
    2. Unfreeze decoder and g_psi
    3. Get dataset
    4. Add denoiser module 
    5. Combine loss
    6. Set optimizer
    """
    print("Starting training script...")
    parser = argparse.ArgumentParser(
        description="Train denoiser + G_psi + fine-tune Decoder"
    )
    parser.add_argument(
        "--p0-checkpoint", type=str, required=True,
        help="Path to Phase 0 best_model.pt checkpoint",
    )
    parser.add_argument(
        "--train-dataset", type=str, required=True,
        help="Path to the training dataset"
    )
    parser.add_argument(
        "--val-dataset", type=str, required=True,
        help="Path to the validation dataset"
    )
    parser.add_argument(
        "--wandb-project", type=str, default="diffusion-as-memory",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-run-name", type=str, required=False,
        help="W&B run name",
    )
    
    print("Parsing arguments...")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = "/project/pi_dagarwal_umass_edu/project_3/issinha/checkpoints/exp1"
    output_dir = "/project/pi_dagarwal_umass_edu/project_3/issinha/output/exp1"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Dataset
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    train_dataset = MSRAugmentedDataset(args.train_dataset, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle = True)
    val_dataset = MSRAugmentedDataset(args.val_dataset, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle = True)
    
    # Configure p0 model
    p0_model = build_p0_model(device, L_SLOTS, U_DIM)
    checkpoint = torch.load(args.p0_checkpoint, map_location=device)
    p0_model.load_state_dict(checkpoint["model_state_dict"])
    p0_trainable_params = set_trainable_params(p0_model)
    print(f"  Loaded from {args.p0_checkpoint} (epoch {checkpoint.get('epoch', '?')})")
    
    # Configure denoiser model (trained from scratch)
    denoiser_model = Denoiser(config=DenoiserConfig())
    denoiser = denoiser_model.to(device)
    noise_schedule = NoiseSchedule(T=denoiser_model.T, schedule_type=denoiser_model.config.schedule)
    
    use_wandb = args.wandb_run_name is not None 
    
    if use_wandb:
        print("  Initializing Weights & Biases...")
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "phase": "P2",
                "epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "T_diffusion": T_DIFFUSION,
                "noise_schedule": denoiser_model.config.schedule,
                "trainable_params": p0_trainable_params,
            },
        )
    
    
    train(
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        p0_model=p0_model,
        denoiser=denoiser_model,
        noise_scheduler=noise_schedule,
        optimizer=torch.optim.AdamW(
            list(p0_model.g_psi.parameters()) + list(p0_model.decoder_x.parameters()) + list(denoiser_model.parameters()),
            lr=LEARNING_RATE
        ),
        device=device,
        num_epochs=NUM_EPOCHS,
        use_wandb=use_wandb,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
    )
    
    

if __name__ == "__main__":
    main()