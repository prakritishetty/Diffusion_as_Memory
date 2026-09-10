"""
Training script for the Denoiser model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Dict, Optional
import os
import sys
import json
from pathlib import Path
from tqdm import tqdm

# Allow importing shared utilities from the project root
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.training_utils import ETATracker
from models.denoiser_module.config import DenoiserConfig
from models.denoiser_module.denoiser import Denoiser, NoiseSchedule, forward_diffusion


class LatentDataset(Dataset):
    """
    Load pre-computed latents (v0, u) from saved PyTorch tensors.
    """
    
    def __init__(self, latent_path: str, L: int, d: int):
        """
        Args:
            latent_path: path to latent .pt file
            L: expected number of slots
            d: expected embedding dimension
        """
        self.latent_path = latent_path
        self.L = L
        self.d = d
        
        # Load latents
        latents_dict = torch.load(latent_path, map_location='cpu')
        v0_loaded = latents_dict['v0']  # [num_samples, L_saved, d]
        self.u_raw = latents_dict['u']  # [num_samples, d_u]
        
        num_samples = v0_loaded.shape[0]
        L_saved = v0_loaded.shape[1]
        d_saved = v0_loaded.shape[2]
        d_u = self.u_raw.shape[1]
        self.v0 = v0_loaded
        
        print(f"Loaded {num_samples} samples from {latent_path}")
        print(f"  v0: {self.v0.shape}, u_raw: {self.u_raw.shape}")
    
    def __len__(self) -> int:
        return self.v0.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (v0, u) pair for a single sample."""
        return self.v0[idx], self.u_raw[idx]


class DenoiserTrainer:
    """Trainer for the Denoiser model."""
    
    def __init__(
        self,
        config: DenoiserConfig,
        checkpoint_dir: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: str = "diffusion-as-memory",
        wandb_run_name: Optional[str] = None,
    ):
        """
        Args:
            config: DenoiserConfig object
            checkpoint_dir: directory to save checkpoints
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize models
        self.denoiser = Denoiser(config).to(self.device)
        self.noise_schedule = NoiseSchedule(config.T, config.schedule)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.denoiser.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,  # adjust based on num_epochs
            eta_min=1e-6
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Checkpoint tracking
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.best_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': []
        }

        # Optional Weights & Biases integration
        self.use_wandb = bool(use_wandb)
        self.wandb = None
        if self.use_wandb:
            try:
                import wandb
                # Build a small config dict from the config object
                try:
                    cfg_dict = {
                        'L': self.config.L,
                        'd': self.config.d,
                        'T': self.config.T,
                        'num_epochs': self.config.num_epochs,
                        'batch_size': self.config.batch_size,
                        'learning_rate': self.config.learning_rate,
                    }
                except Exception:
                    cfg_dict = {}

                wandb.init(project=wandb_project, name=wandb_run_name, config=cfg_dict)
                wandb.watch(self.denoiser, log='all')
                self.wandb = wandb
                print("wandb: initialized")
            except Exception as e:
                print(f"wandb: unavailable or failed to init ({e}); continuing without wandb")
                self.use_wandb = False
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: training DataLoader
        
        Returns:
            average loss over the epoch
        """
        self.denoiser.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for v0_batch, u_batch in pbar:
            # Move to device
            v0_batch = v0_batch.to(self.device)  # [batch_size, L, d]
            u_batch = u_batch.to(self.device)    # [batch_size, u_dim]
            batch_size = v0_batch.shape[0]
            
            # Sample random timesteps
            t = torch.randint(1, self.config.T + 1, (batch_size,), device=self.device)
            
            # Forward diffusion
            vt, eps = forward_diffusion(v0_batch, t, self.noise_schedule)
            
            # Predict noise with Denoiser
            eps_hat = self.denoiser(vt, t, u_batch)
            
            # Compute loss
            loss = self.criterion(eps_hat, eps)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.
        
        Args:
            val_loader: validation DataLoader
        
        Returns:
            average validation loss
        """
        self.denoiser.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating")
            for v0_batch, u_batch in pbar:
                # Move to device
                v0_batch = v0_batch.to(self.device)
                u_batch = u_batch.to(self.device)
                batch_size = v0_batch.shape[0]
                
                # Sample random timesteps
                t = torch.randint(1, self.config.T + 1, (batch_size,), device=self.device)
                
                # Forward diffusion
                vt, eps = forward_diffusion(v0_batch, t, self.noise_schedule)
                
                # Predict noise
                eps_hat = self.denoiser(vt, t, u_batch)
                
                # Compute loss
                loss = self.criterion(eps_hat, eps)
                
                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'val_loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save best model checkpoint only."""
        if not is_best:
            return

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.denoiser.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': {
                'L': self.config.L,
                'd': self.config.d,
                'u_dim': self.config.u_dim,
                'T': self.config.T,
                'N_blocks': self.config.N_blocks,
                'n_heads': self.config.n_heads,
                'd_ff': self.config.d_ff,
                'schedule': self.config.schedule,
            }
        }

        best_path = self.checkpoint_dir / "best_model.pt"
        torch.save(checkpoint, best_path)
        print(f"Saved best model: {best_path} (epoch {epoch})")
        # Upload/save to wandb run if available
        if self.use_wandb and self.wandb is not None:
            try:
                # Ensure file is saved by wandb
                self.wandb.save(str(best_path))
                if hasattr(self.wandb, 'run') and self.wandb.run is not None:
                    self.wandb.run.summary['best_val_loss'] = self.best_loss
                    self.wandb.run.summary['best_epoch'] = epoch
            except Exception:
                print("wandb: unable to save checkpoint to run")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.denoiser.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int
    ):
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: training DataLoader
            val_loader: validation DataLoader
            num_epochs: number of training epochs
        """
        # Open loss log CSV (append mode so resuming doesn't overwrite)
        loss_log_path = self.checkpoint_dir / "losses.csv"
        loss_log_file = open(loss_log_path, 'a')
        # Write header only if file is empty
        if loss_log_path.stat().st_size == 0:
            loss_log_file.write("epoch,train_loss,val_loss,lr\n")

        eta_tracker = ETATracker(total_epochs=num_epochs)

        for epoch in range(1, num_epochs + 1):
            eta_tracker.start_epoch()
            print(f"\n--- Epoch {epoch}/{num_epochs} ---")

            # Train
            train_loss = self.train_epoch(train_loader)
            self.training_history['train_loss'].append(train_loss)
            print(f"Training Loss: {train_loss:.6f}")

            # Validate
            val_loss = self.validate(val_loader)
            self.training_history['val_loss'].append(val_loss)
            print(f"Validation Loss: {val_loss:.6f}")

            # Save best model only
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            self.save_checkpoint(epoch, is_best=is_best)

            # Update learning rate
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning Rate: {lr:.2e}")

            # Compute epoch timing and ETA
            epoch_elapsed, eta_seconds, eta_str = eta_tracker.end_epoch()
            print(f"Epoch time: {epoch_elapsed:.1f}s | ETA: {eta_str}", flush=True)

            # Log to CSV
            loss_log_file.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{lr:.2e}\n")
            loss_log_file.flush()

            # Log to wandb if enabled
            if self.use_wandb and self.wandb is not None:
                try:
                    self.wandb.log({
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'lr': lr,
                        **eta_tracker.wandb_metrics(epoch_elapsed, eta_seconds),
                    }, step=epoch)
                except Exception:
                    print("wandb: error during wandb.log")

        loss_log_file.close()
        print(f"\nLoss log saved to {loss_log_path}")

        # Save training history JSON
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"Training history saved to {history_path}")

        # Finish wandb run if active
        if self.use_wandb and self.wandb is not None:
            try:
                self.wandb.finish()
            except Exception:
                pass
