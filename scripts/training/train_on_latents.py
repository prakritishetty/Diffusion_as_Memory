"""
Full training script for the Denoiser using saved latents.

Usage:
    python train_on_latents.py --train-latents ../data/latents/2.0/train_latents.pt \\
                                --val-latents ../data/latents/2.0/val_latents.pt
"""

import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.denoiser_module.config import DenoiserConfig
from models.denoiser_module.trainer import DenoiserTrainer, LatentDataset


def main():
    parser = argparse.ArgumentParser(description="Train Denoiser on saved latents")
    parser.add_argument("--train-latents", type=str, required=True,
                        help="Path to train_latents.pt file")
    parser.add_argument("--val-latents", type=str, required=True,
                        help="Path to val_latents.pt file")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/p1",
                        help="Directory to save checkpoints")
    parser.add_argument("--wandb-off", action="store_true",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="diffusion-as-memory",
                        help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, required=True,
                        help="W&B run name")
    
    args = parser.parse_args()
    
    # Initialize config
    config = DenoiserConfig()
    
    print("DENOISER TRAINING ON SAVED LATENTS")
    print(f"Config: L={config.L}, d={config.d}, T={config.T}, blocks={config.N_blocks}")
    print(f"Training: epochs={config.num_epochs}, batch_size={config.batch_size}, lr={config.learning_rate}")
    print(f"Device: {config.device}")
    
    # Check file existence
    train_path = Path(args.train_latents)
    val_path = Path(args.val_latents)
    
    if not train_path.exists():
        print(f"ERROR: Train latents file not found: {train_path}")
        sys.exit(1)
    if not val_path.exists():
        print(f"ERROR: Val latents file not found: {val_path}")
        sys.exit(1)
    
    # Create datasets
    print(f"\nLoading datasets...")
    train_dataset = LatentDataset(str(train_path), config.L, config.d)
    val_dataset = LatentDataset(str(val_path), config.L, config.d)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # set to >0 if not on Windows
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Initialize trainer
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    trainer = DenoiserTrainer(config, checkpoint_dir=str(checkpoint_dir))
    # Re-create trainer with wandb if not disabled
    if not args.wandb_off:
        trainer = DenoiserTrainer(
            config,
            checkpoint_dir=str(checkpoint_dir),
            use_wandb=True,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
        )
    
    print(f"\n{'-'*60}")
    print("STARTING TRAINING...")
    
    # Train
    trainer.train(train_loader, val_loader, num_epochs=config.num_epochs)
    
    # Save training history
    history_path = checkpoint_dir / "training_history.json"
    import json
    with open(history_path, 'w') as f:
        json.dump(trainer.training_history, f, indent=2)
    print(f"\nTraining history saved to: {history_path}")
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"Best model saved to: {checkpoint_dir / 'best_model.pt'}")
    print(f"Final checkpoint saved to: {checkpoint_dir / f'checkpoint_epoch_{config.num_epochs}.pt'}")


if __name__ == "__main__":
    main()
