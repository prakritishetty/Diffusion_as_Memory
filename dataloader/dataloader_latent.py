import torch
from torch.utils.data import Dataset
from typing import Tuple


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