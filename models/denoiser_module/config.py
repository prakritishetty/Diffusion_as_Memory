"""
Configuration and hyperparameters for the Denoiser module.
"""

class DenoiserConfig:
    """Hyperparameters for the diffusion denoiser."""
    
    # Latent dimensions
    L = 8                     # number of slots (matches saved latent data)
    d = 512                   # embedding dimension
    u_dim = 128               # raw u dimension before projection to [L, d]
    
    # Diffusion schedule
    T = 1000                  # total diffusion timesteps
    schedule = "cosine"       # noise schedule type: "linear" or "cosine"
    
    # Transformer architecture
    N_blocks = 6              # number of Transformer blocks
    n_heads = 8               # attention heads
    d_ff = 2048               # FFN inner dimension (usually 4×d)
    dropout = 0.1             # dropout rate
    
    # Training
    learning_rate = 1e-4
    batch_size = 16
    num_epochs = 1000
    warmup_steps = 1000 #not used
    weight_decay = 1e-5
    
    # Device
    device = "cuda"           # "cuda" or "cpu"
