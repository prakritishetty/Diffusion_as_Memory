"""
Core denoiser implementation with diffusion pipeline.

This module includes:
- Forward diffusion: corrupts clean latents with noise
- Denoiser Nθ: Transformer-based noise prediction
- One-step estimate: recovers clean latent from noise prediction
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from models.denoiser_module.config import DenoiserConfig


class NoiseSchedule:
    """Pre-computed noise schedule for diffusion."""
    
    def __init__(self, T: int, schedule_type: str = "cosine"):
        """
        Args:
            T: Total number of diffusion timesteps
            schedule_type: "linear" or "cosine"
        """
        self.T = T
        self.schedule_type = schedule_type
        self.alpha_bar = self._compute_alpha_bar()
    
    def _compute_alpha_bar(self) -> torch.Tensor:
        """Compute cumulative product of alphas."""
        if self.schedule_type == "linear":
            betas = torch.linspace(0.0001, 0.02, self.T)
        elif self.schedule_type == "cosine":
            # Cosine schedule as in Improved DDPM
            s = 0.008
            steps = torch.arange(self.T + 1)
            alphas_cumprod = torch.cos(((steps / self.T) + s) / (1 + s) * math.pi * 0.5) ** 2 #instead of product, we directly compute cumulative product using cosine formula
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]) # so alphas_cumprod[1:] reps alpha bar at t which is product of t 1 to t so when divided by alphas_cumprod[:-1] which is product of 1 to t-1 we get alpha at t
            betas = torch.clip(betas, 0.0001, 0.9999) # to avoid smaller or larger betas
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")
        
        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, dim=0) # alphas is calculated alpha at eact timestep but in diffusion alphas are multiplied right so t=1 is alpha1, t=2 is alpha1*alpha2 and so on so we take cumulative product to get alpha bar at each timestep which is product of all alphas from 1 to t
        return alpha_bar # this is basically list of alpha bars for each timestep, so alpha_bar[t-1] gives us alpha bar at timestep t (1-indexed), like a lookup table which is pre calculated 
    
    def get_alpha_bar(self, t: int) -> float:
        """Get alpha_bar value at timestep t (1-indexed)."""
        return self.alpha_bar[t - 1].item()


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding + MLP projection."""
    
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        
        # MLP to project sinusoidal embedding
        self.mlp = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.SiLU(),
            nn.Linear(d * 4, d)
        ) # linear layer to project from d to 4d, then activation, then linear layer back to d, this expand and compress makes it learn richer transformations
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: tensor of shape [batch_size] containing timestep indices (1 to T)
        
        Returns:
            t_emb: tensor of shape [batch_size, d]
        """
        # Sinusoidal encoding
        device = t.device
        batch_size = t.shape[0]
        
        # Create sinusoidal positional encoding
        t_float = t.float()
        half_d = self.d // 2
        emb = math.log(10000) / (half_d - 1)
        emb = torch.exp(torch.arange(half_d, device=device) * -emb)
        emb = t_float[:, None] * emb[None, :]
        
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.d % 2 == 1:
            emb = torch.cat([emb, torch.zeros(batch_size, 1, device=device)], dim=-1)
        
        # Project through MLP
        t_emb = self.mlp(emb)  # [batch_size, d]
        return t_emb


class AdaLN(nn.Module):
    """Adaptive Layer Normalization modulated by timestep embedding."""
    
    def __init__(self, d: int, d_cond: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.affine = nn.Linear(d_cond, 2 * d)  # outputs gamma and beta
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape [batch_size, L, d]
            t_emb: tensor of shape [batch_size, d_cond]
        
        Returns:
            output: tensor of shape [batch_size, L, d]
        """
        normalized = self.norm(x)
        affine_params = self.affine(t_emb)  # [batch_size, 2*d]
        
        # Split into gamma and beta
        gamma, beta = affine_params.chunk(2, dim=-1)  # each [batch_size, d] so d is split into two parts
        
        # Reshape for broadcasting: [batch_size, 1, d]
        if gamma.dim() == 2:
            gamma = gamma.unsqueeze(1)
        if beta.dim() == 2:
            beta = beta.unsqueeze(1)# adds 1 at position (index) of value specifed here it was 1, if 2. was given then it would add 1 at index 2 so the shape would be [batch_size, d, 1]
        # print("normalized", normalized.shape, "gamma", gamma.shape, "beta", beta.shape)
        return gamma * normalized + beta


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        assert d % n_heads == 0, "d must be divisible by n_heads"
        
        self.d = d
        self.n_heads = n_heads
        self.d_head = d // n_heads
        
        self.Q = nn.Linear(d, d)
        self.K = nn.Linear(d, d)
        self.V = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: [batch_size, seq_q, d]
            key: [batch_size, seq_k, d]
            value: [batch_size, seq_k, d]
            mask: optional [batch_size, seq_q, seq_k]
        
        Returns:
            output: [batch_size, seq_q, d]
        """
        batch_size = query.shape[0]
        
        # Project to multiple heads
        Q = self.Q(query)  # [batch_size, seq_q, d]
        K = self.K(key)    # [batch_size, seq_k, d]
        V = self.V(value)  # [batch_size, seq_k, d]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        # [batch_size, n_heads, seq_q, d_head]
        K = K.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        # [batch_size, n_heads, seq_k, d_head]
        V = V.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        # [batch_size, n_heads, seq_k, d_head]
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        # [batch_size, n_heads, seq_q, seq_k]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        # [batch_size, n_heads, seq_q, d_head]
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [batch_size, seq_q, n_heads, d_head]
        attn_output = attn_output.view(batch_size, -1, self.d)
        # [batch_size, seq_q, d]
        
        output = self.out(attn_output)
        return output


class TransformerBlock(nn.Module):
    """Single Transformer block with self-attention, cross-attention, and FFN."""
    
    def __init__(self, d: int, n_heads: int, d_ff: int, u_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        
        # d_cond = d (t_emb) + u_dim (raw u); if u_dim=0 reverts to t-only conditioning
        d_cond = d + u_dim
        self.adalan1 = AdaLN(d, d_cond)
        self.adalan2 = AdaLN(d, d_cond)
        
        # Self-attention
        self.self_attn = MultiHeadAttention(d, n_heads)
        
        # Cross-attention (x attends to u)
        # self.cross_attn = MultiHeadAttention(d, n_heads)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        t_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: main input (vt) [batch_size, L, d]
            u: semantic anchor [batch_size, L, d]
            t_emb: timestep embedding [batch_size, d]
        
        Returns:
            output: [batch_size, L, d]
        
        """
        # fuse timestep and u into a single conditioning vector [B, d + u_dim]
        c = torch.cat([t_emb, u], dim=-1)

        # a. AdaLN + b. Self-Attention
        x_normalized = self.adalan1(x, c)
        x = x + self.dropout(self.self_attn(x_normalized, x_normalized, x_normalized))
        
        # c. Cross-Attention (x attends to u, no AdaLN before it)
        # x = x + self.dropout(self.cross_attn(x, u, u))
        
        # d. AdaLN + e. FFN
        x_normalized = self.adalan2(x, c)
        x = x + self.dropout(self.ffn(x_normalized))
        
        return x


class Denoiser(nn.Module):
    """
    Denoiser Nθ: Predicts noise added during forward diffusion.
    
    Architecture:
    - Timestep embedding (sinusoidal + MLP)
    - N Transformer blocks with AdaLN, self-attention, and cross-attention
    - Output projection to predict noise
    """
    
    def __init__(self, config: DenoiserConfig):
        super().__init__()
        self.config = config
        
        self.d = config.d
        self.L = config.L
        self.T = config.T
        
        # Timestep embedding
        self.timestep_embedding = TimestepEmbedding(config.d)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d=config.d,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                u_dim=config.u_dim,
                dropout=config.dropout
            )
            for _ in range(config.N_blocks)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(config.d)
        self.output_projection = nn.Linear(config.d, config.d)
    
    def forward(
        self,
        vt: torch.Tensor,
        t: torch.Tensor,
        u: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise from noisy latent.
        
        Args:
            vt: noisy latent [batch_size, L, d]
            t: timestep indices [batch_size] (values in [1, T])
            u: semantic anchor [batch_size, L, d]
        
        Returns:
            eps_hat: predicted noise [batch_size, L, d] 
            trying with [batch_size, K]
        """
        # Get timestep embedding
        t_emb = self.timestep_embedding(t)  # [batch_size, d]
        
        # Apply Transformer blocks
        x = vt
        for block in self.transformer_blocks:
            x = block(x, u, t_emb)
        
        # Output projection
        eps_hat = self.output_norm(x)
        eps_hat = self.output_projection(eps_hat)
        
        return eps_hat


def forward_diffusion(
    v0: torch.Tensor,
    t: torch.Tensor,
    noise_schedule: NoiseSchedule
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward diffusion: corrupt clean latent with noise at timestep t.
    
    Args:
        v0: clean latent [batch_size, L, d]
        t: timestep indices [batch_size] (values in [1, T])
        noise_schedule: NoiseSchedule object
    
    Returns:
        vt: noisy latent [batch_size, L, d]
        eps: noise that was added [batch_size, L, d]
    """
    device = v0.device
    batch_size, L, d = v0.shape
    
    # Sample noise
    eps = torch.randn_like(v0)
    
    # Get alpha_bar for each timestep in batch
    alpha_bar = []
    for ti in t:
        alpha_bar.append(noise_schedule.get_alpha_bar(ti.item()))
    alpha_bar = torch.tensor(alpha_bar, device=device, dtype=v0.dtype)
    # [batch_size]
    
    # Reshape for broadcasting
    alpha_bar = alpha_bar.view(-1, 1, 1)  # [batch_size, 1, 1]
    
    # Forward diffusion formula: vt = sqrt(alpha_bar) * v0 + sqrt(1 - alpha_bar) * eps
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
    
    vt = sqrt_alpha_bar * v0 + sqrt_one_minus_alpha_bar * eps
    
    return vt, eps


def one_step_estimate(
    vt: torch.Tensor,
    eps_hat: torch.Tensor,
    t: torch.Tensor,
    noise_schedule: NoiseSchedule
) -> torch.Tensor:
    """
    Recover estimate of clean latent from noise prediction.
    
    Args:
        vt: noisy latent [batch_size, L, d]
        eps_hat: predicted noise [batch_size, L, d]
        t: timestep indices [batch_size]
        noise_schedule: NoiseSchedule object
    
    Returns:
        v0_hat: estimated clean latent [batch_size, L, d]
    """
    device = vt.device
    batch_size = vt.shape[0]
    
    # Get alpha_bar for each timestep
    alpha_bar = []
    for ti in t:
        alpha_bar.append(noise_schedule.get_alpha_bar(ti.item()))
    alpha_bar = torch.tensor(alpha_bar, device=device, dtype=vt.dtype)
    # [batch_size]
    
    # Reshape for broadcasting
    alpha_bar = alpha_bar.view(-1, 1, 1)  # [batch_size, 1, 1]
    
    # Formula: v0_hat = (vt - sqrt(1 - alpha_bar) * eps_hat) / sqrt(alpha_bar)
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
    
    v0_hat = (vt - sqrt_one_minus_alpha_bar * eps_hat) / sqrt_alpha_bar
    
    return v0_hat
