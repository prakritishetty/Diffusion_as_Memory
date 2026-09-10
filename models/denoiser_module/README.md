# Denoiser Module: Diffusion-Based Noise Prediction

This module implements a **Transformer-based denoiser** for a latent memory system using diffusion models.

## Overview

The denoiser is part of a three-stage diffusion pipeline:

```
Forward Diffusion → Denoiser (Nθ) → One-step Clean Estimate
```

### Key Components

1. **Noise Schedule**: Pre-computed $\alpha_{bar}$ values for diffusion timesteps
2. **Timestep Embedding**: Sinusoidal positional encoding + MLP projection
3. **Adaptive Layer Normalization (AdaLN)**: Timestep-conditioned layer norm
4. **Multi-Head Attention**: Self-attention and cross-attention mechanisms
5. **Transformer Blocks**: Combining attention, AdaLN, and FFN
6. **Denoiser Model**: Full architecture combining all components


## Architecture Details

### Forward Diffusion

$$v_t = \sqrt{\bar{\alpha}_t} v_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$

where:
- $v_0$: clean latent
- $\epsilon$: Gaussian noise $\sim \mathcal{N}(0, I)$
- $\bar{\alpha}_t$: cumulative product of alphas from noise schedule

### Denoiser Architecture

1. **Timestep Embedding**
   - Sinusoidal positional encoding
   - MLP projection to dimension $d$

2. **Transformer Blocks** (repeated $N$ times)
   - AdaLN(vt, t_emb)
   - Self-attention: vt → vt
   - Cross-attention: vt ← u (semantic anchor)
   - AdaLN(vt, t_emb)
   - FFN: $x + \text{FFN}(x)$

3. **Output Projection**
   - LayerNorm → Linear → $\epsilon_{hat}$

### One-Step Estimation

$$v_0^{hat} = \frac{v_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_{hat}}{\sqrt{\bar{\alpha}_t}}$$

### Training Loss

$$\mathcal{L} = \text{MSE}(\epsilon_{hat}, \epsilon)$$


## Training

### Training Loop

For each batch:
1. Load clean latents $v_0$ and semantic anchors $u$ (frozen, no gradient)
2. Sample random timestep $t$
3. Apply forward diffusion: $v_t, \epsilon = \text{forward\_diffusion}(v_0, t)$
4. Predict noise: $\epsilon_{hat} = N_\theta(v_t, t, u)$
5. Compute loss: $\mathcal{L} = \text{MSE}(\epsilon_{hat}, \epsilon)$
6. Backpropagation and parameter update


## Inference

Given stored noisy latent $(v_t, u, t)$:

1. Predict noise: $\epsilon_{hat} = N_\theta(v_t, t, u)$
2. Estimate clean latent: $v_0^{hat} = \text{one\_step\_estimate}(v_t, \epsilon_{hat}, t)$
3. Pass to decoder to recover text


## References
- Diffusion Models (Ho et al., 2020)
- Improved DDPM (Nichol & Dhariwal, 2021)
- Transformer-based diffusion (Hoogeboom et al., 2022)
