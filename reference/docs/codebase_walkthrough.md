# Codebase Walkthrough: Latent Diffusion for Language

This document provides a comprehensive overview of the codebase, explaining the architecture, the flow of functions, and the logic behind the "forgetting" and "recall" experiments.

## 1. Original Paper & Objective
**Paper:** [Latent Diffusion for Language Generation](https://arxiv.org/abs/2212.09462) (Lovelace et al., 2022).
**Objective:** The authors aimed to apply diffusion models (which work best in continuous spaces) to discrete language data.
**Method:** 
1.  **Autoencoding:** Train an encoder-decoder (BART) where the encoder maps text to a fixed-length continuous latent representation.
2.  **Diffusion:** Train a diffusion model (using a Transformer denoiser) to model the distribution of these continuous latents.
3.  **Generation:** Sample a latent from the diffusion model and decode it using the pre-trained BART decoder.

---

## 2. Training Chain & Key Functions

### Phase 1: Autoencoding (`train_latent_model.py`)
This script trains the "bridge" between discrete text and continuous latents.

*   **Model:** `BARTForConditionalGenerationLatent` (in `latent_models/bart_latent_model.py`).
*   **Chain of Events:**
    1.  `Trainer.train()`: Standard training loop.
    2.  `model.forward(input_ids, attention_mask)`:
        - Encoder processes tokens.
        - **Latent Bottleneck:** A "Perceiver" or pooling layer reduces the encoder sequence to `num_encoder_latents` (e.g., 32 vectors of size 768).
        - **Latent Space:** The output is a matrix of shape `[batch, 32, 768]`.
    3.  **Loss:** Cross-entropy loss (standard BART reconstruction loss).
*   **Design Decision:** By using a small number of latents, the model is forced to compress the text into a fixed-size continuous representation.

### Phase 2: Diffusion (`train_text_diffusion.py`)
This script trains the model to "know" what a valid text latent looks like.

*   **Model:** `DiffusionTransformer` (in `model/diffusion_transformer.py`).
*   **Diffusion Wrapper:** `GaussianDiffusion` (in `diffusion/text_denoising_diffusion.py`).
*   **Chain of Events:**
    1.  `Trainer.train()`: Iterates through the dataset.
    2.  **Get Latents:** Uses the frozen encoder from Phase 1 to convert text batch $\to x_0$.
    3.  `GaussianDiffusion.forward(x0)`:
        - **Noise Injection:** $x_t = \sqrt{\alpha_t} x_0 + \sqrt{1-\alpha_t} \epsilon$ (where $\epsilon \sim \mathcal{N}(0, I)$).
        - **Denoising Prediction:** The `DiffusionTransformer` tries to predict the noise $\epsilon$ (or the original $x_0$) given $x_t$ and time $t$.
    4.  **Loss:** MSE Loss (L2) between predicted and actual noise.
*   **Diffusion Steps:** Default 250 steps with a **Cosine Schedule**.

---

## 3. Forgetting & Recall (`inference_forget_recall.py`)

This script was added to visualize the "memory" properties of the model.

### Forgetting (Downward Spiral 📉)
*   **Logic:** We take a PII sample, encode it to $x_0$, and then manually add noise to it to reach $x_t$ for $t \in \{0.1, 0.25, 0.5, 0.75, 1.0\}$.
*   **Decoding:** We take the noisy $x_t$ and pass it directly to the BART decoder.
*   **What it shows:** How robust the *decoder* is to noise in the latent space.

### Recall (The Reconstruction 🚀)
*   **Logic:** We start with the noisy $x_t$ and use the *Diffusion Model* to denoise it back to an estimate of $x_0$ (using the DDIM or DDPM sampler).
*   **Decoding:** We decode the denoised estimate.
*   **What it shows:** How well the *Diffusion Model* can "recover" the original sample from a partially destroyed state.

---

## 4. Addressing Specific Concerns

### a.) Truncated Privasis Samples
**Observation:** The original samples in the results appear cut off.
**Diagnosis:** 
- The `max_seq_len` in the dataloader is set to **64** by default (inherited from the ROCStories configuration).
- In `dataset_utils/text_dataset.py:172`, truncation is set to `True`.
- If your Privasis samples (paragraphs) are longer than 64 tokens, they will be clipped.
**Fix:** Increase `--max_seq_len` to 128 or 256 in both training and inference scripts.

### b.) Non-Smooth Transitions (The "Cliff" Effect)
**Observation:** Forgetting is fine at $t=0.5$, but gibberish at $t=0.75$. Recall is perfect at $t=0.5$, but broken earlier.
**Reasons:**

1.  **Cosine Noise Schedule:** The schedule is designed to preserve signal for a long time and then drop off rapidly. This is intentional in diffusion to maximize learning on "hard" denoising steps, but it leads to a non-linear degradation of human-readable text.
2.  **Latent Space Geometry:** The latent space created by the BART encoder is not naturally "smooth" (it's not a VAE with a strong KL penalty). Samples exist as "clusters" or "points". 
    - At $t=0.5$, the noise might still keep the latent within the "neighborhood" of the original sample.
    - At $t=0.75$, the latent crosses a threshold where the decoder no longer recognizes it as a valid starting point for any sentence, leading to gibberish.
3.  **Diffusion "Snapping":** The diffusion model acts as a "manifold projection". When you denoise from $t=0.5$, the model sees enough of the original signal to "snap" the latent back to the specific point it memorized (overfitting). If it's overfitted, it won't give a "partially forgotten" version; it will either give the **exact** sample or **nonsense**.

### c.) Are these Overfitting Examples?
**Likely Yes.** If the model is recalling the text *perfectly* even from $t=0.5$ noise, it suggests the diffusion model has memorized the training samples' latent positions. In a perfectly generalized model, you would expect "smooth" forgetting where names or dates change to other plausible names/dates, rather than turning into gibberish or being perfectly preserved.
