# Forgetting & Recall Table Logging

## Overview

During training validation, the code now logs **visual tables to WandB** that show how the diffusion model learns to:
1. **Forget** (progressively abstract) original text by adding noise
2. **Recall** (progressively recover) specific text by iteratively denoising

This provides intuitive visualization of privacy leakage during the training process.

---

## What Gets Logged

For **each sample** (only for `privasis_abstraction` dataset), a WandB table with 3 rows:

| Original Text | Process | T=0.0 | T=0.25 | T=0.50 | T=0.75 | T=1.0 |
|---|---|---|---|---|---|---|
| "The patient takes medication X daily due to disease Y" | 📉 Forgetting (Progressive Noising) | The patient takes medication X daily due to disease Y | The patient takes some medication | A person takes treatment | A person takes something | Something happens |
| "The patient takes medication X daily due to disease Y" | 🚀 Recall (Iterative Denoising) | The patient takes medication X daily due to disease Y | The patient takes medication for a condition | The patient takes medicine due to health | A person takes treatment for health | Something happens |

### Explanation

**Row 1 (Original)**: Ground truth text at abstraction level 0 (most specific)

**Row 2 (Forgetting - 📉)**:
- Shows what happens when we progressively **add noise** to the clean latent
- At T=0: no noise added → clean latent → should decode to original
- At T=0.5: half noise added → mixed signal → becomes more abstract
- At T=1.0: full noise added → pure noise → completely abstract/unintelligible

**Row 3 (Recall - 🚀)**:
- Shows **iterative denoising** starting from the same noisy latent at each T
- NOT a one-step prediction, but the full multi-step diffusion reverse process
- At T=0: no denoising needed → should recover original
- At T=0.5: model must denoise from halfway point → partial recovery
- At T=1.0: model must denoise from pure noise → harder recovery task

---

## When Is This Logged?

Every `save_and_sample_every` steps (default: 5000 steps)

**Only for `privasis_abstraction` dataset** - determines whether multi-level abstractions are available

---

## How It Works

### Forgetting Process (Adding Noise)

```python
# For each timestep t ∈ [0, 0.25, 0.5, 0.75, 1.0]:
alpha_t = noise_schedule(t)  # How much signal vs noise
noise = randn_like(z0)
z_t_noisy = alpha_t.sqrt() * z0 + (1-alpha_t).sqrt() * noise

# Decode and display the noisy latent
text_noisy = decode(z_t_noisy)
```

**Key insight**: As t increases, the clean signal is "forgotten" and replaced with noise. The decoded text becomes increasingly abstract.

---

### Recall Process (Iterative Denoising)

```python
# For each timestep t ∈ [0, 0.25, 0.5, 0.75, 1.0]:
z_t = alpha_t.sqrt() * z0 + (1-alpha_t).sqrt() * noise  # Start from same noisy z_t as forgetting

# Iteratively denoise from z_t back to z_0
num_steps = int(t * sampling_timesteps)
for timestep in linspace(t, 0, num_steps):
    z_next = diffusion_model(z_t, timestep)  # Single denoising step
    z_t = z_next

# Decode the fully denoised latent
text_recall = decode(z_t)
```

**Key insight**: The model attempts to recover the original text by iteratively predicting what the cleaner version should be. This reveals how much information leaks through the diffusion process.

---

## Interpreting the Results

### Good Model (Privacy Preserved)

```
T=0.0:  "The patient takes medication X daily due to disease Y"
T=1.0:  "Something happens"
```

At T=1.0 (maximum noise):
- **Forgetting row**: Completely abstract/random → privacy preserved ✓
- **Recall row**: Cannot recover original details → model cannot extract info from noise ✓

### Bad Model (Privacy Leaks)

```
T=0.0:  "The patient takes medication X daily due to disease Y"
T=1.0:  "The patient takes medication X daily due to disease Y"  ← LEAK!
```

If the recall row at T=1.0 is too similar to original:
- Model learned to memorize/overfit to original
- Indicates privacy leakage ⚠️

---

## Code Details

### Location

Function: `Trainer.log_forgetting_recall_table()`

File: [diffusion/text_denoising_diffusion.py](diffusion/text_denoising_diffusion.py#L850)

Called from training loop at: [Line ~1640](diffusion/text_denoising_diffusion.py#L1640)

### Parameters

```python
def log_forgetting_recall_table(
    self, 
    num_samples=5,        # How many examples to log
    num_timesteps=6       # How many timestep columns [0, 0.2, 0.4, ..., 1.0]
):
```

### Default Settings (in training loop)

```python
self.log_forgetting_recall_table(num_samples=3, num_timesteps=5)
# Logs 3 example samples with timesteps at T=[0, 0.25, 0.5, 0.75, 1.0]
```

---

## Adjusting the Logging

### More Detailed Trajectories

```python
# In training loop, change:
self.log_forgetting_recall_table(num_samples=5, num_timesteps=11)
# Now shows T=[0, 0.1, 0.2, ..., 1.0] (11 timesteps)
```

### More Samples Per Log

```python
self.log_forgetting_recall_table(num_samples=10, num_timesteps=5)
# Shows 10 examples instead of 3
```

### Different Logging Frequency

```python
# In training loop, wrap with condition:
if self.step % (self.save_and_sample_every * 2) == 0:
    self.log_forgetting_recall_table()
# Now logs every 10,000 steps instead of 5,000
```

---

## Relationship to Privasis Dataset

The Privasis Abstraction dataset has **11 abstraction levels** (L0...L10):

```
L0 (Specific):  "The patient takes medication X daily due to disease Y"
L1 (Abstract):  "The patient takes a specific medication due to a disease"
L2 (Abstract):  "The patient takes medication due to a health issue"
...
L10 (Generic):  "A person takes treatment"
```

The forgetting/recall logging simulates:
- **Forgetting**: Progressive movement from L0 → L10 (adding noise)
- **Recall**: Model's ability to recover from partial abstraction back to L0

This directly tests the model's ability to:
1. Learn privacy-preserving representations (forgetting)
2. Recover information for inference tasks (recall)

---

## WandB Dashboard Viewing

### Finding the Tables

On WandB dashboard:
1. Go to your run
2. Look for tabs: `privasis/forgetting_recall_step{N}`
3. Each table shows one validation checkpoint

### Interpretation Tips

- **Horizontal consistency**: Text should gradually change across timesteps
- **Semantic coherence**: Forgetting text should still be syntactically valid
- **No privacy leaks**: T=1.0 recall should be very different from original
- **Training progress**: Later validation steps should show better recovery at T=0

---

## Performance Considerations

- **GPU memory**: Decoding 3 samples × 5 timesteps × 2 processes = 30 text generations per validation
- **Time**: ~30-60 seconds per validation checkpoint
- **Frequency**: Every 5000 steps = minimal overhead

If you're running tight on memory, reduce `num_samples`:

```python
self.log_forgetting_recall_table(num_samples=1, num_timesteps=5)
```

---

## Standalone Inference Script

For post-training analysis (separate from training loop):

```bash
python inference_forget_recall.py \
    --latent_dir saved_latent_models/privasis_autoencoder \
    --diffusion_dir saved_diff_models/controlled_privasis_diff \
    --num_samples 10 \
    --sampling_timesteps 50
```

See [inference_forget_recall.py](inference_forget_recall.py) for details.

