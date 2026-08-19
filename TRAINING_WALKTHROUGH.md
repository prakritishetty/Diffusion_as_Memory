# Training Walkthrough: Step-by-Step Diffusion Model Training

## Executive Summary
The training code implements a **Latent Diffusion Model** for text generation with privacy-preserving features using the **Privasis Abstraction Dataset**. The issue with WandB logging has been identified and documented below.

---

## 1. Training Entry Point: `train_text_diffusion.py`

### Flow:
```
main(args)
  ├─ Create DiffusionTransformer model
  ├─ Wrap in GaussianDiffusion 
  └─ Create Trainer and call trainer.train()
```

### Key Components:
- **DiffusionTransformer**: The neural network that learns to predict noise at different diffusion timesteps
- **GaussianDiffusion**: Handles the diffusion process (forward & reverse)
- **Trainer**: Orchestrates data loading, optimization, validation, and logging

---

## 2. The Privasis Abstraction Dataset

### Understanding the Data Structure:
The dataset has **multi-level abstractions** for each sample:

```
Original Input: "The patient takes medication X daily due to disease Y"
    ↓
L0 (Most Specific): "The patient takes medication X daily due to disease Y"
    ↓
L1 (Abstract): "The patient takes a specific medication due to a disease"
    ↓
L2 (Abstract): "The patient takes medication due to a health issue"
    ↓
L3 (Most Abstract): "A person takes treatment for health"
```

**Data shape in training:**
```python
data['input_ids']:          [B, max_levels, max_seq_len]
data['attention_mask']:      [B, max_levels, max_seq_len]
data['num_levels']:          [B]  # Actual number of levels per example
```

---

## 3. The Training Loop: Inside `Trainer.train()`

### Step-by-Step Process:

#### **Step 1: Initialize Encoder Latents** (Line ~1210-1290)
For each batch during training:

```python
# Get the CLEAN reference latent (L0)
source_input_ids = data['input_ids'][:, 0, :]  # L0
encoder_outputs_0 = self.bart_model.get_encoder()(...)
source_latent = get_diffusion_latent(encoder_outputs_0)  # Shape: [B, latent_dim]

# Sample a random noise level for the diffusion process
times = torch.rand(B)  # Range: [0, 1]

# Map random time to actual abstraction level
k = (times * (num_levels - 1)).long()  # Which abstraction level to start from
k = torch.min(torch.max(k, 0), num_levels - 1)  # Clamp k

# Get the NOISY latent (at level k - more abstract)
noisy_input_ids = data['input_ids'][:, k, :]
noisy_latent = get_diffusion_latent(encoder(noisy_input_ids))

# Get the TARGET latent (at level k-1 - closer to original)
target_k = torch.clamp(k - 1, min=0)
target_input_ids = data['input_ids'][:, target_k, :]
target_latent = get_diffusion_latent(encoder(target_input_ids))
```

**Key Insight**: The diffusion model learns to map from abstract representations (noisy) back to more specific ones (targets).

---

#### **Step 2: Compute Diffusion Loss** (Line ~1290-1350)

```python
# Normalize latents
noisy_latent_normalized = normalize_latent(noisy_latent)
target_latent_normalized = normalize_latent(target_latent)

# Forward diffusion: add noise based on times
# This simulates the diffusion process at different levels of abstraction
noise = torch.randn_like(target_latent)
alpha_t = alpha_schedule(times)  # Noise level at time t
z_t = alpha_t.sqrt() * target_latent + (1 - alpha_t).sqrt() * noise

# Model prediction: predict the target from noisy
model_out = diffusion_model(z_t, mask, time_cond=alpha_t)

# Loss: L1 or L2 between predicted target and actual target
diffusion_loss = loss_fn(model_out, target_latent)
```

---

#### **Step 3: Optional Decoding Loss** (Line ~1350-1360)

If decoding loss is enabled:
```python
# Generate text from predicted latent and compare to target text
predicted_tokens = decode(model_out)
target_tokens = decode(target_latent)
decoding_loss = seq2seq_loss(predicted_tokens, target_tokens)

total_loss = diffusion_loss + decoding_loss_weight * decoding_loss
```

---

#### **Step 4: Gradient Accumulation** (Line ~1361-1375)

```python
# For gradient_accumulate_every steps, accumulate gradients
total_loss += loss / gradient_accumulate_every
accelerator.backward(loss)

# After accumulation, update weights
if step % gradient_accumulate_every == 0:
    clip_grad_norm(model.parameters(), clip_value)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    step += 1
```

---

#### **Step 5: Statistics Collection** (Warm-up Phase)

```python
# Track statistics for normalization
if stats_count < 100:
    latent_mean += latent.mean(dim=0)
    latent_std += latent.std(dim=0)
    stats_count += 1
    pbar.set_description(f"Warm-up Stats: {stats_count}/100")
```

---

#### **Step 6: Logging Training Metrics** (Line ~1380-1445)

```python
# ISSUE: This only logs when is_main_process=True
if is_main_process:
    if stats_count < 100:
        # Warm-up phase: only show progress bar
        pbar.update(0)
    else:
        # After warm-up: create logs dictionary
        logs = {
            "loss": total_loss,
            "diffusion_loss": diffusion_loss_val,
            "decoding_loss": decoding_loss_val,
            "learning_rate": scheduler.get_last_lr()[0],
            "grad_norm": computed_grad_norm,
            "step": step,
            "epoch": (step * gradient_accum) / len(dataloader),
        }
```

---

#### **Step 7: Validation Loop** (Line ~1410-1440)

Every 50 steps:
```python
with torch.no_grad():
    diffusion.eval()
    
    # Compute validation loss on validation set
    for val_batch in val_dataloader:
        val_latent_noisy = encode(val_batch_abstract)
        val_latent_target = encode(val_batch_specific)
        
        val_loss = diffusion(val_latent_noisy, mask)
        val_ema_loss = ema_model(val_latent_noisy, mask)
        
        logs["val_loss"] = total_val_loss
        logs["val_ema_loss"] = total_val_ema_loss
    
    diffusion.train()
```

---

#### **Step 8: WandB Logging** (Line ~1445)

```python
# THIS IS THE PROBLEM!
accelerator.log(logs, step=self.step)
```

**THE ISSUE**: `logs` is only non-empty when:
1. `stats_count >= 100` (warm-up phase complete)
2. AND `step % 50 == 0` (validation loop ran)

If either condition is false, `logs` is an empty dict! So training metrics aren't logged.

---

## 4. **ROOT CAUSE OF MISSING LOSS CURVES**

### Current Logic (Broken):
```
logs = {}  # Initialize empty

if is_main_process:
    if stats_count < 100:
        pbar.update(0)
        # logs stays empty ❌
    else:
        logs = {...training metrics...}  # Only populated here
    
    if step % 50 == 0:
        # Add validation metrics to logs
        logs["val_loss"] = ...
    
    # Logging happens ALWAYS, but logs might be empty or incomplete
    accelerator.log(logs, step=self.step)  # ❌ PROBLEM
```

### Why Nothing Shows on WandB:
1. **During warmup (steps 0-100)**: `logs` is empty dict
2. **Between validation intervals (e.g., step 1-49)**: `logs` contains only training metrics (good)
3. **BUT**: Validation metrics only added every 50 steps
4. **If not is_main_process**: Nothing is logged at all!

---

## 5. How to Fix the Logging

### Recommended Fix:

```python
# Initialize logs with BOTH training and validation metrics
logs = {}

if accelerator.is_main_process:
    # ALWAYS log training metrics when stats_count >= 100
    if self.diffusion.stats_count >= 100:
        logs = {
            "loss": total_loss,
            "diffusion_loss": diffusion_loss_val,
            "decoding_loss": decoding_loss_val,
            "learning_rate": self.lr_scheduler.get_last_lr()[0],
            "grad_norm": grad_norm if 'grad_norm' in locals() else 0,
            "step": self.step,
            "epoch": (self.step*self.gradient_accumulate_every)/len(self.dataloader),
            "samples": self.step*self.train_batch_size*self.gradient_accumulate_every*self.num_devices
        }
    else:
        # During warmup, log minimal info
        logs = {
            "warmup_step": int(self.diffusion.stats_count.item()),
            "step": self.step
        }
    
    # Validation metrics (separate step, don't reset logs)
    if self.step % 50 == 0:
        self.diffusion.eval()
        self.ema.ema_model.eval()
        with torch.no_grad():
            total_val_loss = 0.
            total_val_ema_loss = 0.
            for grad_accum_step in range(self.gradient_accumulate_every):
                # ... validation code ...
                
            logs["val_loss"] = total_val_loss
            logs["val_ema_loss"] = total_val_ema_loss
            pbar.set_postfix(**logs)
        
        self.diffusion.train()
        self.ema.ema_model.train()

# Always log (but logs might be empty during warmup, which is fine)
accelerator.log(logs, step=self.step)
```

---

## 6. Key Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `gradient_accumulation_steps` | 1-4 | Simulate larger batch size |
| `ema_decay` | 0.9999 | Exponential moving average for model smoothing |
| `ema_update_every` | 1 | Update EMA every N steps |
| `save_and_sample_every` | 5000 | Generate samples and save checkpoint |
| `train_schedule` | 'cosine' | Noise schedule (how alpha_t varies with t) |
| `sampling_schedule` | (optional) | Can differ from training schedule |
| `train_prob_self_cond` | 0.5 | 50% of time, feed previous prediction as input |

---

## 7. Validation Metrics Explained

### `val_loss`: Main diffusion model validation loss
- Measures how well the model predicts target latents from noisy ones
- Should decrease as training progresses

### `val_ema_loss`: EMA model validation loss  
- EMA is a smoothed version of the main model
- Often generalizes better than main model
- Should be lower than `val_loss` once EMA catches up

---

## 8. The EMA (Exponential Moving Average) Model

```python
ema_model = EMA(model, decay=0.9999)

# Each step:
ema_model.update()  # ema_weights = decay * ema_weights + (1-decay) * current_weights
```

**Why use EMA?**
- Smoother, more stable model for better generalization
- Used for validation and sampling
- Weights from entire training history (with exponential decay)

---

## 9. Full Training Workflow Visualization

```
Initialize Diffusion Model & Trainer
         ↓
Load Dataset (Privasis Abstractions)
         ↓
WARM-UP PHASE (step 0-100):
  - Collect latent statistics
  - No weight updates
  - No logging to WandB
         ↓
TRAINING PHASE (step 100+):
  ┌─ For each batch:
  │  ├─ Sample random level k from {0, 1, ..., num_levels-1}
  │  ├─ Get noisy latent at level k
  │  ├─ Get target latent at level k-1
  │  ├─ Compute diffusion loss
  │  ├─ Backward pass & accumulate gradients
  │  └─ Every N steps: update weights & EMA
  │
  ├─ Every 50 steps:
  │  ├─ Run validation loop
  │  ├─ Compute val_loss & val_ema_loss
  │  └─ Log to WandB ← THE BUG IS HERE
  │
  ├─ Every 5000 steps:
  │  ├─ Save checkpoint
  │  └─ Generate samples (if not no_validation)
  │
  └─ Repeat until step >= train_num_steps
         ↓
Save final model
```

---

## 10. Next Steps

1. **Fix the logging** (see Section 5 above)
2. **Verify WandB metrics** appear in your dashboard
3. **Monitor these curves**:
   - `loss`: Should decrease smoothly
   - `val_loss` & `val_ema_loss`: Should eventually be lower than training loss
   - `learning_rate`: Should follow the schedule (cosine decay)

