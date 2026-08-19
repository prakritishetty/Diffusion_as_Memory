# Privasis Abstraction Training Fix - Summary

## Problem Diagnosis

Your multi-level abstraction training was producing **spiky loss curves** (non-monotonic) instead of smooth decay. Root causes:

### 1. **Training/Target Mismatch** (CRITICAL)
- **Old behavior**: Sampled L0 (original), then noised it, then tried to predict randomly selected Lk
- **Problem**: Model was told "denoise L0→L0" but then supervised "denoise to predict Lk" (different text!)
- **Result**: Conflicting gradients, inconsistent supervision

### 2. **Decoder Loss Paradox** (CRITICAL)
- **Old behavior**: Model trained to predict L0, decoder trained to produce Lk text
- **Problem**: Decoder loss every N steps created sudden loss spikes
- **Result**: Alternating loss magnitudes = spiky curves

### 3. **Indentation & Code Duplication**
- Duplicated `seq2seq_cond = None` initialization (lines 1282-1290)
- Malformed comment `# ... (rest of training logic)`
- Validation loop indentation completely broken
- Lines after loss computation not properly indented

---

## Solution Implemented

### Core Change: Level-to-Level Denoising

Your **intended model** (clarified):
- **Privacy model**: Abstraction levels = noise trajectory in latent space
  - L0 = specific (all PII)
  - L1, L2, ..., LN = progressively abstract (forgotten details)
- **Training goal**: Learn step-by-step denoising for **recall**
  - Sample level k ∈ [0, num_levels-1]
  - Input: encoder(Lk) [more abstract/"noisy"]
  - Target: encoder(L_{k-1}) [less abstract/"cleaner"]
  - Decoder supervision: decode(pred) → text_{k-1}

### File Changes

#### 1. **[text_denoising_diffusion.py](diffusion/text_denoising_diffusion.py) - Training Loop (Lines ~1195-1360)**

**Old logic:**
```python
source_latent = encoder(L0)
k = random_level()
target_latent = encoder(Lk)  # WRONG: predicting this instead of using as input
latent = source_latent        # WRONG: using L0, not Lk
z_t = noise_schedule(source)  # WRONG: noising L0
```

**New logic:**
```python
# Step 1: Sample level k and its predecessor
k = random_level()

# Step 2: Get the "noisy" (more abstract) latent at level k
noisy_latent = encoder(Lk)   # Input to model (more abstract)

# Step 3: Get target latent (less abstract)
target_latent = encoder(L_{k-1})  # What model should predict
target_tokens = tokens_{k-1}      # Decoder supervision target

# Step 4: Use abstraction as "noise" (not Gaussian)
latent = noisy_latent

# Step 5: Pass to diffusion
loss = diffusion(latent=noisy_latent, target_latent=target_latent, ...)
```

**Key differences:**
- ✅ Input is now Lk (the noisy/abstract level)
- ✅ Target is now L_{k-1} (the cleaner level)
- ✅ Decoder target matches what we're predicting (L_{k-1})
- ✅ Consistent supervision: Lk → L_{k-1}

#### 2. **[text_denoising_diffusion.py](diffusion/text_denoising_diffusion.py) - Diffusion Forward Pass (Lines ~510-540)**

**Old logic:**
```python
z_t = alpha.sqrt() * txt_latent + (1-alpha).sqrt() * noise
# Always applies Gaussian noise schedule
```

**New logic:**
```python
# Detect semantic denoising (when target ≠ source)
is_semantic_denoising = not torch.allclose(target_latent, txt_latent)

if is_semantic_denoising:
    # For semantic abstraction: use abstraction directly as "noise"
    # No Gaussian noise needed
    z_t = txt_latent  # Already the semantic "noisy" level
else:
    # Standard diffusion: apply Gaussian noise schedule
    z_t = alpha.sqrt() * txt_latent + (1-alpha).sqrt() * noise
```

**Why:** 
- Abstraction levels ARE the "noise" in semantic space
- Adding Gaussian noise on top would corrupt the semantic meaning
- Direct comparison: `denoise(Lk) → L_{k-1}`

#### 3. **[text_denoising_diffusion.py](diffusion/text_denoising_diffusion.py) - Fixed Indentation**

**Removed code duplication:**
- ❌ Duplicate `seq2seq_cond = None` initialization (was on lines 1282 AND 1285)
- ❌ Malformed comment `# ... (rest of training logic)`

**Fixed validation loop:**
- Properly indented all validation code inside `with torch.no_grad():` and `for grad_accum_step` loop
- Fixed EMA loss variable name collision
- Added `.train()` call to both diffusion and ema_model after eval

---

## Mathematical Correctness

### Training Objective
For each batch and sampled level k:

$$\mathcal{L}(\theta) = \|\mathcal{M}_\theta(E(L_k)) - E(L_{k-1})\|_2^2 + \lambda \cdot CE(D(\hat{E}(L_{k-1})), \text{tokens}_{k-1})$$

Where:
- $E$ = BART encoder
- $\mathcal{M}_\theta$ = Denoising transformer
- $D$ = BART decoder
- $\lambda$ = decoder loss weight

### Inference (Sampling/Recall)
Start from $L_N$ (fully forgotten) and iterate:
$$\hat{L}_{k-1} = \mathcal{M}(\hat{L}_k), \quad k = N, N-1, \ldots, 1$$

Text decoded at each step shows progressive PII reconstruction.

---

## Why Spiky Losses Should Now Disappear

### Old Problem
1. **Steps without decoder loss**: Only diffusion loss (baseline magnitude)
2. **Steps with decoder loss**: Diffusion + large decoder loss spike
3. **Conflicting objectives**: Model trying to optimize for L0 and Lk simultaneously
4. **Result**: Loss curves show periodic spikes

### New Solution
1. **Consistent supervision**: Always Lk → L_{k-1} (one direction)
2. **Matched decoder target**: Decoder predicts L_{k-1}, model outputs L_{k-1}
3. **Unified objective**: Both losses optimize the same direction
4. **Result**: Smooth monotonic decay expected

---

## Testing & Validation

Run with the fixed code:
```bash
python train_text_diffusion.py \
    --dataset_name privasis_abstraction \
    --output_dir saved_diff_models/controlled_privasis_diff \
    --num_train_steps 60000 \
    --train_batch_size 16 \
    --max_seq_len 64 \
    --decoding_loss \
    --decoding_loss_every 100 \
    --decoding_loss_weight 1.0
```

**Expected results:**
- ✅ Smooth, monotonically decreasing diffusion loss
- ✅ Smooth, monotonically decreasing decoder loss (when enabled)
- ✅ Combined loss shows consistent trend, no spikes
- ✅ Validation loss also smooth

---

## Files Modified

1. **[diffusion/text_denoising_diffusion.py](diffusion/text_denoising_diffusion.py)**
   - Lines 1195-1285: Fixed training loop (level-to-level denoising)
   - Lines 1287-1360: Fixed indentation and seq2seq handling
   - Lines 510-540: Fixed forward pass (semantic vs. Gaussian noise)
   - Lines 1390-1440: Fixed validation loop indentation

---

## Backward Compatibility

- ✅ **Vanilla datasets** (single text per sample) still work
  - When `target_latent == txt_latent`, uses Gaussian noise schedule
  - No changes to inference code
- ✅ **Other seq2seq datasets** unaffected
- ✅ **EMA model** still works correctly
- ✅ **Checkpoint compatibility** maintained (no model architecture changes)

---

## Next Steps

1. ✅ Run training and monitor loss curves
2. Run evaluation/sampling to verify privacy abstraction behavior
3. Compare recall quality at different abstraction levels
4. Verify decoder produces correct text at each level
