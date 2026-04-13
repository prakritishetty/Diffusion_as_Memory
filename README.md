# Diffusion as Memory

Modelling memory in an SNR landscape to convey 'forgetting' by an LLM. Text is encoded into compact latent representations (a semantic anchor `u` and a detail latent `v0`). A diffusion denoiser then learns to recover stored memories from noisy latents. A semantic projection module (G_psi) maps denoised latents to time-appropriate reconstructions, modelling progressive forgetting.

---

## Three-Phase Training

### Phase 0 (P0) — Autoencoder

Train the full encoding/decoding pipeline end-to-end:
- Encoder, SlotPooling, UHead, VHead, G_psi, DecoderX
- Three losses: InfoNCE + reconstruction x

| Loss | Formula | Purpose |
|------|---------|---------|
| `loss_nce` | InfoNCE(`u`, `upos`) | Contrastive loss, pull together semantically similar inputs |
| `loss_x` | CrossEntropy(DecoderX(`v0`), `x`) | Force `v0` to preserve full text detail |
| **total** | `λ_u · loss_nce + λ_x · loss_x` | Combined objective |

```bash
# Standalone P0 training
python scripts/training/training_dl_augmented.py

```

### Phase 1 (P1) — Denoiser

Freeze everything from P0. Train only the diffusion denoiser:
- Input: frozen `v0` (detail latent) and `u` (semantic anchor)
- Loss: MSE on noise prediction (`ε̂` vs `ε`)

```bash
# Train on pre-extracted latents
python scripts/training/train_on_latents.py
```

### Phase 2 (P2) — Semantic Projection (G_psi) + Decoder Fine-tuning

Train the semantic projection module G_psi and fine-tune DecoderX:
- Maps `(v̂_0, t, v_t, u)` → `ṽ_0` (time-appropriate reconstruction)
- Loss: CrossEntropy(DecoderX(`ṽ_0`), `x_t`)

```bash
python scripts/training/train_phase2.py
```

**The Full Pipeline**: `x → Encoder → SlotPooling → v0 → Forward Diffusion → vt → Denoiser Nθ(vt, t, u) → ε̂ → v̂0 → G_psi(v̂0, t, vt, u) → ṽ0 → DecoderX → x̂_t`

---

## Project Structure

```
Diffusion_as_Memory/
├── README.md
├── requirements.txt
│
├── models/
│   ├── forgetting_model.py                  # ForgettingModel — wraps P0 components
│   │
│   ├── encoder_prep/
│   │   └── encoder.py                       # TextEncoder — T5 encoder wrapper
│   │
│   ├── slot_pooling_prep/
│   │   └── slot_pooling.py                  # SlotPooling — cross-attention pooling
│   │
│   ├── uv_heads_prep/
│   │   ├── u_head.py                        # UHead — semantic anchor projection
│   │   └── v_head.py                        # VHead — detail latent projection
│   │
│   ├── decoder_prep/
│   │   ├── decoder_x.py                     # DecoderX — reconstruct text from v
│   │   └── decoder_y.py                     # DecoderY — reconstruct summary from u
│   │
│   ├── denoiser_module/
│   │   ├── config.py                        # DenoiserConfig hyperparameters
│   │   ├── denoiser.py                      # Denoiser, NoiseSchedule, forward_diffusion
│   │   └── trainer.py                       # DenoiserTrainer, LatentDataset
│   │
│   └── g_psi_module/
│       ├── g_psi_config.py                  # G_psi_config hyperparameters
│       └── semantic_projection.py           # SemanticProjectionModule (G_psi)
│
├── dataloader/
│   ├── dataloader_static.py                 # MSRDataset — static x, x+, y loader
│   ├── dataloader_augmentated.py            # MSRAugmentedDataset — with token dropout
│   └── dataloader_llm/
│       ├── msr_datamodule.py                # MSRGistDataModule — Lightning DataModule
│       ├── msr_gist_dataset.py              # MSRGistDataset — on-the-fly x+ generation
│       └── xplus_gemma.py                   # GemmaXPlusGenerator — Gemma paraphraser
│
├── scripts/
│   ├── training/
│   │   ├── training_dl_augmented.py         # P0 training with augmented dataloader
│   │   ├── train_on_latents.py              # P1 denoiser training on extracted latents
│   │   ├── train_phase2.py                  # P2 G_psi + decoder fine-tuning
│   │   ├── train_phase2_config.py           # P2 configuration constants
│   │   └── archive/
│   │       └── testing.py                   # Legacy testing script
│   │
│   └── inference/
│       ├── forgetting_model_inference.py    # P0 inference + latent extraction
│       └── denoiser_inference.py            # P1 denoiser evaluation at multiple timesteps
│
├── dataset_generation/
│   ├── generate_xt.py                       # GPT-4o progressive detail attenuation
│   ├── system_prompt.py                     # System prompts for xt generation
│   └── archive/
│       ├── summarize_xt.py                  # Gemma-based xt generation
│       └── summarize_y.py                   # Gemma-based summary generation
│
├── dataset_prep/
│   ├── get_first_100.py                     # Train/test split utility
│   ├── load_msr_data.py                     # HuggingFace MSR dataset export
│   └── msr_data.py                          # MSR dataset builder
│
├── evaluation/
│   ├── run_bert_score.py                    # BERTScore evaluation
│   └── run_uni_eval.py                      # UniEval factual consistency
│
├── utils/
│   └── training_utils.py                    # ETATracker, timing utilities
│
├── sbatch_scripts/                          # SLURM job submission scripts
│   ├── sbatch_train_p0.sh                   # P0 training job
│   ├── denoiser_train.sh                    # P1 denoiser training job
│   ├── sbatch_train_p2.sh                   # P2 training job
│   ├── sbatch_inference_p0.sh               # P0 inference job
│   ├── sbatch_inference_p1.sh               # P1 inference job
│   └── sbatch_cmds.sh                       # Common SLURM commands
│
├── checkpoints/                             # Saved model checkpoints & training logs
│
└── data/
    └── final/
        ├── train.json                       # Training data
        ├── validate.json                    # Validation data
        ├── test.json                        # Test data
        └── generated_xt_dataset_with_y.jsonl  # xt trajectories with summaries
```

---

## Component Details

### `models/encoder_prep/encoder.py` — TextEncoder

Wraps `T5EncoderModel` (encoder-only, no decoder). Takes tokenized text and returns hidden states.

| Property | Value |
|----------|-------|
| Base model | `t5-small` |
| Output | `[B, seq_len, 512]` |
| `hidden_dim_size` | 512 (from T5 config) |

**Methods:** `forward(input_ids, attention_mask) → hidden_states`

---

### `models/slot_pooling_prep/slot_pooling.py` — SlotPooling

Compresses variable-length encoder output into a fixed number of slots using cross-attention.

| Input | Output |
|-------|--------|
| `H [B, seq, 512]` | `slots [B, 8, 512]` |

**Methods:** `forward(H, attention_mask=None) → slots`

---

### `models/uv_heads_prep/u_head.py` — UHead

Produces the **semantic anchor `u`** — a compact vector summarizing the input via mean pooling + linear projection.

**Used for:** InfoNCE contrastive loss, DecoderY input, semantic conditioning in denoiser and G_psi.

**Methods:** `forward(h) → u [B, 128]`

---

### `models/uv_heads_prep/v_head.py` — VHead

Produces the **detail latent `v0`** — preserves per-slot fine-grained information.

**Used for:** DecoderX input, diffusion process input.

**Methods:** `forward(h) → v0 [B, 8, 512]`

---

### `models/decoder_prep/decoder_x.py` — DecoderX

Reconstructs original text `x` from slot-pooled latents using T5 decoder.

**Methods:** `forward(encoder_hidden_states, attention_mask, labels) → loss, logits`


---

### `models/forgetting_model.py` — ForgettingModel

Wraps all P0 components (encoder, slot pooling, u/v heads, decoders, optional G_psi) into a single `nn.Module`.

**Methods:**
- `forward(batch) → total_loss, loss_nce, loss_x` — end-to-end forward pass with loss computation
- `encode_latents(batch) → u, v0` — extract latent representations without decoding
- `info_nce_loss(u, upos, temperature=0.1) → loss` — contrastive loss

---

### `models/denoiser_module/denoiser.py` — Denoiser

Diffusion-based denoiser with 6-block Transformer architecture using AdaLN conditioning, self-attention, cross-attention to `u`, and FFN.

**Classes:**
- `NoiseSchedule(T, schedule_type)` — cosine noise schedule with `get_alpha_bar(t)`
- `TimestepEmbedding(d)` — sinusoidal timestep embeddings
- `AdaLN(d, d_cond)` — adaptive layer normalization conditioned on timestep
- `MultiHeadAttention(d, n_heads)` — multi-head attention module
- `TransformerBlock(d, n_heads, d_ff, u_dim, dropout)` — single denoiser block
- `Denoiser(config)` — full denoiser model

**Functions:**
- `forward_diffusion(v0, t, noise_schedule) → (vt, epsilon)` — add noise: `vt = √ᾱ·v0 + √(1-ᾱ)·ε`
- `one_step_estimate(vt, eps_hat, t, noise_schedule) → v0_hat` — recover: `v̂0 = (vt − √(1-ᾱ)·ε̂) / √ᾱ`

---

### `models/denoiser_module/trainer.py` — DenoiserTrainer

Training framework for the denoiser with checkpoint management and W&B logging.

**Classes:**
- `LatentDataset(latent_path, L, d)` — loads pre-extracted latents from `.pt` files
- `DenoiserTrainer(config, checkpoint_dir, use_wandb)` — full training loop with validation and checkpointing

---

### `models/g_psi_module/semantic_projection.py` — SemanticProjectionModule (G_psi)

Semantic projection module that maps denoised latents to time-appropriate reconstructions. Takes `(v̂_0, t, v_t, u)` and produces `ṽ_0`.

**Classes:**
- `SPMBlock(d, d_cond, d_ff, use_attn, n_heads)` — single projection block with FiLM conditioning
- `SemanticProjectionModule(config, no_use_u, no_use_vt)` — full G_psi network with configurable ablations

**Methods:** `forward(v_hat_0, t, v_t=None, u=None) → v_tilde_0`

---

### Dataloaders

| Module | Class | Description |
|--------|-------|-------------|
| `dataloader_static.py` | `MSRDataset` | Loads pre-tokenized `(x, x+, y)` triples from JSON |
| `dataloader_augmentated.py` | `MSRAugmentedDataset` | **Main** Adds token dropout augmentation with `drop_prob` |
| `dataloader_llm/msr_datamodule.py` | `MSRGistDataModule` | Lightning DataModule with on-the-fly x+ via Gemma |
| `dataloader_llm/msr_gist_dataset.py` | `MSRGistDataset` | Dataset with optional xt inclusion and x+ caching |
| `dataloader_llm/xplus_gemma.py` | `GemmaXPlusGenerator` | Generates paraphrased x+ using Gemma model |

---

### Training Scripts

| Script | Phase | Description |
|--------|-------|-------------|
| `scripts/training/training_dl_augmented.py` | P0 | Standalone P0 training with augmented dataloader |
| `scripts/training/train_on_latents.py` | P1 | Denoiser training on pre-extracted latent files |
| `scripts/training/train_phase2.py` | P2 | G_psi + decoder fine-tuning with xt bucket selection |
| `scripts/training/train_phase2_config.py` | P2 | Configuration constants for Phase 2 |


**Key training functions in `train_phase2.py`:**
- `build_p0_model(device)` — load frozen P0 model
- `load_denoiser(checkpoint, device)` — load frozen P1 denoiser
- `select_xt_labels(batch, t, device)` — pick target xt for given diffusion timestep
- `train_epoch(...)` / `validate_epoch(...)` — G_psi training loop

---

### Inference Scripts

| Script | Description |
|--------|-------------|
| `scripts/inference/forgetting_model_inference.py` | Run P0 model inference, extract latents, log decoded samples |
| `scripts/inference/denoiser_inference.py` | Evaluate denoiser at multiple timesteps, compute one-step MSE |

**Key functions in `forgetting_model_inference.py`:**
- `build_model(device)` — construct ForgettingModel
- `load_checkpoint(model, path, device)` — load trained weights
- `run_inference(model, dataloader, tokenizer, ...)` — batch inference with latent extraction

**Key functions in `denoiser_inference.py`:**
- `load_denoiser(checkpoint_path, device)` — load trained denoiser
- `evaluate_one_step(denoiser, noise_schedule, dataloader, timesteps, device)` — MSE at each timestep
- `collect_samples(...)` — gather denoised samples for visualization

---

### Dataset Generation

| Script | Description |
|--------|-------------|
| `dataset_generation/generate_xt.py` | Uses GPT-4o to generate progressive forgetting trajectories (xt sequences) |
| `dataset_generation/system_prompt.py` | System prompts (`SYSTEM_PROMPT`, `SYSTEM_PROMPT_V2`) guiding xt generation |
| `dataset_generation/archive/summarize_xt.py` | Gemma-based xt generation with checkpointing |
| `dataset_generation/archive/summarize_y.py` | Gemma-based summary (y) generation |

---

### Evaluation

| Script | Function | Description |
|--------|----------|-------------|
| `evaluation/run_bert_score.py` | `compute_bert_score(src_list, output_list)` | BERTScore precision/recall/F1 |
| `evaluation/run_uni_eval.py` | `evaluate_factual_consistency(src_list, output_list)` | UniEval factual consistency |

---

### Utilities

| Module | Class/Function | Description |
|--------|---------------|-------------|
| `utils/training_utils.py` | `ETATracker(total_epochs)` | Tracks epoch timing, computes ETA, generates W&B metrics |

---

## Key Dimensions

| Tensor | Shape | Description |
|--------|-------|-------------|
| `H` | `[B, 64, 512]` | T5 encoder hidden states |
| `slots` | `[B, 8, 512]` | Slot-pooled representation |
| `u` | `[B, 128]` | Semantic anchor |
| `upos` | `[B, 128]` | Positive pair semantic anchor |
| `v0` | `[B, 8, 512]` | Detail latent |
| `vt` | `[B, 8, 512]` | Noisy latent at timestep t |
| `v_tilde_0` | `[B, 8, 512]` | G_psi projected latent |

---

## Data Format

Training data is stored in `data/final/` as JSON files. Each sample contains:
- `x` — original source text
- `x+` — paraphrased positive pair (for contrastive learning)
- `y` — semantic summary/gist
- `xt` — progressive forgetting trajectory (list of increasingly abstracted texts, used in P2)

---

## Requirements

See `requirements.txt` for full list.

Before running experiments with Weights & Biases enabled, authenticate with W&B:

```bash
wandb login
```

Or set your API key:

```bash
export WANDB_API_KEY=your_api_key_here
```

---

## SLURM Jobs

Submit training/inference jobs on the cluster:

```bash
sbatch sbatch_scripts/sbatch_train_p0.sh        # Phase 0 training
sbatch sbatch_scripts/denoiser_train.sh          # Phase 1 denoiser training
sbatch sbatch_scripts/sbatch_train_p2.sh         # Phase 2 G_psi training
sbatch sbatch_scripts/sbatch_inference_p0.sh     # Phase 0 inference
sbatch sbatch_scripts/sbatch_inference_p1.sh     # Phase 1 inference
```
