# Baseline for Latent PII Forgetting & Recall

This implementation plan outlines the steps to build an experimental baseline leveraging the innate continuous properties of Latent Diffusion to observe if PIIs are "forgotten" and "recalled" organically during the noise/denoise cycle.

## Goal Description

We will utilize the `nvidia/Privasis-Zero` dataset. We will train the baseline Autoencoder and Diffusion models on this dataset unconditionally. Then, we will create a custom inference/evaluation script to run samples through forward noising and reverse denoising, extracting and decoding the intermediate continuous latents at various timesteps to observe text degradation and reconstruction order. We will evaluate this on both the train split (to check overfitting) and the test split, tracking all progressive texts in Weights & Biases (WandB).

## Proposed Changes

---

### Dataset Processing

We will integrate the Privasis dataset directly into the existing HuggingFace dataset loader routing, ensuring a pristine train-val-test split.

#### [MODIFY] [text_dataset.py](file:///e:/UMass/SPRING%202026/698DS/latent-diffusion-for-language/dataset_utils/text_dataset.py)
- Update `get_dataset` to route `dataset_name == 'privasis'` to a new `process_privasis_dataset` function.
- Create `process_privasis_dataset()` which pulls `nvidia/Privasis-Zero` off huggingface.
- Extract the raw string while discarding PII entity annotations. 
- Implement a robust Train-Val-Test split methodology on the dataset object so that we can evaluate on held-out test data.

---

### Training Configuration

We will author bash scripts to automate the execution of the training pipeline. Both will utilize the native `--wandb_name` argument to track loss and run metrics on Weights & Biases.

#### [NEW] [bart_privasis.sh](file:///e:/UMass/SPRING%202026/698DS/latent-diffusion-for-language/scripts/autoencoder/bart_privasis.sh)
- Configuration to run `train_latent_model.py` with `dataset_name="privasis"`.
- Uses a `facebook/bart-base` encoder/decoder, 32 latents (default), and saves to `saved_latent_models/privasis_autoencoder`.

#### [NEW] [bart_latent_privasis.sh](file:///e:/UMass/SPRING%202026/698DS/latent-diffusion-for-language/scripts/diffusion/bart_latent_privasis.sh)
- Configuration to run `train_text_diffusion.py` unconditionally (`seq2seq=False`).
- Points to the `privasis_autoencoder` checkpoint.

---

### Custom Inference Scripts

The core value of this experiment is inspecting the continuous intermediate representation block and logging the text decoded at each timestamp directly into WandB tables for easy comparison.

#### [NEW] [inference_forget_recall.py](file:///e:/UMass/SPRING%202026/698DS/latent-diffusion-for-language/inference_forget_recall.py)
This file will load both models and contain parameters explicitly for `--split train` and `--split test`.
1. **`simulate_forgetting(val_or_test_batch, timesteps=[10, 25, 50, 100])`**: 
   - Encodes a batch of clean text to $X_0$.
   - Applies the `q_sample(x_start=X_0, t=timestep)` method to simulate varying depths of Gaussian corruption.
   - Decodes those intermediate $X_t$ vectors back into text and **logs the progressive degradation of the sentence at each step**.
2. **`simulate_recall(timesteps=[100, 75, 50, 25, 10, 0])`**:
   - Uses the denoising loop (`p_sample_loop`), but intercepts the $X_t$ array at progressive timestamps as it approaches $T=0$.
   - **Yes, it will decode the text progressively during recall**, extracting what text corresponds to $T=100$, $T=50$, $T=10$, etc.
3. **WandB Logging**: Uses `wandb.Table` to construct unified tables of `[Original Text, T=10 Text, T=50 Text, T=100 Text]` and uploads them to the centralized wandb dashboard for visual inspection.

## Verification Plan

### Automated Tests
- N/A

### Manual Verification
1. Run the Autoencoder script on the Privasis train set and ensure the discrete text-level Cross-Entropy Loss stabilizes and is logged to WandB.
2. Run the Diffusion script to ensure the continuous latent MSE Loss stabilizes and is logged to WandB.
3. **Train Split Check (Overfit baseline):** Run `inference_forget_recall.py` on `--split train` to establish a baseline of how the forgetting/recall behaves when the model has already memorized the data. Verify WandB tables populate correctly.
4. **Test Split Check (Generalization baseline):** Run `inference_forget_recall.py` on `--split test` to evaluate how the system handles unseen PII/syntax dynamics. Verify WandB tables populate correctly.
