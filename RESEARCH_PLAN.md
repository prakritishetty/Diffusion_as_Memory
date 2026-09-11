# Diffusion-as-Memory — Plan v3 (brainstorm + implementation spec)

> **STATUS: IMPLEMENTED.** The plan below was written first and argued through; the code now
> exists. Read §0a for what the implementation *changed about the plan* — three decisions were
> reversed by measurement. Everything else was built as specified.

Grounded on: `models/`, `dataloader/`, `training/`, `evaluation/`, `data/final/*.json`,
and `paper/extended_abstract_v2.tex`.

**Measured dataset facts** (computed from the real data, use these instead of guesses):

| split | records | xt levels min/mean/max | modal level count | x words mean/p95/max | has `y` |
|---|---|---|---|---|---|
| train | 3001 | 3 / 4.88 / 12 | 5 (68% of records) | 36.8 / 56 / 135 | 3001 |
| validate | 1001 | 4 / 4.97 / 10 | 5 (80%) | 37.3 / 56 / 118 | 1001 |
| test | 933 | 1 / 4.85 / 8 | 5 (73%) | 37.2 / 57 / 145 | 933 |

---

## 0a. What measurement changed about this plan

Three things in the plan below turned out to be wrong once the numbers came in. They are corrected
in the code; the original reasoning is left in place so the change is auditable.

### 1. The rung map is `uniform_t`, not log-SNR

§1a argues for defining the timestep→rung map on log-SNR because "halfway through `t` is nowhere
near half the information destroyed". Measured, the cosine schedule is far better behaved than
that implies, and log-SNR is far worse:

| map | rung shares of the schedule (n=6) | top rung |
|---|---|---|
| `bucket` (original) | — | 51% of all timesteps |
| `uniform_t` (**now default**) | 0.167 each, exactly | 16.7% |
| `logsnr`, unclipped | 0.034 / 0.236 / 0.533 / 0.167 / 0.025 / **0.005** | 5 timesteps out of 1000 |

log-SNR spans +9.2 to −13.8, so equal slices of it are dominated by the tails. Equal spacing in
`t` gives per-rung signal fractions `sqrt(ᾱ)` of 1.00 → 0.96 → 0.86 → 0.70 → 0.50 → 0.26 → 0.00 —
every rung a visibly different noise level, which is all the map needs to do. `logsnr` survives as
an ablation arm with a clipped range (`logsnr_clip=6.0`), and `bucket` reproduces the original
defect exactly.

### 2. Nesting alone is not the moat — truncation beats us at it

§10 proposes "graded, nested access" as the headline claim. Two measurements kill that framing:

- **The gold ladders are only 5% nested.** They are paraphrases, not deletions: "dismantled and
  reinstalled at COPIA" → "moved to a museum" introduces new words every rung. No model trained on
  them can exceed that ceiling.
- **Prefix truncation scores 0.73 on nesting and 0.93 on monotonicity.** Cutting a string in half
  trivially adds no information. Scissors win.

What truncation *cannot* do is produce a standalone sentence: it is well formed 8% of the time.
So the claim is not "monotone and nested" — it is **complete at every budget**. The composite
metric `usable_rung_rate` (well formed AND adds nothing specific AND invents nothing) is now the
headline, and it separates cleanly:

| | monotonic | nested | hallucination | well-formed | **usable** |
|---|---|---|---|---|---|
| Gold, LLM-written | 0.92 | 0.05 | 0.24 | 0.86 | **0.33** |
| Gold, rule-based control | 1.00 | 0.03 | 0.09 | 0.99 | **0.56** |
| Truncation, prefix | 0.93 | 0.73 | 0.02 | 0.08 | **0.08** |

### 3. Gate 1 result: it is a modelling problem

The ceiling on the natural ladders is **0.92** monotonicity against the reported **0.325**. The
data is not the bottleneck, so the answer to "regenerate or train?" is train. The control corpus
(§1c-iii) is still worth having, but as the corpus where the consistency property can actually be
demonstrated — not as a replacement.

**One more thing worth knowing:** per-type decay on the natural ladders already shows human-like
forgetting. At the first rung, numbers retain 0.26 and dates 0.28, while people retain 0.53,
places 0.50 and organisations 0.54 — numerics fade roughly twice as fast as named entities. That is
§10's "human-like forgetting order" claim, measured, before any model is trained.

---

## 0. Read this first: 16 defects found in the current code


Ordered by how much they poison the numbers already in the paper. Several of these mean current
results are **not measuring what we think they measure**. I'd fix 1–4 before running anything.

| # | Defect | Where | Consequence |
|---|---|---|---|
| **1** | **Validation "predictions" are teacher-forced.** `validate_epoch` calls `decoder_x(v, mask, labels)` and `evaluate_best_epoch` decodes `argmax(logits)`. The decoder sees the gold prefix at every position. | [train_phase2.py](scripts/training/train_phase2.py), [training_utils.py](utils/training_utils.py) | **BERTScore F1 0.932 in the paper is almost certainly inflated.** Must be free-running `generate()`. |
| **2** | **Contrastive positive == anchor.** `xpos_input_ids = x_input_ids`. `_drop_tokens` is written but commented out. | [dataloader_augmentated.py](dataloader/dataloader_augmentated.py) | InfoNCE reduces to "be unique", satisfiable by any injective map. `u` learns ~nothing. |
| **3** | **`v̂₀` is not `v̂₀`.** P2 calls `step_by_step_estimate(...)` (returns `v_{t−1}`) and assigns the result to `v_hat_0`, then feeds it to G_ψ. | [train_phase2.py](scripts/training/train_phase2.py) | G_ψ is trained on a **one-step-denoised** latent but at inference would see a **fully-run chain**. Train/test mismatch. |
| **4** | **`x` is never a decoder target.** `select_xt_labels` uses `t // 100`, and `t ∈ [1, T]`, so even `t=1` maps to `xt[0]` — already an abstraction. | [training_utils.py](utils/training_utils.py) | The model can never be asked to reproduce full detail. The bottom rung of the ladder is missing. |
| **5** | Supervision collapse (already in the paper): `min(t//100, n−1)` with mean n=4.88 → 61% of timesteps hit the last rung. | same | Objective is near-binary. |
| **6** | **`max_seq_len = 64`.** x averages 36.8 words ≈ 50 T5 tokens; p95 is 56 words ≈ 78 tokens; max 135 words. | [dataloader_augmentated.py](dataloader/dataloader_augmentated.py) | Roughly a third of examples are **silently truncated**. Detail we're claiming to preserve is being cut before the encoder sees it. Flagged in `codebase_walkthrough.md` a year ago, never fixed. |
| **7** | **G_ψ self-attention is off by default** (`use_attn: bool = False`). | [g_psi_config.py](models/g_psi_module/g_psi_config.py) | G_ψ is a **per-slot pointwise MLP**. It cannot mix information across slots. This is a large capacity hole in the module whose whole job is manifold projection. |
| **8** | G_ψ dead code: `input_dim = 2*d` is computed then thrown away (`self.input_proj = nn.Linear(self.d, self.d)`); `v_hat_0` is accepted and unused when `no_use_vt=False`. | [semantic_projection.py](models/g_psi_module/semantic_projection.py) | The intended two-input design never ran. |
| **9** | `LatentDiffusionModel.forward` does `for block in blocks: h = block(vt, u, t_emb)` — **blocks don't chain**, all read `vt`. `h` is also undefined if the list is empty. | [latent_diffusion_model.py](models/latent_diffusion/latent_diffusion_model.py) | Effectively a 1-block model. (This file may just be dead — check before fixing.) |
| **10** | **Config drift.** `train_denoiser_prakriti.py` uses `L_SLOTS=8, U_DIM=128`; `train_phase2_config.py` uses `L_SLOTS=16, U_DIM=512`; `DenoiserConfig` says `L=16, u_dim=512`. | 3 files | Checkpoints silently incompatible across phases. Needs one config source. |
| **11** | NLI is called as `nli(f"{premise} [SEP] {hypothesis}")` on a `text-classification` pipeline. Cross-encoder NLI models expect a **pair** (`{"text":…, "text_pair":…}`). | [custom_metrics.py](evaluation/custom_metrics.py), [run_monotonic_loss_score.py](evaluation/run_monotonic_loss_score.py) | Entailment numbers (0.587/0.628) are suspect until verified. |
| **12** | BERTScore uses `roberta-large` + `rescale_with_baseline=True` in one file, `roberta-base` + no rescale in another. | evaluation/ vs train_phase2.py | The two sets of numbers are not comparable. |
| **13** | **No seed control on reverse-process noise.** `step_by_step_estimate` draws `z = randn_like(vt)` fresh each call. | [denoiser.py](models/denoiser_module/denoiser.py) | Trajectories are **not reproducible**; monotonicity is measured on a different noise draw every time, adding variance to the headline number. |
| **14** | `v₀` scale is never checked. Forward diffusion assumes `v₀ ~ O(1)` variance; `VHead` is a bare `Linear` with no normalization. | [v_head.py](models/uv_heads_prep/v_head.py) | If `std(v₀) ≫ 1`, then `√ᾱ·v₀` dominates `√(1−ᾱ)·ε` until very late → **nothing is forgotten, then everything is at once.** This is a strong candidate explanation for the "cliff effect" in `codebase_walkthrough.md §4b`, independent of the supervision diagnosis. **Cheapest high-value diagnostic in the whole doc: print `v0.std()`.** |
| **15** | `one_step_estimate` divides by `√ᾱ_t → 0`. | [denoiser.py](models/denoiser_module/denoiser.py) | You asked to delete it. Agreed — and §6 shows how to remove the *need* for it. |
| **16** | `raise ValueError(f"Unknown schedule_type: {schedule_type}")` — undefined local, should be `self.schedule_type`. | [denoiser.py](models/denoiser_module/denoiser.py) | `NameError` instead of a useful message. Trivial. |

Also worth noting: `forward_diffusion` and `one_step_estimate` loop over the batch calling `.item()`
per element — a GPU→CPU sync per sample. Vectorize (index the buffer on-device).

---

## 0b. Repo reorganisation

Working tree is clean on `prakriti_sept26`, so this is fully reversible (`git reset --hard` or `git mv` back).

**Root keeps only the live stack:**

```
data/  models/  dataloader/  evaluation/  training/  configs/  utils/  paper/
README.md  requirements.txt  environment.yml  RESEARCH_PLAN.md  .gitignore  LICENSE
```

**Everything else → `reference/` with a `reference/README.md` index:**

| Move | From | Why keep it |
|---|---|---|
| `reference/runs/output/` | `output/` (725 files, 77 MB) | per-epoch sample JSONs — the qualitative evidence |
| `reference/runs/saved_diffusion_outputs/` | `saved_diffusion_outputs/` (92 MB) | denoiser checkpoints/outputs |
| `reference/runs/saved_latent_models_outputs/` | `saved_latent_models_outputs/` (131 MB) | P0 checkpoints — **needed for the `v0.std()` diagnostic (#14)** |
| `reference/runs/wandb/`, `reference/runs/slurm/` | `wandb/`, `slurm-*.out`, `scripts/sbatch/logs/` | run configs + hyperparameters actually used |
| `reference/scripts/` | `scripts/{autoencoder,diffusion,inference,sbatch,archive}/`, `sbatch_scripts/`, `run_pipeline.sh`, `FINAL_commands_to_run.sh` | the exact UMass launch params you'll want to re-read |
| `reference/baseline_ldlm/` | `diffusion/`, `latent_models/`, `model/`, `dataset_utils/`, `datasets_1/`, `train_latent_model.py`, `train_text_diffusion.py`, `inference_forget_recall.py`, `CONSTANTS.py`, `generate_xt.py` | the Lovelace fork we started from; not part of the new stack |
| `reference/dataset_prep/` | `dataset_prep/`, `privasis_*.json/csv` | provenance of `data/final/*` |
| `reference/docs/` | `Analyzing Latent Diffusion Architectures.md`, `codebase_walkthrough.md`, `implementation_plan.md`, `Running Privasis Diffusion Baseline.md`, `Final_Report___698DS__Bala_s_Copy_.pdf`, `figures/` | prior write-ups |
| `training/` | `scripts/training/*.py` | promoted to root — this is live code |
| `configs/` | new; absorbs `train_phase2_config.py`, `DenoiserConfig`, `G_psi_config` | fixes defect #10 |

Notes:
- `scripts/training/*.py` compute `ROOT = ../..`; after promotion to `training/` that becomes `..`.
  One line per file.
- `saved_latent_models_outputs/` and `*.pt` are gitignored, so those move with `Move-Item`, not
  `git mv`. Simplest: `Move-Item` everything, then `git add -A` — git detects the renames.
- Add `reference/**` to `.gitattributes` as `linguist-vendored` so it doesn't dominate the repo stats.
- I'll write `tools/reorg.ps1` (idempotent, dry-run flag) rather than issuing 30 loose commands.

---

## 1. Dataset

### 1a. Even distribution of timesteps → abstraction rungs

**Now.** `k = min(⌊t/100⌋, n_i − 1)`, `B=100`, `T=1000`. Assumes 10 rungs; the data has 4.88.
61% of timesteps land on the last rung. Plus defect #4: `x` itself is never a target.

**Fix — three parts, all cheap:**

**(i) Ladder includes the source.** Define `levels_i = [x] + xt_i`, so `n_i = len(xt_i) + 1`
(mean 5.88). Level 0 = verbatim source, level n−1 = most abstract. Now `t≈0` legitimately means
"recall in full".

**(ii) Per-example normalised map, defined on log-SNR not on `t`.**

The naive fix is `k = ⌊(t/T)·n_i⌋`. That's exactly the equal split you asked for — with `n_i = 6`,
each rung owns 167 timesteps and gets 16.7% of the probability mass. But `t` is not the natural
axis: under a cosine schedule, `ᾱ_t` stays near 1 for a long stretch and then collapses, so "50% of
the way through `t`" is nowhere near "50% of the information destroyed". Mapping on
signal-to-noise ratio fixes that:

```python
# utils/level_map.py  — ONE definition, imported by training, eval, and inference.
def level_index(t, n_levels, schedule, T):
    """Map diffusion timestep -> abstraction rung, equal mass per rung in log-SNR."""
    lam   = schedule.log_snr(t)                      # log(ᾱ/(1−ᾱ)), monotone decreasing in t
    lo,hi = schedule.log_snr(T), schedule.log_snr(1)
    frac  = (hi - lam) / (hi - lo)                   # 0 at t=1, 1 at t=T
    return torch.clamp((frac * n_levels).long(), max=n_levels - 1)
```

**(iii) Stratified sampling of `t`.** Don't sample `t ~ U{1,T}` and hope the rungs come out even —
at batch size 16 they won't. Sample the rung first, then a timestep inside it:

```python
k = torch.randint(0, n_i, (B,))                       # exactly uniform over rungs
t = sample_uniform_within_bucket(k, n_i, schedule)     # invert level_index
```

This makes the "equal distribution" a guarantee per batch, not a statistical hope.

**Do we need to regenerate for a fixed rung count?** Not yet. 68–80% of records already have exactly
5 rungs; with `x` prepended that's 6. Normalised mapping handles the spread. If we later want
constant `K`, `K=6` is nearly free (pad short ladders by repeating, trim long ones by dropping
adjacent near-duplicates) — and can be done offline with no API calls.

**Cost:** 0 GPU, 0 API. ~150 lines. **Do this first.**

### 1b. Using `y` to supervise the gist `u`

**Now.** `decoder_y` exists and is fully commented out in `ForgettingModel.forward`; only
`L_x` (reconstruct x from v₀) and `L_nce` are live.

**Your original worry was right but avoidable.** The discrepancy was: `DecoderX` consumes
`[B, L, 512]` slot-structured input, while `DecoderY` tiles a single `u` into `[B, L, 512]`
constant-across-slots. If those go through *the same* T5 weights, the decoder sees two very
different input distributions and P2 inherits the mess. **Solution: they were never the same
module.** `DecoderY` instantiates its own `T5ForConditionalGeneration`. Keep it that way, use it
**only in P0**, and **drop it entirely at P1/P2/inference.** There is then no discrepancy by
construction, because `D_y` never touches the P2 graph.

**Proposed P0 objective:**

```
L_P0 = λ_x·L_x        # CE: decode v₀ → x                  (v₀ holds the detail)
     + λ_y·L_y        # CE: decode u  → y   via D_y        (u holds the gist)
     + λ_a·L_align    # cos(proj(u), E_frozen(y))          (u lands in a *measurable* space)
     + λ_n·L_nce      # InfoNCE, positives from §1d
     + λ_o·L_orth     # (cos(u, mean_L v₀))²               (u ≠ v₀)
```

Three comments on why this composition, not just `L_y`:

- **`L_align` is the sleeper.** Regress `u` onto a frozen sentence-embedding of `y`
  (`intfloat/e5-base-v2` or `gte-base`). Cost: one frozen forward, free. Payoff: `u` now lives in a
  *known retrieval space*, which means at QA time you can encode the **question** with the same
  encoder and it lands next to the right `u` — cue-conditioned recall works with zero extra
  machinery. This is the bridge from §1b straight to §9. Without it, `u` is an arbitrary learned
  space and the question→cue mapping has to be learned from scratch on a small QA set.
- **`L_orth` is what actually buys the separation claim.** `L_x` and `L_y` alone don't stop `u`
  from also encoding details. A squared-cosine decorrelation between `u` and pooled `v₀` is one
  line, stable, and gives you something to point at. I'd avoid gradient-reversal adversarial
  probes — they're finicky and a reviewer will ask about training stability.
- **`L_y` and `L_nce` mildly fight each other.** Two memories with the same gist want the *same*
  `u` under `L_y` and *different* `u` under InfoNCE. Keep `λ_n` small (0.1) and use §1d's positives.

**How you prove the separation** (this is the actual paper claim, and it's measurement, not loss):

| Probe | Expected if separation holds |
|---|---|
| decode(`v₀`) vs `x` | high IU-recall (§7), high fluency |
| decode(`u`) vs `y` | matches gist, **low** IU-recall on names/numbers |
| **shuffle `u` across batch, then recall** | quality collapses (if it doesn't, `u` is decoration) |
| linear probe: predict entity presence from `u` | near chance |
| linear probe: predict entity presence from `v₀` | well above chance |

Those five rows are a self-contained ablation table. **Cost: ~0.5 GPU-h.**

### 1c. Slot-pooling ablation — can we prove slots matter?

**Honest answer up front: slots in a learned-query cross-attention pool are permutation-arbitrary
and not type-aligned.** Slot 3 does not reliably mean "names". Masking will most likely give you a
mushy, seed-dependent result. So run it as **two experiments, not one.**

**(i) Discovery (no data change, no retraining).** For each slot `j`, replace it with the
dataset-mean slot (better than zeroing — zeroing is off-distribution) and decode. Score the drop in
*information units by type* (§7):

```
A[j, c] = mean over examples of  IU_recall(full, type=c) − IU_recall(mask_j, type=c)
```

`A` is an `L × C` heatmap and is a good figure **either way**:
- sparse `A` → slots specialise; you have controllable, targeted forgetting;
- flat `A` → the representation is distributed/redundant. That's a real finding, it's publishable
  as such, and it *justifies* diffusing all slots jointly rather than dropping them.

Cost: `L × n` decodes; `16 × 200 = 3200` generations ≈ 15 min. Do this before anything else in 1c.

**(ii) Enforcement (if you want the masking story to be causal).** Add per-slot auxiliary heads that
force type assignment during P0:

```python
# slot j is assigned entity-type c_j; predict which units of that type are present in x
L_slot = Σ_j BCE(head_j(h[:, j, :]),  bag_of_units(x, type=c_j))
```

with `c_j` cycling over `{PERSON, GPE/LOC, DATE/TIME, CARDINAL/MONEY, ORG, PREDICATE, MODIFIER, other}`
(8 types, so `L=16` gives 2 slots per type). Now masking slot `j` *must* remove that type, and the
ablation demonstrates a mechanism rather than hunting for one. Framing for the paper: "*specialisation
does not emerge (i), but it can be induced at negligible cost, and induction gives type-controllable
forgetting (ii)*" — that's a stronger two-part result than either half alone.

**(iii) Your data question — "make xt forget names first, places second, consistently."**
Yes, that makes sense, and it's achievable **without any LLM**. Build a second, *controlled* corpus
with a deterministic rule cascade:

```
MSR-Ladder-Ctrl:  spaCy parse x, then delete/generalise in FIXED priority order:
  rung 1: CARDINAL, MONEY, PERCENT, QUANTITY, DATE, TIME   → "some", "a number of", "once"
  rung 2: PERSON                                            → "someone", "a person"
  rung 3: GPE, LOC, FAC                                     → "somewhere", "a place"
  rung 4: ORG, PRODUCT, WORK_OF_ART                         → "an organisation", "something"
  rung 5: ADJ / ADV modifiers                               → deleted
  rung 6: head nouns → WordNet hypernym (1 hop)             → "seabass" → "fish"
  rung 7: main verb → generic                               → "happened", "did something"
```

Why this is worth the effort:
- **fixed removal order** → the slot-masking ablation becomes interpretable;
- **monotonic by construction** → this *is* your missing ground-truth ceiling (paper §3, `\tbd{}`),
  and it costs nothing;
- **exact ground truth for what was removed at each rung** → the countable metric in §7 gets a
  perfect reference instead of an inferred one;
- **$0 and reproducible** — a reviewer can rerun it.

Downside: the rungs read stiffly ("A person went somewhere"). So keep the LLM ladders as the
*natural* corpus for headline results, and use `MSR-Ladder-Ctrl` for the ceiling + ablations.
Two corpora, two clearly-stated jobs.

Layout: `ablations/slot_pooling/{mask_sweep.py, build_ctrl_ladders.py, analyze_heatmap.py}`.

### 1d. Fixing `x⁺` (contrastive positives)

**Now.** `xpos = x`, identical. `_drop_tokens` written but disabled. There's a
`data/processed/train_part2_with_xplus.jsonl` and `scripts/archive/generate_xplus_gemma.py`, so a
Gemma-based generation existed at some point — worth checking coverage before regenerating.

**The best positive is already in the dataset and costs nothing.**

Use the ladder itself: `u(x) ≈ u(xt[k])` for every `k`. This is not a hack — it *is* the design
claim. The gist must be **invariant along the forgetting trajectory**, otherwise it cannot serve as
a stable cue during recall. Making that an explicit training signal is strictly better than a
generic paraphrase, and it's free:

```python
k_pos = randint(0, n_i)          # any rung, including x itself
x_pos = levels[k_pos]            # "same memory, different detail level"
```

Then `L_nce` teaches: *`u` is constant along the ladder, and distinct across memories.* That is
exactly the property §9 depends on.

**Secondary positive (surface paraphrase), all free and local:**

| Method | Model | Quality | Notes |
|---|---|---|---|
| token dropout | — | low | already written; use as fallback |
| WordNet synonyms | `nlpaug` | low–mid | wrong word senses |
| contextual substitution | `roberta-base` via `nlpaug` | mid | sense-aware, GPU, fast |
| **back-translation EN→DE→EN** | `Helsinki-NLP/opus-mt-*` | **high** | ~300 MB × 2, offline, best quality/cost |
| **paraphrase model** | `humarin/chatgpt_paraphraser_on_T5_base` | **high** | purpose-built, one forward |
| NVIDIA NIM | `llama-3.1-8b-instruct` | high | you have keys — but use for *audit*, not bulk |

Recommendation: **precompute** 2–3 paraphrases per `x` with back-translation + the T5 paraphraser,
store as `x_plus: [...]` in the JSON, sample one at training time. Precompute, don't generate on the
fly — reproducibility and speed.

**Critical filter.** A paraphrase that drops entities teaches `u` to ignore entities. That's fine for
the gist head but it contaminates `v₀` if you ever reuse `x⁺` there. So validate: keep a paraphrase
only if every NER span and every number in `x` survives (normalised match). Reject otherwise. Expect
~60–70% acceptance from back-translation; generate 4, keep the best 2.

**Also fix the loss itself:** make InfoNCE symmetric (both directions), L2-normalise `u`
(see §3), and use `τ = 0.07` or make it learnable. Current version is one-directional with a hard
`τ = 0.1`.

**Cost: 0 GPU-h of training, ~20 min of one-time preprocessing, $0 API.**

---

## 2. Text encoders — going beyond t5-small

**Now.** `T5EncoderModel("t5-small")`, `d=512`, 6 layers, ~35M params, `max_seq_len=64`.

**The important structural point first:** the encoder is **not** locked to the decoder. `v₀` is a
learned latent (slot pooling → VHead), and the decoder consumes it via `encoder_outputs=`. There's a
`Linear` in between. So you can swap the encoder freely; only `hidden_dim` changes. That frees up the
design a lot.

**Free options worth considering:**

| Model | Enc params | `d` | Context | Why |
|---|---|---|---|---|
| `google/flan-t5-base` | 110M | 768 | 512 | drop-in, instruction-tuned, much better semantics |
| `google/flan-t5-large` | 340M | 1024 | 512 | strong; fine frozen, heavy to fine-tune |
| `answerdotai/ModernBERT-base` | 149M | 768 | 8192 | 2024 SOTA encoder, fast, long context |
| `microsoft/deberta-v3-base` | 86M | 768 | 512 | best-in-class NLU, strong on entities/negation |
| `intfloat/e5-base-v2` / `thenlper/gte-base` | 110M | 768 | 512 | retrieval-tuned → excellent **gist** space |
| `nomic-ai/nomic-embed-text-v1.5` | 137M | 768 | 8192 | Matryoshka dims (pick `u_dim` for free) |
| `google/byt5-small` | — | — | — | char-level, robust to entity spelling; slow — skip |

**Recommendation: use two encoders, because `u` and `v₀` want different things.**

- **Detail path (`v₀`):** upgrade `t5-small` → **`flan-t5-base`** (`d=768`) and raise
  `max_seq_len` 64 → **128** (defect #6). Biggest single-line quality win available.
- **Gist path (`u`):** a **frozen** `e5-base-v2` / `gte-base` over `x`, then a small projection to
  `u_dim`. Frozen = free, fast, no extra training. And it makes §9 work: the question encodes into
  the same space, so `u ← E(q)` is meaningful on day one instead of needing a learned adapter.

**Don't** jump to `flan-t5-large` until (a) `max_seq_len` is fixed and (b) the level map is fixed.
Those two are worth more than 3× the parameters.

Ordering by value/cost: `max_seq_len 64→128` **>** `t5-small→flan-t5-base` **>** frozen retrieval
encoder for `u` **>** everything else.

---

## 3. The `u` and `v₀` heads

### Current architecture, stated plainly

```python
UHead:  h[B,L,512] → mean over L → Linear(512, u_dim)      # ONE linear layer on an average
VHead:  h[B,L,512] → Linear(512, 512)                       # per-slot, ~identity
```

**So today `u` and `v₀` are both essentially linear functions of the same pooled tensor `h`.** They
are not architecturally distinct and not independently expressive. Any separation must come entirely
from the losses — and right now only `L_x` exists, so `v₀` does all the work and `u` is decoration.
Which is precisely why the paper's shuffled-cue control is the sharpest available test: I expect it
to show `u` currently does very little.

Mean-pooling over slots is also the wrong pooling: it throws away *which* slot carried what, right
after we spent a cross-attention layer creating slot structure.

### Proposed `UHead` v2 — attention pooling + MLP + normalisation

```python
class UHead(nn.Module):
    def __init__(self, d, u_dim, n_heads=8):
        self.query = nn.Parameter(torch.randn(1, 1, d) * d**-0.5)   # learned CLS-like query
        self.attn  = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.mlp   = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4*d),
                                   nn.GELU(), nn.Linear(4*d, u_dim))

    def forward(self, h):                       # h: [B, L, d]
        q            = self.query.expand(h.size(0), -1, -1)
        pooled, attw = self.attn(q, h, h, need_weights=True)   # [B,1,d], [B,1,L]
        u            = self.mlp(pooled.squeeze(1))
        return F.normalize(u, dim=-1), attw     # unit-norm u  +  free interpretability signal
```

Three reasons, each concrete:
1. **Attention pooling can select** which slots are gist-bearing instead of averaging everything.
2. **`attw` is free interpretability** — it directly feeds the §1c slot heatmap.
3. **L2-normalising `u` matters twice over.** InfoNCE is well-behaved on the unit sphere; and `u` is
   concatenated into AdaLN conditioning in both the denoiser and G_ψ, where an unnormalised 512-dim
   vector with drifting scale is a genuine training-stability hazard.

### Proposed `VHead` v2 — slot-aware, and *scale-controlled*

```python
class VHead(nn.Module):
    def __init__(self, d, L, n_blocks=2, n_heads=8):
        self.slot_pos = nn.Parameter(torch.randn(1, L, d) * d**-0.5)  # break permutation symmetry
        self.blocks   = nn.ModuleList([PreNormTransformerBlock(d, n_heads) for _ in range(n_blocks)])
        self.out      = nn.Linear(d, d)
        self.norm     = nn.LayerNorm(d, elementwise_affine=False)     # <-- see below

    def forward(self, h):
        x = h + self.slot_pos
        for blk in self.blocks: x = blk(x)      # slots exchange information
        return self.norm(self.out(x))           # v₀ with std ≈ 1 per slot
```

**The `norm` line is the most important change in this whole section.** The forward process is
`v_t = √ᾱ·v₀ + √(1−ᾱ)·ε` with `ε ~ N(0, I)`. That equation only produces graded corruption if
`v₀` has roughly unit variance. If `std(v₀) = 8`, the noise term is negligible until `ᾱ` is tiny,
so **nothing degrades and then everything degrades at once** — exactly the "cliff effect" documented
in `codebase_walkthrough.md §4b` and blamed there on the cosine schedule and latent geometry.
Nobody has checked. Two acceptable fixes: (a) `LayerNorm` at the VHead output as above, or (b) the
Stable-Diffusion approach — measure `σ_data = std(v₀)` over the training set once and scale by
`1/σ_data` before diffusing. **Diagnostic first** (load a P0 checkpoint from
`reference/runs/saved_latent_models_outputs/`, print `v0.std()`), then choose.

Optionally give `u` a little structure (`u ∈ R^{n_cue × d}`, `n_cue = 2–4`) — it costs nothing and
makes the cross-attention conditioning in §4 richer than a single broadcast vector.

---

## 4. G_ψ, the semantic projection module

### Current architecture

`input_proj: Linear(d,d)` → 3 × `SPMBlock` → `v_tilde = x_ini + out_proj(x)` with `out_proj`
zero-initialised. `SPMBlock` = `AdaLN(d, d_cond=u_dim+d)` + FFN, **plus optional self-attention that
is off by default**.

### What's wrong

1. **`use_attn=False` → G_ψ is a pointwise MLP.** No slot mixing at all. The module whose job is
   "map an off-manifold latent back onto the encoder manifold" cannot look at more than one slot at
   a time. Turn it on. (Defect #7.)
2. **`u` enters only through AdaLN** — a global scale and shift. That channel can *modulate*
   features; it cannot *inject content*. For cue-conditioned recall, where `u` is supposed to supply
   the semantic anchor, that's far too thin. (The denoiser has the same hole — its cross-attention
   is also commented out.)
3. **`v̂₀` is accepted and unused**; `input_dim = 2*d` is computed then discarded. (Defect #8.)
4. **No `t`-dependent output gate.** Zero-init makes G_ψ ≈ identity *at initialisation*, not
   *per timestep*. We want ≈identity at `t→0` and a large correction at `t→T`.
5. **No standalone loss** (the paper says so explicitly). All gradient arrives through a distant
   cross-entropy at the decoder output. That's a weak, high-variance signal for a module doing
   geometry.

### Proposed G_ψ v2

```python
class SPMBlock:
    self_attn  = AdaLN + MHA(slots ↔ slots)                       # ON
    cross_attn = AdaLN + MHA(q=slots, kv=cue_tokens)              # NEW: content injection
    ffn        = AdaLN + FFN
    # every residual scaled by a learned, t-conditioned gate α_i(t)

class G_psi:
    def forward(self, v_t, v_hat_0, t, u, q_emb=None):
        cue = self.to_cue(torch.cat([u, q_emb], -1) if q_emb is not None else u)  # → [B, n_cue, d]
        x   = self.in_proj(torch.cat([v_t, v_hat_0], -1)) + self.slot_pos          # [B, L, d]
        c   = torch.cat([self.t_emb(t), u], -1)
        for blk in self.blocks:                    # 6 blocks, not 3
            x = blk(x, cue, c)
        return v_t + self.gate(t) * self.out_proj(x)
```

### 4b. Inputs — you're right that this is where cue-conditioning lives

Feed **`(v_t, v̂₀, t, u)`**:

- `v_t` = where we actually are on the trajectory.
- `v̂₀` = where the denoiser *thinks* the clean latent is.
  Giving both lets G_ψ decide **how far to trust the denoiser at this noise level** — which is a
  `t`-dependent decision, hence AdaLN on `t` is exactly the right conditioning for it. This is the
  two-input design that was intended and never wired up.
- `u` through **cross-attention** (content) **and** AdaLN (modulation). Belt and braces.
- Later, for QA: `cue = [u_memory ; E(q)]`. **Same code path, no architecture change.** This is the
  clean upgrade route to §9, and it's why `L_align` in §1b matters.

### 4a. Taking load off the decoder — add a real loss to G_ψ

This is the change I'd argue hardest for.

> **`L_proj = ‖ G_ψ(v_t, v̂₀, t, u) − E_frozen(levels[k(t)]) ‖²`**

Encode the *ground-truth rung text* `levels[k(t)]` with the frozen P0 encoder + slot pool + VHead.
That vector is, by definition, a point the decoder was trained to read. Regressing G_ψ's output onto
it is **literally "project onto the manifold"** — the module's stated purpose — as a dense, well-posed
target instead of a distant cross-entropy.

Consequences:
- The decoder stops having to compensate for off-manifold inputs. Its job shrinks to "read a
  latent", which is exactly what it learned in P0.
- **You can then try freezing the decoder entirely in P2.** If that works, the P0/P2 input-discrepancy
  worry from §1b evaporates *and* you get a much stronger claim: *the same decoder that reads clean
  latents reads projected noisy latents, with no fine-tuning.* Fall back to fine-tuning only if the
  frozen version underperforms.
- Precomputable: 5k examples × ~6 rungs × 16 slots × 512 dims in fp16 ≈ **1 GB on disk**. Cache once,
  reuse across every P2 run. Also removes the encoder forward from the P2 training loop → faster.

Total P2 objective: `L = λ_proj·L_proj + λ_ce·L_CE(decoder)`. Start `λ_proj=1.0, λ_ce=1.0`, then
sweep — the ratio is the interesting knob (pure `L_proj` = "G_ψ does everything", pure `L_CE` =
today's setup).

---

## 5. The decoder

**Now.** `t5-small` `T5ForConditionalGeneration`, fed `encoder_outputs=(v,)` with an all-ones
attention mask over `L` slots. Two jobs: P0 `v₀ → x` (clean), P2 `ṽ₀ → xt[k]` (abstracted).

### Can it do both? Yes — but only if we stop asking it to guess.

The failure mode ("gibberish at high `t`") has three separable causes, and only one of them is
the decoder's fault:

1. **Off-manifold input** → fix in G_ψ (§4a `L_proj`).
2. **Scale mismatch in `v₀`** → fix in VHead (§3, defect #14).
3. **The decoder is not told how abstract to be.** ← *this* one is the decoder's.

### Fix 3: level conditioning (cheap, high leverage)

Prepend a learned **level token** to the cross-attention memory:

```python
memory = torch.cat([level_emb(k)[:, None, :], v_tilde], dim=1)     # [B, 1+L, d]
mask   = torch.ones(B, 1 + L)
loss, logits = decoder(encoder_outputs=(memory,), attention_mask=mask, labels=labels)
```

One extra memory slot. Now "produce an abstract sentence" is an **instruction**, not something the
decoder must infer from a degraded latent. Today the decoder gets a noisy latent and a short target
and has to work out on its own whether the shortness is intentional or the latent is broken. That
ambiguity is a direct cause of degenerate outputs.

Optionally add a **length-bucket embedding** alongside it — abstract rungs are shorter, and without
length guidance the decoder tends to pad abstract outputs with invented detail.

### Other decoder changes

- **`t5-small` → `flan-t5-base`.** `t5-small`'s decoder is genuinely weak; this is the second-biggest
  quality lever after `max_seq_len`.
- **`L=16` slots is a tight bottleneck** for a ~50–80 token detail-heavy sentence — T5 cross-attends
  to only 16 vectors. If IU-recall on numbers/names is the bottleneck, try `L=32`. Cost: diffusion
  scales linearly in `L`, and `L` is small, so this is cheap. Worth one sweep point.
- **Generation config:** `num_beams=4` is fine, but also report greedy; set
  `no_repeat_ngram_size=3` and `min_new_tokens=4` to stop the degenerate one-word collapse observed
  at high `t`.
- **Never report teacher-forced outputs again** (defect #1).

### Where the load should sit

> **G_ψ carries the latent-space burden (`L_proj`). The level token carries the "how abstract"
> burden. The decoder just reads.** If that division works, the decoder can stay frozen after P0 —
> the cleanest possible story.

---

## 6. Diffusion and denoising

You're right that the math is mostly settled. Five changes are still worth making, and one of them
is a genuine research contribution rather than a fix.

### 6.1 Delete one-step estimation — and remove the *need* for it

Done, as instructed. But note *why* it broke: `v̂₀ = (v_t − √(1−ᾱ)·ε̂)/√ᾱ` divides by something
going to zero. **Switching the denoiser to predict `v₀` directly (x₀-parameterisation) removes the
division entirely.** No ill-conditioning at high `t`, and the output is what every downstream
consumer (G_ψ, the decoder) actually wants. DDIM sampling works unchanged in x₀ form.
This is a strictly better fix than clipping `t`.

```python
v0_hat = denoiser(v_t, t, u)                                   # predicts v₀, not ε
loss    = F.mse_loss(v0_hat, v0) * snr_weight(t)               # min-SNR-γ weighting, γ=5
```

### 6.2 Fix the train/inference mismatch (defect #3)

`step_by_step_estimate` returns `v_{t−1}`, and P2 treats it as `v̂₀`. So G_ψ trains on a
barely-denoised latent while inference would give it a fully-run chain.

Fix: use **DDIM with `K` steps (K = 20–50), the same `K` in training and evaluation.** Cheap enough
to run inside the training loop (the denoiser is ~19M params on a `[B,16,512]` tensor), and it makes
the two distributions identical by construction. Implement once in `diffusion/sampler.py`; import it
everywhere.

### 6.3 Schedule: define levels on log-SNR, not on `t`

Covered in §1a(ii). Cosine keeps `ᾱ` high then drops fast, so equal spacing in `t` is very unequal
spacing in information destroyed. Ablation candidates: cosine (current), sigmoid/laplace, and
Kingma's learned schedule. But the log-SNR level map is a 5-line change that captures most of the
benefit without touching the schedule at all — do that first.

### 6.4 **Gist-anchored diffusion** ← the interesting one

The paper already flags it (§5): "standard diffusion drives `v_T` to pure noise, so the most
abstract end of the ladder is anchored on nothing." Right. And there's a one-line fix that keeps all
the DDPM math intact:

```python
m     = self.gist_to_slots(u)          # [B, L, d]  — gist broadcast into slot space
w0    = v0 - m                         # residual DETAIL only
wt, ε = forward_diffusion(w0, t, sched)
vt    = wt + m                         # ⇒ v_T ≈ m(u) + noise, NOT pure noise
```

Diffuse the **residual** `w₀ = v₀ − m(u)` instead of `v₀`. Then:
- forward process = *details dissolve, gist remains* — structurally, not by hope;
- `t=T` decodes to the gist, not gibberish, so the top of the ladder is anchored;
- the "most abstract rung" now has a semantic floor, which is exactly what §1a needs at `k = n−1`;
- zero new math — it's DDPM on a shifted variable (and it's the same trick as
  cold-diffusion / I2SB / diffusion bridges, so it's citable).

**This is the change that makes the memory metaphor structurally true instead of aspirational.**
I'd rank it as the highest-value single experiment in the doc, alongside the level-map fix.

### 6.5 Standard hygiene, all cheap

- **EMA of denoiser weights** (decay 0.999). Standard, materially improves sample quality, ~5 lines.
- **Classifier-free guidance on `u`**: drop `u` with `p=0.1` during training; at inference
  `v̂₀ = v̂₀(∅) + w·(v̂₀(u) − v̂₀(∅))`. Gives a **second dial — cue strength — alongside depth**, and
  makes the shuffled-cue control far more informative (you can plot quality vs. `w`).
- **Seed the reverse noise** (defect #13) so trajectories are reproducible.
- **Vectorise** `forward_diffusion` (defect: per-element `.item()` loop).
- **Cache `u, v₀` to disk** after P0. P1 currently re-encodes every step for a frozen encoder —
  pure waste. Caching turns P1 from ~2 GPU-h into <0.5 GPU-h per run.
- Fix or delete `models/latent_diffusion/latent_diffusion_model.py` (defect #9 — blocks don't chain;
  it looks like a dead duplicate of `Denoiser`).

---

## 7. Evaluation metrics

### 7.1 How the current metrics actually work, and why they're wrong here

**BERTScore.** Embeds every token of candidate and reference with a BERT-family model, greedily
matches each candidate token to its most-similar reference token by cosine, averages (optionally
IDF-weighted), then rescales against a random baseline. Precision = how much of the candidate is
supported; recall = how much of the reference is covered.
*What it measures:* distributional token overlap. *What it does not measure:* whether the **specific
facts** survived. Swapping "Paris"→"London" or "16,000 tons"→"a few tons" barely moves it, because
the substitutes are near neighbours in embedding space. For a project whose entire claim is *which
specific details survive*, BERTScore is close to useless as a primary metric — and the paper's own
observation ("reconstructions preserve the sentence frame while corrupting content words") is
exactly this failure. Plus we currently use `recall` for monotonicity and mix `roberta-base` /
`roberta-large` across files (defect #12).

**UniEval.** A T5 model reframed as boolean QA: "Is this claim consistent with the document?
yes/no", score = `P(yes)`. Trained on synthetic positive/negative pairs, calibrated on news
summarization. Issues here: out-of-domain, no explanation, sensitive to input formatting, and its
consistency dimension penalises *any* added information while giving no credit for *appropriate*
abstraction. A good abstract rung and a hallucination can land in the same range.

**NLI entailment asymmetry.** The idea is right — a specific sentence should entail a general one but
not vice versa. The implementation passes `f"{premise} [SEP] {hypothesis}"` as a single string to a
`text-classification` pipeline; cross-encoder NLI models expect a `text`/`text_pair` argument
(defect #11). Verify before trusting 0.587/0.628.

Also: **monotonicity as a binary flag over a 5-point trajectory is brutally lossy.** One blip
anywhere zeroes the whole trajectory. Spearman ρ between rung index and detail retained tells you
far more, and degrades gracefully.

### 7.2 What to replace them with

**Tier 1 — countable, deterministic, no model. This is the backbone.**

Extract typed **information units (IUs)** from `x` with spaCy: `PERSON, ORG, GPE/LOC, DATE/TIME,
CARDINAL/MONEY/PERCENT/QUANTITY`, plus content lemmas and numeric literals. Normalise (lowercase,
lemmatise, canonicalise numbers). Then:

| Metric | Definition | Answers |
|---|---|---|
| **IU-Recall(k)** | `|IU(x) ∩ IU(ŷ_k)| / |IU(x)|` | how much detail survives at rung k |
| **Decay monotonicity** | Spearman ρ(k, IU-Recall) **and** the strict non-increasing fraction | *does detail fall as t rises* |
| **Per-type decay curves** | IU-Recall split by type vs k | **the money figure**: numbers fade before names before predicates = human-like forgetting |
| **Nesting violation rate** | fraction of adjacent pairs where `IU(ŷ_{k+1}) ⊄ IU(ŷ_k)` | **the early-exit safety guarantee**, countable |
| **Hallucination rate** | `|IU(ŷ_k) \ IU(x)| / |IU(ŷ_k)|` | separates *forgot* from *made up* |
| **Abstraction validity** | % of noun replacements that are WordNet hypernyms of the original | **separates abstraction from corruption** — nothing we currently run measures this, and it is the distinction the entire project rests on |
| **Fluency guardrail** | GPT-2 perplexity | catches gibberish; guardrail, not headline |

**Tier 2 — model-based but targeted.**

- **QA-based faithfulness (strongest single metric).** Offline, generate one question per IU from `x`
  (templates or a local model). Ask each question of each rung with an extractive QA model. Three
  outcomes:
  - answered correctly → **detail retained**
  - answered *incorrectly* → **corruption** ← the failure we care about most
  - unanswerable → **abstraction** ← the behaviour we *want*

  BERTScore and UniEval both conflate the last two. This metric is the one that finally separates
  them, and it's the same family as QAGS / FEQA / QuestEval so it's defensible in review.
- **Directional NLI, done properly** (`text_pair` API, both directions, report the asymmetry). Keep
  as secondary evidence.

**Tier 3 — LLM judge, used narrowly.**

Only for what cannot be counted: *"is rung k+1 a plausible human-like abstraction of rung k?"*
Protocol matters more than the model:
- **pairwise A/B against the ground-truth rung**, not an absolute 1–5 score (absolute Likert from an
  LLM is uncalibrated and drifts);
- **randomise presentation order** and report position-bias;
- ~200 items, not the whole set;
- **report agreement with a human-labelled subset** — this is what makes a judge credible to
  reviewers, and it's cheap: 50 items × 4 of you.

Free backends: NVIDIA NIM (`llama-3.1-70b`, `nemotron`) or a local `Qwen2.5-14B-Instruct`.

**Tier 4 — small human eval.** 50 trajectories, 4 annotators, two questions: *"which rung would you
stop at to answer this?"* and *"is any rung inconsistent with the one above it?"* This is the only
thing that directly validates the early-exit premise, and it costs an afternoon.

### 7.3 The metric the project needs and doesn't have

**The early-exit curve.** The headline claim is "you can stop early", so the headline figure should be:

> **task quality (IU-coverage / QA accuracy) on the y-axis, budget (reverse steps, or tokens
> emitted, or NFE) on the x-axis** — with the truncation baseline, the level-token baseline, and the
> full-generation ceiling on the same axes.

Everything else in §7 is supporting evidence for that one plot.

---

## 8. Baselines and how to frame the motivation

### 8.1 The baselines, concretely

| # | Baseline | What it settles | Cost |
|---|---|---|---|
| **1** | **Level-token seq2seq.** Same `flan-t5-base`, input `"level {k}: {x}"`, output `xt[k]`. Identical data. | **Does diffusion buy anything at all?** If this matches us on IU-decay + nesting, the diffusion stack is unjustified *on this task*. **Run this first — it decides whether there's a paper.** | ~1 GPU-h |
| **2** | **Prompted LLM at k budgets.** Local Qwen2.5-7B / NIM, zero-shot and 5-shot. | Can an off-the-shelf model already do this? Probably yes on *quality*, probably **no on nesting** — measure violations, that's the opening. | ~2 GPU-h |
| **3** | **Truncated full generation.** Generate the full answer, cut to budget. | Does early exit beat post-hoc cutting? Our edge: a truncated prefix is incomplete *and possibly misleading*; our rungs are complete-but-coarse. | ~0 |
| **4** | **Length-controlled summarizer** (BART/LED at budget, or LLMLingua-style compression). | The practical incumbent for "less detail, fewer tokens". Reviewers will ask. | ~1 GPU-h |
| **5** | **RAG** (vector store + top-k + LLM). | The real competitor for the *memory* claim (§9). Measure quality vs. context tokens, and storage bytes. | ~1 GPU-h |
| **6** | **Shuffled cue `u`.** Permute `u` across the batch at recall. | Is recall genuinely cue-conditioned? **Cheap and potentially project-invalidating — run it early.** | ~0.2 GPU-h |
| **7** | **No-G_ψ.** Decode `v_t` directly. | Isolates G_ψ's contribution. | ~0.2 GPU-h |
| **8** | **Ground-truth ceiling.** Run every metric on the reference `xt` ladders themselves. | If GT monotonicity is 0.6, our 0.325 is a *completely* different story than if GT is 0.98. **Data-only, zero GPU, fills the paper's `\tbd{}`. Do it in the first hour.** | **0** |

Note on #1: even if quality ties, we retain two structural advantages — (a) the level-token model
regenerates from scratch per budget, so it has no *amortisation* across budgets, and (b) it has no
**nesting guarantee**. Measure both explicitly; that's the fallback claim.

### 8.2 Framing against summarization — how to word the motivation

Do **not** claim "we summarise better". Claim this:

> A summarizer gives you one compression ratio per call, and no guarantee that its short output is
> consistent with its long output. RAG gives you a stored passage whole, or not at all. A memory
> system needs three things neither provides: **(i)** many levels of detail from one stored object,
> **(ii)** a guarantee that a coarser recall is a *subset* of a finer one — so stopping early is
> safe rather than merely short, and **(iii)** an access cost that scales with the detail you
> *asked for*, not the detail you *stored*. We ask whether a diffusion trajectory over a latent
> memory provides them.

That is defensible, testable by the metrics in §7, and it doesn't overclaim.

---

## 9. Downstream task: long-form / multi-turn QA

**Verdict: right setting, and your day-0 / month-3 framing is exactly the correct intuition.** Two
caveats: it's a real engineering lift, and the memory claim can get confounded with plain QA
ability. Both are manageable.

### 9.1 Your questions, answered directly

**"Do we need to store the noised latent too?"** **No.** Noise is *sampled*, not stored. Per memory
item you store `u` (small) and — if you want veridical detail — `v₀`. You never store `v_t`.

**"Then how are we cutting costs?"** If you store `v₀`, **you aren't.** The paper already concedes
this ("the latent is larger than the text it encodes"). **So stop trying to make storage the moat.**
But there is a real and different cost story:

> RAG's read cost scales with the detail you **stored**: embed query → ANN search → fetch the *whole*
> passage → put every one of its tokens in the context window → pay attention over all of them.
> Ours scales with the detail you **requested**: `K` reverse steps on a `16×512` latent, then decode
> a *short* answer. In a long conversation history where most queries need only the gist, that's an
> asymptotic difference in prompt tokens.

Quantify it as `context tokens/query` and `$/query` vs. answer quality. That's a plot reviewers
understand immediately. But I'd still keep cost **secondary** and lead with graded, consistent
access (§10) — the cost argument is contingent on workload mix, the consistency argument isn't.

**The genuinely compressive variant (optional, riskier).** Store *only* `u`; at recall run the
reverse process from `v_T = m(u) + noise` (§6.4) and let the model **generate plausible detail
conditioned on the gist**. That truly compresses — but the recalled details are **confabulated, not
veridical**. Which is, of course, exactly what human memory does (Bartlett's reconstructive memory).
Fascinating framing, terrible headline result, great *discussion* section. Run it as a controlled
measurement — "how much detail recovered from gist alone is *correct*?" — and expect the answer to
be "plausible names, wrong names". Don't build the paper on it.

**"How to handle contradictions as time condenses memories?"** Honestly: that's a second paper.
Minimal version if you must touch it — memories are keyed by `u`; when a new item arrives with
`cos(u_new, u_old) > τ` you have a merge candidate; either keep both with timestamps and let a
recency-conditioned cue disambiguate, or re-encode the concatenation. **Scope it out explicitly in
Limitations.** Do not put it in the main claim.

### 9.2 Building the dataset

**Route A — reuse an existing benchmark. Strongly recommended; near-zero cost.**

| Dataset | Why |
|---|---|
| **LOCOMO** (2024) | Purpose-built for *exactly* your scenario: very long multi-session dialogues with QA about earlier sessions, labelled by reasoning type (single-hop / multi-hop / temporal / open-domain). This is *the* benchmark for the long-term-memory claim. |
| **LongMemEval** (2024) | 500 questions over long chat histories, categorised by memory ability. Complements LOCOMO. |
| **MSC (Multi-Session Chat)** | Person talks in session 1, is asked about it in session 5. Your scenario, already built. |
| **ASQA** | Ambiguous questions with **multiple gold sub-answers** → *coverage of gold sub-answers vs. depth* is a graded metric with real ground truth and zero generated supervision. Exactly what the paper's §7 asks for. |

Recommendation: **LOCOMO (or LongMemEval) as the primary downstream benchmark, ASQA as the graded-coverage
measurement bed.** This removes almost all dataset-construction cost and sidesteps "you evaluated on
data you generated yourself".

**Route B — build our own.** Only if A is insufficient. Schema:

```jsonc
{
  "memory_id": "...", "session_id": 3, "timestamp": "2024-01-14", "speaker": "user",
  "text": "...",                       // the memory item (x)
  "gist": "...",                       // y
  "ladder": ["...", "..."],            // rungs (LLM + rule-based)
  "question": "...", "question_time": "2024-04-02",
  "question_type": "single-hop | multi-hop | temporal | aggregation",
  "gold_answer_full": "...",
  "gold_answer_at_budget": {"1": "...", "3": "...", "5": "..."},   // ← key field
  "required_IUs": ["PERSON:Nelvis", "DATE:November 13"],
  "min_depth": 3,                                                   // ← the early-exit label
  "distractor_ids": ["...", "..."]     // similar memories — this is what breaks RAG
}
```

The two fields that make it *our* dataset rather than a generic QA set are **`min_depth`** and
**`gold_answer_at_budget`**. With `min_depth`, early exit stops being a vibe and becomes a
supervised, measurable prediction.

Design considerations while building:
- **Temporal gap distribution** — vary day-0-to-query lag, it's the independent variable.
- **Distractor density** — near-duplicate memories are what make retrieval fail and cue-conditioning
  matter. Without them the RAG baseline wins trivially.
- **Balance gist-sufficient vs. detail-required questions.** You need *both* or you can't show a
  tradeoff curve at all.
- **Contamination check** — run a *no-memory* baseline. Any question answerable from world knowledge
  alone must be dropped.

---

## 10. Motivation and moat

Ranked by defensibility:

1. **Graded, nested access — the consistency guarantee.** *Every coarser recall is a subset of every
   finer recall of the same memory.* Nothing in RAG or summarization offers this. It's directly
   measurable (nesting violation rate, §7), and it is **what makes early exit safe rather than
   merely short**. This is the moat.
2. **Early exit / compute-for-detail dial.** Strong — but only with a real quality-vs-budget curve
   beating truncation *and* the level-token baseline.
3. **Human-like forgetting order** (numbers → names → predicates). The most *interesting* and most
   *ownable* claim; your per-type IU decay curve measures it directly; it's cognitively grounded and
   RAG cannot produce it. **Underrated — I'd elevate this to co-headline.**
4. **Read cost scales with requested, not stored, detail.** Good secondary, workload-dependent.
5. ~~Storage compression~~ — **drop it.** It's false as currently designed, and claiming it invites
   an easy rejection.

One-line version:

> **Retrieval is a switch; we want a dial. And a dial is only useful if turning it down never makes
> you wrong.**

---

## 11. Training vs. inference — behaviour and code changes

### Training

| Phase | Trains | Frozen | Loss | Notes |
|---|---|---|---|---|
| **P0** | encoder, slot pool, U/V heads, `D_x`, `D_y` | — | `λ_x L_x + λ_y L_y + λ_a L_align + λ_n L_nce + λ_o L_orth` | teacher forcing ON; level token = 0; **cache `u, v₀` at the end** |
| **P1** | denoiser | all of P0 | x₀-pred MSE, min-SNR-γ weighting, CFG dropout `p=0.1`, EMA | runs on **cached** latents → very fast |
| **P2** | G_ψ (+ decoder only if needed) | encoder side, denoiser | `λ_proj L_proj + λ_ce L_CE` | K-step DDIM **inside** the loop so train ≈ inference |

### Inference

- **Free-running `generate()`. No labels. Ever.** (defect #1)
- **No ground-truth `xt`** — the rung comes from the requested budget `b`, mapped to `t` by the
  *same* `level_map`.
- The reverse chain actually runs; decode at chosen checkpoints along it.
- Two modes, one engine:
  - **forget(`v₀`, t)** — forward-noise a stored latent and read out (simulates elapsed time);
  - **recall(`v_t`, `u`, budgets=[...])** — reverse with the cue (simulates remembering).
- Fixed seed per trajectory → reproducible (defect #13).

### The single most important code change here

There are currently **five** near-duplicate inference scripts in `scripts/inference/`. Collapse them
into **one** `RecallEngine` used by evaluation, ablations, *and* the downstream QA harness:

```python
class RecallEngine:
    def __init__(self, p0, denoiser, g_psi, sched, K=50, guidance=1.0, seed=0): ...
    def encode(self, text)                 -> (u, v0)
    def forget(self, v0, t, u)             -> v_t                     # forward, gist-anchored
    def recall(self, v_t, u, budgets, q=None) -> {b: (v_tilde, text)} # reverse + G_ψ + decode
    def ladder(self, text, budgets)        -> [(b, text_b), ...]      # convenience: full ladder
```

If the trajectory logic exists in exactly one place, train/eval/downstream can't drift apart — which
is the class of bug that produced defects #1, #3 and #10.

### How we test without a cluster

- **Unit tests**: level map (bijectivity, bucket masses, `n_i` edge cases), noise schedule
  (`ᾱ` monotone, endpoints), `L_orth`/InfoNCE shapes.
- **Overfit-10 smoke test per phase** — 10 examples, loss should approach 0. Catches wiring bugs
  (like #3 and #9) in minutes on CPU, no GPU needed. **This is how we validate everything before you
  have GPUs.**
- **Shape/dtype contract test** across the whole P0→P1→P2 chain.
- **Golden-file test** for the metrics: hand-written trajectories with known IU counts.

All four are runnable on a laptop, which matters given the current constraint.

---

## 12. Cost estimate

### GPU hours (per run, single A100-40G / L40S / 4090-class; dataset is tiny — 3k train)

| Phase | Config | Time/run |
|---|---|---|
| P0 | flan-t5-base, bs 16, seq 128, ~188 steps/epoch, 200 epochs | **2–3 h** |
| P1 | denoiser on **cached** latents (currently re-encodes — fix that) | **< 0.5 h** |
| P2 | G_ψ + K=20 DDIM in-loop, 100 epochs | **~2 h** |
| Full eval | 933 × 6 rungs ≈ 5.6k beam generations + metrics | **0.5–1 h** |

| Bucket | Runs | GPU-h |
|---|---|---|
| Main grid (8 configs × full pipeline) | 8 × 6.5 | **52** |
| Baselines (§8.1 #1–#5, ×3 seeds where relevant) | — | **~8** |
| Ablations (slot mask, shuffled cue, no-G_ψ, schedules, λ sweeps) | — | **~15** |
| Downstream QA (LOCOMO/LongMemEval, 7B answerer + judge inference) | — | **20–40** |
| Debug / failed runs (×1.5 overhead on the above) | — | **~50** |
| **Total** | | **≈ 150–220 GPU-h** |

Everything fits in **24 GB** (flan-t5-base, bs 16, seq 128), so you are *not* forced onto A100s.
Rough rental: 4090 ≈ $0.35–0.50/h → **$55–110**; L40S ≈ $0.80–1.10/h → **$120–240**;
A100-40G ≈ $1.00–1.50/h → **$150–330**. Free if any UMass Unity allocation survives.

### API credits

| Item | Approach | Cost |
|---|---|---|
| `x⁺` paraphrases | local back-translation + T5 paraphraser | **$0** |
| Rule-based control ladders + ceiling | spaCy + WordNet | **$0** |
| IU / QA-faithfulness metrics | local spaCy + extractive QA | **$0** |
| LLM judge (200 items × 2 comparisons ≈ 600k tokens) | NIM / local Qwen → $0; gpt-4o-mini → <$1; gpt-4o → $3–5 | **$0–5** |
| Regenerate all 5k ladders at fixed K (*if* we decide to) | local Qwen2.5-14B / NIM → $0; gpt-4o-mini ≈ $2.70; gpt-4o ≈ $44 | **$0–3** |
| QA dataset generation (only under Route B) | NIM / local → $0; gpt-4o-mini ≈ $5–15 | **$0–15** |
| **Total** | | **$0–25** |

**Conclusion: money is not the constraint. GPU time and person-time are.** Which means the ordering
in §13 matters much more than the budget.

---

## 13. Splitting the work — 7 parallel streams

Designed so four people rarely block each other. **A is on everyone's critical path; do it first.**

| Stream | Owner-sized chunk | GPU? | Depends on | Contents |
|---|---|---|---|---|
| **A — Data & Ceiling** | 1 person, ~2 days | **No** | — | repo reorg (§0b); `level_map` (§1a); ladder = `[x]+xt`; stratified `t`; `x⁺` precompute (§1d); rule-based control ladders (§1c-iii); **ground-truth ceiling numbers (§8.1 #8)** |
| **B — Metrics & Baselines** | 1 person | light | A | IU/nesting/hallucination/hypernym metrics (§7.1); QA-faithfulness (§7.2); fix NLI + BERTScore configs; **level-token baseline (§8.1 #1)**; truncation; prompted LLM |
| **C — Model core (P0)** | 1 person | yes | A | UHead/VHead v2 (§3); `L_y + L_align + L_orth` (§1b); flan-t5-base; `max_seq_len=128`; **`v0.std()` diagnostic**; latent caching |
| **D — Diffusion (P1)** | 1 person | yes | C (cached latents) | x₀-parameterisation (§6.1); **gist-anchored bridge (§6.4)**; DDIM sampler; CFG; EMA; vectorised schedule |
| **E — G_ψ + decoder (P2)** | 1 person | yes | C + D | cross-attention cue + self-attn ON (§4); **`L_proj` (§4a)**; level token (§5); frozen-decoder test; `RecallEngine` (§11) |
| **F — Downstream QA** | 1 person | light | — (independent) | pick LOCOMO/LongMemEval; harness; RAG baseline; `min_depth` labels; cost accounting |
| **G — Ablations** | whoever's free | yes | E | slot masking heatmap (§1c-i); shuffled cue; no-G_ψ; schedule variants; λ sweeps |

**Dependency graph:** `A → {B, C}`, `C → D → E → G`, `F` runs in parallel throughout.

### Three go/no-go gates

- **Gate 1 (end of A).** If the **ground-truth ceiling** on monotonicity/nesting is low, the *data*
  is the problem — regenerate ladders before training anything. Costs one afternoon, could save two
  weeks.
- **Gate 2 (after B's level-token baseline).** If a plain level-token seq2seq already achieves good
  IU-decay *and* nesting, **the diffusion stack has no justification on this task.** Pivot the claim
  to amortisation/cost or move straight to the QA setting. Do not spend GPU-weeks discovering this
  in month two.
- **Gate 3 (after the shuffled-cue control).** If shuffling `u` doesn't hurt recall, `u` is
  decoration — fix §1b/§3 before investing further in cue-conditioning claims.

### Suggested first week

| Day | A | B | C | F |
|---|---|---|---|---|
| 1 | reorg + `level_map` | IU metric | `v0.std()` diagnostic | pick benchmark |
| 2 | ladder `[x]+xt`, stratified `t` | **ceiling numbers → Gate 1** | UHead/VHead v2 | harness skeleton |
| 3 | `x⁺` precompute | level-token baseline | `L_y/L_align/L_orth` | RAG baseline |
| 4 | control ladders | **baseline eval → Gate 2** | P0 run #1 | `min_depth` labelling |
| 5 | overfit-10 smoke tests | fix NLI/BERTScore | cache latents → hand to D | cost accounting |

---

## 14. Open questions for you

1. **Do we fix the current stack, or restart P0 with flan-t5-base?** My call: restart. `max_seq_len`
   (#6) and the head redesign (§3) both invalidate existing P0 checkpoints anyway. But keep the old
   checkpoints in `reference/` — we need one to run the `v0.std()` diagnostic.
2. **MSR forever, or pivot to LOCOMO/LongMemEval now?** The paper already signals the pivot. Doing
   both doubles the eval surface. My call: keep MSR as the controlled testbed (it's where the
   countable metrics are clean) and add **one** external benchmark for the downstream claim.
3. **Freeze the decoder in P2, or fine-tune?** I want to *try* frozen (§4a) because it makes the
   claim much cleaner. Needs `L_proj` working first.
4. **Is `MSR-Ladder-Ctrl` (rule-based) worth building?** I think yes — it's $0 and it single-handedly
   fills the ceiling `\tbd{}`, the ablation ground truth, and the abstraction-vs-corruption metric.
5. **Gist-anchored diffusion (§6.4) — in or out?** It's the most interesting idea in this doc and
   it's ~15 lines. I'd make it a headline experiment, not an ablation.
