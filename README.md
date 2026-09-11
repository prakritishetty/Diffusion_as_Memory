# Diffusion as Memory

Store a memory once, read it back at any level of detail, and stop as soon as enough detail has
surfaced.

A memory is split into a **gist** `u` (the general point) and a **detail latent** `v0` (names,
numbers, dates). Only the details are diffused, so forward diffusion is graded forgetting and the
reverse process — conditioned on the gist as a cue — is graded recall. Reading out at any point
along the trajectory gives a shorter, vaguer, but still *true* version of the memory.

> Retrieval is a switch; we want a dial. And a dial is only useful if turning it down never makes
> you wrong, only less specific.

Design rationale, the full audit of the previous implementation, and the experiment plan are in
[RESEARCH_PLAN.md](RESEARCH_PLAN.md). The previous codebase — including the Lovelace
latent-diffusion fork this started from — is preserved under [reference/](reference/).

---

## Quick start

```bash
uv venv --python 3.12 .venv          # or: python -m venv .venv
.venv/Scripts/activate               # Linux/macOS: source .venv/bin/activate
uv pip install -r requirements.txt
python -m spacy download en_core_web_sm

python -m pytest tests/ -q           # 133 tests, ~50s, no GPU, no network
python -m tools.smoke                # full P0->P1->P2->eval pipeline on CPU, ~40s
```

`tools.smoke` is the pre-GPU gate: it runs every training phase and the evaluation end to end with
a ~1M-parameter stand-in model. If it passes, the remaining unknowns are scientific rather than
mechanical.

---

## Layout

```
configs/        one source of truth for every hyperparameter (tiny/dev/small/full/legacy)
data/           final/ = LLM ladders, control/ = rule-based deletion ladders
data_pipeline/  build control ladders, generate contrastive positives
dataloader/     ladder dataset: positives, labels, cached tokenisation
diffusion/      schedule, timestep->rung map, forward process, DDIM sampler
models/         encoder, slot pooling, u/v heads, denoiser, G_psi, decoders, losses
training/       train_p0 (autoencoder), train_p1 (denoiser), train_p2 (projection)
inference/      RecallEngine -- the single implementation of forget/recall
evaluation/     information units, metrics, ceiling, NLI, evaluate
baselines/      level-token seq2seq, truncation, prompted LLM, RAG
ablations/      cue controls, oracle/noise brackets, slot masking
tools/          smoke test, v0-scale diagnostic, ladder comparison, deck, Azure setup
tests/          math, shapes, gradient flow, contracts, end to end
reference/      previous runs, scripts, docs, and the original fork
```

---

## Running it

```bash
# Gate 0 -- is the latent at the scale the diffusion process assumes? (10 min)
python -m tools.diagnose_v0_scale --p0 <checkpoint>

# Gate 1 -- what do the gold ladders themselves achieve? (no GPU)
python -m evaluation.ceiling --data-dir data/final

# Gate 2 -- does a plain conditional seq2seq already do this? (~1 GPU-h)
python -m baselines.level_token --epochs 30

# Train
python -m training.train_p0 --preset full --name main
python -m training.train_p1 --p0-checkpoint runs/main/p0/best_model.pt
python -m training.train_p2 --p0-checkpoint runs/main/p0/best_model.pt \
                            --p1-checkpoint runs/main/p1/best_model.pt

# Evaluate and ablate
python -m evaluation.evaluate --p0 ... --p1 ... --p2 ... --split test
python -m ablations.runner    --p0 ... --p1 ... --p2 ... --only shuffled_cue oracle_latent
```

Reference points, computable from data alone:

```bash
python -m tools.compare_ladders
```

---

## How it is evaluated

Detail is **counted**, not embedded. BERTScore cannot distinguish "Paris" from "London", which
makes it close to useless for a claim about *which* specific details survive. Typed information
units — numbers, dates, people, places, organisations, content words — are extracted with spaCy
and compared as sets.

| Metric | Question |
|---|---|
| retention by rung | how much source detail survives at each depth |
| decay Spearman | does detail fall monotonically with depth |
| nesting violations | does a coarser rung ever assert something the finer one did not |
| hallucination | did it *forget* a detail or *invent* one |
| abstraction validity | were replacements genuine hypernyms, or corruption |
| well-formedness | is the rung a standalone sentence |
| **usable rung rate** | **all of the above at once — the headline** |

Monotonicity and nesting alone are not a claim: prefix truncation scores 0.93 and 0.73 on them
while being a sentence fragment ~92% of the time. Measured on the MSR test split:

| Ladder source | monotonic | nested | hallucination | well-formed | **usable** |
|---|---|---|---|---|---|
| Gold, LLM-written | 0.92 | 0.05 | 0.24 | 0.86 | **0.33** |
| Gold, rule-based control | 1.00 | 0.03 | 0.09 | 0.99 | **0.56** |
| Truncation (prefix) | 0.93 | 0.73 | 0.02 | 0.08 | **0.08** |

The model has to clear 0.33 to be interesting.

---

## Caching

Everything expensive and deterministic is content-addressed on disk: tokenisation, frozen gist
embeddings, `(u, v0)` after P0, per-rung latents for `L_proj`, and DDIM trajectories. Cache keys
include a code version and the source checkpoint hash, so a stale entry cannot silently feed the
wrong latents into training. `--no-cache` disables it; `CacheStore.purge()` clears it.

---

## Compute

Roughly 150–220 GPU-hours for the full programme, all of which fits in 16–24 GB. See
[tools/azure/README.md](tools/azure/README.md) for setup — note that a fresh Azure subscription has
**zero GPU quota** and approval takes time, so request it first.
