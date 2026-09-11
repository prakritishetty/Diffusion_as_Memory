# Azure setup

Nothing here has been run — it is written so that when you have a subscription you execute it in
order. Two paths: a **single GPU VM** (simplest, start here) or **Azure ML** (autoscaling
cluster, better for the sweeps).

---

## 0. Do this first — it is the long pole

**A fresh subscription has zero GPU quota.** You cannot create an NC/ND/NV VM until a quota
increase is approved, and that can take a day or more. Request it before anything else; it is free
to ask.

Portal → *Subscriptions* → your subscription → *Usage + quotas* → filter by region and by
`Standard NCASv3_T4 Family vCPUs` (or the family you want) → *Request increase*. Ask for at least
8 vCPUs.

Also check what you already have: **Azure for Students** gives ~$100, and a Microsoft internal
subscription changes the maths entirely. Confirm before paying for anything.

---

## 1. Pick a VM size

This workload fits in 24 GB (flan-t5-base, batch 16, sequence 128), so you are not forced onto
A100s.

| Size | GPU | VRAM | Rough $/h | Notes |
|---|---|---|---|---|
| `Standard_NC4as_T4_v3` | T4 | 16 GB | ~0.55 | cheapest workable; **fp16 only, no bf16** |
| `Standard_NC8as_T4_v3` | T4 | 16 GB | ~0.75 | more CPU for dataloading |
| `Standard_NV36ads_A10_v5` | A10 | 24 GB | ~3.2 | bf16, comfortable |
| `Standard_NC24ads_A100_v4` | A100 | 80 GB | ~3.7 | overkill here |

The T4 tier is the right default. If you use T4, set `train.amp = False` or switch the autocast
dtype to fp16 — `training/common.autocast_ctx` currently requests bf16, which T4 does not support.

**Spot / low-priority cuts 60–90%** but can be evicted mid-run, so only use it once
checkpoint-resume is wired (see §6).

---

## 2. Create the resources

```bash
az login
az account set --subscription "<SUBSCRIPTION_ID>"

RG=diffusion-memory
LOC=eastus            # pick by GPU availability; keep storage in the SAME region
az group create -n $RG -l $LOC

# Storage for data, cache and checkpoints
az storage account create -n diffmemstore -g $RG -l $LOC --sku Standard_LRS
az storage container create --account-name diffmemstore -n data
az storage container create --account-name diffmemstore -n cache
az storage container create --account-name diffmemstore -n checkpoints

# Cost guardrail on day one, not day thirty
az consumption budget create --budget-name diffmem --amount 150 --category Cost \
  --time-grain Monthly --resource-group $RG
```

---

## 3a. Path A — a single GPU VM

```bash
az vm create -g $RG -n diffmem-gpu \
  --image Ubuntu2204 --size Standard_NC4as_T4_v3 \
  --admin-username azureuser --generate-ssh-keys \
  --os-disk-size-gb 128

az vm open-port -g $RG -n diffmem-gpu --port 22
az vm show -d -g $RG -n diffmem-gpu --query publicIps -o tsv
```

On the VM:

```bash
# NVIDIA driver + CUDA
sudo apt update && sudo apt install -y python3.11-venv build-essential
curl -fsSL https://raw.githubusercontent.com/Azure/azhpc-images/master/ubuntu/install_nvidiagpudriver.sh | sudo bash
nvidia-smi   # confirm before going further

# Mount blob storage so cache and checkpoints survive a rebuild
sudo apt install -y blobfuse2
# see blobfuse2.yaml in this folder
sudo blobfuse2 mount /mnt/blob --config-file=blobfuse2.yaml

git clone <repo> && cd Diffusion_as_Memory
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

python -m tools.smoke          # ~1 min; verifies the wiring on this machine
```

**Deallocate when idle.** A merely *stopped* VM still bills:

```bash
az vm deallocate -g $RG -n diffmem-gpu
```

---

## 3b. Path B — Azure ML (use this for the sweeps)

Right for §13 of the research plan, which is many short runs: the cluster scales to zero between
jobs and supports spot pricing.

```bash
az extension add -n ml
az ml workspace create -n diffmem-ws -g $RG -l $LOC

az ml compute create -n gpu-cluster -g $RG -w diffmem-ws \
  --type AmlCompute --size Standard_NC4as_T4_v3 \
  --min-instances 0 --max-instances 4 --tier low_priority

az ml data create -n msr-ladders -g $RG -w diffmem-ws \
  --type uri_folder --path ./data/final
```

Then submit jobs with the YAML in this folder:

```bash
az ml job create -f aml/p0.yaml -g $RG -w diffmem-ws
az ml job create -f aml/sweep_p2.yaml -g $RG -w diffmem-ws
```

---

## 4. Storage layout

```
blob://data/         data/final, data/control          (read-mostly)
blob://cache/        tokenised, gist embeddings, p0_latents, rung_latents, trajectories
blob://checkpoints/  runs/<name>/{p0,p1,p2}/best_model.pt
```

`blobfuse2` is slow on many small files, which is why the cache is a handful of large tensor files
rather than thousands of little ones. Keep `cache.root` on the mounted share so a spot eviction
does not throw away hours of precomputation.

---

## 5. Cost estimate

Per run, from the plan: P0 ≈ 2–3 h, P1 < 0.5 h (cached latents), P2 ≈ 2 h, eval ≈ 0.5–1 h.
Whole programme ≈ 150–220 GPU-hours.

| Tier | $/h | Total |
|---|---|---|
| T4 low-priority | ~0.15 | **$25–35** |
| T4 on-demand | ~0.55 | **$85–120** |
| A10 on-demand | ~3.2 | **$480–700** |

Use low-priority T4 for sweeps and on-demand only for the final runs.

---

## 6. Before using spot instances

Eviction is a *when*, not an *if*. Required:

1. periodic checkpointing (every epoch, not only on improvement);
2. `--resume` handling in the training scripts;
3. checkpoints written to mounted blob storage, not the OS disk.

`training/common.save_checkpoint` already writes atomically (`.tmp` then `os.replace`), so a
killed job never leaves a half-written file. Resume support is the remaining piece.

---

## 7. Things that will bite you

- GPU quota is zero by default — see §0.
- A *stopped* VM still bills; you must **deallocate**.
- Storage in a different region from compute costs egress and latency.
- Not every region has every GPU family; pick the region by availability, not habit.
- T4 has no bf16. Set `train.amp = False` or switch to fp16.
- Never put credentials in code or in a notebook. Use `az login`, managed identity, or an
  environment variable read at runtime (`baselines/prompted_llm.py` reads `$NIM_API_KEY` and
  refuses to run without it).
