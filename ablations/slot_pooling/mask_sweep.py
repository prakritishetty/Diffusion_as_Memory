"""Slot-masking ablation.

Two experiments, because "do slots specialise?" and "can slots be made to specialise?" are
different questions and the honest answer to the first is probably no.

**Discovery** (`mask_sweep`). Replace slot j with the dataset-mean slot and measure the drop in
retained information units *by type*. Zeroing a slot is off-distribution, so the mean is used
instead. The result is an `L x C` heatmap and it is a usable figure either way:
  * sparse  -> slots specialise, and forgetting is targetable;
  * flat    -> the representation is distributed and redundant, which is a real finding and
               actively justifies diffusing all slots jointly rather than dropping them.

**Enforcement** lives in the P0 loss (`models/losses.slot_type_loss`): per-slot heads predict the
entity types present, so masking a slot *must* remove that type. The control ladders give this a
ground truth, since they record which rung removed which attribute class.

    python -m ablations.slot_pooling.mask_sweep --p0 runs/full/p0/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from tqdm import tqdm


@torch.no_grad()
def mean_slot(model, loader, device, max_batches: int = 20) -> torch.Tensor:
    acc, n = None, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        slots = model.encode(batch["x_input_ids"].to(device),
                             batch["x_attention"].to(device)).slots
        acc = slots.sum(dim=(0, 1)) if acc is None else acc + slots.sum(dim=(0, 1))
        n += slots.shape[0] * slots.shape[1]
    return acc / max(n, 1)


@torch.no_grad()
def decode_with_mask(model, batch, device, slot_ids, replacement):
    from models.slot_pooling_prep.slot_pooling import mask_slots

    H = model.encoder(input_ids=batch["x_input_ids"].to(device),
                      attention_mask=batch["x_attention"].to(device))
    slots = model.slot_pooling(H, batch["x_attention"].to(device))
    if slot_ids:
        slots = mask_slots(slots, slot_ids, replacement)
    v0 = model.v_head(slots)
    level = torch.zeros(v0.shape[0], dtype=torch.long, device=device)
    return model.generate_from_latent(v0, level=level)


def run(p0_path: str, data_dir: str | None, split: str, limit: int, device: str, out: str):
    from dataloader.memory_dataset import MemoryLadderDataset, collate
    from evaluation.info_units import UNIT_TYPES, Extractor
    from models.memory_model import MemoryModel
    from training.common import build_tokenizer, config_from_checkpoint, load_checkpoint
    from torch.utils.data import DataLoader

    cfg = config_from_checkpoint(p0_path)
    dev = torch.device(device if device != "auto"
                       else ("cuda" if torch.cuda.is_available() else "cpu"))
    tok = build_tokenizer(cfg)
    model = MemoryModel(cfg).to(dev)
    model.load_state_dict(load_checkpoint(p0_path, dev)["model_state_dict"], strict=False)
    model.drop_gist_decoder()
    model.eval()

    ds = MemoryLadderDataset(os.path.join(data_dir or cfg.data.data_dir, f"{split}.json"),
                             tok, cfg, split=split)
    idxs = list(range(min(limit, len(ds))))
    loader = DataLoader([ds[i] for i in idxs], batch_size=8, collate_fn=collate)
    batches = list(loader)

    ex = Extractor()
    replacement = mean_slot(model, batches, dev)
    L = cfg.model.n_slots

    def retention(slot_ids):
        got: dict[str, list[float]] = {t: [] for t in UNIT_TYPES}
        for batch in batches:
            ids = decode_with_mask(model, batch, dev, slot_ids, replacement)
            preds = tok.batch_decode(ids, skip_special_tokens=True)
            for src, pred in zip(batch["x_text"], preds):
                s, p = ex.by_type(src), ex.by_type(pred)
                for t in UNIT_TYPES:
                    if s[t]:
                        got[t].append(len(p[t] & s[t]) / len(s[t]))
        return {t: (sum(v) / len(v) if v else None) for t, v in got.items()}

    base = retention([])
    heat: dict[int, dict[str, float | None]] = {}
    for j in tqdm(range(L), desc="slot sweep"):
        masked = retention([j])
        heat[j] = {t: (None if base[t] is None or masked[t] is None
                       else round(base[t] - masked[t], 4))
                   for t in UNIT_TYPES}

    types = [t for t in UNIT_TYPES if base[t] is not None]
    drops = [[heat[j][t] for t in types] for j in range(L)]
    flat = [d for row in drops for d in row if d is not None]
    spread = (max(flat) - min(flat)) if flat else 0.0

    result = {
        "checkpoint": p0_path, "split": split, "n_examples": len(idxs),
        "baseline_retention": {t: base[t] for t in types},
        "heatmap": heat, "types": types,
        "specialisation_spread": round(spread, 4),
        "verdict": ("slots specialise" if spread > 0.15 else
                    "representation is distributed/redundant"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'slot':<6}" + "".join(f"{t:>9}" for t in types))
    print("-" * (6 + 9 * len(types)))
    for j in range(L):
        print(f"{j:<6}" + "".join(
            f"{(heat[j][t] if heat[j][t] is not None else 0):>9.3f}" for t in types))
    print(f"\nspread {spread:.3f} -> {result['verdict']}")
    print("A flat heatmap is a publishable result: it says the latent is redundant, which is")
    print("itself an argument for diffusing all slots jointly rather than dropping them.")
    print(f"\nwritten to {out}")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="Slot-masking attribution sweep")
    ap.add_argument("--p0", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="runs/ablations/slot_mask.json")
    a = ap.parse_args(argv)
    return run(a.p0, a.data_dir, a.split, a.limit, a.device, a.out)


if __name__ == "__main__":
    main()
