"""Gate 2 -- the level-token seq2seq baseline.

Input `"level {k}: {x}"`, output `xt[k]`. Same model, same data, no diffusion, no latents, no
gist code. About one GPU-hour.

This is the baseline that decides whether the project has a claim. If a plain conditional
seq2seq matches the diffusion stack on detail decay *and* nesting, then the machinery is
unjustified on this task and the honest move is to pivot -- to amortisation across budgets, or to
the downstream QA setting -- rather than to spend GPU-weeks discovering it later.

Two things it structurally cannot do, which is where our fallback claim lives:
  * it regenerates from scratch per budget, so producing K levels costs K full forward passes
    rather than one shared descent;
  * it has no nesting guarantee -- nothing ties level k+1 to level k.
Both are measured here so the comparison is on the record either way.

    python -m baselines.level_token --preset full --epochs 30
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class LevelTokenDataset(Dataset):
    """Flattens each ladder into (level-prefixed source -> rung) pairs."""

    def __init__(self, path: str, tokenizer, max_seq_len: int = 128,
                 include_source_as_level0: bool = True):
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        self.pairs: list[tuple[str, str, int]] = []
        for r in records:
            x = (r.get("x") or "").strip()
            xt = [s.strip() for s in (r.get("xt") or []) if s and s.strip()]
            if not x or not xt:
                continue
            levels = ([x] + xt) if include_source_as_level0 else xt
            for k, target in enumerate(levels):
                self.pairs.append((f"level {k}: {x}", target, k))
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        src, tgt, k = self.pairs[i]
        enc = self.tokenizer(src, padding="max_length", truncation=True,
                             max_length=self.max_seq_len, return_tensors="pt")
        lab = self.tokenizer(tgt, padding="max_length", truncation=True,
                             max_length=self.max_seq_len, return_tensors="pt")["input_ids"][0]
        lab[lab == self.tokenizer.pad_token_id] = -100
        return {"input_ids": enc["input_ids"][0], "attention_mask": enc["attention_mask"][0],
                "labels": lab, "level": k}


def train(model, loader, device, epochs, lr=1e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        total = 0.0
        for b in tqdm(loader, desc=f"level-token ep{ep+1}", leave=False):
            opt.zero_grad(set_to_none=True)
            out = model(input_ids=b["input_ids"].to(device),
                        attention_mask=b["attention_mask"].to(device),
                        labels=b["labels"].to(device))
            out.loss.backward()
            opt.step()
            total += float(out.loss)
        print(f"  epoch {ep+1}: loss {total/max(len(loader),1):.4f}")
    return model


@torch.no_grad()
def generate_ladders(model, tokenizer, sources, n_levels, device, max_seq_len=128,
                     batch_size=16, num_beams=4):
    """One full forward pass per level -- the cost the diffusion path amortises."""
    model.eval()
    ladders = [[] for _ in sources]
    nfe = 0
    t0 = time.time()
    for k in range(n_levels):
        for i in range(0, len(sources), batch_size):
            chunk = sources[i:i + batch_size]
            enc = tokenizer([f"level {k}: {s}" for s in chunk], padding=True, truncation=True,
                            max_length=max_seq_len, return_tensors="pt").to(device)
            ids = model.generate(**enc, max_new_tokens=64, num_beams=num_beams,
                                 no_repeat_ngram_size=3)
            for j, txt in enumerate(tokenizer.batch_decode(ids, skip_special_tokens=True)):
                ladders[i + j].append(txt)
            nfe += len(chunk)
    return ladders, {"decoder_calls": nfe, "wall_seconds": round(time.time() - t0, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Gate 2: level-token seq2seq baseline")
    ap.add_argument("--model", default="google/flan-t5-base")
    ap.add_argument("--data-dir", default="data/final")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--n-levels", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="runs/baselines/level_token.json")
    ap.add_argument("--nli", action="store_true")
    args = ap.parse_args(argv)

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from evaluation.ceiling import load_split
    from evaluation.evaluate import add_nli, budget_quality, score_ladders

    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)

    ds = LevelTokenDataset(os.path.join(args.data_dir, "train.json"), tok, args.max_seq_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    print(f"{len(ds)} (level, source) training pairs")
    train(model, loader, device, args.epochs, args.lr)

    test = load_split(os.path.join(args.data_dir, "test.json"))
    if args.limit:
        test = test[: args.limit]
    sources = [r["x"] for r in test]
    ladders, cost = generate_ladders(model, tok, sources, args.n_levels, device,
                                     args.max_seq_len)

    results = score_ladders(sources, ladders)
    results["early_exit"] = budget_quality(sources, ladders)
    results["cost"] = cost
    results["baseline"] = "level_token_seq2seq"
    if args.nli:
        add_nli(results, ladders)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": results,
                   "samples": [{"source": s, "ladder": l}
                               for s, l in zip(sources[:25], ladders[:25])]}, f, indent=2)

    print("\n" + "=" * 70)
    print("GATE 2 -- level-token seq2seq")
    print(f"  monotonic          {results['frac_strictly_monotonic']:.3f}")
    print(f"  decay spearman     {results['mean_decay_spearman']:.3f}")
    print(f"  perfectly nested   {results['frac_perfectly_nested']:.3f}")
    print(f"  hallucination      {results['mean_hallucination']:.3f}")
    print(f"  decoder calls      {cost['decoder_calls']} for {args.n_levels} levels "
          f"({cost['decoder_calls'] // max(len(sources),1)} per example)")
    print("\nIf this matches the diffusion stack on monotonicity AND nesting, the diffusion")
    print("machinery is unjustified on this task -- pivot to amortisation or to the QA setting.")
    print(f"\nwritten to {args.out}")
    return results


if __name__ == "__main__":
    main()
