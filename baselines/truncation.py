"""Truncation baseline -- the sharpest test of the early-exit claim, and it costs no GPU.

Take the full detailed text and cut it to the token budget. Reviewers will ask "why not just
generate the whole thing and truncate?", and without this the early-exit argument is unfalsifiable.

Our expected advantage is not length but *form*: a truncated prefix is incomplete **and
potentially misleading** (it can end mid-clause, and it keeps early specifics while silently
dropping later ones), whereas a rung is complete-but-coarse. That difference shows up as
well-formedness and as which information types survive, not as raw token count.

Three variants:
  prefix    -- first N tokens
  sentence  -- first N tokens rounded down to a sentence boundary (a charitable version)
  head_tail -- half from the start and half from the end (the LLMLingua-style heuristic)

    python -m baselines.truncation --data-dir data/final --split test
"""

from __future__ import annotations

import argparse
import json
import os
import re

MODES = ("prefix", "sentence", "head_tail")


def truncate(text: str, n_tokens: int, mode: str = "prefix") -> str:
    words = text.split()
    if n_tokens >= len(words):
        return text
    if n_tokens <= 0:
        return ""
    if mode == "prefix":
        return " ".join(words[:n_tokens])
    if mode == "head_tail":
        h = max(1, n_tokens // 2)
        return " ".join(words[:h]) + " ... " + " ".join(words[-(n_tokens - h):])
    if mode == "sentence":
        cut = " ".join(words[:n_tokens])
        parts = re.split(r"(?<=[.!?])\s+", cut)
        return " ".join(parts[:-1]).strip() if len(parts) > 1 else cut
    raise ValueError(mode)


def is_well_formed(text: str) -> bool:
    """Cheap proxy: ends on terminal punctuation and is not a bare fragment."""
    t = text.strip()
    return bool(t) and t[-1] in ".!?" and len(t.split()) >= 3


def build_ladders(sources, reference_lengths, mode="prefix"):
    """Match each rung's budget to the gold rung length, so the comparison is at equal cost."""
    ladders = []
    for src, lens in zip(sources, reference_lengths):
        ladders.append([truncate(src, n, mode) for n in lens])
    return ladders


def main(argv=None):
    ap = argparse.ArgumentParser(description="Truncation baseline for early exit")
    ap.add_argument("--data-dir", default="data/final")
    ap.add_argument("--split", default="test")
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="runs/baselines/truncation.json")
    args = ap.parse_args(argv)

    from evaluation.ceiling import load_split
    from evaluation.evaluate import budget_quality, score_ladders

    records = load_split(os.path.join(args.data_dir, f"{args.split}.json"))
    if args.limit:
        records = records[: args.limit]
    sources = [r["x"] for r in records]
    ref_lengths = [[len(s.split()) for s in r["levels"]] for r in records]

    out: dict = {}
    for mode in args.modes:
        ladders = build_ladders(sources, ref_lengths, mode)
        res = score_ladders(sources, ladders)
        res["early_exit"] = budget_quality(sources, ladders)
        res["well_formed_rate"] = sum(
            is_well_formed(r) for lad in ladders for r in lad
        ) / max(sum(len(l) for l in ladders), 1)
        res["baseline"] = f"truncation_{mode}"
        out[mode] = res

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"{'mode':<12} {'monotonic':>10} {'nested':>8} {'halluc':>8} {'well-formed':>12}")
    print("-" * 54)
    for mode, r in out.items():
        print(f"{mode:<12} {r['frac_strictly_monotonic']:>10.3f} "
              f"{r['frac_perfectly_nested']:>8.3f} {r['mean_hallucination']:>8.3f} "
              f"{r['well_formed_rate']:>12.3f}")
    print("\nTruncation is trivially monotonic and nested -- a prefix cannot add information.")
    print("The comparison that matters is well-formedness and which detail types survive:")
    print("truncation keeps whatever came first, not whatever matters.")
    print(f"\nwritten to {args.out}")
    return out


if __name__ == "__main__":
    main()
