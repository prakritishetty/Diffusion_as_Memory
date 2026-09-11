"""Side-by-side comparison of ladder sources on the graded-recall metrics.

Runs entirely on data -- no model, no GPU. Use it to sanity-check that a new metric or corpus
behaves before spending GPU time, and to keep the reference points (gold ladders, truncation) on
the record next to model output.

    python -m tools.compare_ladders
    python -m tools.compare_ladders --predictions runs/full/eval_test.json
"""

from __future__ import annotations

import argparse
import json
import os

COLUMNS = [
    ("monot", "frac_strictly_monotonic"),
    ("spear", "mean_decay_spearman"),
    ("nested", "frac_perfectly_nested"),
    ("halluc", "mean_hallucination"),
    ("wellfrm", "mean_well_formed"),
    ("USABLE", "mean_usable_rung_rate"),
]


def _print_table(rows):
    head = f"{'setting':<20}" + "".join(f"{name:>9}" for name, _ in COLUMNS)
    print(head)
    print("-" * len(head))
    for name, r in rows:
        line = f"{name:<20}" + "".join(f"{r.get(key, float('nan')):>9.3f}" for _, key in COLUMNS)
        print(line)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare ladder sources on graded-recall metrics")
    ap.add_argument("--natural", default="data/final")
    ap.add_argument("--control", default="data/control")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--predictions", nargs="*", default=[],
                    help="eval_*.json files produced by evaluation.evaluate")
    ap.add_argument("--hypernyms", action="store_true")
    args = ap.parse_args(argv)

    from baselines.truncation import build_ladders
    from evaluation.ceiling import load_split
    from evaluation.evaluate import score_ladders
    from evaluation.info_units import Extractor

    ex = Extractor()
    rows = []

    for label, d in (("gold: natural", args.natural), ("gold: control", args.control)):
        path = os.path.join(d, f"{args.split}.json")
        if not os.path.exists(path):
            continue
        recs = load_split(path)[: args.limit]
        rows.append((label, score_ladders([r["x"] for r in recs], [r["levels"] for r in recs],
                                          ex, args.hypernyms)))

    nat = os.path.join(args.natural, f"{args.split}.json")
    if os.path.exists(nat):
        recs = load_split(nat)[: args.limit]
        src = [r["x"] for r in recs]
        lens = [[len(s.split()) for s in r["levels"]] for r in recs]
        for mode in ("prefix", "sentence", "head_tail"):
            rows.append((f"truncate: {mode}",
                         score_ladders(src, build_ladders(src, lens, mode), ex, args.hypernyms)))

    for p in args.predictions:
        with open(p, encoding="utf-8") as f:
            blob = json.load(f)
        if "summary" in blob:
            rows.append((os.path.basename(p)[:20], blob["summary"]))

    _print_table(rows)
    print("\nMonotonicity and nesting alone are achievable with scissors: prefix truncation")
    print("scores well on both while being a sentence fragment ~90% of the time. USABLE")
    print("(well formed AND adds nothing specific AND invents nothing) is the column that")
    print("separates graded recall from cutting a string.")
    return rows


if __name__ == "__main__":
    main()
