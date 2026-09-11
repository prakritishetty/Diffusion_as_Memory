"""Gate 1 -- the ground-truth ceiling.

Runs the whole metric suite on the *reference* ladders rather than on model output. No GPU, no
model, a couple of minutes.

This is the highest value-per-hour item in the plan and the paper currently carries it as a
`\\tbd{}`. Without it, "32.5% of trajectories are monotonic" is uninterpretable: if the gold
ladders themselves are only 60% monotonic then we have a data problem, and if they are 98%
monotonic then we have a modelling problem. Those two readings imply completely different work.

It also reports the supervision-map coverage under each rung map, which turns the paper's
diagnosis of supervision collapse from an argument into a measurement.

    python -m evaluation.ceiling --data-dir data/final --out runs/ceiling.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from diffusion.level_map import LevelMap
from diffusion.schedule import NoiseSchedule
from evaluation.info_units import Extractor
from evaluation.metrics import aggregate, evaluate_trajectory


def load_split(path: str, include_source: bool = True) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    out = []
    for i, r in enumerate(records):
        x = (r.get("x") or "").strip()
        xt = [s.strip() for s in (r.get("xt") or []) if s and s.strip()]
        if not x or not xt:
            continue
        out.append({
            "id": str(r.get("id", i)),
            "x": x,
            "y": (r.get("y") or "").strip(),
            "levels": ([x] + xt) if include_source else xt,
        })
    return out


def ladder_stats(records: list[dict]) -> dict:
    counts = Counter(len(r["levels"]) for r in records)
    n = max(len(records), 1)
    lengths = [len(r["x"].split()) for r in records]
    return {
        "n_records": len(records),
        "mean_levels": round(sum(len(r["levels"]) for r in records) / n, 3),
        "min_levels": min(counts) if counts else 0,
        "max_levels": max(counts) if counts else 0,
        "level_histogram": dict(sorted(counts.items())),
        "x_words_mean": round(sum(lengths) / n, 1),
        "x_words_p95": sorted(lengths)[int(0.95 * len(lengths))] if lengths else 0,
        "x_words_max": max(lengths) if lengths else 0,
    }


def supervision_coverage(records: list[dict], T: int = 1000) -> dict:
    """How the timestep -> rung map distributes supervision, per map variant."""
    sched = NoiseSchedule(T=T, schedule_type="cosine")
    level_counts = [len(r["levels"]) for r in records]
    out: dict = {}
    for kind in ("bucket", "uniform_t", "logsnr"):
        lm = LevelMap(sched, kind=kind)
        per_n = lm.coverage_report(level_counts)
        weighted_final, total = 0.0, 0
        worst = 0.0
        for n, c in Counter(level_counts).items():
            weighted_final += per_n[n]["fraction_on_final_level"] * c
            worst = max(worst, per_n[n]["max_minus_min"])
            total += c
        out[kind] = {
            "frac_timesteps_on_final_rung": round(weighted_final / max(total, 1), 4),
            "worst_rung_imbalance": round(worst, 4),
        }
    return out


def ceiling_for_split(records: list[dict], extractor: Extractor, hypernyms: bool) -> dict:
    metrics = [
        evaluate_trajectory(r["x"], r["levels"], extractor, check_hypernyms=hypernyms)
        for r in records
    ]
    agg = aggregate(metrics)
    agg["ladder_stats"] = ladder_stats(records)
    agg["supervision_coverage"] = supervision_coverage(records)
    agg["extractor_backend"] = extractor.backend
    return agg


def main(argv=None):
    ap = argparse.ArgumentParser(description="Gate 1: ground-truth ceiling on the gold ladders")
    ap.add_argument("--data-dir", default="data/final")
    ap.add_argument("--splits", nargs="+", default=["train", "validate", "test"])
    ap.add_argument("--out", default="runs/ceiling.json")
    ap.add_argument("--hypernyms", action="store_true",
                    help="also score abstraction validity (needs nltk wordnet)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    extractor = Extractor()
    print(f"information-unit backend: {extractor.backend}")
    if extractor.backend == "heuristic":
        print("  NOTE: install spaCy + en_core_web_sm for the reported numbers")

    results: dict = {}
    for split in args.splits:
        path = os.path.join(args.data_dir, f"{split}.json")
        if not os.path.exists(path):
            print(f"  skip {split} (missing)")
            continue
        records = load_split(path)
        if args.limit:
            records = records[: args.limit]
        print(f"  {split}: {len(records)} trajectories")
        results[split] = ceiling_for_split(records, extractor, args.hypernyms)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 78)
    print("GROUND-TRUTH CEILING  (what the gold ladders themselves achieve)")
    print("=" * 78)
    hdr = f"{'split':<10} {'monotonic':>10} {'spearman':>10} {'nested':>9} {'halluc':>8}"
    print(hdr)
    print("-" * 78)
    for split, r in results.items():
        print(f"{split:<10} {r['frac_strictly_monotonic']:>10.3f} "
              f"{r['mean_decay_spearman']:>10.3f} {r['frac_perfectly_nested']:>9.3f} "
              f"{r['mean_hallucination']:>8.3f}")

    for split, r in results.items():
        print(f"\n[{split}] detail retention by rung: {r['retention_curve']}")
        for t, curve in r["retention_curve_by_type"].items():
            print(f"    {t:<8} {curve}")
        print(f"    ladder levels: mean {r['ladder_stats']['mean_levels']}, "
              f"histogram {r['ladder_stats']['level_histogram']}")
        print("    supervision coverage (fraction of timesteps landing on the final rung):")
        for kind, cov in r["supervision_coverage"].items():
            print(f"      {kind:<10} {cov['frac_timesteps_on_final_rung']:.3f}  "
                  f"(worst rung imbalance {cov['worst_rung_imbalance']:.3f})")
        break

    print(f"\nwritten to {args.out}")
    return results


if __name__ == "__main__":
    main()
