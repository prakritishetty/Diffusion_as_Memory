"""Ablation runner.

One entry point for every controlled comparison, so they are defined once and named consistently
in the paper. Grouped by what they are for:

**Gates** -- run these first; each can invalidate a claim cheaply.
    shuffled_cue    permute u across the batch. If recall barely changes, `u` is decoration and
                    the cue-conditioning story is dead. Run before anything else.
    no_cue          drop `u` entirely (classifier-free unconditional branch).
    no_gpsi         decode `v_t` directly, isolating G_psi's contribution.

**Brackets** -- every other number should land between these two.
    oracle_latent   encode the *gold* rung text and decode it. Upper bound on the reader, so any
                    remaining gap is attributable to the diffusion path rather than the decoder.
    noise_only      recall from pure noise plus the gist, with no stored detail latent. Lower
                    bound, and simultaneously the "store only u" compression variant: whatever
                    detail appears is reconstructed, not remembered.

**Design choices**
    cue_strength    classifier-free guidance sweep -- a continuous version of shuffled_cue.
    level_map       bucket / uniform_t / logsnr, which turns the supervision-collapse diagnosis
                    from an argument into a measurement.
    repeated_recall noise -> decode -> re-encode -> noise again. Does the memory drift under
                    repeated retrieval? (Reconsolidation.)
    interference    store N memories and recall one by its cue as N and distractor similarity
                    grow. The failure mode RAG is supposed to be good at.

    python -m ablations.runner --p0 ... --p1 ... --p2 ... --only shuffled_cue oracle_latent
"""

from __future__ import annotations

import argparse
import json
import os

import torch

GATES = ("shuffled_cue", "no_cue", "no_gpsi")
BRACKETS = ("oracle_latent", "noise_only")
DESIGN = ("cue_strength", "repeated_recall", "interference")
ALL = ("baseline",) + GATES + BRACKETS + DESIGN


def _summarise(sources, ladders, ex):
    from evaluation.evaluate import budget_quality, score_ladders
    r = score_ladders(sources, ladders, ex)
    r["early_exit"] = budget_quality(sources, ladders, ex)
    r["samples"] = [{"source": s, "ladder": l} for s, l in zip(sources[:5], ladders[:5])]
    return r


def _ladders(engine, records, n_levels, batch_size=8, **kw):
    from evaluation.evaluate import generate_ladders
    return generate_ladders(engine, records, n_levels, batch_size, **kw)


@torch.no_grad()
def noise_only_ladders(engine, records, n_levels, batch_size=8):
    """Recall with the gist but no stored detail latent -- the gist-only storage variant."""
    out: list[list[str]] = []
    T = engine.T
    for i in range(0, len(records), batch_size):
        chunk = records[i:i + batch_size]
        u, v0 = engine.encode_texts([r["x"] for r in chunk])
        anchor = engine.diffusion.anchor(u)
        start = torch.randn_like(v0) if anchor is None else anchor + torch.randn_like(v0)
        stops = [engine.level_map.t_for_budget(k / (n_levels - 1)) for k in range(n_levels)]
        _, rec = engine.diffusion.recall(start, T, min(stops), u, seed=0, record_at=stops)
        per = [[] for _ in chunk]
        for t in sorted(stops, reverse=True):
            v = engine.project(rec[t], t, u)
            lvl = int(engine.level_map.level_index(
                torch.tensor([t], device=v.device),
                torch.tensor([n_levels], device=v.device)).item())
            for j, txt in enumerate(engine.decode(v, lvl)):
                per[j].append(txt)
        out.extend([list(reversed(p)) for p in per])
    return out


@torch.no_grad()
def repeated_recall(engine, records, rounds=3, budget=0.5, batch_size=8):
    """Recall, re-encode, recall again. Measures drift under repeated retrieval."""
    texts = [r["x"] for r in records]
    history = [texts]
    t = engine.level_map.t_for_budget(budget)
    for _ in range(rounds):
        nxt: list[str] = []
        for i in range(0, len(texts), batch_size):
            chunk = history[-1][i:i + batch_size]
            u, v0 = engine.encode_texts(chunk)
            v_t = engine.forget(v0, t, u=u, seed=0)
            v_rec, _ = engine.diffusion.recall(v_t, t, 1, u, seed=0)
            nxt.extend(engine.decode(engine.project(v_rec, 1, u), level=0))
        history.append(nxt)
    return history


@torch.no_grad()
def interference(engine, records, sizes=(8, 32, 128), batch_size=8):
    """Does recall degrade as the store grows and cues become less distinctive?"""
    import torch.nn.functional as F
    out = {}
    for n in sizes:
        subset = records[:n]
        if len(subset) < 2:
            continue
        u, _ = engine.encode_texts([r["x"] for r in subset])
        sim = F.normalize(u, dim=-1) @ F.normalize(u, dim=-1).t()
        sim.fill_diagonal_(-1)
        nearest = sim.max(dim=1)
        correct = int((nearest.indices == torch.arange(len(subset), device=u.device)).sum())
        out[n] = {
            "mean_nearest_neighbour_similarity": round(float(nearest.values.mean()), 4),
            "cue_collisions": correct,
            "collision_rate": round(correct / len(subset), 4),
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run controlled ablations")
    ap.add_argument("--p0", required=True)
    ap.add_argument("--p1", required=True)
    ap.add_argument("--p2", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-levels", type=int, default=6)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=list(ALL))
    ap.add_argument("--cue-scales", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="runs/ablations/summary.json")
    args = ap.parse_args(argv)

    from evaluation.ceiling import load_split
    from evaluation.evaluate import build_engine
    from evaluation.info_units import Extractor

    engine, cfg = build_engine(args.p0, args.p1, args.p2, args.device)
    records = load_split(os.path.join(args.data_dir or cfg.data.data_dir,
                                      f"{args.split}.json"))[: args.limit]
    sources = [r["x"] for r in records]
    ex = Extractor()
    res: dict = {}

    def want(name):
        return name in args.only

    if want("baseline"):
        res["baseline"] = _summarise(
            sources, _ladders(engine, records, args.n_levels, args.batch_size), ex)

    if want("shuffled_cue"):
        res["shuffled_cue"] = _summarise(
            sources, _ladders(engine, records, args.n_levels, args.batch_size,
                              shuffle_cue=True), ex)

    if want("no_cue"):
        res["no_cue"] = _summarise(
            sources, _ladders(engine, records, args.n_levels, args.batch_size,
                              cfg_scale=0.0), ex)

    if want("no_gpsi"):
        e2, _ = build_engine(args.p0, args.p1, None, args.device)
        res["no_gpsi"] = _summarise(
            sources, _ladders(e2, records, args.n_levels, args.batch_size), ex)

    if want("oracle_latent"):
        lad = [[r.text for r in l]
               for l in engine.oracle_ladder([r["levels"] for r in records])]
        res["oracle_latent"] = _summarise(sources, lad, ex)

    if want("noise_only"):
        res["noise_only"] = _summarise(
            sources, noise_only_ladders(engine, records, args.n_levels, args.batch_size), ex)

    if want("cue_strength"):
        res["cue_strength"] = {
            str(w): _summarise(sources, _ladders(engine, records, args.n_levels,
                                                 args.batch_size, cfg_scale=w), ex)
            for w in args.cue_scales
        }

    if want("repeated_recall"):
        hist = repeated_recall(engine, records[:50], batch_size=args.batch_size)
        res["repeated_recall"] = {
            "rounds": [
                _summarise(hist[0], [[h] for h in rnd], ex) for rnd in hist[1:]
            ],
            "samples": [{"original": hist[0][i], "recalls": [r[i] for r in hist[1:]]}
                        for i in range(min(5, len(hist[0])))],
        }

    if want("interference"):
        res["interference"] = interference(engine, records)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    print(f"\n{'ablation':<18}{'monot':>8}{'nested':>8}{'halluc':>8}{'wellfrm':>9}{'USABLE':>8}")
    print("-" * 59)
    for name, r in res.items():
        if isinstance(r, dict) and "frac_strictly_monotonic" in r:
            print(f"{name:<18}{r['frac_strictly_monotonic']:>8.3f}"
                  f"{r['frac_perfectly_nested']:>8.3f}{r['mean_hallucination']:>8.3f}"
                  f"{r['mean_well_formed']:>9.3f}{r['mean_usable_rung_rate']:>8.3f}")
    if "baseline" in res and "shuffled_cue" in res:
        d = (res["baseline"]["mean_usable_rung_rate"]
             - res["shuffled_cue"]["mean_usable_rung_rate"])
        print(f"\nGATE: shuffling the cue costs {d:+.3f} usable-rung rate.")
        print("      If that is ~0, `u` is decoration -- fix it before claiming cue-conditioning.")
    print(f"\nwritten to {args.out}")
    return res


if __name__ == "__main__":
    main()
