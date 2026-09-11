"""Evaluation runner: generate ladders, score them, write a report.

Every number here comes from free-running `generate()`. The previous evaluation decoded
`argmax(logits)` from a teacher-forced pass -- the decoder had already been handed the gold prefix
at every position -- which is why the reported BERTScore was so high.

    python -m evaluation.evaluate --p0 ... --p1 ... --p2 ... --split test
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Sequence

import torch
from tqdm import tqdm

from evaluation.info_units import Extractor
from evaluation.metrics import aggregate, early_exit_curve, evaluate_trajectory


def score_ladders(
    sources: Sequence[str],
    predicted: Sequence[Sequence[str]],
    extractor: Extractor | None = None,
    hypernyms: bool = True,
) -> dict:
    ex = extractor or Extractor()
    metrics = [evaluate_trajectory(s, p, ex, check_hypernyms=hypernyms)
               for s, p in zip(sources, predicted)]
    out = aggregate(metrics)
    out["extractor_backend"] = ex.backend
    return out


def budget_quality(sources: Sequence[str], predicted: Sequence[Sequence[str]],
                   extractor: Extractor | None = None) -> dict:
    """Detail retained at each depth -- the raw material for the early-exit curve."""
    ex = extractor or Extractor()
    n_rungs = min(len(p) for p in predicted) if predicted else 0
    per_budget: dict[float, list[float]] = {}
    for s, p in zip(sources, predicted):
        src = ex.keys(s)
        for i in range(n_rungs):
            b = round(i / max(n_rungs - 1, 1), 3)
            hit = len(ex.keys(p[i]) & src) / max(len(src), 1)
            per_budget.setdefault(b, []).append(hit)
    cost = {b: sum(len(p[i].split()) for p in predicted) / max(len(predicted), 1)
            for i, b in enumerate(sorted(per_budget))}
    return early_exit_curve(per_budget, cost)


def add_nli(results: dict, predicted: Sequence[Sequence[str]], model: str | None = None,
            limit: int = 200) -> dict:
    """Entailment-based nesting, which is the right test on paraphrastic ladders."""
    try:
        from evaluation.nli import NLIScorer, aggregate_nli
        scorer = NLIScorer(model) if model else NLIScorer()
        per = [scorer.trajectory(p) for p in tqdm(predicted[:limit], desc="NLI", leave=False)]
        results["nli"] = aggregate_nli(per)
    except Exception as exc:
        results["nli"] = {"error": str(exc)}
    return results


def build_engine(p0_path: str, p1_path: str, p2_path: str | None, device: str):
    from inference.recall_engine import RecallEngine
    from models.g_psi_module.semantic_projection import SemanticProjectionModule
    from models.latent_diffusion import LatentDiffusion
    from models.memory_model import MemoryModel
    from training.common import build_tokenizer, config_from_checkpoint, load_checkpoint

    cfg = config_from_checkpoint(p0_path)
    dev = torch.device(device if device != "auto"
                       else ("cuda" if torch.cuda.is_available() else "cpu"))

    model = MemoryModel(cfg).to(dev)
    model.load_state_dict(load_checkpoint(p0_path, dev)["model_state_dict"], strict=False)
    model.drop_gist_decoder()

    diffusion = LatentDiffusion(cfg, model.d_model).to(dev)
    diffusion.load_state_dict(load_checkpoint(p1_path, dev)["diffusion_state_dict"])

    g_psi = None
    if p2_path:
        ck = load_checkpoint(p2_path, dev)
        g_psi = SemanticProjectionModule(model.d_model, cfg.model.n_slots,
                                         cfg.model.u_dim, cfg.gpsi).to(dev)
        g_psi.load_state_dict(ck["g_psi_state_dict"])
        if "decoder_state_dict" in ck:
            model.decoder_x.load_state_dict(ck["decoder_state_dict"])

    tok = build_tokenizer(cfg)
    return RecallEngine(model, diffusion, g_psi, tok, cfg, dev), cfg


def generate_ladders(engine, records: list[dict], n_levels: int, batch_size: int = 8,
                     **kw) -> list[list[str]]:
    out: list[list[str]] = []
    for i in tqdm(range(0, len(records), batch_size), desc="recall"):
        chunk = records[i:i + batch_size]
        ladders = engine.ladder([r["x"] for r in chunk], n_levels=n_levels,
                                example_ids=[r["id"] for r in chunk], **kw)
        out.extend([[rung.text for rung in lad] for lad in ladders])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate graded recall")
    ap.add_argument("--p0", required=True)
    ap.add_argument("--p1", required=True)
    ap.add_argument("--p2", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-levels", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--cfg-scale", type=float, default=None)
    ap.add_argument("--shuffle-cue", action="store_true")
    ap.add_argument("--no-gpsi", action="store_true", help="decode v_t directly (ablation)")
    ap.add_argument("--oracle", action="store_true", help="decode gold rung latents (upper bound)")
    ap.add_argument("--nli", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from evaluation.ceiling import load_split

    engine, cfg = build_engine(args.p0, args.p1, None if args.no_gpsi else args.p2, args.device)
    data_dir = args.data_dir or cfg.data.data_dir
    records = load_split(os.path.join(data_dir, f"{args.split}.json"))
    if args.limit:
        records = records[: args.limit]
    print(f"{len(records)} records from {data_dir}/{args.split}.json")

    if args.oracle:
        ladders = [[r.text for r in lad]
                   for lad in engine.oracle_ladder([r["levels"] for r in records])]
    else:
        ladders = generate_ladders(engine, records, args.n_levels, args.batch_size,
                                   cfg_scale=args.cfg_scale, shuffle_cue=args.shuffle_cue)

    sources = [r["x"] for r in records]
    ex = Extractor()
    results = score_ladders(sources, ladders, ex)
    results["early_exit"] = budget_quality(sources, ladders, ex)
    results["setting"] = {
        "split": args.split, "n_levels": args.n_levels, "oracle": args.oracle,
        "no_gpsi": args.no_gpsi, "shuffle_cue": args.shuffle_cue, "cfg_scale": args.cfg_scale,
    }
    if args.nli:
        add_nli(results, ladders)

    out = args.out or os.path.join(cfg.out_dir, cfg.name, f"eval_{args.split}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": results,
                   "samples": [{"source": s, "ladder": l}
                               for s, l in zip(sources[:25], ladders[:25])]}, f, indent=2)

    print("\n" + "=" * 70)
    print(f"monotonic (strict)      {results['frac_strictly_monotonic']:.3f}")
    print(f"decay spearman          {results['mean_decay_spearman']:.3f}")
    print(f"perfectly nested        {results['frac_perfectly_nested']:.3f}")
    print(f"nesting violation rate  {results['mean_nesting_violation_rate']:.3f}")
    print(f"hallucination           {results['mean_hallucination']:.3f}")
    print(f"abstraction validity    {results['mean_abstraction_validity']:.3f}")
    print(f"retention curve         {results['retention_curve']}")
    if "nli" in results:
        print(f"entailment nesting      {results['nli'].get('mean_entailment_nesting_rate')}")
    print(f"\nwritten to {out}")
    return results


if __name__ == "__main__":
    main()
