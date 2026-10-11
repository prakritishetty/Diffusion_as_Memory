"""Group D -- dataset probes. No GPU, no model, no training.

`tests/` already asserts these contracts, but it asserts them against a 1M-parameter stub on
synthetic data from `tests/stubs.py`. That proves the code is wired correctly; it proves nothing
about flan-t5-base on the 4935 real MSR records. These probes run the same checks on the real
files and *record the numbers* rather than returning pass/fail, so two runs can be diffed.

    python -m tools.probe_data --data-dir data/final
    python -m tools.probe_data --data-dir data/control --out runs/probes/data_control.json
    python -m tools.probe_data --only ladder map_coverage
    python -m tools.probe_data --gist            # adds D7, needs the frozen e5 encoder
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import torch

from configs import get as get_preset
from dataloader.memory_dataset import build_examples
from diffusion.level_map import LevelMap
from diffusion.schedule import NoiseSchedule
from utils import ledger

PROBES = ("ladder", "tokenisation", "map_coverage", "rung_separation", "sampler",
          "positives", "gist")
DEFAULT = ("ladder", "tokenisation", "map_coverage", "rung_separation", "sampler", "positives")


def _read(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pct(x: float) -> float:
    return round(float(x), 4)


# -- D1 ---------------------------------------------------------------------------------------
def probe_ladder(records: list[dict], cfg) -> tuple[dict, str]:
    """Does every ladder start at the verbatim source, and does detail actually fall?"""
    examples = build_examples(records, cfg.data.include_source_as_level0, cfg.data.max_levels)

    n = len(examples)
    source_ok = sum(e.levels[0] == e.x for e in examples)
    empty = sum(any(not s.strip() for s in e.levels) for e in examples)
    shortening = sum(
        all(b <= a for a, b in zip([len(s.split()) for s in e.levels],
                                   [len(s.split()) for s in e.levels][1:]))
        for e in examples
    )
    dup = sum(any(a == b for a, b in zip(e.levels, e.levels[1:])) for e in examples)
    counts = Counter(len(e.levels) for e in examples)
    words = sorted(len(e.x.split()) for e in examples)
    has_y = sum(bool(e.y.strip()) for e in examples)
    has_xplus = sum(bool(e.x_plus) for e in examples)

    m = {
        "n_records_in_file": len(records),
        "n_examples_built": n,
        "n_dropped_empty_x": len(records) - n,
        "source_at_rung0_rate": _pct(source_ok / max(n, 1)),
        "any_empty_rung_rate": _pct(empty / max(n, 1)),
        "strictly_shortening_rate": _pct(shortening / max(n, 1)),
        "adjacent_duplicate_rung_rate": _pct(dup / max(n, 1)),
        "n_levels_mean": round(sum(len(e.levels) for e in examples) / max(n, 1), 3),
        "n_levels_min": min(counts) if counts else 0,
        "n_levels_max": max(counts) if counts else 0,
        "n_levels_histogram": dict(sorted(counts.items())),
        "x_words_mean": round(sum(words) / max(n, 1), 1),
        "x_words_p95": words[int(0.95 * len(words))] if words else 0,
        "x_words_max": words[-1] if words else 0,
        "has_y_rate": _pct(has_y / max(n, 1)),
        "has_xplus_rate": _pct(has_xplus / max(n, 1)),
    }
    v = ledger.verdict_of(m["source_at_rung0_rate"] == 1.0, m["any_empty_rung_rate"] == 0.0)
    return m, v


# -- D2 ---------------------------------------------------------------------------------------
def probe_tokenisation(records: list[dict], cfg, tokenizer) -> tuple[dict, str]:
    """How much of the source is silently cut before the encoder sees it?

    64 truncated roughly a third of MSR; the point of this probe is to show that 128 does not.
    """
    examples = build_examples(records, cfg.data.include_source_as_level0, cfg.data.max_levels)
    xs = [e.x for e in examples]
    ys = [e.y for e in examples]
    levels = [s for e in examples for s in e.levels]

    def lengths(texts: list[str]) -> list[int]:
        out: list[int] = []
        for i in range(0, len(texts), 256):
            enc = tokenizer(texts[i:i + 256], add_special_tokens=True)["input_ids"]
            out.extend(len(s) for s in enc)
        return out

    lx, ly, ll = lengths(xs), lengths(ys), lengths(levels)
    S, G = cfg.data.max_seq_len, cfg.data.max_gist_len
    sx = sorted(lx)

    m = {
        "tokenizer": getattr(tokenizer, "name_or_path", "?"),
        "max_seq_len": S,
        "max_gist_len": G,
        "x_tokens_mean": round(sum(lx) / max(len(lx), 1), 1),
        "x_tokens_p95": sx[int(0.95 * len(sx))] if sx else 0,
        "x_tokens_max": max(lx) if lx else 0,
        "x_truncated_rate_at_max_seq_len": _pct(sum(v > S for v in lx) / max(len(lx), 1)),
        "x_truncated_rate_at_64": _pct(sum(v > 64 for v in lx) / max(len(lx), 1)),
        "level_tokens_max": max(ll) if ll else 0,
        "level_truncated_rate": _pct(sum(v > S for v in ll) / max(len(ll), 1)),
        "y_tokens_max": max(ly) if ly else 0,
        "y_truncated_rate_at_max_gist_len": _pct(sum(v > G for v in ly) / max(len(ly), 1)),
    }
    v = ledger.verdict_of(m["x_truncated_rate_at_max_seq_len"] < 0.01,
                          m["level_truncated_rate"] < 0.01)
    return m, v


# -- D3 ---------------------------------------------------------------------------------------
def probe_map_coverage(level_counts: list[int], cfg) -> tuple[dict, str]:
    """Supervision mass per rung under each map, weighted by the real ladder-length mix."""
    sched = NoiseSchedule(T=cfg.diffusion.T, schedule_type=cfg.diffusion.schedule)
    freq = Counter(level_counts)
    total = sum(freq.values())

    m: dict = {"T": cfg.diffusion.T, "schedule": cfg.diffusion.schedule,
               "active_map": cfg.diffusion.level_map}
    for kind in ("bucket", "uniform_t", "logsnr"):
        lm = LevelMap(sched, kind=kind, bucket_size=cfg.diffusion.bucket_size,
                      logsnr_clip=cfg.diffusion.logsnr_clip)
        rep = lm.coverage_report(sorted(freq))
        final = sum(rep[n]["fraction_on_final_level"] * c for n, c in freq.items()) / max(total, 1)
        worst = max(rep[n]["max_minus_min"] for n in freq)
        m[f"{kind}.frac_timesteps_on_final_rung"] = _pct(final)
        m[f"{kind}.worst_rung_imbalance"] = _pct(worst)
    modal = freq.most_common(1)[0][0] if freq else 6
    lm = LevelMap(sched, kind=cfg.diffusion.level_map, bucket_size=cfg.diffusion.bucket_size,
                  logsnr_clip=cfg.diffusion.logsnr_clip)
    m["modal_n_levels"] = modal
    m["active_map.per_rung_fraction_at_modal_n"] = lm.coverage_report([modal])[modal][
        "per_level_fraction"]

    v = ledger.verdict_of(m["uniform_t.worst_rung_imbalance"] < 0.02)
    return m, v


# -- D4 ---------------------------------------------------------------------------------------
def probe_rung_separation(level_counts: list[int], cfg) -> tuple[dict, str]:
    """Adjacent rungs must sit at visibly different noise levels or the objective cannot
    separate them. Reported as the signal fraction sqrt(alpha_bar) at each rung midpoint."""
    sched = NoiseSchedule(T=cfg.diffusion.T, schedule_type=cfg.diffusion.schedule)
    lm = LevelMap(sched, kind=cfg.diffusion.level_map, bucket_size=cfg.diffusion.bucket_size,
                  logsnr_clip=cfg.diffusion.logsnr_clip)

    common = [n for n, _ in Counter(level_counts).most_common(4)]
    m: dict = {"map": cfg.diffusion.level_map}
    worst = 1.0
    for n in sorted(common):
        sig, mids = [], []
        for k in range(n):
            t_mid = lm.t_for_level(k, n, where="mid")
            mids.append(t_mid)
            sig.append(round(float(sched.alpha_bar(torch.tensor([t_mid])).sqrt()), 4))
        gaps = [round(a - b, 4) for a, b in zip(sig, sig[1:])]
        m[f"n={n}.t_midpoints"] = mids
        m[f"n={n}.signal_fraction"] = sig
        m[f"n={n}.min_adjacent_gap"] = min(gaps) if gaps else 1.0
        worst = min(worst, min(gaps) if gaps else 1.0)
    m["worst_adjacent_gap"] = round(worst, 4)
    return m, ledger.verdict_of(worst > 0.05)


# -- D5 ---------------------------------------------------------------------------------------
def probe_sampler(level_counts: list[int], cfg, draws: int = 200_000) -> tuple[dict, str]:
    """Stratified sampling must guarantee uniform rung coverage *per batch*, not on average."""
    sched = NoiseSchedule(T=cfg.diffusion.T, schedule_type=cfg.diffusion.schedule)
    lm = LevelMap(sched, kind=cfg.diffusion.level_map, bucket_size=cfg.diffusion.bucket_size,
                  logsnr_clip=cfg.diffusion.logsnr_clip)

    base = torch.tensor(level_counts, dtype=torch.long)
    n_levels = base.repeat(max(1, draws // max(len(base), 1)))
    g = torch.Generator().manual_seed(cfg.seed)

    m: dict = {"draws": int(n_levels.numel()), "stratified_in_config": cfg.diffusion.stratified_t}
    for name, fn in (("stratified", lm.sample_stratified), ("uniform_t", lm.sample_uniform_t)):
        t, k = fn(n_levels, g)
        roundtrip = float((lm.level_index(t, n_levels) == k).float().mean())
        worst = 0.0
        for n in sorted(set(level_counts)):
            sel = n_levels == n
            cnt = int(sel.sum())
            if cnt < 500:
                continue
            hist = torch.bincount(k[sel], minlength=n).float() / cnt
            worst = max(worst, float((hist - 1.0 / n).abs().max()))
        m[f"{name}.rung_roundtrip_rate"] = _pct(roundtrip)
        m[f"{name}.worst_deviation_from_uniform"] = _pct(worst)
        m[f"{name}.t_mean"] = round(float(t.float().mean()), 1)

    # One real batch is the case that actually matters: averages hide per-batch imbalance.
    B = cfg.train.batch_size
    idx = torch.randint(0, len(base), (B,), generator=g)
    t_b, k_b = lm.sample_stratified(base[idx], g)
    m["one_batch.batch_size"] = B
    m["one_batch.distinct_rungs"] = int(torch.unique(k_b).numel())
    m["one_batch.rungs"] = sorted(k_b.tolist())

    v = ledger.verdict_of(m["stratified.rung_roundtrip_rate"] == 1.0,
                          m["stratified.worst_deviation_from_uniform"] < 0.02)
    return m, v


# -- D6 ---------------------------------------------------------------------------------------
def probe_positives(data_path: str, cfg, tokenizer, cache, split: str = "train",
                    draws: int = 40, limit: int = 512) -> tuple[dict, str]:
    """Which rung does the InfoNCE positive actually come from?

    `xplus_source="ladder"` samples uniformly from rungs 0..ladder_positive_max_level, and rung 0
    *is* the source -- so a predictable share of positives are the anchor itself, which is a
    valid but uninformative pair. This reports that share instead of assuming it.
    """
    from dataloader.memory_dataset import MemoryLadderDataset

    # Same split name as training, so this warms the cache P0 will read rather than duplicating it.
    ds = MemoryLadderDataset(data_path, tokenizer, cfg, cache, split="probe")
    n = min(limit, len(ds))
    ids = ds.enc["x_input_ids"]
    lvl = ds.enc["level_input_ids"]

    same = 0
    rung_hist: Counter = Counter()
    unresolved = 0
    for i in range(n):
        for _ in range(draws):
            pos, _mask = ds._positive(i)
            if torch.equal(pos, ids[i]):
                same += 1
            hit = [j for j in range(int(ds.enc["n_levels"][i]))
                   if torch.equal(pos, lvl[i, j])]
            if hit:
                rung_hist[hit[0]] += 1
            else:
                unresolved += 1

    total = n * draws
    m = {
        "xplus_source": cfg.data.xplus_source,
        "ladder_positive_max_level": cfg.data.ladder_positive_max_level,
        "examples_probed": n,
        "draws_per_example": draws,
        "positive_equals_anchor_rate": _pct(same / max(total, 1)),
        "positive_differs_rate": _pct(1 - same / max(total, 1)),
        "rung_histogram": {k: _pct(c / max(total, 1)) for k, c in sorted(rung_hist.items())},
        "unresolved_rate": _pct(unresolved / max(total, 1)),
    }
    v = ledger.verdict_of(m["positive_differs_rate"] > 0.5)
    return m, v


# -- D7 ---------------------------------------------------------------------------------------
def probe_gist(records: list[dict], cfg, device: str = "auto") -> tuple[dict, str]:
    """Does the frozen retrieval space actually put `x` next to its own gist `y`?

    `L_align` is only worth its weight if this holds -- it is the whole argument for anchoring
    `u` in e5 space rather than letting it be an arbitrary learned space.
    """
    from transformers import AutoTokenizer

    from models.encoder_prep.encoder import FrozenGistEncoder

    name = cfg.model.gist_encoder_name
    if not name:
        return {"skipped": "model.gist_encoder_name is None"}, ledger.WARN

    dev = torch.device(device if device != "auto"
                       else ("cuda" if torch.cuda.is_available() else "cpu"))
    examples = build_examples(records, cfg.data.include_source_as_level0, cfg.data.max_levels)
    examples = [e for e in examples if e.y.strip()]
    tok = AutoTokenizer.from_pretrained(name)
    enc = FrozenGistEncoder(name).to(dev)

    def embed(texts: list[str], max_len: int) -> torch.Tensor:
        out = []
        for i in range(0, len(texts), 64):
            b = tok(texts[i:i + 64], padding=True, truncation=True, max_length=max_len,
                    return_tensors="pt").to(dev)
            out.append(enc(b["input_ids"], b["attention_mask"]).float().cpu())
        return torch.cat(out)

    ex = embed([e.x for e in examples], cfg.data.max_seq_len)
    ey = embed([e.y for e in examples], cfg.data.max_gist_len)
    exn = torch.nn.functional.normalize(ex, dim=-1)
    eyn = torch.nn.functional.normalize(ey, dim=-1)

    sim = exn @ eyn.t()
    nearest = sim.argmax(dim=1)
    gold = torch.arange(len(examples))
    top1 = float((nearest == gold).float().mean())
    rank = (sim > sim.gather(1, gold[:, None])).sum(1)
    top5 = float((rank < 5).float().mean())

    m = {
        "gist_encoder": name,
        "n_pairs": len(examples),
        "embedding_dim": ex.shape[-1],
        "x_norm_mean": round(float(ex.norm(dim=-1).mean()), 4),
        "y_norm_mean": round(float(ey.norm(dim=-1).mean()), 4),
        "mean_cos_x_to_own_y": round(float(sim.diagonal().mean()), 4),
        "mean_cos_x_to_random_y": round(float(sim.mean()), 4),
        "retrieval_top1": _pct(top1),
        "retrieval_top5": _pct(top5),
        "chance_top1": _pct(1 / max(len(examples), 1)),
    }
    v = ledger.verdict_of(top1 > 20 * m["chance_top1"], top1 > 0.2)
    return m, v


# -- runner -------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Group D: dataset probes (no GPU, no model)")
    ap.add_argument("--preset", default="full")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--splits", nargs="+", default=["train", "validate", "test"])
    ap.add_argument("--only", nargs="*", default=list(DEFAULT), choices=list(PROBES))
    ap.add_argument("--gist", action="store_true", help="also run D7 (loads the e5 encoder)")
    ap.add_argument("--draws", type=int, default=200_000, help="D5 sample count")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="runs/probes/data.json")
    ap.add_argument("--ledger", default=ledger.DEFAULT_PATH)
    args = ap.parse_args(argv)

    from training.common import apply_set_overrides, build_cache, build_tokenizer

    cfg = apply_set_overrides(get_preset(args.preset), args.set)
    if args.data_dir:
        cfg = cfg.override(**{"data.data_dir": args.data_dir})
    data_dir = cfg.data.data_dir
    chosen = list(args.only) + (["gist"] if args.gist else [])
    chash = cfg.hash("data", "diffusion")

    print(f"preset={args.preset}  data_dir={data_dir}  config_hash={chash}")
    print(f"probes: {', '.join(chosen)}\n")

    results: dict = {"config_hash": chash, "data_dir": data_dir, "splits": {}}
    verdicts: list[tuple[str, str, str]] = []

    files = {"train": cfg.data.train_file, "validate": cfg.data.val_file,
             "test": cfg.data.test_file}
    for split in args.splits:
        path = os.path.join(data_dir, files.get(split, f"{split}.json"))
        if not os.path.exists(path):
            print(f"skip {split}: {path} not found")
            continue
        records = _read(path)
        examples = build_examples(records, cfg.data.include_source_as_level0,
                                  cfg.data.max_levels)
        level_counts = [len(e.levels) for e in examples]
        split_out: dict = {}

        print(f"\n{'=' * 78}\n{split}  ({len(records)} records)\n{'=' * 78}")

        if "ladder" in chosen:
            m, v = probe_ladder(records, cfg)
            ledger.print_table(m, f"D1 ladder integrity  [{v}]")
            split_out["ladder"] = m
            verdicts.append((split, "D1 ladder", v))
            ledger.append("D", f"{split}.ladder", m, config_hash=chash, verdict=v,
                          path=args.ledger)

        if "tokenisation" in chosen:
            tok = build_tokenizer(cfg)
            m, v = probe_tokenisation(records, cfg, tok)
            ledger.print_table(m, f"D2 tokenisation  [{v}]")
            split_out["tokenisation"] = m
            verdicts.append((split, "D2 tokenisation", v))
            ledger.append("D", f"{split}.tokenisation", m, config_hash=chash, verdict=v,
                          path=args.ledger)

        if "map_coverage" in chosen:
            m, v = probe_map_coverage(level_counts, cfg)
            ledger.print_table(m, f"D3 level-map coverage  [{v}]")
            split_out["map_coverage"] = m
            verdicts.append((split, "D3 map coverage", v))
            ledger.append("D", f"{split}.map_coverage", m, config_hash=chash, verdict=v,
                          path=args.ledger)

        if "rung_separation" in chosen:
            m, v = probe_rung_separation(level_counts, cfg)
            ledger.print_table(m, f"D4 rung separation  [{v}]")
            split_out["rung_separation"] = m
            verdicts.append((split, "D4 rung separation", v))
            ledger.append("D", f"{split}.rung_separation", m, config_hash=chash, verdict=v,
                          path=args.ledger)

        if "sampler" in chosen:
            m, v = probe_sampler(level_counts, cfg, args.draws)
            ledger.print_table(m, f"D5 stratified sampler  [{v}]")
            split_out["sampler"] = m
            verdicts.append((split, "D5 sampler", v))
            ledger.append("D", f"{split}.sampler", m, config_hash=chash, verdict=v,
                          path=args.ledger)

        if "positives" in chosen:
            tok = build_tokenizer(cfg)
            cache_split = {"validate": "val"}.get(split, split)
            m, v = probe_positives(path, cfg, tok, build_cache(cfg), cache_split)
            ledger.print_table(m, f"D6 contrastive positives  [{v}]")
            split_out["positives"] = m
            verdicts.append((split, "D6 positives", v))
            ledger.append("D", f"{split}.positives", m, config_hash=chash, verdict=v,
                          path=args.ledger)

        if "gist" in chosen:
            m, v = probe_gist(records, cfg, args.device)
            ledger.print_table(m, f"D7 gist alignment  [{v}]")
            split_out["gist"] = m
            verdicts.append((split, "D7 gist", v))
            ledger.append("D", f"{split}.gist", m, config_hash=chash, verdict=v,
                          path=args.ledger)

        results["splits"][split] = split_out

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    for split, probe, v in verdicts:
        print(f"  {split:<10} {probe:<22} {v}")
    failed = [p for _, p, v in verdicts if v == ledger.FAIL]
    print(f"\nwrote {args.out}  and {len(verdicts)} ledger rows -> {args.ledger}")
    print("ALL PASS" if not failed else f"FAILED: {', '.join(sorted(set(failed)))}")
    return results


if __name__ == "__main__":
    main()
