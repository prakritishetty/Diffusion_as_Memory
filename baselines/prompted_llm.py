"""Prompted-LLM baseline: ask an instruct model for the ladder directly.

Settles "can an off-the-shelf model already do this?". On quality, probably yes. The interesting
column is nesting: nothing in a per-level prompt ties level k+1 to level k, so an LLM asked for
five independent rewrites has no reason to produce a consistent refinement sequence.

Two prompting modes, because they test different things:
  ``independent`` -- each level requested separately (what a naive system would do)
  ``chained``     -- each level is asked to abstract the *previous* level, which is the charitable
                     version and the one that could plausibly be nested

Backends: a local HF model, or an OpenAI-compatible endpoint (NVIDIA NIM works). Prefer an
open-weights judge/generator so a reviewer can reproduce it.

    python -m baselines.prompted_llm --backend hf --model Qwen/Qwen2.5-7B-Instruct
    python -m baselines.prompted_llm --backend openai --model meta/llama-3.1-8b-instruct \
        --base-url https://integrate.api.nvidia.com/v1
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Sequence

SYSTEM = (
    "You model human forgetting. Given a sentence, rewrite it at a requested level of "
    "abstraction. Remove fine-grained details first (numbers, dates, names, places), keep the "
    "core of who did what. Never introduce information that was not present. Reply with the "
    "rewritten sentence only."
)


def _independent_prompt(text: str, k: int, n: int) -> str:
    return (f"Sentence: {text}\n\n"
            f"Rewrite it at abstraction level {k} of {n - 1}, where 0 is the sentence verbatim "
            f"and {n - 1} is the most abstract summary that still makes sense.")


def _chained_prompt(prev: str) -> str:
    return (f"Sentence: {prev}\n\n"
            "Rewrite it one step more abstract: drop the most specific remaining detail and keep "
            "everything else unchanged. Do not add anything new.")


class HFBackend:
    def __init__(self, model: str, device: str = "auto", max_new_tokens: int = 64):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device if device != "cpu" else None,
        )
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompts: Sequence[str]) -> list[str]:
        import torch
        outs = []
        for p in prompts:
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}]
            text = self.tok.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True)
            enc = self.tok(text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                ids = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                          do_sample=False,
                                          pad_token_id=self.tok.eos_token_id)
            outs.append(self.tok.decode(ids[0][enc["input_ids"].shape[1]:],
                                        skip_special_tokens=True).strip())
        return outs


class OpenAIBackend:
    """Works with any OpenAI-compatible endpoint, including NVIDIA NIM."""

    def __init__(self, model: str, base_url: str | None = None, api_key_env: str = "NIM_API_KEY"):
        from openai import OpenAI
        key = os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(f"set ${api_key_env} (never hard-code the key)")
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model = model

    def __call__(self, prompts: Sequence[str]) -> list[str]:
        outs = []
        for p in prompts:
            r = self.client.chat.completions.create(
                model=self.model, temperature=0.0, max_tokens=96,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": p}],
            )
            outs.append(r.choices[0].message.content.strip())
        return outs


def _clean(s: str) -> str:
    s = re.sub(r"^\s*(?:level\s*\d+\s*[:.\-]\s*)", "", s, flags=re.IGNORECASE)
    s = s.strip().strip('"').strip()
    return s.split("\n")[0].strip()


def build_ladders(backend, sources: Sequence[str], n_levels: int, mode: str) -> list[list[str]]:
    from tqdm import tqdm

    ladders: list[list[str]] = []
    for src in tqdm(sources, desc=f"prompted-{mode}"):
        if mode == "independent":
            prompts = [_independent_prompt(src, k, n_levels) for k in range(n_levels)]
            rungs = [src] + [_clean(r) for r in backend(prompts[1:])]
        else:
            rungs, cur = [src], src
            for _ in range(n_levels - 1):
                cur = _clean(backend([_chained_prompt(cur)])[0]) or cur
                rungs.append(cur)
        ladders.append(rungs[:n_levels])
    return ladders


def main(argv=None):
    ap = argparse.ArgumentParser(description="Prompted-LLM ladder baseline")
    ap.add_argument("--backend", choices=["hf", "openai"], default="hf")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-env", default="NIM_API_KEY")
    ap.add_argument("--data-dir", default="data/final")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-levels", type=int, default=6)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--modes", nargs="+", default=["independent", "chained"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="runs/baselines/prompted_llm.json")
    args = ap.parse_args(argv)

    from evaluation.ceiling import load_split
    from evaluation.evaluate import budget_quality, score_ladders

    backend = (HFBackend(args.model, args.device) if args.backend == "hf"
               else OpenAIBackend(args.model, args.base_url, args.api_key_env))

    records = load_split(os.path.join(args.data_dir, f"{args.split}.json"))[: args.limit]
    sources = [r["x"] for r in records]

    out: dict = {}
    for mode in args.modes:
        ladders = build_ladders(backend, sources, args.n_levels, mode)
        res = score_ladders(sources, ladders)
        res["early_exit"] = budget_quality(sources, ladders)
        res["baseline"] = f"prompted_llm_{mode}"
        res["model"] = args.model
        res["samples"] = [{"source": s, "ladder": l}
                          for s, l in zip(sources[:10], ladders[:10])]
        out[mode] = res

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'mode':<14}{'monot':>8}{'nested':>8}{'halluc':>8}{'wellfrm':>9}{'USABLE':>8}")
    print("-" * 55)
    for mode, r in out.items():
        print(f"{mode:<14}{r['frac_strictly_monotonic']:>8.3f}{r['frac_perfectly_nested']:>8.3f}"
              f"{r['mean_hallucination']:>8.3f}{r['mean_well_formed']:>9.3f}"
              f"{r['mean_usable_rung_rate']:>8.3f}")
    print(f"\nwritten to {args.out}")
    return out


if __name__ == "__main__":
    main()
