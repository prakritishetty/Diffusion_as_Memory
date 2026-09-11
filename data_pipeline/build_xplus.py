"""Contrastive positives for `u`.

The previous loader set `xpos_input_ids = x_input_ids`, so the InfoNCE positive was the anchor
itself. That objective reduces to "be unique", which any injective encoder satisfies without
learning anything -- which is why `u` was effectively decoration.

The default positive needs no generation at all: **another rung of the same ladder**. That is not
a workaround, it is the design claim -- the gist must stay fixed while detail fades, otherwise it
cannot act as a stable cue during recall. It is handled directly in the dataloader.

This script produces the *secondary*, surface-form positive: paraphrases that keep every entity
and number but change the wording, so `u` is pushed to be invariant to phrasing rather than to
content. Everything runs locally; no paid API and nothing whose version we cannot pin.

  round-trip  EN->DE->EN with Helsinki-NLP/opus-mt (best quality per unit of effort)
  paraphrase  humarin/chatgpt_paraphraser_on_T5_base (purpose-built)

Generated candidates are filtered: a paraphrase is kept only if every named entity and number
from the source survives. Without that filter the positives teach `u` to ignore exactly the
details we are trying to measure.

    python -m data_pipeline.build_xplus --data-dir data/final --n 2
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from tqdm import tqdm


def _device(spec: str) -> torch.device:
    return torch.device(spec if spec != "auto"
                        else ("cuda" if torch.cuda.is_available() else "cpu"))


class BackTranslator:
    def __init__(self, pivot: str = "de", device: str = "auto", batch_size: int = 32):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.dev = _device(device)
        self.batch_size = batch_size
        fwd, back = f"Helsinki-NLP/opus-mt-en-{pivot}", f"Helsinki-NLP/opus-mt-{pivot}-en"
        self.t1 = AutoTokenizer.from_pretrained(fwd)
        self.m1 = AutoModelForSeq2SeqLM.from_pretrained(fwd).to(self.dev).eval()
        self.t2 = AutoTokenizer.from_pretrained(back)
        self.m2 = AutoModelForSeq2SeqLM.from_pretrained(back).to(self.dev).eval()

    @torch.no_grad()
    def _translate(self, texts, tok, model, num_beams):
        out = []
        for i in range(0, len(texts), self.batch_size):
            b = tok(texts[i:i + self.batch_size], return_tensors="pt", padding=True,
                    truncation=True, max_length=256).to(self.dev)
            ids = model.generate(**b, num_beams=num_beams, max_new_tokens=256)
            out.extend(tok.batch_decode(ids, skip_special_tokens=True))
        return out

    def __call__(self, texts: list[str], n: int = 2) -> list[list[str]]:
        pivoted = self._translate(texts, self.t1, self.m1, num_beams=max(n, 2))
        back = self._translate(pivoted, self.t2, self.m2, num_beams=max(n, 2))
        return [[b] for b in back]


class ParaphraseModel:
    def __init__(self, model_name: str = "humarin/chatgpt_paraphraser_on_T5_base",
                 device: str = "auto", batch_size: int = 16):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.dev = _device(device)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.dev).eval()
        self.batch_size = batch_size

    @torch.no_grad()
    def __call__(self, texts: list[str], n: int = 2) -> list[list[str]]:
        out: list[list[str]] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = [f"paraphrase: {t}" for t in texts[i:i + self.batch_size]]
            b = self.tok(chunk, return_tensors="pt", padding=True, truncation=True,
                         max_length=256).to(self.dev)
            ids = self.model.generate(
                **b, num_beams=max(n * 2, 4), num_beam_groups=max(n, 2),
                diversity_penalty=0.7, num_return_sequences=n, max_new_tokens=256,
            )
            dec = self.tok.batch_decode(ids, skip_special_tokens=True)
            out.extend([dec[j * n:(j + 1) * n] for j in range(len(chunk))])
        return out


def preserves_details(source: str, candidate: str, extractor) -> bool:
    """Keep a paraphrase only if every entity and number survives."""
    from evaluation.info_units import EPISODIC_TYPES

    s, c = extractor.by_type(source), extractor.by_type(candidate)
    for t in EPISODIC_TYPES:
        if s[t] - c[t]:
            return False
    return True


def is_distinct(source: str, candidate: str, min_edit_ratio: float = 0.15) -> bool:
    """Reject near-copies: an identical positive is the defect we are fixing."""
    a, b = source.lower().split(), candidate.lower().split()
    if not b or a == b:
        return False
    overlap = len(set(a) & set(b)) / max(len(set(a) | set(b)), 1)
    return (1 - overlap) >= min_edit_ratio


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate entity-preserving paraphrase positives")
    ap.add_argument("--data-dir", default="data/final")
    ap.add_argument("--splits", nargs="+", default=["train", "validate", "test"])
    ap.add_argument("--methods", nargs="+", default=["backtranslate", "paraphrase"])
    ap.add_argument("--n", type=int, default=2, help="candidates to keep per example")
    ap.add_argument("--candidates", type=int, default=4, help="candidates to generate")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--in-place", action="store_true",
                    help="write x_plus back into the source files")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    from evaluation.info_units import Extractor

    ex = Extractor()
    generators = []
    if "backtranslate" in args.methods:
        generators.append(("bt", BackTranslator(device=args.device)))
    if "paraphrase" in args.methods:
        generators.append(("pp", ParaphraseModel(device=args.device)))

    out_dir = args.out_dir or (args.data_dir if args.in_place else "data/with_xplus")
    os.makedirs(out_dir, exist_ok=True)

    for split in args.splits:
        path = os.path.join(args.data_dir, f"{split}.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        if args.limit:
            records = records[: args.limit]
        sources = [(r.get("x") or "").strip() for r in records]

        pools: list[list[str]] = [[] for _ in sources]
        for name, gen in generators:
            for i in tqdm(range(0, len(sources), 64), desc=f"{split}:{name}"):
                chunk = sources[i:i + 64]
                for j, cands in enumerate(gen(chunk, args.candidates)):
                    pools[i + j].extend(cands)

        kept_counts = []
        for r, src, pool in zip(records, sources, pools):
            good = [c for c in dict.fromkeys(pool)
                    if is_distinct(src, c) and preserves_details(src, c, ex)]
            r["x_plus"] = good[: args.n]
            kept_counts.append(len(r["x_plus"]))

        dst = os.path.join(out_dir, f"{split}.json")
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=1)
        have = sum(c > 0 for c in kept_counts)
        print(f"  {split}: {have}/{len(records)} have >=1 usable positive "
              f"(mean {sum(kept_counts)/max(len(kept_counts),1):.2f}) -> {dst}")

    print("\nThe ladder-rung positive needs no generation and is the default; these paraphrases")
    print("are the secondary, surface-form positive (data.xplus_source='paraphrase').")
    return out_dir


if __name__ == "__main__":
    main()
