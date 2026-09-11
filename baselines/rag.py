"""RAG baseline -- the real competitor for the memory claim.

Store every memory in a vector index, retrieve top-k for a query, and count what it costs to put
those passages into a prompt.

This is where the honest cost argument lives. We do **not** claim storage savings: if `v0` is
stored then the latent is larger than the text it encodes. The claim is about *read* cost --
RAG's context grows with the detail that was **stored**, ours with the detail that was
**requested**. This script measures the first half of that comparison so the second half has
something to be plotted against.

    python -m baselines.rag --data-dir data/final --split test
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from typing import Sequence


class BM25:
    """Dependency-free lexical retriever, so the baseline runs without a GPU or an index server."""

    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.docs = list(docs)
        self.toks = [d.lower().split() for d in self.docs]
        self.k1, self.b = k1, b
        self.len = [len(t) for t in self.toks]
        self.avgdl = sum(self.len) / max(len(self.len), 1)
        self.df: Counter = Counter()
        for t in self.toks:
            self.df.update(set(t))
        self.N = len(self.toks)
        self.tf = [Counter(t) for t in self.toks]

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        q = query.lower().split()
        scores = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for term in q:
                if term not in tf:
                    continue
                num = tf[term] * (self.k1 + 1)
                den = tf[term] + self.k1 * (1 - self.b + self.b * self.len[i] / self.avgdl)
                s += self._idf(term) * num / den
            scores.append((i, s))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]


def dense_index(texts: Sequence[str], model_name: str, device: str = "auto"):
    """Optional dense retriever over the same gist encoder that anchors `u`.

    Using the same space is the point: one vector can serve both "which memory" and "what is the
    cue", which is what makes the RAG comparison like-for-like.
    """
    import torch
    from transformers import AutoTokenizer

    from models.encoder_prep.encoder import FrozenGistEncoder

    dev = torch.device(device if device != "auto"
                       else ("cuda" if torch.cuda.is_available() else "cpu"))
    enc = FrozenGistEncoder(model_name).to(dev)
    tok = AutoTokenizer.from_pretrained(model_name)
    vecs = []
    for i in range(0, len(texts), 64):
        b = tok(list(texts[i:i + 64]), padding=True, truncation=True, max_length=128,
                return_tensors="pt").to(dev)
        vecs.append(enc(b["input_ids"], b["attention_mask"]).cpu())
    return torch.cat(vecs), enc, tok, dev


def main(argv=None):
    ap = argparse.ArgumentParser(description="RAG read-cost baseline")
    ap.add_argument("--data-dir", default="data/final")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--topk", nargs="+", type=int, default=[1, 3, 5, 10])
    ap.add_argument("--dense-model", default=None,
                    help="e.g. intfloat/e5-base-v2; omit for BM25 only")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="runs/baselines/rag.json")
    args = ap.parse_args(argv)

    from evaluation.ceiling import load_split

    records = load_split(os.path.join(args.data_dir, f"{args.split}.json"))
    if args.limit:
        records = records[: args.limit]
    corpus = [r["x"] for r in records]
    # Query with the gist: "what do I remember about this?" -- the cue our system also uses.
    queries = [r["y"] or r["levels"][-1] for r in records]

    bm25 = BM25(corpus)
    results: dict = {"n_memories": len(corpus), "retrievers": {}}

    def _eval(name, ranker):
        per_k = {}
        for k in args.topk:
            hits, ctx_tokens = 0, 0
            for i, q in enumerate(queries):
                idxs = ranker(q, k)
                hits += int(i in idxs)
                ctx_tokens += sum(len(corpus[j].split()) for j in idxs)
            per_k[k] = {
                "recall_at_k": round(hits / max(len(queries), 1), 4),
                "mean_context_tokens": round(ctx_tokens / max(len(queries), 1), 1),
            }
        results["retrievers"][name] = per_k

    _eval("bm25", lambda q, k: [i for i, _ in bm25.search(q, k)])

    if args.dense_model:
        import torch
        mat, enc, tok, dev = dense_index(corpus, args.dense_model, args.device)

        def dense(q, k):
            b = tok([q], padding=True, truncation=True, max_length=128,
                    return_tensors="pt").to(dev)
            v = enc(b["input_ids"], b["attention_mask"]).cpu()
            return torch.topk(mat @ v.T.squeeze(-1), k).indices.tolist()

        _eval(f"dense:{args.dense_model}", dense)

    # Storage accounting, stated plainly rather than claimed as an advantage.
    mean_words = sum(len(c.split()) for c in corpus) / max(len(corpus), 1)
    results["storage_bytes_per_memory"] = {
        "raw_text_utf8": round(sum(len(c.encode()) for c in corpus) / max(len(corpus), 1), 1),
        "note": ("A 16x512 fp16 detail latent is 16384 bytes, far larger than the text. "
                 "Storing only the 256-dim gist is 512 bytes, but then recalled detail is "
                 "reconstructed rather than veridical -- measured separately as an ablation."),
        "gist_only_fp16": 512,
        "detail_latent_fp16": 16 * 512 * 2,
        "mean_words_per_memory": round(mean_words, 1),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"{len(corpus)} memories")
    for name, per_k in results["retrievers"].items():
        print(f"\n{name}")
        print(f"  {'k':>3} {'recall@k':>10} {'ctx tokens':>12}")
        for k, v in per_k.items():
            print(f"  {k:>3} {v['recall_at_k']:>10.3f} {v['mean_context_tokens']:>12.1f}")
    print("\nRAG context grows with the detail that was stored. The comparison to plot is")
    print("answer quality against context tokens, not storage bytes.")
    print(f"\nwritten to {args.out}")
    return results


if __name__ == "__main__":
    main()
