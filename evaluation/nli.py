"""NLI-based directional consistency.

Two fixes over the previous implementation:

* **The model is called correctly.** The old code did
  `nli(f"{premise} [SEP] {hypothesis}")` against a `text-classification` pipeline. Cross-encoder
  NLI models expect a *pair* (`{"text": ..., "text_pair": ...}`); a manually joined string with a
  literal `[SEP]` is not encoded as two segments, so the reported entailment numbers were
  measuring something other than entailment.

* **It is used for the right thing.** Unit-level nesting is exact for deletion-based ladders but
  cannot work for paraphrastic ones: measured on the gold MSR ladders, strict containment holds
  for only ~6% of trajectories, purely because abstraction *rewrites* ("dismantled and reinstalled
  at COPIA" -> "moved to a museum") rather than deleting. Entailment tolerates that substitution,
  so it is the right operationalisation of "the coarse answer is implied by the fine one" on
  paraphrastic data.

The property we want between adjacent rungs is asymmetric:
    finer |= coarser        (nothing new is asserted)
    coarser |/= finer       (detail was genuinely lost)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-base"


@dataclass
class PairResult:
    forward_label: str
    forward_score: float
    backward_label: str
    backward_score: float

    @property
    def detail_loss(self) -> bool:
        """Finer entails coarser, and coarser does not entail finer."""
        return self.forward_label == "entailment" and self.backward_label != "entailment"

    @property
    def contradiction(self) -> bool:
        return "contradiction" in (self.forward_label, self.backward_label)

    @property
    def asymmetry(self) -> float:
        return self.forward_score - self.backward_score


class NLIScorer:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: int | None = None,
                 batch_size: int = 32):
        from transformers import pipeline
        import torch

        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("text-classification", model=model_name, device=device,
                             top_k=None)
        self.batch_size = batch_size

    def _score(self, pairs: Sequence[tuple[str, str]]) -> list[tuple[str, float]]:
        if not pairs:
            return []
        # The pair form is what makes this an NLI call rather than a single-sequence call.
        inputs = [{"text": p, "text_pair": h} for p, h in pairs]
        raw = self.pipe(inputs, batch_size=self.batch_size, truncation=True)
        out = []
        for r in raw:
            scores = r if isinstance(r, list) else [r]
            best = max(scores, key=lambda s: s["score"])
            out.append((best["label"].lower(), float(best["score"])))
        return out

    def pair(self, premise: str, hypothesis: str) -> PairResult:
        return self.pairs([(premise, hypothesis)])[0]

    def pairs(self, items: Sequence[tuple[str, str]]) -> list[PairResult]:
        fwd = self._score(items)
        bwd = self._score([(h, p) for p, h in items])
        return [PairResult(f[0], f[1], b[0], b[1]) for f, b in zip(fwd, bwd)]

    def trajectory(self, rungs: Sequence[str]) -> dict:
        """Adjacent-rung directional test over one ladder (least abstract first)."""
        if len(rungs) < 2:
            return {"detail_loss_rate": 0.0, "contradiction_rate": 0.0,
                    "entailment_nesting_rate": 0.0, "pairs": []}
        res = self.pairs(list(zip(rungs, rungs[1:])))
        n = len(res)
        return {
            "detail_loss_rate": sum(r.detail_loss for r in res) / n,
            "contradiction_rate": sum(r.contradiction for r in res) / n,
            # "nothing new asserted": the coarse rung is implied by the fine one
            "entailment_nesting_rate": sum(
                r.forward_label == "entailment" for r in res) / n,
            "mean_asymmetry": sum(r.asymmetry for r in res) / n,
            "pairs": [
                {"from": i, "to": i + 1, "forward": (r.forward_label, round(r.forward_score, 3)),
                 "backward": (r.backward_label, round(r.backward_score, 3)),
                 "detail_loss": r.detail_loss}
                for i, r in enumerate(res)
            ],
        }


def aggregate_nli(results: Sequence[dict]) -> dict:
    if not results:
        return {}
    n = len(results)
    return {
        "mean_detail_loss_rate": sum(r["detail_loss_rate"] for r in results) / n,
        "mean_entailment_nesting_rate": sum(r["entailment_nesting_rate"] for r in results) / n,
        "mean_contradiction_rate": sum(r["contradiction_rate"] for r in results) / n,
        "frac_fully_nested": sum(r["entailment_nesting_rate"] >= 1.0 for r in results) / n,
    }
