"""Rule-based control ladders: MSR-Ladder-Ctrl.

Why a second corpus:

1. **A real ceiling.** The LLM ladders are paraphrases, so strict information-nesting holds for
   only ~6% of them (measured). Deletion-based ladders are nested *by construction*, which gives
   the early-exit consistency claim a corpus where it can actually be demonstrated rather than
   merely hoped for.

2. **An interpretable slot ablation.** Details are removed in a fixed priority order --
   numbers/dates, then people, then places, then organisations, then modifiers, then nouns to
   hypernyms, then the verb. So "which slot holds names" has a ground truth to check against.

3. **Free and reproducible.** No API calls; a reviewer can regenerate it.

The rungs read stiffly ("A person went somewhere"), which is exactly why this is a *control*
corpus rather than a replacement for the natural one.

    python -m data_pipeline.build_control_ladders --data-dir data/final --out data/control
"""

from __future__ import annotations

import argparse
import json
import os
import re

from evaluation.info_units import _MONTHS, _NUM_RE, _YEAR_RE

# Removal priority: most episodic first. One rung per stage that actually changes the text.
STAGES = ("NUM", "DATE", "PERSON", "GPE", "ORG", "MODIFIER", "HYPERNYM")

PLACEHOLDER = {
    "NUM": "some", "DATE": "at some point", "PERSON": "someone",
    "GPE": "somewhere", "ORG": "an organisation",
}

# Words inserted by an earlier stage must be immune to later ones, otherwise the MODIFIER stage
# deletes the very placeholders that replaced the entities.
PROTECTED = frozenset("some someone somewhere point organisation".split())

# Removing these does not lose detail, it changes the claim. Dropping "never" turns
# "has never demanded" into "has demanded", which is a contradiction, not forgetting.
NEGATIONS = frozenset("not never no nor none neither n't cannot without nothing nobody".split())
DEGREE = frozenset("more most less least very too so much many few only just also".split())

# WordNet's first hypernym is frequently an unusably abstract or wrong-sense node.
HYPERNYM_BLOCKLIST = frozenset([
    "entity", "object", "abstraction", "whole", "matter", "physical entity", "causal agent",
    "body", "part", "component", "legal document", "attribute", "measure", "psychological feature",
    "state", "relation", "act", "event", "thing", "unit", "group", "substance", "artifact",
])

_SPACY_STAGE = {
    "CARDINAL": "NUM", "MONEY": "NUM", "PERCENT": "NUM", "QUANTITY": "NUM", "ORDINAL": "NUM",
    "DATE": "DATE", "TIME": "DATE",
    "PERSON": "PERSON",
    "GPE": "GPE", "LOC": "GPE", "FAC": "GPE",
    "ORG": "ORG", "NORP": "ORG", "PRODUCT": "ORG", "WORK_OF_ART": "ORG", "EVENT": "ORG",
}

_TITLES = ("mr", "mrs", "ms", "dr", "prof", "sen", "rep", "gov", "pres")


def _load_spacy():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def _tidy(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    # Collapse placeholders that ended up adjacent after a multi-entity replacement.
    text = re.sub(r"\b(an organisation)(\s+\1\b)+", r"\1", text)
    text = re.sub(r"\b(someone|somewhere|some)(\s+\1\b)+", r"\1", text)
    text = re.sub(r"\b(at some point)(\s*,?\s*\1\b)+", r"\1", text)
    # Articles do not survive substitution: "the an organisation" / "a someone".
    text = re.sub(r"\b(?:a|an|the)\s+(an organisation|someone|somewhere)\b", r"\1",
                  text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r",\s*,+", ",", text)
    text = re.sub(r"\s+", " ", text).strip(" ,")
    text = re.sub(r"\ba\s+([aeiou])", r"an \1", text)
    if text and text[-1] not in ".!?":
        text += "."
    return text[0].upper() + text[1:] if text else text


def _replace_spans(text: str, spans: list[tuple[int, int, str]]) -> str:
    out, last = [], 0
    for start, end, rep in sorted(spans):
        if start < last:
            continue
        out.append(text[last:start])
        out.append(rep)
        last = end
    out.append(text[last:])
    return _tidy("".join(out))


def _hypernym(word: str) -> str | None:
    """One conservative generalisation step, or None when WordNet cannot give a clean one."""
    try:
        from nltk.corpus import wordnet as wn
        syns = wn.synsets(word, pos="n")
        if not syns:
            return None
        hyp = syns[0].hypernyms()
        if not hyp:
            return None
        name = hyp[0].lemmas()[0].name().replace("_", " ").lower()
    except Exception:
        return None
    if name in HYPERNYM_BLOCKLIST or " " in name or name == word.lower():
        return None
    return name


class ControlLadderBuilder:
    def __init__(self):
        self.nlp = _load_spacy()
        self.backend = "spacy" if self.nlp else "heuristic"

    # -- stage span finders ------------------------------------------------------------------
    def _spans_spacy(self, text: str, stage: str) -> list[tuple[int, int, str]]:
        doc = self.nlp(text)
        spans: list[tuple[int, int, str]] = []
        if stage in PLACEHOLDER:
            for ent in doc.ents:
                if _SPACY_STAGE.get(ent.label_) != stage:
                    continue
                if any(w in PROTECTED for w in ent.text.lower().split()):
                    continue
                spans.append((ent.start_char, ent.end_char, PLACEHOLDER[stage]))
        elif stage == "MODIFIER":
            for tok in doc:
                low = tok.lower_
                if low in PROTECTED or low in NEGATIONS or low in DEGREE:
                    continue
                if tok.dep_ == "neg" or any(c.dep_ == "neg" for c in tok.children):
                    continue
                if tok.pos_ == "ADJ" and tok.dep_ == "amod":
                    spans.append((tok.idx, tok.idx + len(tok.text) + 1, ""))
                elif tok.pos_ == "ADV" and tok.dep_ == "advmod" and low.endswith("ly"):
                    spans.append((tok.idx, tok.idx + len(tok.text) + 1, ""))
        elif stage == "HYPERNYM":
            for tok in doc:
                if tok.lower_ in PROTECTED or tok.ent_type_:
                    continue
                if tok.pos_ == "NOUN" and tok.dep_ in ("nsubj", "dobj", "pobj", "nsubjpass"):
                    h = _hypernym(tok.lemma_)
                    if h and h != tok.lemma_.lower():
                        spans.append((tok.idx, tok.idx + len(tok.text), h))
        return spans

    def _spans_heuristic(self, text: str, stage: str) -> list[tuple[int, int, str]]:
        spans: list[tuple[int, int, str]] = []
        if stage == "NUM":
            spans += [(m.start(), m.end(), "some") for m in _NUM_RE.finditer(text)]
        elif stage == "DATE":
            spans += [(m.start(), m.end(), "at some point") for m in _YEAR_RE.finditer(text)]
            for mon in _MONTHS:
                for m in re.finditer(rf"\b{mon}\b", text, re.IGNORECASE):
                    spans.append((m.start(), m.end(), "at some point"))
        elif stage in ("PERSON", "GPE", "ORG"):
            # Without NER, all capitalised multi-token names collapse in the PERSON stage.
            if stage == "PERSON":
                for m in re.finditer(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text):
                    spans.append((m.start(), m.end(), "someone"))
                for t in _TITLES:
                    for m in re.finditer(rf"\b{t}\.?\s+[A-Z][a-z]+", text, re.IGNORECASE):
                        spans.append((m.start(), m.end(), "someone"))
            elif stage == "GPE":
                for m in re.finditer(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}\b", text):
                    spans.append((m.start(), m.end(), "somewhere"))
        elif stage == "MODIFIER":
            for m in re.finditer(r"\b\w+(?:ly|ous|ive|ful|able)\b", text):
                if m.group().lower() in PROTECTED | NEGATIONS | DEGREE:
                    continue
                spans.append((m.start(), m.end() + 1, ""))
        return spans

    def spans(self, text: str, stage: str):
        return self._spans_spacy(text, stage) if self.nlp else self._spans_heuristic(text, stage)

    # -- ladder ---------------------------------------------------------------------------------
    def build(self, text: str, max_levels: int = 8) -> list[str]:
        """Returns [source, rung1, ...]; each rung removes exactly one attribute class."""
        levels = [_tidy(text)]
        current = levels[0]
        for stage in STAGES:
            if len(levels) >= max_levels:
                break
            spans = self.spans(current, stage)
            if not spans:
                continue
            nxt = _replace_spans(current, spans)
            if nxt and nxt != current and len(nxt.split()) >= 2:
                levels.append(nxt)
                current = nxt
        return levels

    def removal_map(self, text: str) -> dict[str, int]:
        """Which rung index removed each attribute class -- the ablation's ground truth."""
        out, current, idx = {}, _tidy(text), 0
        for stage in STAGES:
            spans = self.spans(current, stage)
            if not spans:
                continue
            nxt = _replace_spans(current, spans)
            if nxt and nxt != current:
                idx += 1
                out[stage] = idx
                current = nxt
        return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build deletion-based control ladders")
    ap.add_argument("--data-dir", default="data/final")
    ap.add_argument("--out", default="data/control")
    ap.add_argument("--splits", nargs="+", default=["train", "validate", "test"])
    ap.add_argument("--max-levels", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    builder = ControlLadderBuilder()
    print(f"backend: {builder.backend}")
    if builder.backend == "heuristic":
        print("  NOTE: install spaCy + en_core_web_sm for proper entity typing")
    os.makedirs(args.out, exist_ok=True)

    for split in args.splits:
        src = os.path.join(args.data_dir, f"{split}.json")
        if not os.path.exists(src):
            continue
        with open(src, encoding="utf-8") as f:
            records = json.load(f)
        if args.limit:
            records = records[: args.limit]

        out_records, kept = [], 0
        for i, r in enumerate(records):
            x = (r.get("x") or "").strip()
            if not x:
                continue
            levels = builder.build(x, args.max_levels)
            if len(levels) < 3:  # nothing to forget -> useless as a ladder
                continue
            kept += 1
            out_records.append({
                "id": str(r.get("id", i)),
                "x": levels[0],
                "xt": levels[1:],
                "y": levels[-1],
                "removal_map": builder.removal_map(x),
            })

        dst = os.path.join(args.out, f"{split}.json")
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(out_records, f, indent=1)
        mean_levels = sum(len(r["xt"]) + 1 for r in out_records) / max(kept, 1)
        print(f"  {split}: {kept}/{len(records)} usable, mean {mean_levels:.2f} levels -> {dst}")

    return args.out


if __name__ == "__main__":
    main()
