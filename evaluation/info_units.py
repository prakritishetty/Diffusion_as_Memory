"""Information units: the countable backbone of the new evaluation.

BERTScore cannot tell "Paris" from "London" -- both are near neighbours in embedding space -- so
for a project whose entire claim is about *which specific details survive* it is close to useless
as a primary metric. UniEval returns one opaque number and gives no credit for *appropriate*
abstraction. Both conflate "forgot a detail" (what we want) with "invented a wrong detail" (the
failure we most need to catch).

So detail is counted instead. An information unit is a typed, normalised piece of content:
numbers, dates, people, places, organisations and content words. Everything downstream --
retention curves, nesting, hallucination rate -- is set arithmetic over these.

spaCy is used when available; otherwise a regex/heuristic extractor keeps the pipeline runnable
(and the test suite offline). `Extractor.backend` records which was used, because the two are not
interchangeable for reported numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

# Coarse types shared by both backends, ordered from most to least episodic.
UNIT_TYPES = ("NUM", "DATE", "PERSON", "GPE", "ORG", "NAME", "CONTENT")
EPISODIC_TYPES = ("NUM", "DATE", "PERSON", "GPE", "ORG", "NAME")

_SPACY_MAP = {
    "CARDINAL": "NUM", "MONEY": "NUM", "PERCENT": "NUM", "QUANTITY": "NUM", "ORDINAL": "NUM",
    "DATE": "DATE", "TIME": "DATE",
    "PERSON": "PERSON",
    "GPE": "GPE", "LOC": "GPE", "FAC": "GPE",
    "ORG": "ORG", "NORP": "ORG", "PRODUCT": "ORG", "WORK_OF_ART": "ORG", "EVENT": "ORG",
}

STOPWORDS = frozenset("""
a an the and or but if then than that this these those of to in on at by for with from as is are
was were be been being do does did have has had will would shall should can could may might must
it its it's he she they them his her their we our you your i me my not no nor so such only own
same too very s t just don now also into over under again further once here there when where why
how all any both each few more most other some who whom which what while about against between
during before after above below up down out off through during
""".split())

_NUM_RE = re.compile(r"(?<![\w])(?:\$|£|€)?\d[\d,]*(?:\.\d+)?%?(?![\w])")
_MONTHS = ("january february march april may june july august september october november december "
           "jan feb mar apr jun jul aug sep sept oct nov dec").split()
_DAYS = "monday tuesday wednesday thursday friday saturday sunday".split()
_YEAR_RE = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")


@dataclass(frozen=True)
class InfoUnit:
    key: str     # normalised form, used for set comparison
    type: str
    surface: str

    def __repr__(self) -> str:
        return f"{self.type}:{self.key}"


def normalise_number(s: str) -> str:
    """Canonical key for a numeric mention.

    spaCy returns spans like "more than 16,000 tons" for QUANTITY, so the numeral is extracted
    and the unit words are left to be picked up as ordinary content units. Without this,
    "16,000 tons" and "16000" are different keys and never match across rungs.
    """
    s = s.strip().lower()
    m = _NUM_RE.search(s)
    core = (m.group() if m else s).replace(",", "").lstrip("$£€")
    pct = core.endswith("%") or "%" in s or "percent" in s
    core = core.rstrip("%")
    try:
        v = float(core)
        core = str(int(v)) if v == int(v) else str(v)
    except ValueError:
        core = re.sub(r"[^a-z0-9.]+", " ", core).strip()
    return core + ("%" if pct else "")


def normalise_word(w: str) -> str:
    w = w.lower().strip("'-")
    if len(w) > 3 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("es") and not w.endswith("ses"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


@lru_cache(maxsize=1)
def _load_spacy():
    try:
        import spacy
        return spacy.load("en_core_web_sm", disable=["lemmatizer"])
    except Exception:
        return None


class Extractor:
    """Extracts typed information units from text."""

    def __init__(self, prefer_spacy: bool = True):
        self.nlp = _load_spacy() if prefer_spacy else None
        self.backend = "spacy" if self.nlp is not None else "heuristic"

    # -- heuristic backend -------------------------------------------------------------------
    def _heuristic(self, text: str) -> list[InfoUnit]:
        units: list[InfoUnit] = []
        seen: set[str] = set()

        def add(key: str, typ: str, surface: str):
            k = f"{typ}:{key}"
            if key and k not in seen:
                seen.add(k)
                units.append(InfoUnit(key, typ, surface))

        for m in _NUM_RE.finditer(text):
            add(normalise_number(m.group()), "NUM", m.group())
        for m in _YEAR_RE.finditer(text):
            add(m.group(), "DATE", m.group())

        tokens = list(_TOKEN_RE.finditer(text))
        for i, m in enumerate(tokens):
            w = m.group()
            lower = w.lower()
            if lower in _MONTHS or lower in _DAYS:
                add(lower, "DATE", w)
                continue
            # Proper nouns: capitalised and not merely sentence-initial.
            prev_char = text[:m.start()].rstrip()[-1:] if m.start() else ""
            sentence_initial = (i == 0) or prev_char in ".!?\"'"
            if w[0].isupper() and not sentence_initial and lower not in STOPWORDS:
                add(normalise_word(w), "NAME", w)
            elif lower not in STOPWORDS and len(lower) > 2 and not w[0].isupper():
                add(normalise_word(w), "CONTENT", w)
            elif w[0].isupper() and sentence_initial and lower not in STOPWORDS:
                add(normalise_word(w), "CONTENT", w)
        return units

    # -- spaCy backend -------------------------------------------------------------------------
    def _spacy(self, text: str) -> list[InfoUnit]:
        doc = self.nlp(text)
        units: list[InfoUnit] = []
        seen: set[str] = set()
        covered: set[int] = set()

        def add(key: str, typ: str, surface: str):
            k = f"{typ}:{key}"
            if key and k not in seen:
                seen.add(k)
                units.append(InfoUnit(key, typ, surface))

        for ent in doc.ents:
            typ = _SPACY_MAP.get(ent.label_)
            if not typ:
                continue
            key = normalise_number(ent.text) if typ == "NUM" else normalise_word(ent.text)
            add(key, typ, ent.text)
            covered.update(range(ent.start, ent.end))

        for tok in doc:
            if tok.i in covered or tok.is_punct or tok.is_space:
                continue
            low = tok.text.lower()
            if tok.like_num:
                add(normalise_number(tok.text), "NUM", tok.text)
            elif tok.pos_ == "PROPN":
                add(normalise_word(tok.text), "NAME", tok.text)
            elif tok.pos_ in ("NOUN", "VERB", "ADJ", "ADV") and low not in STOPWORDS:
                add(normalise_word(tok.text), "CONTENT", tok.text)
        return units

    # -- public ---------------------------------------------------------------------------------
    def extract(self, text: str) -> list[InfoUnit]:
        if not text or not text.strip():
            return []
        return self._spacy(text) if self.nlp is not None else self._heuristic(text)

    def keys(self, text: str) -> set[str]:
        return {f"{u.type}:{u.key}" for u in self.extract(text)}

    def by_type(self, text: str) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {t: set() for t in UNIT_TYPES}
        for u in self.extract(text):
            out[u.type].add(u.key)
        return out


_DEFAULT: Extractor | None = None


# Placeholders that abstraction *introduces* rather than adds as new information. Treating them as
# new units would make every genuine abstraction look like a nesting violation and a
# hallucination: measured on the gold MSR ladders, strict lexical nesting holds for only ~2.5% of
# trajectories purely because "Washington" gets replaced by "someone".
GENERIC_UNITS = frozenset("""
someone somebody something somewhere sometime someplace anyone anybody anything anywhere
person people place thing things stuff item event group organisation organization entity
number amount quantity many several few various certain some other others one
happen happened happening occur occurred take took make made do did go went
company agency official authority body member area region location site
time date day year period moment
""".split())


def is_generic(key: str) -> bool:
    return key.split(":")[-1] in GENERIC_UNITS


def default_extractor() -> Extractor:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Extractor()
    return _DEFAULT


@lru_cache(maxsize=200_000)
def is_hypernym(specific: str, general: str) -> bool:
    """True when `general` is a WordNet ancestor of `specific`.

    This is what separates *abstraction* ("seabass" -> "fish") from *corruption*
    ("seabass" -> "tractor"), a distinction no metric in the previous evaluation measured.
    """
    if specific == general:
        return False
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return False
    try:
        s_syns, g_syns = wn.synsets(specific), wn.synsets(general)
    except LookupError:
        return False
    if not s_syns or not g_syns:
        return False
    g_set = set(g_syns)
    for s in s_syns[:3]:
        for path in s.hypernym_paths():
            if g_set & set(path):
                return True
    return False
