"""Graded-recall metrics.

What each one answers:

* **retention(k)**      -- how much of the source's detail survives at rung k
* **decay_spearman**    -- does detail fall monotonically with depth? (a rank correlation, not a
                           binary flag: one blip in a 5-rung trajectory used to zero the whole
                           thing, which threw away almost all the signal)
* **nesting violations**-- is each rung a *subset* of the one below it? This is the early-exit
                           safety guarantee, and it is the property that makes stopping early
                           "incomplete but not wrong" rather than merely short
* **hallucination**     -- units present in a rung but absent from the source: separates
                           *forgot* from *made up*, which BERTScore and UniEval both conflate
* **abstraction validity** -- of the units that were replaced, how many were replaced by a
                           genuine hypernym

`early_exit_curve` is the headline figure: quality against budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Sequence

from evaluation.info_units import (
    EPISODIC_TYPES, UNIT_TYPES, Extractor, default_extractor, is_generic, is_hypernym,
)


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(list(xs)), rank(list(ys))
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def length_matched_retention(
    source: str,
    n_words: int,
    extractor: Extractor,
    n_draws: int = 8,
    seed: int = 0,
) -> dict[str, float]:
    """Detail retained by deleting words at random down to `n_words`.

    The control for a confound that is easy to miss: on the LLM ladders, retention correlates
    with length ratio at Spearman 0.92, so most of the apparent "detail decay" is just the text
    getting shorter. Any model that writes tersely scores well without abstracting anything.

    Comparing against a length-matched random deletion isolates *selection* from *brevity*.
    """
    import random

    words = source.split()
    if n_words >= len(words) or n_words <= 0:
        by_type = extractor.by_type(source)
        return {t: 1.0 if by_type[t] else float("nan") for t in UNIT_TYPES}

    rng = random.Random(seed)
    src = extractor.by_type(source)
    acc = {t: 0.0 for t in UNIT_TYPES}
    for _ in range(n_draws):
        keep = sorted(rng.sample(range(len(words)), n_words))
        sample = extractor.by_type(" ".join(words[i] for i in keep))
        for t in UNIT_TYPES:
            if src[t]:
                acc[t] += len(sample[t] & src[t]) / len(src[t])
    return {t: (acc[t] / n_draws if src[t] else float("nan")) for t in UNIT_TYPES}


def selectivity(
    source: str,
    rungs: Sequence[str],
    extractor: Extractor | None = None,
    n_draws: int = 8,
) -> dict[str, list[float]]:
    """Per-type forgetting *above chance*, holding length fixed.

    `selectivity[type][k] = retention_random(type, k) - retention_actual(type, k)`

    Positive means the rung dropped that type *more* than shortening alone would, i.e. it chose
    to forget it. This is the length-independent version of the human-like-forgetting claim:
    numbers and dates should be positive, core predicates negative.
    """
    ex = extractor or default_extractor()
    src = ex.by_type(source)
    out: dict[str, list[float]] = {t: [] for t in UNIT_TYPES if src[t]}
    for k, rung in enumerate(rungs):
        actual = ex.by_type(rung)
        chance = length_matched_retention(source, len(rung.split()), ex, n_draws, seed=k)
        for t in out:
            a = len(actual[t] & src[t]) / len(src[t])
            c = chance[t]
            out[t].append(round(c - a, 4) if c == c else 0.0)
    return out


def _is_non_increasing(xs: Sequence[float], tol: float = 1e-9) -> bool:
    return all(b <= a + tol for a, b in zip(xs, xs[1:]))


def is_well_formed(text: str) -> bool:
    """Cheap syntactic sanity check: a standalone clause, not a cut-off fragment.

    This is the axis that separates graded recall from scissors. Measured on MSR, prefix
    truncation scores 0.93 monotonicity and 0.73 nesting -- better than the gold ladders -- while
    being well formed only 8.5% of the time. Monotonicity and nesting alone are therefore not a
    claim; producing rungs that are *complete at every budget* is.
    """
    t = (text or "").strip()
    if len(t.split()) < 3:
        return False
    if "..." in t or "…" in t:  # head-tail compression looks terminated but is spliced
        return False
    if t.endswith(","):
        return False
    if t[-1] not in ".!?":
        return False
    return any(c.isalpha() for c in t)


@dataclass
class TrajectoryMetrics:
    retention: list[float] = field(default_factory=list)
    retention_by_type: dict[str, list[float]] = field(default_factory=dict)
    decay_spearman: float = 0.0
    strictly_monotonic: bool = False
    # Headline: a rung introduces no *specific* information the finer rung lacked.
    nesting_violations: int = 0
    nesting_rate: float = 0.0
    # Diagnostic: strict set containment, which abstraction breaks by construction.
    lexical_nesting_rate: float = 0.0
    hallucination: list[float] = field(default_factory=list)
    abstraction_validity: float = 0.0
    well_formed: list[bool] = field(default_factory=list)
    # Composite: a rung is usable if it is well formed, adds nothing specific, and invents nothing.
    usable_rate: float = 0.0
    n_rungs: int = 0
    length: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _specific(units: set[str]) -> set[str]:
    """Drop placeholder units, which abstraction introduces rather than asserts."""
    return {u for u in units if not is_generic(u)}


def evaluate_trajectory(
    source: str,
    rungs: Sequence[str],
    extractor: Extractor | None = None,
    check_hypernyms: bool = False,
) -> TrajectoryMetrics:
    """Score one ladder, ordered least-abstract first."""
    ex = extractor or default_extractor()
    src_units = ex.by_type(source)
    src_all = {f"{t}:{k}" for t, ks in src_units.items() for k in ks}
    rung_units = [ex.by_type(r) for r in rungs]
    rung_all = [{f"{t}:{k}" for t, ks in ru.items() for k in ks} for ru in rung_units]

    m = TrajectoryMetrics(n_rungs=len(rungs))
    m.length = [len(r.split()) for r in rungs]

    denom = max(len(src_all), 1)
    m.retention = [len(ru & src_all) / denom for ru in rung_all]
    m.hallucination = [
        len(_specific(ru) - src_all) / max(len(_specific(ru)), 1) for ru in rung_all
    ]

    for t in UNIT_TYPES:
        if src_units[t]:
            m.retention_by_type[t] = [
                len(ru[t] & src_units[t]) / len(src_units[t]) for ru in rung_units
            ]

    m.decay_spearman = -_spearman(list(range(len(rungs))), m.retention)
    m.strictly_monotonic = _is_non_increasing(m.retention)

    # Nesting: a coarser rung must not assert anything specific the finer rung did not.
    # Generalising something that *was* there ("seabass" -> "fish") is valid abstraction rather
    # than new information, so hypernyms of dropped units are exempt.
    if len(rung_all) > 1:
        viol = 0
        for a, b in zip(rung_all, rung_all[1:]):
            new = _specific(b) - a
            if new and check_hypernyms:
                dropped = {u.split(":", 1)[1] for u in a - b}
                new = {u for u in new
                       if not any(is_hypernym(d, u.split(":", 1)[1]) for d in dropped)}
            viol += len(new)
        total = sum(max(len(_specific(b)), 1) for b in rung_all[1:])
        m.nesting_violations = viol
        m.nesting_rate = viol / max(total, 1)
        lex = sum(len(b - a) for a, b in zip(rung_all, rung_all[1:]))
        m.lexical_nesting_rate = lex / max(sum(max(len(b), 1) for b in rung_all[1:]), 1)

    if check_hypernyms and len(rung_all) > 1:
        replaced, valid = 0, 0
        for a_u, b_u in zip(rung_units, rung_units[1:]):
            dropped = a_u["CONTENT"] - b_u["CONTENT"]
            added = b_u["CONTENT"] - a_u["CONTENT"]
            for d in dropped:
                replaced += 1
                if any(is_hypernym(d, g) for g in added):
                    valid += 1
        m.abstraction_validity = valid / replaced if replaced else 0.0

    m.well_formed = [is_well_formed(r) for r in rungs]
    usable = 0
    for i, r in enumerate(rungs):
        ok = m.well_formed[i] and m.hallucination[i] == 0.0
        if i > 0:
            ok = ok and not (_specific(rung_all[i]) - rung_all[i - 1])
        usable += bool(ok)
    m.usable_rate = usable / max(len(rungs), 1)
    return m


def aggregate(metrics: Sequence[TrajectoryMetrics]) -> dict:
    """Corpus-level summary. Curves are averaged over trajectories of matching length."""
    if not metrics:
        return {}
    n = len(metrics)
    out: dict = {
        "n_trajectories": n,
        "mean_decay_spearman": sum(m.decay_spearman for m in metrics) / n,
        "frac_strictly_monotonic": sum(m.strictly_monotonic for m in metrics) / n,
        "mean_nesting_violation_rate": sum(m.nesting_rate for m in metrics) / n,
        "frac_perfectly_nested": sum(m.nesting_violations == 0 for m in metrics) / n,
        "mean_lexical_nesting_violation_rate": sum(m.lexical_nesting_rate for m in metrics) / n,
        "mean_hallucination": sum(
            sum(m.hallucination) / max(len(m.hallucination), 1) for m in metrics
        ) / n,
        "mean_abstraction_validity": sum(m.abstraction_validity for m in metrics) / n,
        "mean_well_formed": sum(
            sum(m.well_formed) / max(len(m.well_formed), 1) for m in metrics
        ) / n,
        "mean_usable_rung_rate": sum(m.usable_rate for m in metrics) / n,
    }

    max_len = max(m.n_rungs for m in metrics)
    curve, counts = [0.0] * max_len, [0] * max_len
    for m in metrics:
        for i, v in enumerate(m.retention):
            curve[i] += v
            counts[i] += 1
    out["retention_curve"] = [round(c / k, 4) if k else None for c, k in zip(curve, counts)]

    by_type: dict[str, list[float]] = {}
    for t in UNIT_TYPES:
        acc, cnt = [0.0] * max_len, [0] * max_len
        for m in metrics:
            for i, v in enumerate(m.retention_by_type.get(t, [])):
                acc[i] += v
                cnt[i] += 1
        if any(cnt):
            by_type[t] = [round(a / c, 4) if c else None for a, c in zip(acc, cnt)]
    out["retention_curve_by_type"] = by_type

    # Do episodic details fade faster than the rest? The human-like-forgetting claim.
    ep = [by_type[t] for t in EPISODIC_TYPES if t in by_type]
    if ep and "CONTENT" in by_type:
        def drop(curve_):
            vals = [v for v in curve_ if v is not None]
            return (vals[0] - vals[-1]) if len(vals) > 1 else 0.0
        out["episodic_minus_content_decay"] = round(
            sum(drop(c) for c in ep) / len(ep) - drop(by_type["CONTENT"]), 4
        )
    return out


def early_exit_curve(
    per_budget: dict[float, list[float]],
    cost: dict[float, float] | None = None,
) -> dict:
    """Quality against budget -- the headline figure.

    `per_budget` maps budget -> per-example quality; `cost` maps budget -> tokens or NFE.
    """
    budgets = sorted(per_budget)
    quality = [sum(per_budget[b]) / max(len(per_budget[b]), 1) for b in budgets]
    out = {"budgets": budgets, "quality": [round(q, 4) for q in quality]}
    if cost:
        out["cost"] = [cost.get(b) for b in budgets]
        full = quality[-1] if quality else 0.0
        for frac in (0.9, 0.95, 0.99):
            hit = next((b for b, q in zip(budgets, quality) if full and q >= frac * full), None)
            out[f"budget_for_{int(frac*100)}pct_quality"] = hit
    return out
