"""Metric tests.

These lock in the behaviour that the ceiling run exposed:
  * abstraction rewrites, so *lexical* nesting is the wrong test on paraphrastic ladders
  * truncation scores well on monotonicity and nesting while being a fragment, so those two
    metrics alone are not a claim
"""

import pytest

from evaluation.info_units import Extractor, is_generic, normalise_number, normalise_word
from evaluation.metrics import (
    aggregate, early_exit_curve, evaluate_trajectory, is_well_formed,
)


@pytest.fixture(scope="module")
def ex():
    return Extractor()


# -- extraction ------------------------------------------------------------------------------
def test_numbers_are_normalised():
    assert normalise_number("16,000") == "16000"
    assert normalise_number("$1,500.00") == "1500"
    assert normalise_number("61%") == "61%"


def test_plural_normalisation():
    assert normalise_word("walls") == "wall"
    assert normalise_word("museums") == "museum"
    assert normalise_word("class") == "class"


def test_extractor_finds_numbers_and_names(ex):
    units = ex.by_type("The NMFS caught 16,000 tons near Washington in 2000.")
    assert "16000" in units["NUM"]
    assert any("washington" in k for k in units["GPE"] | units["NAME"])
    assert units["DATE"] or "2000" in units["NUM"]


def test_generic_placeholders_are_recognised():
    assert is_generic("CONTENT:someone")
    assert is_generic("CONTENT:something")
    assert not is_generic("CONTENT:seabass")


# -- trajectory ------------------------------------------------------------------------------
def test_retention_falls_along_a_good_ladder(ex):
    m = evaluate_trajectory(
        "Washington demanded governance from Cairo in 1999.",
        ["Washington demanded governance from Cairo in 1999.",
         "Washington demanded governance from Cairo.",
         "Washington demanded something.",
         "Someone demanded something."],
        ex,
    )
    assert m.retention[0] == pytest.approx(1.0)
    assert m.retention[-1] < m.retention[0]
    assert m.strictly_monotonic
    assert m.decay_spearman > 0.8


def test_reappearing_detail_breaks_monotonicity(ex):
    m = evaluate_trajectory(
        "Washington demanded governance from Cairo in 1999.",
        ["Washington demanded governance.",
         "Someone demanded something.",
         "Washington demanded governance from Cairo in 1999."],
        ex,
    )
    assert not m.strictly_monotonic


def test_invented_detail_is_counted_as_hallucination_not_forgetting(ex):
    m = evaluate_trajectory(
        "Washington demanded governance.",
        ["Washington demanded governance.", "London demanded governance in 1789."],
        ex,
    )
    assert m.hallucination[1] > 0
    assert m.nesting_violations > 0


def test_placeholders_do_not_count_as_new_information(ex):
    m = evaluate_trajectory(
        "Washington demanded governance from Cairo.",
        ["Washington demanded governance from Cairo.",
         "Someone demanded governance from somewhere.",
         "Someone demanded something."],
        ex,
    )
    assert m.nesting_violations == 0, "generic placeholders must not be nesting violations"
    assert m.nesting_rate < m.lexical_nesting_rate or m.lexical_nesting_rate == 0


def test_spearman_degrades_gracefully_unlike_a_binary_flag(ex):
    src = "Washington demanded good governance from Cairo in 1999."
    blipped = [src,
               "Washington demanded governance from Cairo.",
               "Washington demanded governance from Cairo in 1999.",  # one blip
               "Someone demanded something."]
    m = evaluate_trajectory(src, blipped, ex)
    assert not m.strictly_monotonic
    assert m.decay_spearman > 0.3, "a single blip should not zero the whole trajectory"


# -- well-formedness / composite ----------------------------------------------------------------
def test_well_formedness_rejects_fragments_and_splices():
    assert is_well_formed("Someone demanded something.")
    assert not is_well_formed("Washington has never demanded good")
    assert not is_well_formed("The senior of ... and New Jersey.")
    assert not is_well_formed("Hi.")


def test_truncation_is_nested_but_not_usable(ex):
    """The result that reframes the claim: scissors pass nesting, not usability."""
    src = "Washington has never demanded good governance and democracy from Cairo in return."
    truncated = [src, " ".join(src.split()[:8]), " ".join(src.split()[:4]),
                 " ".join(src.split()[:2])]
    m = evaluate_trajectory(src, truncated, ex)
    assert m.nesting_violations == 0, "a prefix cannot add information"
    assert m.usable_rate < 0.5, "but prefixes are not standalone rungs"


def test_good_ladder_beats_truncation_on_usable_rate(ex):
    src = "Washington has never demanded good governance from Cairo."
    good = [src, "Washington has never demanded governance from Cairo.",
            "Someone has never demanded governance.", "Someone demanded nothing."]
    trunc = [src, " ".join(src.split()[:6]), " ".join(src.split()[:4]),
             " ".join(src.split()[:2])]
    assert (evaluate_trajectory(src, good, ex).usable_rate
            > evaluate_trajectory(src, trunc, ex).usable_rate)


# -- aggregation ------------------------------------------------------------------------------
def test_aggregate_produces_curves(ex):
    src = "Washington demanded governance from Cairo in 1999."
    ms = [evaluate_trajectory(src, [src, "Washington demanded governance.", "Someone demanded."],
                              ex) for _ in range(3)]
    agg = aggregate(ms)
    assert agg["n_trajectories"] == 3
    assert len(agg["retention_curve"]) == 3
    assert agg["retention_curve"][0] == pytest.approx(1.0)
    assert "mean_usable_rung_rate" in agg and "mean_well_formed" in agg


def test_aggregate_handles_ragged_lengths(ex):
    src = "Washington demanded governance from Cairo."
    a = evaluate_trajectory(src, [src, "Washington demanded governance."], ex)
    b = evaluate_trajectory(src, [src, "Washington demanded governance.", "Someone demanded."], ex)
    agg = aggregate([a, b])
    assert len(agg["retention_curve"]) == 3


def test_early_exit_curve_reports_budget_for_target_quality():
    per_budget = {0.0: [0.2], 0.25: [0.5], 0.5: [0.8], 0.75: [0.95], 1.0: [1.0]}
    out = early_exit_curve(per_budget, cost={b: b * 40 for b in per_budget})
    assert out["budgets"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert out["budget_for_90pct_quality"] == 0.75
    assert out["cost"][-1] == 40


def test_empty_input_does_not_crash(ex):
    assert aggregate([]) == {}
    m = evaluate_trajectory("", [], ex)
    assert m.n_rungs == 0


# -- length confound -----------------------------------------------------------------------------
def test_length_matched_baseline_tracks_the_cut(ex):
    from evaluation.metrics import length_matched_retention

    src = "Washington demanded good governance from Cairo in 1999 for many reasons."
    full = length_matched_retention(src, len(src.split()), ex)
    half = length_matched_retention(src, len(src.split()) // 2, ex, n_draws=16)
    assert full["CONTENT"] == pytest.approx(1.0)
    assert half["CONTENT"] < full["CONTENT"]


def test_selectivity_detects_targeted_removal(ex):
    """Dropping numbers specifically must score above a length-matched random deletion."""
    from evaluation.metrics import selectivity

    src = "Washington demanded 16,000 tons of governance from Cairo in 1999."
    targeted = [src, "Washington demanded tons of governance from Cairo."]
    s = selectivity(src, targeted, ex, n_draws=16)
    assert s["NUM"][1] > 0.2, s
    assert s["NUM"][1] > s["CONTENT"][1], "numbers should be dropped more selectively than words"


def test_selectivity_is_near_zero_for_random_shortening(ex):
    """A rung that just truncates should show no selectivity -- that is the point of the control."""
    from evaluation.metrics import selectivity

    src = "Washington demanded 16,000 tons of governance from Cairo in 1999 for reasons."
    words = src.split()
    trunc = [src, " ".join(words[: len(words) // 2])]
    s = selectivity(src, trunc, ex, n_draws=16)
    assert abs(s["CONTENT"][1]) < 0.35, s
