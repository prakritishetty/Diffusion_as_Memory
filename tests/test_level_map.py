"""The level map is the fix for the supervision collapse, so it gets the strictest tests."""

import pytest
import torch

from diffusion import LevelMap, NoiseSchedule


@pytest.fixture
def sched():
    return NoiseSchedule(T=1000, schedule_type="cosine")


# -- the defect we are fixing -------------------------------------------------------------
def test_legacy_bucket_map_reproduces_the_collapse(sched):
    """min(t//100, n-1) with the observed mean ladder length puts ~60% on the final rung."""
    lm = LevelMap(sched, kind="bucket", bucket_size=100)
    report = lm.coverage_report([5])
    assert report[5]["fraction_on_final_level"] > 0.55


@pytest.mark.parametrize("n", [2, 4, 5, 6, 8, 13])
def test_uniform_t_map_gives_every_rung_an_equal_share(sched, n):
    lm = LevelMap(sched, kind="uniform_t")
    frac = lm.coverage_report([n])[n]["per_level_fraction"]
    assert len(frac) == n
    assert abs(sum(frac) - 1.0) < 1e-6
    for f in frac:
        assert abs(f - 1.0 / n) < 0.02, f"rung mass {f} far from {1/n} for n={n}"


@pytest.mark.parametrize("n", [4, 6, 8])
def test_clipped_logsnr_map_is_not_degenerate(sched, n):
    """Unclipped log-SNR gave the top rung 5 timesteps out of 1000; clipping keeps it usable."""
    lm = LevelMap(sched, kind="logsnr", logsnr_clip=6.0)
    frac = lm.coverage_report([n])[n]["per_level_fraction"]
    assert min(frac) > 0.02, frac
    assert max(frac) / min(frac) < 8.0, frac


def test_unclipped_logsnr_would_be_degenerate(sched):
    """Documents why `logsnr_clip` exists."""
    lm = LevelMap(sched, kind="logsnr", logsnr_clip=1e9)
    frac = lm.coverage_report([6])[6]["per_level_fraction"]
    assert min(frac) < 0.01


def test_every_rung_sits_at_a_distinguishable_noise_level(sched):
    """Adjacent rungs must differ in signal fraction, else the objective cannot separate them."""
    lm = LevelMap(sched, kind="uniform_t")
    n = 6
    sig = []
    for k in range(n):
        lo, hi = lm.level_bounds(torch.tensor([k]), torch.tensor([n]))
        mid = (lo + hi) // 2
        sig.append(float(sched.alpha_bar(mid).sqrt()))
    for a, b in zip(sig, sig[1:]):
        assert a - b > 0.05, f"adjacent rungs too similar: {sig}"


# -- invariants ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["uniform_t", "logsnr", "bucket"])
@pytest.mark.parametrize("n", [1, 3, 5, 12])
def test_index_in_range(sched, kind, n):
    lm = LevelMap(sched, kind=kind)
    t = torch.arange(1, sched.T + 1)
    k = lm.level_index(t, torch.full_like(t, n))
    assert int(k.min()) >= 0 and int(k.max()) <= n - 1


@pytest.mark.parametrize("kind", ["uniform_t", "logsnr", "bucket"])
def test_index_is_monotone_in_t(sched, kind):
    lm = LevelMap(sched, kind=kind)
    t = torch.arange(1, sched.T + 1)
    k = lm.level_index(t, torch.full_like(t, 6))
    assert torch.all(k[1:] >= k[:-1]), "more noise must never mean a less abstract rung"


def test_t_equals_one_is_the_verbatim_rung(sched):
    """Under the old rule t=1 mapped to xt[0], so the source was never a decoder target."""
    lm = LevelMap(sched, kind="uniform_t")
    k = lm.level_index(torch.tensor([1]), torch.tensor([6]))
    assert int(k.item()) == 0


def test_t_equals_T_is_the_most_abstract_rung(sched):
    lm = LevelMap(sched, kind="uniform_t")
    k = lm.level_index(torch.tensor([sched.T]), torch.tensor([6]))
    assert int(k.item()) == 5


@pytest.mark.parametrize("kind", ["uniform_t", "logsnr", "bucket"])
@pytest.mark.parametrize("n", [2, 5, 9])
def test_bounds_round_trip(sched, kind, n):
    """Every t inside [lo, hi] for rung k must map back to k."""
    lm = LevelMap(sched, kind=kind)
    for k in range(n):
        kk = torch.tensor([k])
        nn_ = torch.tensor([n])
        lo, hi = lm.level_bounds(kk, nn_)
        lo, hi = int(lo), int(hi)
        assert 1 <= lo <= hi <= sched.T
        probe = torch.tensor([lo, (lo + hi) // 2, hi])
        back = lm.level_index(probe, torch.full_like(probe, n))
        assert torch.all(back == k), f"k={k} n={n} bounds=({lo},{hi}) -> {back.tolist()}"


def test_bounds_tile_the_schedule_without_gaps(sched):
    lm = LevelMap(sched, kind="uniform_t")
    n = 6
    spans = []
    for k in range(n):
        lo, hi = lm.level_bounds(torch.tensor([k]), torch.tensor([n]))
        spans.append((int(lo), int(hi)))
    assert spans[0][0] == 1
    assert spans[-1][1] == sched.T
    for (_, hi), (lo, _) in zip(spans, spans[1:]):
        assert lo == hi + 1, f"gap or overlap between {hi} and {lo}"


# -- stratified sampling -------------------------------------------------------------------
def test_stratified_sampling_is_uniform_over_rungs(sched):
    lm = LevelMap(sched, kind="uniform_t")
    n_levels = torch.full((20000,), 5)
    t, k = lm.sample_stratified(n_levels)
    counts = torch.bincount(k, minlength=5).float() / len(k)
    assert torch.all((counts - 0.2).abs() < 0.02), counts.tolist()
    assert torch.all(lm.level_index(t, n_levels) == k), "sampled t must land in its own rung"


def test_stratified_handles_mixed_ladder_lengths(sched):
    lm = LevelMap(sched, kind="uniform_t")
    n_levels = torch.tensor([2, 4, 5, 5, 6, 13] * 200)
    t, k = lm.sample_stratified(n_levels)
    assert torch.all(k < n_levels) and torch.all(k >= 0)
    assert torch.all(lm.level_index(t, n_levels) == k)


def test_uniform_t_sampling_is_biased_which_is_why_we_stratify(sched):
    """Documents the difference: uniform t under the legacy map over-weights the last rung."""
    lm_bucket = LevelMap(sched, kind="bucket", bucket_size=100)
    n_levels = torch.full((20000,), 5)
    _, k = lm_bucket.sample_uniform_t(n_levels)
    counts = torch.bincount(k, minlength=5).float() / len(k)
    assert counts[-1] > 0.5


def test_sampling_is_reproducible_with_a_generator(sched):
    lm = LevelMap(sched, kind="uniform_t")
    n_levels = torch.full((64,), 5)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    t1, k1 = lm.sample_stratified(n_levels, generator=g1)
    t2, k2 = lm.sample_stratified(n_levels, generator=g2)
    assert torch.equal(t1, t2) and torch.equal(k1, k2)


# -- inference helpers ----------------------------------------------------------------------
def test_budget_zero_is_full_detail_and_one_is_most_abstract(sched):
    lm = LevelMap(sched, kind="uniform_t")
    assert lm.t_for_budget(0.0) == 1
    assert lm.t_for_budget(1.0) >= sched.T - 1


def test_t_for_level_is_inside_its_bounds(sched):
    lm = LevelMap(sched, kind="uniform_t")
    for k in range(6):
        t = lm.t_for_level(k, 6)
        assert int(lm.level_index(torch.tensor([t]), torch.tensor([6]))) == k
