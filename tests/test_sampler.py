import pytest
import torch

from diffusion import DDIMSampler, NoiseSchedule, q_sample, strided_timesteps, target_for, to_v0


@pytest.fixture
def sched():
    return NoiseSchedule(T=200, schedule_type="cosine")


# -- timestep striding ---------------------------------------------------------------------
def test_strided_timesteps_descend_and_hit_both_ends():
    ts = strided_timesteps(200, 1, 20)
    assert ts[0] == 200 and ts[-1] == 1
    assert all(a > b for a, b in zip(ts, ts[1:]))
    assert len(ts) <= 22


def test_strided_timesteps_degenerate_cases():
    assert strided_timesteps(5, 5, 10) == [5]
    assert strided_timesteps(3, 1, 10) == [3, 2, 1]


# -- forward process -----------------------------------------------------------------------
def test_q_sample_is_reproducible_with_a_seed(sched):
    v0 = torch.randn(2, 4, 8)
    t = torch.tensor([50, 120])
    a, _ = q_sample(v0, t, sched, seed=11)
    b, _ = q_sample(v0, t, sched, seed=11)
    c, _ = q_sample(v0, t, sched, seed=12)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_q_sample_preserves_signal_at_small_t(sched):
    v0 = torch.randn(4, 4, 8)
    v_lo, _ = q_sample(v0, torch.full((4,), 1), sched, seed=0)
    v_hi, _ = q_sample(v0, torch.full((4,), sched.T), sched, seed=0)
    assert (v_lo - v0).abs().mean() < (v_hi - v0).abs().mean()


def test_injected_noise_is_uncorrelated_with_data_at_the_same_seed(sched):
    """set_seed(s) followed by q_sample(seed=s) must not draw the same stream twice."""
    from utils.seed import randn_like_seeded, set_seed

    set_seed(0)
    v0 = torch.randn(64, 8, 16)
    noise = randn_like_seeded(v0, 0)
    corr = torch.corrcoef(torch.stack([v0.flatten(), noise.flatten()]))[0, 1]
    assert corr.abs() < 0.1, f"injected noise correlates with data: r={corr:.3f}"


def test_gist_anchored_terminal_state_is_gist_plus_noise(sched):
    """v_T should sit around m(u), not around 0 -- the ladder's abstract end needs a floor."""
    v0 = torch.randn(64, 4, 8)
    anchor = torch.full_like(v0, 5.0)
    t = torch.full((64,), sched.T)
    v_anch, _ = q_sample(v0, t, sched, anchor=anchor, seed=3)
    v_plain, _ = q_sample(v0, t, sched, anchor=None, seed=3)
    assert abs(v_anch.mean().item() - 5.0) < 0.3
    assert abs(v_plain.mean().item()) < 0.3


def test_gist_anchored_is_identity_at_t_zero_limit(sched):
    v0 = torch.randn(4, 4, 8)
    anchor = torch.randn(4, 4, 8)
    v1, _ = q_sample(v0, torch.full((4,), 1), sched, anchor=anchor, seed=0)
    assert (v1 - v0).abs().max() < 0.15


# -- parameterisation round trips -----------------------------------------------------------
@pytest.mark.parametrize("param", ["x0", "eps", "v"])
@pytest.mark.parametrize("anchored", [False, True])
def test_target_and_inverse_are_consistent(sched, param, anchored):
    v0 = torch.randn(3, 4, 8)
    anchor = torch.randn(3, 4, 8) if anchored else None
    t = torch.tensor([10, 100, 190])
    v_t, noise = q_sample(v0, t, sched, anchor=anchor, seed=5)
    pred = target_for(param, v0, noise, t, sched, anchor=anchor)
    recovered = to_v0(pred, v_t, t, sched, param, anchor=anchor)
    assert torch.allclose(recovered, v0, atol=1e-3), (recovered - v0).abs().max()


def test_x0_parameterisation_is_stable_at_the_top_of_the_schedule(sched):
    """eps-prediction divides by sqrt(alpha_bar); x0-prediction does not."""
    v0 = torch.randn(4, 4, 8)
    t = torch.full((4,), sched.T)
    v_t, noise = q_sample(v0, t, sched, seed=1)

    x0_pred = target_for("x0", v0, noise, t, sched)
    rec_x0 = to_v0(x0_pred, v_t, t, sched, "x0")
    assert torch.allclose(rec_x0, v0, atol=1e-5)

    perturbed = target_for("eps", v0, noise, t, sched) + 0.01 * torch.randn_like(noise)
    rec_eps = to_v0(perturbed, v_t, t, sched, "eps")
    assert (rec_eps - v0).abs().max() > 10 * (rec_x0 - v0).abs().max()


# -- sampler -------------------------------------------------------------------------------
def _oracle(v0_true, anchor=None):
    """A denoiser that already knows the answer; DDIM on it must be exact."""
    w0 = v0_true if anchor is None else v0_true - anchor
    return lambda v_t, t, u: w0


@pytest.mark.parametrize("anchored", [False, True])
def test_ddim_with_an_oracle_reproduces_the_forward_trajectory(sched, anchored):
    torch.manual_seed(0)
    v0 = torch.randn(2, 4, 8)
    anchor = torch.randn(2, 4, 8) if anchored else None
    t_start = sched.T

    v_t, noise = q_sample(v0, torch.full((2,), t_start), sched, anchor=anchor, seed=9)
    sampler = DDIMSampler(sched, n_steps=25, eta=0.0, parameterization="x0")
    _, rec = sampler.sample(
        _oracle(v0, anchor), v_t, t_start, t_end=1, anchor=anchor, record_at=[150, 100, 50, 1]
    )
    for t_chk, v_chk in rec.items():
        expected, _ = q_sample(
            v0, torch.full((2,), t_chk), sched, anchor=anchor, noise=noise
        )
        assert torch.allclose(v_chk, expected, atol=1e-4), f"mismatch at t={t_chk}"


def test_ddim_recovers_v0_at_the_bottom_of_the_chain(sched):
    torch.manual_seed(0)
    v0 = torch.randn(2, 4, 8)
    v_t, noise = q_sample(v0, torch.full((2,), sched.T), sched, seed=2)
    sampler = DDIMSampler(sched, n_steps=50, eta=0.0)
    out, _ = sampler.sample(_oracle(v0), v_t, sched.T, t_end=1)
    # t=1 still carries sqrt(1-alpha_bar_1) of noise, so compare against the exact q_sample there.
    expected, _ = q_sample(v0, torch.full((2,), 1), sched, noise=noise)
    assert torch.allclose(out, expected, atol=1e-4)
    assert (out - v0).abs().mean() < 0.05


def test_train_and_eval_chains_agree_when_step_counts_match(sched):
    """The property that fixes the P2 train/inference mismatch."""
    torch.manual_seed(0)
    v0 = torch.randn(2, 4, 8)
    v_t, _ = q_sample(v0, torch.full((2,), sched.T), sched, seed=4)
    a = DDIMSampler(sched, n_steps=20).sample(_oracle(v0), v_t, sched.T, 80)[0]
    b = DDIMSampler(sched, n_steps=20).sample(_oracle(v0), v_t, sched.T, 80)[0]
    assert torch.equal(a, b)


def test_ladder_harvests_every_requested_stop(sched):
    torch.manual_seed(0)
    v0 = torch.randn(2, 4, 8)
    v_t, _ = q_sample(v0, torch.full((2,), sched.T), sched, seed=6)
    stops = [180, 140, 90, 40, 5]
    out = DDIMSampler(sched, n_steps=40).ladder(_oracle(v0), v_t, sched.T, stops)
    assert sorted(out.keys(), reverse=True) == stops
    assert all(v.shape == v0.shape for v in out.values())


def test_stochastic_sampling_is_seeded(sched):
    torch.manual_seed(0)
    v0 = torch.randn(2, 4, 8)
    v_t, _ = q_sample(v0, torch.full((2,), sched.T), sched, seed=8)
    s = DDIMSampler(sched, n_steps=20, eta=1.0)
    a, _ = s.sample(_oracle(v0), v_t, sched.T, 1, seed=42)
    b, _ = s.sample(_oracle(v0), v_t, sched.T, 1, seed=42)
    c, _ = s.sample(_oracle(v0), v_t, sched.T, 1, seed=43)
    assert torch.equal(a, b), "trajectories must be reproducible"
    assert not torch.equal(a, c)


def test_cfg_interpolates_between_conditional_and_unconditional(sched):
    v0_cond, v0_uncond = torch.zeros(1, 2, 4), torch.ones(1, 2, 4)

    def fn(v_t, t, u):
        return v0_cond if u is not None else v0_uncond

    s = DDIMSampler(sched, n_steps=2, parameterization="x0")
    u = torch.zeros(1, 3)
    a, _ = s._predict(fn, torch.zeros(1, 2, 4), 100, u, None, 1.0)
    b, _ = s._predict(fn, torch.zeros(1, 2, 4), 100, u, None, 2.0)
    assert torch.allclose(a, v0_cond)
    assert torch.allclose(b, v0_uncond + 2.0 * (v0_cond - v0_uncond))
