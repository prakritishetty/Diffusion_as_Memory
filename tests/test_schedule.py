import pytest
import torch

from diffusion import NoiseSchedule


@pytest.mark.parametrize("kind", ["cosine", "linear", "sigmoid"])
def test_alpha_bar_is_monotone_and_bounded(kind):
    s = NoiseSchedule(T=200, schedule_type=kind)
    ab = s.alpha_bar_table
    assert ab.shape == (200,)
    assert torch.all(ab > 0) and torch.all(ab < 1)
    assert torch.all(ab[1:] <= ab[:-1] + 1e-7), "alpha_bar must not increase with t"
    assert ab[0] > ab[-1]


@pytest.mark.parametrize("kind", ["cosine", "linear", "sigmoid"])
def test_log_snr_decreases(kind):
    s = NoiseSchedule(T=200, schedule_type=kind)
    lam = s.log_snr(s.timesteps())
    assert torch.all(lam[1:] <= lam[:-1] + 1e-5)


def test_unknown_schedule_raises_useful_message():
    # The original raised NameError here because it referenced an undefined local.
    with pytest.raises(ValueError, match="Unknown schedule_type"):
        NoiseSchedule(T=10, schedule_type="nope")


def test_lookup_is_vectorised_and_batched():
    s = NoiseSchedule(T=100)
    t = torch.tensor([1, 50, 100])
    assert s.alpha_bar(t).shape == (3,)
    assert torch.allclose(s.alpha_bar(t)[0], s.alpha_bar_table[0])
    assert torch.allclose(s.alpha_bar(t)[-1], s.alpha_bar_table[-1])


def test_timesteps_are_clamped_not_wrapped():
    s = NoiseSchedule(T=100)
    assert torch.allclose(s.alpha_bar(torch.tensor([0])), s.alpha_bar(torch.tensor([1])))
    assert torch.allclose(s.alpha_bar(torch.tensor([999])), s.alpha_bar(torch.tensor([100])))


def test_min_snr_weight_is_finite_everywhere():
    s = NoiseSchedule(T=1000)
    w = s.min_snr_weight(s.timesteps(), gamma=5.0)
    assert torch.all(torch.isfinite(w))
    assert torch.all(w > 0)


def test_schedule_moves_with_module():
    s = NoiseSchedule(T=50)
    s.to("cpu")
    assert s.alpha_bar_table.device.type == "cpu"
