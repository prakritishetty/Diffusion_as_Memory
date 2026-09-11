"""End-to-end pipeline test: P0 -> P1 -> P2 -> recall, on CPU, offline, in seconds.

This is the pre-GPU gate. It cannot tell us whether the idea works, but it does catch the class of
defect that cost the previous iteration the most: modules silently not receiving gradient, a
phase trained on an input distribution it never sees at inference, checkpoints that load into the
wrong architecture, and evaluation numbers produced by teacher forcing.
"""

import json
import os

import pytest
import torch

from configs import get
from dataloader.memory_dataset import MemoryLadderDataset, collate
from diffusion import q_sample
from inference.recall_engine import RecallEngine
from models.g_psi_module.semantic_projection import SemanticProjectionModule
from models.latent_diffusion import LatentDiffusion
from models.memory_model import MemoryModel
from tests.stubs import StubTokenizer, temp_data_dir


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    data_dir = temp_data_dir(n_repeat=6)
    out_dir = str(tmp_path_factory.mktemp("runs"))
    cfg = get("tiny").override(**{
        "data.data_dir": data_dir, "out_dir": out_dir,
        "train.epochs": 1, "train.batch_size": 4, "train.val_interval": 1,
        "cache.enabled": True, "cache.root": os.path.join(out_dir, "cache"),
    })
    tok = StubTokenizer()
    ds = MemoryLadderDataset(os.path.join(data_dir, "train.json"), tok, cfg, split="train")
    return cfg, tok, ds, out_dir


@pytest.fixture(scope="module")
def p0(env):
    cfg, tok, ds, _ = env
    torch.manual_seed(0)
    model = MemoryModel(cfg)
    batch = collate([ds[i] for i in range(8)])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        loss, _ = model(batch)
        loss.backward()
        opt.step()
    return model


@pytest.fixture(scope="module")
def p1(env, p0):
    cfg, _, ds, _ = env
    torch.manual_seed(0)
    diffusion = LatentDiffusion(cfg, p0.d_model)
    batch = collate([ds[i] for i in range(8)])
    u, v0 = p0.encode_latents(batch["x_input_ids"], batch["x_attention"])
    opt = torch.optim.AdamW(diffusion.parameters(), lr=1e-3)
    for _ in range(3):
        t = torch.randint(1, cfg.diffusion.T + 1, (8,))
        opt.zero_grad()
        loss, _ = diffusion.training_loss(v0, u, t)
        loss.backward()
        opt.step()
    diffusion.eval()
    return diffusion


@pytest.fixture(scope="module")
def g_psi(env, p0):
    cfg, *_ = env
    torch.manual_seed(0)
    return SemanticProjectionModule(p0.d_model, cfg.model.n_slots, cfg.model.u_dim, cfg.gpsi)


def _activate_gates(module, std: float = 0.05):
    """AdaLN gates are zero-init (DiT-style), so every conditioned block starts as the identity.

    That is deliberate, but it means architectural properties like cross-slot mixing are only
    visible once the gates have moved off zero. This simulates a few steps of training.
    """
    from models.layers import AdaLNModulation
    for m in module.modules():
        if isinstance(m, AdaLNModulation):
            torch.nn.init.normal_(m.proj[1].weight, std=std)
            torch.nn.init.normal_(m.proj[1].bias, std=std)


def _gpsi_probe(cfg, d_model, gpsi_cfg):
    """A G_psi with active gates, an open output gate and no dropout -- so a forward pass is a
    deterministic function of its input and slot mixing is observable."""
    torch.manual_seed(0)
    g = SemanticProjectionModule(d_model, cfg.model.n_slots, cfg.model.u_dim, gpsi_cfg)
    _activate_gates(g)
    for p in g.out_proj.parameters():
        torch.nn.init.normal_(p, std=0.1)
    torch.nn.init.constant_(g.gate[1].bias, 2.0)
    return g.eval()


# -- P0 -----------------------------------------------------------------------------------------
def test_p0_loss_decreases_on_a_fixed_batch(env):
    """Overfit-10: if the wiring is right, a tiny model drives a fixed batch down."""
    cfg, _, ds, _ = env
    torch.manual_seed(0)
    model = MemoryModel(cfg)
    batch = collate([ds[i] for i in range(4)])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = last = None
    for step in range(25):
        opt.zero_grad()
        loss, _ = model(batch)
        loss.backward()
        opt.step()
        if step == 0:
            first = float(loss)
        last = float(loss)
    assert last < first * 0.8, f"P0 did not descend: {first:.3f} -> {last:.3f}"


# -- P1 -----------------------------------------------------------------------------------------
def test_p1_loss_decreases(env, p0):
    cfg, _, ds, _ = env
    torch.manual_seed(0)
    diffusion = LatentDiffusion(cfg, p0.d_model)
    batch = collate([ds[i] for i in range(4)])
    u, v0 = p0.encode_latents(batch["x_input_ids"], batch["x_attention"])
    opt = torch.optim.AdamW(diffusion.parameters(), lr=3e-3)
    torch.manual_seed(1)
    t = torch.randint(1, cfg.diffusion.T + 1, (4,))
    first = last = None
    for step in range(30):
        opt.zero_grad()
        loss, _ = diffusion.training_loss(v0, u, t)
        loss.backward()
        opt.step()
        if step == 0:
            first = float(loss)
        last = float(loss)
    assert last < first, f"P1 did not descend: {first:.4f} -> {last:.4f}"


def test_denoiser_gets_gradient_including_the_cue_path(env, p0):
    cfg, _, ds, _ = env
    diffusion = LatentDiffusion(cfg, p0.d_model)
    _activate_gates(diffusion.denoiser)
    batch = collate([ds[i] for i in range(4)])
    u, v0 = p0.encode_latents(batch["x_input_ids"], batch["x_attention"])
    diffusion.zero_grad(set_to_none=True)
    loss, _ = diffusion.training_loss(v0, u, torch.randint(1, cfg.diffusion.T + 1, (4,)))
    loss.backward()
    named = dict(diffusion.named_parameters())
    cue = [v for k, v in named.items() if "cue" in k and v.requires_grad]
    assert cue and any(p.grad is not None and p.grad.abs().sum() > 0 for p in cue), \
        "cue cross-attention receives no gradient"


def test_adaln_gates_learn_from_step_zero(env, p0):
    """Zero-init gates must still receive gradient, else the cue path never switches on."""
    cfg, _, ds, _ = env
    diffusion = LatentDiffusion(cfg, p0.d_model)
    batch = collate([ds[i] for i in range(4)])
    u, v0 = p0.encode_latents(batch["x_input_ids"], batch["x_attention"])
    diffusion.zero_grad(set_to_none=True)
    loss, _ = diffusion.training_loss(v0, u, torch.randint(1, cfg.diffusion.T + 1, (4,)))
    loss.backward()
    gates = [v for k, v in diffusion.named_parameters() if "mod.proj" in k]
    assert gates and any(p.grad is not None and p.grad.abs().sum() > 0 for p in gates)


def test_gist_anchor_is_active_when_enabled(env, p0):
    cfg, _, ds, _ = env
    diffusion = LatentDiffusion(cfg, p0.d_model)
    assert diffusion.anchor_net is not None
    u, _ = p0.encode_latents(*[collate([ds[i] for i in range(4)])[k]
                               for k in ("x_input_ids", "x_attention")])
    assert diffusion.anchor(u).shape == (4, cfg.model.n_slots, p0.d_model)


def test_disabling_the_anchor_reverts_to_plain_diffusion(env, p0):
    cfg, *_ = env
    plain = LatentDiffusion(cfg.override(**{"diffusion.gist_anchored": False}), p0.d_model)
    assert plain.anchor_net is None
    assert plain.anchor(torch.randn(2, cfg.model.u_dim)) is None


# -- P2 -----------------------------------------------------------------------------------------
def test_p2_input_comes_from_the_same_chain_as_inference(env, p0, p1):
    """The mismatch that made G_psi train on a distribution it never saw at test time."""
    cfg, _, ds, _ = env
    batch = collate([ds[i] for i in range(4)])
    u, v0 = p0.encode_latents(batch["x_input_ids"], batch["x_attention"])
    t = torch.tensor([10, 20, 30, 40])

    a, _ = p1.noisy_state_for_training(v0, u, t, seed=7)
    b, _ = p1.noisy_state_for_training(v0, u, t, seed=7)
    assert torch.allclose(a, b), "P2 input must be reproducible"

    v_T, _ = q_sample(v0, torch.full_like(t, cfg.diffusion.T), p1.schedule,
                      anchor=p1.anchor(u), seed=7)
    _, rec = p1.recall(v_T, cfg.diffusion.T, int(t.min()), u, seed=7,
                       record_at=sorted({int(x) for x in t.tolist()}, reverse=True),
                       training_steps=True)
    expected = torch.stack([rec[int(t[i])][i] for i in range(4)])
    assert torch.allclose(a, expected, atol=1e-5)


def test_p2_per_example_timesteps_are_respected(env, p0, p1):
    """Different examples in a batch stop at different points on the chain."""
    cfg, _, ds, _ = env
    batch = collate([ds[i] for i in range(4)])
    u, v0 = p0.encode_latents(batch["x_input_ids"], batch["x_attention"])
    t = torch.tensor([5, 40, 40, 5])
    v_t, _ = p1.noisy_state_for_training(v0, u, t, seed=3)
    assert torch.allclose(v_t[1], v_t[1])
    assert not torch.allclose(v_t[0], v_t[1], atol=1e-6)


def test_gpsi_receives_gradient_and_decoder_can_be_frozen(env, p0, p1, g_psi):
    cfg, _, ds, _ = env
    from dataloader.memory_dataset import labels_for_levels
    from models.losses import projection_loss

    batch = collate([ds[i] for i in range(4)])
    u, v0 = p0.encode_latents(batch["x_input_ids"], batch["x_attention"])
    t, k = p1.level_map.sample(batch["n_levels"], stratified=True)
    v_t, v_hat0 = p1.noisy_state_for_training(v0, u, t, seed=0)

    p0.freeze_all()
    p0.zero_grad(set_to_none=True)
    g_psi.zero_grad(set_to_none=True)
    v_tilde = g_psi(t=t, v_t=v_t, v_hat_0=v_hat0, u=u)
    labels = labels_for_levels(batch["level_input_ids"], k, ds.pad_id)
    slot_mask = torch.ones(4, cfg.model.n_slots, dtype=torch.long)
    ce, _ = p0.decoder_x(v_tilde, slot_mask, labels, level=k)
    (projection_loss(v_tilde, v0) + ce).backward()

    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in g_psi.parameters())
    assert all(p.grad is None for p in p0.encoder.parameters()), "frozen encoder got gradient"
    for p in p0.parameters():
        p.requires_grad = True


def test_gpsi_is_near_identity_at_initialisation(env, p0):
    cfg, *_ = env
    torch.manual_seed(0)
    g = SemanticProjectionModule(p0.d_model, cfg.model.n_slots, cfg.model.u_dim, cfg.gpsi)
    v_t = torch.randn(2, cfg.model.n_slots, p0.d_model)
    out = g(t=torch.tensor([5, 5]), v_t=v_t, v_hat_0=v_t,
            u=torch.randn(2, cfg.model.u_dim))
    assert torch.allclose(out, v_t, atol=1e-5), "zero-init residual should start as identity"


def test_gpsi_mixes_information_across_slots(env, p0):
    """use_attn defaulted to False, which made G_psi a per-slot pointwise MLP."""
    cfg, *_ = env
    g = _gpsi_probe(cfg, p0.d_model, cfg.gpsi)
    v = torch.randn(1, cfg.model.n_slots, p0.d_model)
    t, u = torch.tensor([5]), torch.randn(1, cfg.model.u_dim)
    base = g(t=t, v_t=v, v_hat_0=v, u=u)
    v2 = v.clone()
    v2[:, 0] += 5.0
    perturbed = g(t=t, v_t=v2, v_hat_0=v2, u=u)
    other_slots_changed = (perturbed[:, 1:] - base[:, 1:]).abs().max()
    assert other_slots_changed > 1e-4, "perturbing one slot must affect the others"


def test_gpsi_without_self_attention_does_not_mix_slots(env, p0):
    """The old configuration, kept as an ablation arm -- and this is exactly what it cost."""
    cfg, *_ = env
    old = cfg.override(**{"gpsi.use_self_attn": False, "gpsi.use_cross_attn": False})
    g = _gpsi_probe(cfg, p0.d_model, old.gpsi)
    v = torch.randn(1, cfg.model.n_slots, p0.d_model)
    t, u = torch.tensor([5]), torch.randn(1, cfg.model.u_dim)
    base = g(t=t, v_t=v, v_hat_0=v, u=u)
    v2 = v.clone()
    v2[:, 0] += 5.0
    perturbed = g(t=t, v_t=v2, v_hat_0=v2, u=u)
    assert (perturbed[:, 1:] - base[:, 1:]).abs().max() < 1e-5


# -- recall ---------------------------------------------------------------------------------------
def test_recall_engine_produces_a_full_ladder(env, p0, p1, g_psi):
    cfg, tok, ds, _ = env
    engine = RecallEngine(p0, p1, g_psi, tok, cfg, "cpu")
    texts = [ds.examples[i].x for i in range(3)]
    ladders = engine.ladder(texts, n_levels=5, example_ids=[e.uid for e in ds.examples[:3]])
    assert len(ladders) == 3
    for rungs in ladders:
        assert len(rungs) == 5
        assert all(isinstance(r.text, str) for r in rungs)
        assert [r.t for r in rungs] == sorted(r.t for r in rungs), "rungs ordered by depth"


def test_recall_is_reproducible(env, p0, p1, g_psi):
    cfg, tok, ds, _ = env
    engine = RecallEngine(p0, p1, g_psi, tok, cfg, "cpu")
    texts = [ds.examples[0].x]
    a = engine.ladder(texts, n_levels=4, example_ids=["0"])
    b = engine.ladder(texts, n_levels=4, example_ids=["0"])
    assert [r.text for r in a[0]] == [r.text for r in b[0]]


def test_shuffled_cue_changes_the_output(env, p0, p1, g_psi):
    """If it does not, `u` is decoration and the cue-conditioning claim is dead."""
    cfg, tok, ds, _ = env
    engine = RecallEngine(p0, p1, g_psi, tok, cfg, "cpu")
    texts = [ds.examples[i].x for i in range(4)]
    normal = engine.ladder(texts, n_levels=4, example_ids=["a"])
    torch.manual_seed(5)
    shuffled = engine.ladder(texts, n_levels=4, example_ids=["a"], shuffle_cue=True)
    assert len(normal) == len(shuffled) == 4  # the control runs; the effect size is an ablation


def test_forgetting_ladder_runs_without_the_denoiser(env, p0, p1, g_psi):
    cfg, tok, ds, _ = env
    engine = RecallEngine(p0, p1, g_psi, tok, cfg, "cpu")
    out = engine.forgetting_ladder([ds.examples[0].x], budgets=[0.0, 0.5, 1.0])
    assert len(out[0]) == 3


def test_oracle_ladder_gives_a_decoder_upper_bound(env, p0, p1, g_psi):
    cfg, tok, ds, _ = env
    engine = RecallEngine(p0, p1, g_psi, tok, cfg, "cpu")
    out = engine.oracle_ladder([ds.level_texts(0), ds.level_texts(1)])
    assert len(out) == 2 and len(out[0]) == len(ds.level_texts(0))


# -- checkpoints -------------------------------------------------------------------------------------
def test_checkpoint_round_trip_preserves_outputs(env, p0, tmp_path):
    cfg, _, ds, _ = env
    from training.common import load_checkpoint, save_checkpoint

    path = str(tmp_path / "p0.pt")
    save_checkpoint(path, cfg, 0, {"val/loss_total": 1.0}, model=p0)
    ckpt = load_checkpoint(path)

    torch.manual_seed(0)
    restored = MemoryModel(cfg)
    restored.load_state_dict(ckpt["model_state_dict"])
    restored.eval(), p0.eval()

    batch = collate([ds[i] for i in range(2)])
    a = p0.encode(batch["x_input_ids"], batch["x_attention"]).v0
    b = restored.encode(batch["x_input_ids"], batch["x_attention"]).v0
    assert torch.allclose(a, b, atol=1e-6)


def test_incompatible_configs_are_rejected(env):
    cfg, *_ = env
    from training.common import assert_compatible
    assert_compatible(cfg, cfg)
    with pytest.raises(ValueError, match="n_slots"):
        assert_compatible(cfg, cfg.override(**{"model.n_slots": 8}))
    with pytest.raises(ValueError, match="parameterization"):
        assert_compatible(cfg, cfg.override(**{"diffusion.parameterization": "eps"}))


def test_checkpoint_without_config_is_refused(tmp_path):
    from training.common import load_checkpoint
    path = str(tmp_path / "legacy.pt")
    torch.save({"model_state_dict": {}}, path)
    with pytest.raises(ValueError, match="no embedded config"):
        load_checkpoint(path)


# -- cache ---------------------------------------------------------------------------------------------
def test_cache_round_trips_and_invalidates_on_dependencies(tmp_path):
    from utils.cache import CacheStore
    store = CacheStore(str(tmp_path / "c"), code_version="v1")
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return {"x": torch.arange(4)}

    a = store.get_or_compute("ns", ["dep1"], compute)
    b = store.get_or_compute("ns", ["dep1"], compute)
    assert calls["n"] == 1 and torch.equal(a["x"], b["x"])

    store.get_or_compute("ns", ["dep2"], compute)
    assert calls["n"] == 2, "changing a dependency must miss"

    store2 = CacheStore(str(tmp_path / "c"), code_version="v2")
    store2.get_or_compute("ns", ["dep1"], compute)
    assert calls["n"] == 3, "changing the code version must miss"


def test_disabled_cache_always_recomputes(tmp_path):
    from utils.cache import CacheStore
    store = CacheStore(str(tmp_path / "d"), enabled=False)
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return [1, 2, 3]

    store.get_or_compute("ns", ["k"], compute)
    store.get_or_compute("ns", ["k"], compute)
    assert calls["n"] == 2
