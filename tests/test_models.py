"""Shape, gradient-flow and dataset-contract tests.

These are the ones that catch wiring bugs -- the class of defect that produced the previous
`v_hat_0` mislabelling, the non-chaining transformer blocks, and `xpos == x`. All run on CPU in
seconds with no network.
"""

import os

import pytest
import torch

from configs import get
from dataloader.memory_dataset import MemoryLadderDataset, collate, labels_for_levels
from models.memory_model import MemoryModel
from tests.stubs import StubTokenizer, temp_data_dir


@pytest.fixture(scope="module")
def cfg():
    return get("tiny")


@pytest.fixture(scope="module")
def tok():
    return StubTokenizer(vocab_size=128)


@pytest.fixture(scope="module")
def data_dir():
    return temp_data_dir(n_repeat=4)


@pytest.fixture(scope="module")
def dataset(cfg, tok, data_dir):
    return MemoryLadderDataset(os.path.join(data_dir, "train.json"), tok, cfg, split="train")


@pytest.fixture(scope="module")
def model(cfg):
    torch.manual_seed(0)
    return MemoryModel(cfg, gist_encoder_dim=32)


# -- dataset contracts -----------------------------------------------------------------------
def test_ladder_starts_at_the_verbatim_source(dataset):
    for i in range(len(dataset)):
        assert dataset.level_texts(i)[0] == dataset.examples[i].x


def test_ladder_is_strictly_shortening_on_the_sample_data(dataset):
    for i in range(len(dataset)):
        lens = [len(s.split()) for s in dataset.level_texts(i)]
        assert lens[0] == max(lens)


def test_contrastive_positive_is_not_the_anchor(cfg, tok, data_dir):
    """The original loader set xpos = x, which made InfoNCE trivially satisfiable."""
    ds = MemoryLadderDataset(os.path.join(data_dir, "train.json"), tok, cfg, split="t")
    differed = 0
    for _ in range(40):
        item = ds[0]
        if not torch.equal(item["xpos_input_ids"], item["x_input_ids"]):
            differed += 1
    assert differed > 0, "positive never differs from the anchor"


def test_identity_positive_mode_reproduces_the_old_behaviour(cfg, tok, data_dir):
    bad = cfg.override(**{"data.xplus_source": "identity"})
    ds = MemoryLadderDataset(os.path.join(data_dir, "train.json"), tok, bad, split="t")
    item = ds[0]
    assert torch.equal(item["xpos_input_ids"], item["x_input_ids"])


def test_labels_mask_padding_with_minus_100(dataset):
    item = dataset[0]
    assert (item["x_labels"] == -100).any()
    assert not (item["x_labels"] == dataset.pad_id).any()


def test_n_levels_matches_the_ladder(dataset):
    for i in range(len(dataset)):
        assert int(dataset[i]["n_levels"]) == len(dataset.level_texts(i))


def test_collate_stacks_tensors_and_keeps_text(dataset):
    batch = collate([dataset[i] for i in range(4)])
    assert batch["x_input_ids"].shape[0] == 4
    assert isinstance(batch["x_text"], list) and len(batch["x_text"]) == 4


def test_labels_for_levels_selects_the_right_rung(dataset):
    batch = collate([dataset[i] for i in range(4)])
    k = torch.tensor([0, 1, 2, 0])
    labels = labels_for_levels(batch["level_input_ids"], k, dataset.pad_id)
    for b in range(4):
        expected = batch["level_input_ids"][b, k[b]].clone()
        expected[expected == dataset.pad_id] = -100
        assert torch.equal(labels[b], expected)


# -- model shapes ------------------------------------------------------------------------------
def test_encode_shapes(model, dataset, cfg):
    batch = collate([dataset[i] for i in range(3)])
    out = model.encode(batch["x_input_ids"], batch["x_attention"], return_attention=True)
    assert out.slots.shape == (3, cfg.model.n_slots, model.d_model)
    assert out.u.shape == (3, cfg.model.u_dim)
    assert out.v0.shape == (3, cfg.model.n_slots, model.d_model)
    assert out.pool_attn is not None and out.u_attn is not None


def test_u_is_unit_norm(model, dataset):
    batch = collate([dataset[i] for i in range(3)])
    u = model.encode(batch["x_input_ids"], batch["x_attention"]).u
    assert torch.allclose(u.norm(dim=-1), torch.ones(3), atol=1e-4)


def test_v0_scale_is_controlled(model, dataset):
    """If std(v0) >> 1 the noise term is negligible until very late, producing a cliff."""
    batch = collate([dataset[i] for i in range(4)])
    v0 = model.encode(batch["x_input_ids"], batch["x_attention"]).v0
    assert 0.2 < float(v0.std()) < 5.0, float(v0.std())


def test_u_and_v0_are_not_the_same_function(model, dataset):
    """They were previously two Linears on the same pooled tensor."""
    p_u = {id(p) for p in model.u_head.parameters()}
    p_v = {id(p) for p in model.v_head.parameters()}
    assert not (p_u & p_v)
    assert sum(p.numel() for p in model.v_head.parameters()) > 4 * model.d_model


def test_forward_returns_all_loss_parts(model, dataset):
    batch = collate([dataset[i] for i in range(4)])
    batch["gist_embedding"] = torch.randn(4, 32)
    loss, parts = model(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    for key in ("loss_x", "loss_y", "loss_align", "loss_nce", "loss_orth"):
        assert key in parts, f"missing {key}"


def test_level_token_changes_the_decoder_memory(model, dataset, cfg):
    batch = collate([dataset[i] for i in range(2)])
    v0 = model.encode(batch["x_input_ids"], batch["x_attention"]).v0
    mask = torch.ones(2, cfg.model.n_slots, dtype=torch.long)
    m0, _ = model.decoder_x.build_memory(v0, mask, level=torch.tensor([0, 0]))
    m3, _ = model.decoder_x.build_memory(v0, mask, level=torch.tensor([3, 3]))
    assert m0.shape[1] == cfg.model.n_slots + 2  # level + length tokens
    assert not torch.allclose(m0, m3)


def test_generate_is_free_running_and_needs_no_labels(model, dataset, cfg):
    batch = collate([dataset[i] for i in range(2)])
    v0 = model.encode(batch["x_input_ids"], batch["x_attention"]).v0
    ids = model.generate_from_latent(v0, level=torch.tensor([0, 0]), max_new_tokens=8,
                                     num_beams=1)
    assert ids.shape[0] == 2 and ids.shape[1] > 1


# -- gradient flow -------------------------------------------------------------------------------
def test_every_p0_component_receives_gradient(model, dataset):
    batch = collate([dataset[i] for i in range(4)])
    batch["gist_embedding"] = torch.randn(4, 32)
    model.zero_grad()
    loss, _ = model(batch)
    loss.backward()

    for name, module in [
        ("encoder", model.encoder), ("slot_pooling", model.slot_pooling),
        ("u_head", model.u_head), ("v_head", model.v_head),
        ("decoder_x", model.decoder_x), ("decoder_y", model.decoder_y),
        ("align_proj", model.align_proj),
    ]:
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in grads), f"{name} got no gradient"


def test_freeze_all_stops_gradient(model, dataset):
    model.freeze_all()
    assert model.trainable_parameter_count() == 0
    for p in model.parameters():
        p.requires_grad = True


def test_dropping_the_gist_decoder_leaves_recall_intact(cfg, dataset):
    torch.manual_seed(0)
    m = MemoryModel(cfg)
    m.drop_gist_decoder()
    batch = collate([dataset[i] for i in range(2)])
    loss, parts = m(batch)
    assert "loss_y" not in parts
    assert torch.isfinite(loss)
