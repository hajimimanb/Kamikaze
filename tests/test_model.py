# -*- coding: utf-8 -*-
"""Unit tests for MultiHeadRiichiNet (shapes, masking, gradients, params)."""
import torch

from model.features import feature_channels
from model.model import MultiHeadRiichiNet, get_device


def test_stage1_forward_shapes_and_mask():
    c = feature_channels("simple")
    net = MultiHeadRiichiNet(c, channels=32, n_blocks=2, binary_heads=(),
                             include_value=False)
    x = torch.randn(4, c, 34, 1)
    mask = torch.zeros(4, 34, dtype=torch.bool)
    mask[:, 7] = True  # only kind 7 legal
    out = net(x, masks={"discard": mask})
    assert out["discard"].shape == (4, 34)
    assert set(out.keys()) == {"discard"}
    assert (out["discard"].argmax(dim=1) == 7).all()
    probs = net.discard_probs(x, masks={"discard": mask})
    assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)
    assert (probs[:, 7] == 1.0).all()


def test_stage2_all_heads_and_candidates():
    c = feature_channels("full")
    net = MultiHeadRiichiNet(c, channels=32, n_blocks=2)
    x = torch.randn(4, c, 34, 1)
    cands = {h: torch.zeros(4, 34) for h in ("riichi", "chow", "pon", "kan")}
    for h in cands:
        cands[h][:, 3] = 1.0
    out = net(x, candidates=cands)
    assert set(out.keys()) == {"discard", "riichi", "chow", "pon", "kan", "value"}
    for h in ("riichi", "chow", "pon", "kan"):
        assert out[h].shape == (4, 2)
    assert out["value"].shape == (4,)
    # missing candidate -> zeros, no crash
    out2 = net(x)
    assert out2["riichi"].shape == (4, 2)


def test_binary_availability_masking():
    """Unavailable binary heads get both logits masked to -inf."""
    c = feature_channels("full")
    net = MultiHeadRiichiNet(c, channels=32, n_blocks=2)
    x = torch.randn(4, c, 34, 1)
    avail = torch.tensor([True, False, True, True])
    out = net(x, masks={"riichi": avail})
    assert (out["riichi"][~avail] == float("-inf")).all()
    assert torch.isfinite(out["riichi"][avail]).all()


def test_gradients_flow():
    c = feature_channels("simple")
    net = MultiHeadRiichiNet(c, channels=32, n_blocks=2, binary_heads=(),
                             include_value=False)
    x = torch.randn(2, c, 34, 1)
    loss = net(x)["discard"].sum()
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)


def test_param_counts():
    c = feature_channels("simple")
    small = MultiHeadRiichiNet(c, channels=32, n_blocks=2, binary_heads=(),
                               include_value=False)
    assert small.num_parameters() < 2_000_000  # discard head FCs dominate
    big = MultiHeadRiichiNet(c, channels=256, n_blocks=50)
    # trunk+discard ~21.7M (phase0_plan.md); +4 binary heads + value ~26.5M
    assert 20e6 < big.num_parameters() < 30e6


def test_model_profile():
    """Profile prints params (~21M trunk), FLOPs, CPU forward timing."""
    from model.model import model_profile
    c = feature_channels("full")
    net = MultiHeadRiichiNet(c, channels=256, n_blocks=50)
    prof = model_profile(net, device="cpu", batch=1, warmup=1, iters=3)
    assert 20e6 < prof["params"] < 30e6
    assert 1.0 < prof["flops_fwd_G"] < 3.0   # ~1.37G forward
    assert prof["flops_fwd_bwd_G"] > prof["flops_fwd_G"]
    assert prof["fwd_ms"] > 0
    # FLOPs scale linearly with batch
    prof2 = model_profile(net, device="cpu", batch=2, warmup=1, iters=1)
    assert abs(prof2["flops_fwd_G"] - 2 * prof["flops_fwd_G"]) < 1e-6


def test_get_device_fallback():
    dev = get_device("cuda")
    assert dev.type in ("cuda", "cpu")
    assert get_device("cpu").type == "cpu"
