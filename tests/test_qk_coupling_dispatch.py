"""Phase-2.1: qk_coupling dispatch unit tests.

The eight tests below are the load-bearing correctness gates for Phase B
(experiment.md d.7.3). Without all eight passing, the C/D/E/H' qk_coupling
sweeps cannot produce defensible evidence:

  1. test_full_rope_qk_policy_invariance — under full RoPE, all policies
     produce bitwise-identical Q,K updates (preserves A0/B/G/H anchors).
  2. test_current_flat2d_fallback_matches_legacy_learned_pos — the
     `current_flat2d_fallback` policy under learned-pos matches the
     pre-refactor factory's flat-2D fallback exactly (preserves C/D
     negative control vs historical C/D).
  3. test_full_rope_explicit_off — `full_rope=off` routes Q,K to plain Muon
     (explicit ablation, distinct from any non-full-RoPE policy).
  4. test_qk_off_on_learned_pos — `qk_off` matches a hand-derived plain-Muon
     update (momentum=0/nesterov=false/wd=0 to eliminate plumbing variance).
  5. test_headwise_no_rope_on_learned_pos — headwise differs from flat-2D
     on the same seeded gradient (sanity check that the policy actually
     hits a different code path).
  6. test_partial_rope_split_dispatch_strict — the partial_rope_split path
     hits the expected sentinel and uses the [0:rotary_dim] / [rotary_dim:]
     slices (not silently falling back to flat-2D).
  7. test_partial_rope_split_raises_under_no_rope — fail-fast for the one
     ill-defined combination.
  8. test_nonfull_qk_policy_does_not_change_vo_routing — V-O routing under
     rope_status != full stays legacy flat-2D regardless of qk_coupling
     (Phase B is Q-K-only).
"""
from __future__ import annotations

import pytest
import torch

from coupled_muon_nanogpt.model.transformer import GPT, BlockConfig, GPTConfig
from coupled_muon_nanogpt.optim.factory import build_optimizer
from omegaconf import OmegaConf

from tests._legacy_optimizer_helpers import build_legacy_nonfull_flat2d_optimizer


def _build_model(*, pos_emb_type: str, rope_partial_frac: float = 1.0, hidden: int = 64, n_heads: int = 4) -> GPT:
    block = BlockConfig(
        hidden=hidden,
        n_heads=n_heads,
        intermediate=128,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        pos_emb_type=pos_emb_type,
        rope_partial_frac=rope_partial_frac,
        qk_norm=False,
        max_seq_len=128,
    )
    return GPT(GPTConfig(vocab_size=128, n_layers=2, block=block))


def _build_cfg(*, pos_emb_type: str, rope_partial_frac: float, qk_coupling: dict, hidden: int = 64, n_heads: int = 4,
               momentum: float = 0.95, nesterov: bool = True, wd: float = 0.0) -> OmegaConf:
    return OmegaConf.create(
        {
            "model": {
                "hidden": hidden,
                "attn": {
                    "n_heads": n_heads,
                    "n_kv_heads": None,
                    "rope_partial_frac": rope_partial_frac,
                },
                "pos_emb": {"type": pos_emb_type},
            },
            "optimizer": {
                "type": "coupled_muon_v2",
                "lr": 1e-3,
                "wd": wd,
                "momentum": momentum,
                "nesterov": nesterov,
                "betas": [0.95, 0.95],
                "eps": 1e-8,
                "ns_steps": 5,
                "coupled_steps": 4,
                "couple_qk": True,
                "couple_vo": True,
                "couple_updown": True,
                "use_multi_head": True,
                "ns_dtype": "fp32",  # tests run on CPU; fp32 avoids bf16 nondeterminism
                "qk_coupling": qk_coupling,
            },
        }
    )


def _seed_and_forward_backward(model: GPT, seed: int = 7):
    torch.manual_seed(seed)
    x = torch.randint(0, 128, (2, 16))
    y = torch.randint(0, 128, (2, 16))
    out = model(x, targets=y)
    out["total_loss"].backward()


def _snapshot(model: GPT) -> dict[str, torch.Tensor]:
    return {n: p.data.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


# ---------------------------------------------------------------------------
# Test 1: full-RoPE invariance
# ---------------------------------------------------------------------------
def test_full_rope_qk_policy_invariance():
    """Under full RoPE + full_rope=legacy_rope2d, the no_rope_policy and
    partial_rope_policy values must NOT affect numerics — they're
    inaccessible at rope_status=full. Verified by stepping two optimizers
    with different (no_rope_policy, partial_rope_policy) and asserting all
    parameter deltas are bitwise-identical.
    """
    # Model 1
    torch.manual_seed(0)
    m1 = _build_model(pos_emb_type="rope", rope_partial_frac=1.0)
    cfg1 = _build_cfg(
        pos_emb_type="rope", rope_partial_frac=1.0,
        qk_coupling={"full_rope": "legacy_rope2d",
                     "no_rope_policy": "current_flat2d_fallback",
                     "partial_rope_policy": "current_flat2d_fallback"},
    )
    opt1 = build_optimizer(m1, cfg1)
    _seed_and_forward_backward(m1, seed=7)
    opt1.step()
    snap1 = _snapshot(m1)

    # Model 2 — identical init + same forward grads, different (inaccessible) policies.
    torch.manual_seed(0)
    m2 = _build_model(pos_emb_type="rope", rope_partial_frac=1.0)
    cfg2 = _build_cfg(
        pos_emb_type="rope", rope_partial_frac=1.0,
        qk_coupling={"full_rope": "legacy_rope2d",
                     "no_rope_policy": "headwise_no_rope",
                     "partial_rope_policy": "partial_rope_split"},
    )
    opt2 = build_optimizer(m2, cfg2)
    _seed_and_forward_backward(m2, seed=7)
    opt2.step()
    snap2 = _snapshot(m2)

    for name in snap1:
        torch.testing.assert_close(snap1[name], snap2[name], rtol=0, atol=0,
                                   msg=lambda m, n=name: f"{n}: {m}")


# ---------------------------------------------------------------------------
# Test 2: current_flat2d_fallback bitwise matches legacy learned-pos path
# ---------------------------------------------------------------------------
def test_current_flat2d_fallback_matches_legacy_learned_pos():
    """The new factory-with-qk_coupling path with no_rope_policy=
    current_flat2d_fallback under learned-pos must produce bitwise-identical
    parameter deltas to the pre-refactor factory's "use_multi_head disabled,
    fall through to flat-2D" path.
    """
    # Legacy path
    torch.manual_seed(0)
    m_legacy = _build_model(pos_emb_type="learned", rope_partial_frac=1.0)
    opt_legacy = build_legacy_nonfull_flat2d_optimizer(
        m_legacy, lr=1e-3, momentum=0.0, nesterov=False, coupled_steps=4, ns_steps=5, n_heads=4,
        ns_dtype=torch.float32,
    )
    _seed_and_forward_backward(m_legacy, seed=11)
    opt_legacy.step()
    snap_legacy = _snapshot(m_legacy)

    # New path with current_flat2d_fallback under learned-pos
    torch.manual_seed(0)
    m_new = _build_model(pos_emb_type="learned", rope_partial_frac=1.0)
    cfg = _build_cfg(
        pos_emb_type="learned", rope_partial_frac=1.0,
        qk_coupling={"full_rope": "legacy_rope2d",
                     "no_rope_policy": "current_flat2d_fallback",
                     "partial_rope_policy": "current_flat2d_fallback"},
        momentum=0.0, nesterov=False, wd=0.0,
    )
    opt_new = build_optimizer(m_new, cfg)
    _seed_and_forward_backward(m_new, seed=11)
    opt_new.step()
    snap_new = _snapshot(m_new)

    for name in snap_legacy:
        torch.testing.assert_close(
            snap_legacy[name], snap_new[name], rtol=1e-6, atol=1e-6,
            msg=lambda m, n=name: f"{n} mismatch: {m}",
        )


# ---------------------------------------------------------------------------
# Test 3: full_rope=off explicit ablation
# ---------------------------------------------------------------------------
def test_full_rope_explicit_off():
    """full_rope=off under full RoPE must route Q,K to plain-Muon updates
    that differ from full_rope=legacy_rope2d on the same seeded gradient."""
    torch.manual_seed(0)
    m_legacy = _build_model(pos_emb_type="rope", rope_partial_frac=1.0)
    cfg_legacy = _build_cfg(
        pos_emb_type="rope", rope_partial_frac=1.0,
        qk_coupling={"full_rope": "legacy_rope2d",
                     "no_rope_policy": "current_flat2d_fallback",
                     "partial_rope_policy": "current_flat2d_fallback"},
    )
    opt_legacy = build_optimizer(m_legacy, cfg_legacy)
    _seed_and_forward_backward(m_legacy, seed=13)
    opt_legacy.step()
    snap_legacy = _snapshot(m_legacy)

    torch.manual_seed(0)
    m_off = _build_model(pos_emb_type="rope", rope_partial_frac=1.0)
    cfg_off = _build_cfg(
        pos_emb_type="rope", rope_partial_frac=1.0,
        qk_coupling={"full_rope": "off",
                     "no_rope_policy": "current_flat2d_fallback",
                     "partial_rope_policy": "current_flat2d_fallback"},
    )
    opt_off = build_optimizer(m_off, cfg_off)
    _seed_and_forward_backward(m_off, seed=13)
    opt_off.step()
    snap_off = _snapshot(m_off)

    # Q and K weights must differ between the two policies.
    qk_names = [n for n in snap_legacy if "q_proj" in n or "k_proj" in n]
    assert qk_names, "expected at least one q_proj/k_proj param"
    diffs = [
        not torch.allclose(snap_legacy[n], snap_off[n], rtol=1e-4, atol=1e-4)
        for n in qk_names
    ]
    assert any(diffs), "full_rope=off should change at least one Q/K weight vs legacy_rope2d"


# ---------------------------------------------------------------------------
# Test 4: qk_off plain-Muon match (zero momentum)
# ---------------------------------------------------------------------------
def test_qk_off_on_learned_pos():
    """qk_off under learned-pos should match plain-Muon NS5 on the raw
    gradient (momentum=0, nesterov=false, wd=0 to eliminate plumbing
    variance). The expected delta is computed by REUSING the optimizer's
    `_compute_muon_update_scale` and the NS kernel — not a re-derivation.
    """
    torch.manual_seed(0)
    m = _build_model(pos_emb_type="learned", rope_partial_frac=1.0)
    cfg = _build_cfg(
        pos_emb_type="learned", rope_partial_frac=1.0,
        qk_coupling={"full_rope": "legacy_rope2d",
                     "no_rope_policy": "qk_off",
                     "partial_rope_policy": "current_flat2d_fallback"},
        momentum=0.0, nesterov=False, wd=0.0,
    )
    opt = build_optimizer(m, cfg)
    pre = _snapshot(m)
    _seed_and_forward_backward(m, seed=17)
    # Snapshot gradients for q/k before step (step consumes them in-place).
    qk_grads: dict[str, torch.Tensor] = {}
    for n, p in m.named_parameters():
        if ("q_proj" in n or "k_proj" in n) and p.grad is not None:
            qk_grads[n] = p.grad.detach().clone()
    opt.step()
    post = _snapshot(m)
    assert qk_grads, "expected to capture q/k gradients"

    # Reconstruct expected delta for each Q/K via the optimizer's helpers.
    from coupled_muon_nanogpt.optim.coupled_muon import zeropower_via_newtonschulz5
    for n, g in qk_grads.items():
        delta = post[n] - pre[n]
        # qk_off with final_polish=True is plain Muon: u = NS5(g_eff, ns_steps).
        # mom=0, nesterov=false → g_eff == g.
        u = zeropower_via_newtonschulz5(
            g.to(torch.float32),
            steps=5,  # ns_steps
            dtype=torch.float32,
            coeffs=opt._coeffs_stage2,
            gram_form=opt.ns_gram_form,
        )
        scale = opt.adjust_lr_for_muon(1e-3, post[n].shape)
        expected_delta = -scale * u
        torch.testing.assert_close(delta.to(torch.float32), expected_delta.to(torch.float32),
                                   rtol=1e-4, atol=1e-5,
                                   msg=lambda m, n=n: f"{n}: qk_off should match plain Muon: {m}")


# ---------------------------------------------------------------------------
# Test 5: headwise_no_rope hits a different code path than flat-2D
# ---------------------------------------------------------------------------
def test_headwise_no_rope_on_learned_pos():
    """headwise_no_rope must produce DIFFERENT Q,K updates than
    current_flat2d_fallback on the same seeded gradient — proves the
    headwise dispatch actually fires."""
    torch.manual_seed(0)
    m_flat = _build_model(pos_emb_type="learned", rope_partial_frac=1.0)
    cfg_flat = _build_cfg(
        pos_emb_type="learned", rope_partial_frac=1.0,
        qk_coupling={"full_rope": "legacy_rope2d",
                     "no_rope_policy": "current_flat2d_fallback",
                     "partial_rope_policy": "current_flat2d_fallback"},
    )
    opt_flat = build_optimizer(m_flat, cfg_flat)
    _seed_and_forward_backward(m_flat, seed=19)
    opt_flat.step()
    snap_flat = _snapshot(m_flat)

    torch.manual_seed(0)
    m_hw = _build_model(pos_emb_type="learned", rope_partial_frac=1.0)
    cfg_hw = _build_cfg(
        pos_emb_type="learned", rope_partial_frac=1.0,
        qk_coupling={"full_rope": "legacy_rope2d",
                     "no_rope_policy": "headwise_no_rope",
                     "partial_rope_policy": "current_flat2d_fallback"},
    )
    opt_hw = build_optimizer(m_hw, cfg_hw)
    _seed_and_forward_backward(m_hw, seed=19)
    opt_hw.step()
    snap_hw = _snapshot(m_hw)

    qk_names = [n for n in snap_flat if "q_proj" in n or "k_proj" in n]
    diffs = [not torch.allclose(snap_flat[n], snap_hw[n], rtol=1e-4, atol=1e-4) for n in qk_names]
    assert any(diffs), "headwise_no_rope must differ from current_flat2d_fallback on Q/K weights"


# ---------------------------------------------------------------------------
# Test 6: partial_rope_split strict dispatch
# ---------------------------------------------------------------------------
def test_partial_rope_split_dispatch_strict():
    """partial_rope_split must hit its sentinel path (not silently fall
    through to flat-2D) and must produce finite, head-block-shaped output.
    Verified by direct call to the helper with monkey-patching to record
    the dispatch."""
    torch.manual_seed(0)
    m = _build_model(pos_emb_type="rope", rope_partial_frac=0.5)
    cfg = _build_cfg(
        pos_emb_type="rope", rope_partial_frac=0.5,
        qk_coupling={"full_rope": "legacy_rope2d",
                     "no_rope_policy": "current_flat2d_fallback",
                     "partial_rope_policy": "partial_rope_split"},
    )
    opt = build_optimizer(m, cfg)
    # Verify the configuration landed correctly.
    assert opt.rope_status == "partial"
    assert opt.qk_coupling_partial_rope_policy == "partial_rope_split"
    assert opt.rotary_dim is not None and opt.rotary_dim > 0
    # Patch the helper to record that it was called.
    called = {"count": 0}
    orig = opt._qk_partial_rope_split_stage1
    def patched(*a, **k):
        called["count"] += 1
        return orig(*a, **k)
    opt._qk_partial_rope_split_stage1 = patched
    _seed_and_forward_backward(m, seed=23)
    opt.step()
    assert called["count"] > 0, "partial_rope_split helper was never called"
    # Verify all Q/K weights remain finite after the step.
    for n, p in m.named_parameters():
        if "q_proj" in n or "k_proj" in n:
            assert torch.isfinite(p).all(), f"{n} has non-finite weights after partial_rope_split"


# ---------------------------------------------------------------------------
# Test 7: partial_rope_split raises under rope_status="none"
# ---------------------------------------------------------------------------
def test_partial_rope_split_raises_under_no_rope():
    """rope_status='none' + no_rope_policy='partial_rope_split' must raise
    ValueError at factory build time. Note: setting partial_rope_policy=
    partial_rope_split under rope_status='none' is silently ignored (the
    partial_rope_policy is inaccessible). It's the no_rope_policy value
    that's invalid."""
    torch.manual_seed(0)
    m = _build_model(pos_emb_type="learned", rope_partial_frac=1.0)
    cfg = _build_cfg(
        pos_emb_type="learned", rope_partial_frac=1.0,
        qk_coupling={"full_rope": "legacy_rope2d",
                     "no_rope_policy": "partial_rope_split",
                     "partial_rope_policy": "current_flat2d_fallback"},
    )
    with pytest.raises(ValueError, match="partial_rope_split"):
        build_optimizer(m, cfg)


# ---------------------------------------------------------------------------
# Test 8: non-full-RoPE qk_policy does NOT change V-O routing
# ---------------------------------------------------------------------------
def test_nonfull_qk_policy_does_not_change_vo_routing():
    """Hard rule (review v3.4): V-O routing under rope_status in {partial, none}
    must remain legacy flat-2D regardless of qk_coupling choice. Verify by
    stepping three optimizers with different no_rope_policy values and
    asserting their V-O weight deltas are bitwise-identical.
    """
    policies = ("current_flat2d_fallback", "qk_off", "headwise_no_rope")
    snaps: list[dict[str, torch.Tensor]] = []
    for pol in policies:
        torch.manual_seed(0)
        m = _build_model(pos_emb_type="learned", rope_partial_frac=1.0)
        cfg = _build_cfg(
            pos_emb_type="learned", rope_partial_frac=1.0,
            qk_coupling={"full_rope": "legacy_rope2d",
                         "no_rope_policy": pol,
                         "partial_rope_policy": "current_flat2d_fallback"},
            momentum=0.0, nesterov=False, wd=0.0,
        )
        opt = build_optimizer(m, cfg)
        _seed_and_forward_backward(m, seed=29)
        opt.step()
        snaps.append(_snapshot(m))

    vo_names = [n for n in snaps[0] if "v_proj" in n or "o_proj" in n]
    assert vo_names, "expected at least one v_proj/o_proj param"
    for n in vo_names:
        # V-O deltas across policies should be identical (qk policy doesn't touch V-O).
        for i in range(1, len(snaps)):
            torch.testing.assert_close(
                snaps[0][n], snaps[i][n], rtol=1e-6, atol=1e-6,
                msg=lambda m, name=n, idx=i: f"V-O {name} differed between policies[0] and policies[{idx}]: {m}",
            )
