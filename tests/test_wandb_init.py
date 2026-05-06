"""Wandb integration policy tests.

Monkeypatches the `wandb` module so the tests never hit the network and don't
require a wandb account. Validates: identity (project/group/job_type/name/tags),
define_metric x-axis registration, log calls drop the `step=` kwarg whenever
`tokens` is in the payload, mode resolution (cfg → env → online), and the
sweep-override path for `cfg.run.wandb_group` interpolation.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Any

import pytest
from omegaconf import OmegaConf


@pytest.fixture
def fake_wandb(monkeypatch):
    """Replace `wandb` in sys.modules with a recording mock."""
    init_calls: list[dict[str, Any]] = []
    define_calls: list[dict[str, Any]] = []
    log_calls: list[tuple[dict[str, Any], dict[str, Any]]] = []
    finish_calls: list[bool] = []

    class _Settings:
        def __init__(self, **kw):
            self.kw = kw

    class _Histogram:
        def __init__(self, values):
            self.values = list(values)

    class _Run:
        def __init__(self):
            self.summary: dict[str, Any] = {}

    def _init(**kwargs):
        init_calls.append(kwargs)
        return _Run()

    def _define_metric(name, step_metric=None, **kw):
        define_calls.append({"name": name, "step_metric": step_metric, **kw})

    def _log(payload, **kwargs):
        log_calls.append((dict(payload), dict(kwargs)))

    def _finish():
        finish_calls.append(True)

    fake = SimpleNamespace(
        init=_init,
        define_metric=_define_metric,
        log=_log,
        finish=_finish,
        Settings=_Settings,
        Histogram=_Histogram,
    )

    monkeypatch.setitem(sys.modules, "wandb", fake)
    return SimpleNamespace(
        module=fake,
        init_calls=init_calls,
        define_calls=define_calls,
        log_calls=log_calls,
        finish_calls=finish_calls,
    )


def _sample_cfg(**overrides) -> Any:
    base = OmegaConf.create(
        {
            "name": "ladder_A0_llama60m",
            "run": {
                "output_dir": "results",
                "wandb": True,
                "wandb_project": "coupled-muon-nanogpt",
                "wandb_group": None,
                "wandb_mode": None,
                "wandb_tags": [],
            },
            "model": {
                "moe": {
                    "enabled": False,
                    "num_experts": 8,
                    "top_k": 2,
                    "balancing_type": "aux_loss",
                    "shared_expert": False,
                },
            },
            "optimizer": {
                "type": "coupled_muon_v2",
                "lr": 3.0e-3,
                "coupled_steps": 4,
                "couple_qk": True,
                "couple_vo": True,
                "couple_updown": True,
                "final_polish": True,
                "use_multi_head": True,
            },
        }
    )
    if overrides:
        return OmegaConf.merge(base, OmegaConf.create(overrides))
    return base


def test_init_wandb_identity(fake_wandb, monkeypatch, tmp_path):
    from coupled_muon_nanogpt import wandb_utils

    # Ensure a clean baseline regardless of the host environment (WANDB_MODE
    # is set by my_env.sh on the Inspire cluster; would otherwise leak in).
    monkeypatch.delenv("WANDB_MODE", raising=False)

    cfg = _sample_cfg()
    handle = wandb_utils.init_wandb(cfg, rid="ladder_A0_llama60m-s0-abc", seed=0, out_dir=tmp_path)

    assert handle is not None
    assert len(fake_wandb.init_calls) == 1
    kw = fake_wandb.init_calls[0]
    assert kw["project"] == "coupled-muon-nanogpt"
    # Fallback group formula varies on rung × optimizer × lr.
    assert kw["group"] == "ladder_A0_llama60m/coupled_muon_v2/lr3e-03"
    assert kw["job_type"] == "coupled_muon_v2"
    assert "ladder_A0_llama60m·coupled_muon_v2·lr3e-03·s0" in kw["name"]
    # Tag set carries the slicing axes.
    tags = kw["tags"]
    assert "ladder_A0_llama60m" in tags
    assert "coupled_muon_v2" in tags
    assert "seed0" in tags
    assert "couple_qk" in tags and "couple_vo" in tags and "couple_updown" in tags
    assert "cs4" in tags
    assert "final_polish" in tags
    assert "multi_head" in tags
    assert "moe" not in tags  # dense run
    assert "smoke" not in tags  # not a smoke config
    # No mode specified ⇒ wandb default (online); init kw should not contain mode.
    assert "mode" not in kw


def test_smoke_config_gets_smoke_tag(fake_wandb, tmp_path):
    from coupled_muon_nanogpt import wandb_utils

    cfg = _sample_cfg(name="smoke_tiny_dev")
    wandb_utils.init_wandb(cfg, rid="smoke_tiny_dev-s0-abc", seed=0, out_dir=tmp_path)
    tags = fake_wandb.init_calls[0]["tags"]
    assert "smoke" in tags


def test_moe_tags(fake_wandb, tmp_path):
    from coupled_muon_nanogpt import wandb_utils

    cfg = _sample_cfg(
        model={
            "moe": {
                "enabled": True,
                "num_experts": 16,
                "top_k": 2,
                "balancing_type": "deepseek_bias",
                "shared_expert": True,
            }
        }
    )
    wandb_utils.init_wandb(cfg, rid="rid", seed=0, out_dir=tmp_path)
    tags = fake_wandb.init_calls[0]["tags"]
    assert "moe" in tags
    assert "experts16" in tags
    assert "topk2" in tags
    assert "balance:deepseek_bias" in tags
    assert "shared_expert" in tags


def test_define_metric_registers_tokens_axis(fake_wandb, tmp_path):
    from coupled_muon_nanogpt import wandb_utils

    cfg = _sample_cfg()
    wandb_utils.init_wandb(cfg, rid="rid", seed=0, out_dir=tmp_path)
    by_name = {c["name"]: c for c in fake_wandb.define_calls}
    # Cross-run metrics: tokens-axis.
    assert by_name["loss"]["step_metric"] == "tokens"
    assert by_name["val_loss"]["step_metric"] == "tokens"
    assert by_name["probe/svd/*"]["step_metric"] == "tokens"
    # Within-run metrics: no step_metric.
    assert by_name["lr"]["step_metric"] is None
    assert by_name["opt_step_s"]["step_metric"] is None


def test_log_step_does_not_pass_step_kwarg(fake_wandb, tmp_path):
    """The single most important correctness test: passing `step=` overrides
    `step_metric="tokens"` and silently breaks the cross-run x-axis. Catch any
    regression that adds it back."""
    from coupled_muon_nanogpt import wandb_utils

    cfg = _sample_cfg()
    handle = wandb_utils.init_wandb(cfg, rid="rid", seed=0, out_dir=tmp_path)
    fake_wandb.log_calls.clear()

    wandb_utils.log_step(handle, {"step": 10, "tokens": 1024, "loss": 5.0})
    wandb_utils.log_eval(handle, {"step": 10, "tokens": 1024, "val_loss": 4.5})
    wandb_utils.log_probe(handle, {"svd": {"linear.frob": 3.14}}, tokens=2048)

    assert len(fake_wandb.log_calls) == 3
    for payload, kwargs in fake_wandb.log_calls:
        assert "step" not in kwargs, (
            f"wandb.log was called with step={kwargs.get('step')!r}; this "
            f"silently overrides step_metric='tokens'. Drop the kwarg."
        )
        assert "tokens" in payload


def test_mode_resolution_env_var(fake_wandb, monkeypatch, tmp_path):
    from coupled_muon_nanogpt import wandb_utils

    cfg = _sample_cfg()
    monkeypatch.setenv("WANDB_MODE", "offline")
    wandb_utils.init_wandb(cfg, rid="rid", seed=0, out_dir=tmp_path)
    assert fake_wandb.init_calls[-1]["mode"] == "offline"


def test_mode_resolution_cfg_overrides_env(fake_wandb, monkeypatch, tmp_path):
    from coupled_muon_nanogpt import wandb_utils

    cfg = _sample_cfg(run={"wandb_mode": "disabled"})
    monkeypatch.setenv("WANDB_MODE", "offline")
    wandb_utils.init_wandb(cfg, rid="rid", seed=0, out_dir=tmp_path)
    assert fake_wandb.init_calls[-1]["mode"] == "disabled"


def test_explicit_wandb_group_overrides_fallback(fake_wandb, tmp_path):
    from coupled_muon_nanogpt import wandb_utils

    cfg = _sample_cfg(run={"wandb_group": "ladder_A0/pair_truefalsetrue"})
    wandb_utils.init_wandb(cfg, rid="rid", seed=0, out_dir=tmp_path)
    assert fake_wandb.init_calls[-1]["group"] == "ladder_A0/pair_truefalsetrue"


def test_probe_flatten_attn_logit_per_layer(fake_wandb, tmp_path):
    from coupled_muon_nanogpt import wandb_utils

    cfg = _sample_cfg()
    handle = wandb_utils.init_wandb(cfg, rid="rid", seed=0, out_dir=tmp_path)
    fake_wandb.log_calls.clear()

    probe_out = {
        "attn_logit": {"per_layer": [10.0, 20.0, 30.0], "global_max": 30.0},
    }
    wandb_utils.log_probe(handle, probe_out, tokens=1000)

    assert len(fake_wandb.log_calls) == 1
    payload, _ = fake_wandb.log_calls[0]
    # Per-layer scalars expanded with /L{i}, NOT collapsed to max+mean.
    assert payload["probe/attn_logit/L0"] == 10.0
    assert payload["probe/attn_logit/L1"] == 20.0
    assert payload["probe/attn_logit/L2"] == 30.0
    assert payload["probe/attn_logit/global_max"] == 30.0
    assert payload["tokens"] == 1000


def test_probe_flatten_moe_load_emits_histogram(fake_wandb, tmp_path):
    from coupled_muon_nanogpt import wandb_utils

    cfg = _sample_cfg()
    handle = wandb_utils.init_wandb(cfg, rid="rid", seed=0, out_dir=tmp_path)
    fake_wandb.log_calls.clear()

    probe_out = {
        "moe_load": {
            "layer.0": {
                "loads": [10.0, 20.0, 30.0, 40.0],
                "imbalance": 1.5,
                "router_entropy": 2.1,
                "grad_norms": [0.1, 0.2, 0.3, 0.4],
                "grad_norm_var": 0.05,
            }
        }
    }
    wandb_utils.log_probe(handle, probe_out, tokens=5000)
    payload, _ = fake_wandb.log_calls[0]

    # Per-expert scalars + histogram.
    assert payload["probe/moe_load/layer.0/loads/E0"] == 10.0
    assert payload["probe/moe_load/layer.0/loads/E3"] == 40.0
    assert hasattr(payload["probe/moe_load/layer.0/loads_hist"], "values")
    assert payload["probe/moe_load/layer.0/loads_hist"].values == [10.0, 20.0, 30.0, 40.0]
    assert payload["probe/moe_load/layer.0/imbalance"] == 1.5
    assert payload["probe/moe_load/layer.0/router_entropy"] == 2.1


def test_finalize_writes_summary(fake_wandb, tmp_path):
    from coupled_muon_nanogpt import wandb_utils

    cfg = _sample_cfg()
    handle = wandb_utils.init_wandb(cfg, rid="my_rid_123", seed=0, out_dir=tmp_path)
    wandb_utils.finalize(
        handle,
        summary={"final_val_loss": 3.21, "wall_clock_s": 1234.0, "diverged": False},
    )
    assert handle.run.summary["final_val_loss"] == 3.21
    assert handle.run.summary["wall_clock_s"] == 1234.0
    assert handle.run.summary["diverged"] is False
    # rid was set in init_wandb's summary block and preserved by finalize.
    assert handle.run.summary["rid"] == "my_rid_123"
    assert fake_wandb.finish_calls == [True]


def test_divergence_detector():
    from coupled_muon_nanogpt import wandb_utils

    assert wandb_utils.probe_indicates_divergence({}) is False
    assert wandb_utils.probe_indicates_divergence({"attn_logit": {"global_max": 50.0}}) is False
    assert wandb_utils.probe_indicates_divergence({"attn_logit": {"global_max": 5000.0}}) is True
    assert wandb_utils.probe_indicates_divergence(
        {"ns_internal": {"qk0": {"diverged": True}}}
    ) is True
    assert wandb_utils.probe_indicates_divergence(
        {"ns_internal": {"qk0": {"diverged": False}}}
    ) is False


def test_init_returns_none_when_wandb_missing(monkeypatch, tmp_path):
    """If the wandb package isn't installed, init_wandb returns None and the
    caller (train.py) silently continues."""
    from coupled_muon_nanogpt import wandb_utils

    # Force ImportError by removing the (real or mocked) wandb module and
    # blocking re-imports.
    monkeypatch.setitem(sys.modules, "wandb", None)
    cfg = _sample_cfg()
    handle = wandb_utils.init_wandb(cfg, rid="rid", seed=0, out_dir=tmp_path)
    assert handle is None
    # Subsequent log calls are no-ops.
    wandb_utils.log_step(handle, {"tokens": 1, "loss": 5.0})
    wandb_utils.log_probe(handle, {"svd": {}}, tokens=1)
    wandb_utils.finalize(handle, summary={"final": 1})
