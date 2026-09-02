"""Stage 3.6: per-expert SNR probe — unit-test the metric on a tiny model."""
from __future__ import annotations

import torch

from coupled_muon_nanogpt.model.transformer import GPT, BlockConfig, GPTConfig
from coupled_muon_nanogpt.probes.expert_snr import estimate_per_expert_snr


class _SyntheticLoader:
    """Fixed-vocab loader serving the same x,y pair each call. Lets us compare
    the noise-floor regime (every batch identical → cos≈1) and the noise regime
    (random batches → cos near 1/sqrt(P) for P parameters)."""

    def __init__(self, vocab: int, B: int, T: int, identical: bool, seed: int = 0):
        self.vocab = vocab
        self.B = B
        self.T = T
        self.identical = identical
        self.gen = torch.Generator().manual_seed(seed)
        if identical:
            self._x = torch.randint(0, vocab, (B, T), generator=self.gen)
            self._y = torch.randint(0, vocab, (B, T), generator=self.gen)
        # For the loader interface only:
        self.seq_len = T
        self.local_batch_size = B

    def next_batch(self):
        if self.identical:
            return self._x, self._y
        x = torch.randint(0, self.vocab, (self.B, self.T), generator=self.gen)
        y = torch.randint(0, self.vocab, (self.B, self.T), generator=self.gen)
        return x, y


def _moe_model() -> GPT:
    block = BlockConfig(
        hidden=32,
        n_heads=2,
        intermediate=64,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        pos_emb_type="rope",
        qk_norm=False,
        max_seq_len=64,
        moe_enabled=True,
        moe_cfg=dict(num_experts=2, top_k=1, expert_mlp_type="swiglu"),
    )
    return GPT(GPTConfig(vocab_size=64, n_layers=1, block=block))


def test_identical_batches_high_cos():
    """If every micro-batch is identical, gradient is deterministic ⇒ cos≈1."""
    torch.manual_seed(0)
    model = _moe_model()
    loader = _SyntheticLoader(vocab=64, B=2, T=16, identical=True, seed=42)
    report = estimate_per_expert_snr(model, loader, num_micro=4)
    for layer_key, layer in report.items():
        if not layer_key.startswith("layer."):
            continue
        for expert_key, m in layer.items():
            assert m["cos_to_mean_avg"] > 0.99, f"identical-batches cos < 0.99: {m}"


def test_random_batches_lower_cos():
    """Different micro-batches ⇒ cos drops well below 1.0."""
    torch.manual_seed(0)
    model = _moe_model()
    loader = _SyntheticLoader(vocab=64, B=2, T=16, identical=False, seed=42)
    report = estimate_per_expert_snr(model, loader, num_micro=8)
    avg_cos = []
    for layer_key, layer in report.items():
        if not layer_key.startswith("layer."):
            continue
        for m in layer.values():
            avg_cos.append(m["cos_to_mean_avg"])
    # Tight assertion: cos should drop below the identical-batch level.
    assert all(c < 0.99 for c in avg_cos), f"cos didn't drop with random batches: {avg_cos}"


def test_dense_model_returns_no_op_note():
    torch.manual_seed(0)
    block = BlockConfig(
        hidden=32, n_heads=2, intermediate=64, mlp_type="swiglu", max_seq_len=64
    )
    model = GPT(GPTConfig(vocab_size=64, n_layers=1, block=block))
    loader = _SyntheticLoader(vocab=64, B=2, T=16, identical=False)
    report = estimate_per_expert_snr(model, loader, num_micro=2)
    assert "_note" in report
