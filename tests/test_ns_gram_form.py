"""NS Gram-form smoke test.

Phase 2 wires `ns_gram_form` through the optimizer interface but currently
implements it as a passthrough to the standard form. This test asserts the
flag does not break numerics — the two paths produce identical outputs
(passthrough). Replace once the actual Gram-NS kernel lands in Phase 2.5.
"""
from __future__ import annotations

import torch

from coupled_muon_nanogpt.optim.coupled_muon import zeropower_via_newtonschulz5


def test_gram_form_passthrough_equals_standard():
    torch.manual_seed(0)
    G = torch.randn(40, 60)
    out_std = zeropower_via_newtonschulz5(
        G, steps=5, dtype=torch.float32, gram_form=False
    )
    out_gram = zeropower_via_newtonschulz5(
        G, steps=5, dtype=torch.float32, gram_form=True
    )
    diff = (out_std - out_gram).norm().item() / (out_std.norm().item() + 1e-12)
    # Passthrough ⇒ identical outputs. Once the real Gram-NS kernel lands,
    # this threshold becomes 0.05 (rel-error) per suggestion.md §1.3.
    assert diff < 1e-6, f"gram_form passthrough diverged: {diff}"
