"""Shared fixtures for tests."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on sys.path so `coupled_muon_nanogpt` is importable when running
# pytest from the repo root without `pip install -e .`
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Some Newton-Schulz kernels are `@torch.compile`'d. On CPU-only test runs the
# inductor backend can fail with internal-buffer assertions for our specific
# shape patterns. Production runs on CUDA (H200/4090) where compile works
# fine. Suppress compile errors in tests so failures fall back to eager.
import torch._dynamo  # noqa: E402

torch._dynamo.config.suppress_errors = True
