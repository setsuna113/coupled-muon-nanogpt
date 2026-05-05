"""Shared fixtures for tests."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on sys.path so `coupled_muon_nanogpt` is importable when running
# pytest from the repo root without `pip install -e .`
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
