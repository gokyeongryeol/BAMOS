"""A module of scripts/isaac/ loaded by its path, for the HSSD scripts that reuse the Isaac pipeline's functions
(scripts/hssd/ has files of the same names, so a plain import would find the HSSD one)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ISAAC = Path(__file__).resolve().parents[1] / "isaac"


def load(name: str):
    key = f"isaac_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ISAAC / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]
