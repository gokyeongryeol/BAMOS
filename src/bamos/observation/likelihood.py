"""The detection likelihood's constants (spec 7.2): r = eta_f + (1 - eta_f) * eta_t * h(a)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectionLikelihood:
    gain: float  # eta_t: the detection ceiling, P(DET) for a fully visible target
    false_positive_rate: float = 0.0  # eta_f: the false-positive floor, P(DET) with no visible target

    def __post_init__(self):
        for name in ("gain", "false_positive_rate"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name}={v} outside [0,1]")
