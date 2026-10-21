"""The view utility U_t(q) = I(Z;X) (spec 9)."""
from __future__ import annotations

import numpy as np
import torch

LOG2 = float(np.log(2.0))


def binary_entropy_t(p: torch.Tensor) -> torch.Tensor:
    a = torch.where(p > 0, -p * torch.log(torch.clamp(p, min=1e-300)), torch.zeros_like(p))
    q = 1.0 - p
    b = torch.where(q > 0, -q * torch.log(torch.clamp(q, min=1e-300)), torch.zeros_like(p))
    return a + b


def ig_loc(pi: torch.Tensor, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """I(Z;X) = H_b(P+) - sum_x pi(x) H_b(r(x)) in nats, for r [P, n_X] and pi [n_X] over the joint
    hypotheses x = (v, s). Returns (I(Z;X), P+)."""
    p_plus = (r * pi[None, :]).sum(dim=1)
    ig = binary_entropy_t(p_plus) - (pi[None, :] * binary_entropy_t(r)).sum(dim=1)
    return ig, p_plus
