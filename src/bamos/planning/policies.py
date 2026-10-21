"""View-selection rule of BAMOS (Bayesian Active Mask-based Object Search): q_{t+1} = argmax_q I(Z;X) (spec 9).
Ties within tie_tolerance are broken by lower path cost, then lower pose id; path cost is never added to the
utility."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bamos.planning.candidates import CandidateBatch
from bamos.planning.scorer import ScoreBatch


def method_flags(method_id: str) -> dict:
    """The one method this build implements: `bamos` — Bayesian Active Mask-based Object Search.

    A joint (location, size) belief carries the target; the scene geometry is not modelled. A detection is read as two things at once: the event itself, through a likelihood r driven by the
    hypothesis' apparent area, and the mask's centroid, through an isotropic kappa centred beta s towards
    the camera.  The DET update is pi r kappa, i.e. exact Bayes on the mixed observation space, and poses
    are scored by I(Z;X), the detection event's mutual information with the hypothesis.
    """
    if method_id != "bamos":
        raise ValueError(f"unsupported method {method_id!r}: this build implements 'bamos' only")
    return {"name": "bamos"}


@dataclass
class PolicyConfig:
    method: str
    tie_tolerance: float = 1e-12
    # What the planner maximizes. "ig" is BAMOS: I(Z;X), the mutual information between the detection event
    # and the hypothesis (spec 9). "coverage" reads no observation model and no belief at all (outline S4).
    # "scene_coverage" (2026-10-06) reads the scene's occupancy belief but not the target belief: the candidate cells a
    # pose would newly bring into view, each weighted by its expected transmittance (a scene-model next-best view)
    utility_mode: str = "ig"  # | "coverage" | "scene_coverage"


def _select_max(score: np.ndarray, path_cost: np.ndarray | None, pose_ids: np.ndarray, tol: float) -> tuple[int, int]:
    best = np.max(score)
    ties = np.nonzero(score >= best - tol)[0]
    n_tie = len(ties)
    if n_tie > 1 and path_cost is not None:
        pc = path_cost[ties]
        ties = ties[pc <= pc.min() + 1e-9]
    if len(ties) > 1:
        ties = ties[np.argsort(pose_ids[ties])][:1]
    return int(ties[0]), n_tie


class Policy:
    MODES = ("ig", "coverage", "scene_coverage")

    def __init__(self, cfg: PolicyConfig):
        self.cfg = cfg
        method_flags(cfg.method)
        if cfg.utility_mode not in self.MODES:  # fail at start-up, not at the first active view
            raise ValueError(f"utility_mode={cfg.utility_mode!r} must be one of {self.MODES}")

    def utility(self, sb: ScoreBatch) -> np.ndarray:
        """U_t(q) under the configured mode.

        "ig" is exact on the joint belief. The DET slice also carries a centroid, worth P+ I_kappa, which is
        left out: I_kappa barely depends on the pose, so that term is close to a monotone function of P+, and
        I(Z;X) already rises with P+ while P+ is far below 1/2 (spec 9).

        "coverage" counts candidate cells the pose would newly bring into view, so it reads neither the belief nor
        the observation model.
        """
        mode = self.cfg.utility_mode
        if mode == "ig":
            return np.asarray(sb.ig_normalized, dtype=np.float64)
        if mode == "coverage":
            # newly seen candidate cells, as a fraction of the candidate set
            return np.asarray(sb.extra["coverage"], dtype=np.float64)
        if mode == "scene_coverage":
            # expected newly seen candidate cells through the occupancy belief, as a fraction of the candidate set
            return np.asarray(sb.extra["scene_coverage"], dtype=np.float64)
        raise ValueError(f"unknown utility_mode {mode!r}")

    def select(self, sb: ScoreBatch, cand: CandidateBatch, rng: np.random.Generator) -> tuple[int, dict]:
        u = self.utility(sb)
        k, n_tie = _select_max(u, cand.path_cost, cand.pose_ids, self.cfg.tie_tolerance)
        return k, {"utility": float(u[k]), "tie_count": n_tie}
