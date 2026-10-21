"""Algorithmic stop decision S_hat (spec 10). Evaluated only on the movable robot camera's current view."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bamos.belief.state import BeliefState
from bamos.observation.types import DetectionStatus, PlannerObservation


@dataclass
class StopConfig:
    # Declare when the belief says a declaration is more likely right than wrong: pi(v_hat), the marginal
    # probability of the MAP cell itself. (The mass within the success radius of the MAP was tried and was
    # overconfident: a face neighbour's centre lies exactly tau away, but most of its cell does not.)
    success_radius_m: float = 0.2
    tau_pi: float = 0.5
    # "joint": declare as soon as both conditions hold.
    # "none":  evaluate and log the very same decision but never act on it, so every episode runs the full
    #          view budget. Metrics are then read off the trajectory (anytime curves) instead of a
    #          self-declared moment, which keeps the stop rule from confounding comparisons.
    mode: str = "joint"
    # Confirmation ablation: a declaration needs this many steps with S_hat = 1 whose MAP lies within
    # success_radius_m of the MAP at the first of them (not necessarily consecutive). 1 is spec 10 itself.
    confirmations: int = 1


@dataclass
class StopDecision:
    declared: bool
    max_pi: float
    pi_map: float
    detected: bool
    normalized_area: float
    loc_ok: bool
    det_ok: bool


def mass_within_radius(state: BeliefState, radius_m: float) -> float:
    """Posterior mass on candidates within radius_m of the MAP candidate (a diagnostic; not the stop rule)."""
    pi = state.location.probs_loc()
    centers = state.candidate_centers()
    k = int(state.location.map_index())
    near = np.linalg.norm(centers - centers[k][None, :], axis=1) <= radius_m + 1e-9
    return float(pi[near].sum())


def omega_table(step: float, radius_m: float, n: int = 40) -> dict[tuple[int, int, int], float]:
    """omega by a cell's lattice offset from the declared cell, as the sorted absolute offsets in steps per axis ((0, 0, 0):
    itself, (0, 0, 1): a face neighbour, ...): the probability that a target centred uniformly in that cell lies within
    radius_m of the declared cell's centre (formulation §10). The same uniform-in-cell assumption as the cell average
    (§7.3); a midpoint rule on n^3 points of the cell. Offsets of more than ceil(radius_m / step) steps on an axis lie
    farther than radius_m, so they are 0 (radius_m <= step: one step, the face, edge and corner neighbours)."""
    u = (np.arange(n) + 0.5) / n * step - step / 2
    X, Y, Z = np.meshgrid(u, u, u, indexing="ij")
    m = int(np.ceil(radius_m / step - 1e-9))
    out = {}
    for a in range(m + 1):
        for b in range(a, m + 1):
            for c in range(b, m + 1):
                d = np.sqrt((X + a * step) ** 2 + (Y + b * step) ** 2 + (Z + c * step) ** 2)
                out[(a, b, c)] = float(np.mean(d <= radius_m + 1e-12))
    return out


def success_probability(state: BeliefState, radius_m: float, step: float) -> float:
    """P(the MAP cell's centre is within radius_m of the target) under the belief and a target centre uniform in its
    cell: sum over the cells c of omega(c - v_hat) pi(c) (formulation §10)."""
    table = _OMEGA.setdefault((round(step, 6), round(radius_m, 6)), omega_table(step, radius_m))
    m = max(max(k) for k in table)
    pi = state.location.probs_loc()
    centers = state.candidate_centers()
    k = int(state.location.map_index())
    d = np.rint((centers - centers[k][None, :]) / step).astype(int)
    near = np.all(np.abs(d) <= m, axis=1)
    w = np.array([table[tuple(sorted(np.abs(x).tolist()))] for x in d[near]])
    return float(pi[near] @ w)


_OMEGA: dict = {}


class StopTracker:
    """Counts S_hat = 1 steps per location and says when a declaration is made (StopConfig.confirmations).

    A location is anchored at the MAP centre of the first S_hat = 1 step not within success_radius_m of an
    existing anchor; a later S_hat = 1 step whose MAP centre is within that radius of an anchor adds one to it.
    """

    def __init__(self, cfg: StopConfig):
        self.cfg = cfg
        self.anchors: list[list] = []  # [centre, count]

    def update(self, state: BeliefState, stop: "StopDecision") -> bool:
        if not stop.declared:
            return False
        x = np.asarray(state.map_center(), dtype=float)
        for a in self.anchors:
            if np.linalg.norm(x - a[0]) <= self.cfg.success_radius_m + 1e-9:
                a[1] += 1
                return a[1] >= self.cfg.confirmations
        self.anchors.append([x, 1])
        return self.cfg.confirmations <= 1


def evaluate_stop(cfg: StopConfig, state: BeliefState, obs: PlannerObservation) -> StopDecision:
    pi_map = state.location.max_prob()
    det = obs.detection == DetectionStatus.DET
    loc_ok = pi_map >= cfg.tau_pi
    return StopDecision(bool(det and loc_ok), pi_map, pi_map, det,
                        obs.normalized_mask_area, loc_ok, det)
