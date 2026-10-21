"""Target-location belief pi_t over the fixed candidate set C (spec 3, 5, 8)."""
from __future__ import annotations

import numpy as np


def entropy_from_probs(p: np.ndarray) -> float:
    """H(p) with 0 log 0 = 0, natural log."""
    p = np.asarray(p, dtype=np.float64)
    nz = p > 0
    return float(-np.sum(p[nz] * np.log(p[nz])))



class LocationBelief:
    """log_pi over the candidate set, optionally jointly with a discrete size axis (spec 3).

    n_sizes = 1 reproduces v1.0-v2.2 exactly. With n_sizes > 1 the array is the joint pi(v, s) flattened as
    v * n_sizes + j, and every location-level quantity (entropy, MAP, max, evaluator mass) uses the marginal
    sum_s pi(v, s). Likelihoods passed to the update methods must have the joint length.
    """

    def __init__(self, log_pi: np.ndarray, candidate_voxels: np.ndarray, n_sizes: int = 1):
        assert log_pi.shape[0] == candidate_voxels.shape[0] * n_sizes
        self.log_pi = log_pi
        self.candidate_voxels = np.asarray(candidate_voxels, dtype=np.int64)
        self.n_sizes = int(n_sizes)

    @staticmethod
    def uniform(candidate_voxels: np.ndarray, n_sizes: int = 1) -> "LocationBelief":
        n = len(candidate_voxels) * int(n_sizes)
        return LocationBelief(np.full(n, -np.log(n)), np.asarray(candidate_voxels), n_sizes)

    @property
    def n(self) -> int:
        return self.log_pi.shape[0]

    def probs(self) -> np.ndarray:
        """Joint pi(v, s) flattened (length n_C * n_sizes)."""
        return np.exp(self.log_pi)

    def probs_loc(self) -> np.ndarray:
        """Location marginal sum_s pi(v, s) (length n_C)."""
        if self.n_sizes == 1:
            return np.exp(self.log_pi)
        return np.exp(self.log_pi).reshape(-1, self.n_sizes).sum(axis=1)

    def probs_size(self) -> np.ndarray:
        """Size marginal sum_v pi(v, s) (length n_sizes)."""
        if self.n_sizes == 1:
            return np.ones(1)
        return np.exp(self.log_pi).reshape(-1, self.n_sizes).sum(axis=0)

    def entropy(self) -> float:
        return entropy_from_probs(self.probs_loc())

    def map_index(self) -> int:
        """Index into the candidate set (not into the joint array)."""
        return int(np.argmax(self.probs_loc()))

    def max_prob(self) -> float:
        return float(np.max(self.probs_loc()))

    def update_with_log_likelihood(self, loglik: np.ndarray) -> dict:
        """pi <- pi * exp(loglik) normalized, in log space: a likelihood whose every entry underflows in linear
        space (a Gaussian evaluated far from its mean) still orders the hypotheses correctly."""
        loglik = np.asarray(loglik, dtype=np.float64)
        if loglik.shape != self.log_pi.shape or np.isnan(loglik).any():
            raise ValueError("log-likelihood must be finite or -inf, one entry per candidate")
        unnorm = self.log_pi + loglik
        m = np.max(unnorm)
        if not np.isfinite(m):
            return {"model_inconsistency": True}
        lse = m + np.log(np.sum(np.exp(unnorm - m)))
        self.log_pi = unnorm - lse
        return {"model_inconsistency": False, "log_evidence": float(lse)}

    def update(self, r: np.ndarray, detected: bool) -> dict:
        """pi <- pi * r (DET) or pi * (1-r) (NO_DET), normalized (spec 8.3).

        r must be the pre-observation likelihood snapshot. If the normalizer is 0 the belief is left
        unchanged and 'model_inconsistency' is flagged (never silently reset).
        """
        r = np.asarray(r, dtype=np.float64)
        if r.shape != self.log_pi.shape:
            raise ValueError("r must have one entry per candidate")
        if np.any(r < 0) or np.any(r > 1):
            raise ValueError("r outside [0,1]")
        lik = r if detected else 1.0 - r
        with np.errstate(divide="ignore"):
            unnorm = self.log_pi + np.log(lik)
        m = np.max(unnorm)
        if not np.isfinite(m):
            return {"model_inconsistency": True, "predictive_prob": 0.0}
        lse = m + np.log(np.sum(np.exp(unnorm - m)))
        self.log_pi = unnorm - lse
        return {"model_inconsistency": False, "predictive_prob": float(np.exp(lse))}
