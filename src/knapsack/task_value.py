"""
Value functions from the Knapsack RL paper.

  "Knapsack RL: Unlocking Exploration of LLMs via Optimizing Budget Allocation"
  Ziniu Li et al., ByteDance Seed, arXiv:2509.25849
"""

import numpy as np
from numba import njit


def prob_nonzero_gradient(N: int, p: float) -> float:
    """
    Probability of non-zero gradient in GRPO with N rollouts and success rate p.
    P(non-zero) = 1 - p^N - (1-p)^N
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return 1.0 - p**N - (1.0 - p)**N


def info_gain(p: float) -> float:
    """
    Expected improvement in success rate after one gradient step.
    InfoGain(p) = p * (1 - p)^2  (Proposition 1, Taylor approximation)
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return p * (1.0 - p) ** 2


def task_value(N: int, p: float) -> float:
    """
    Value of allocating N rollouts to a prompt with success rate p.
    Value(N, p) = ProbNonZeroGradient(N, p) * InfoGain(p)
    """
    return prob_nonzero_gradient(N, p) * info_gain(p)


@njit
def _compute_values_table(
    M: int,
    n_options: int,
    N_low: int,
    success_rates: np.ndarray,
) -> np.ndarray:
    """Pre-compute value table for all (prompt, budget) pairs. Numba-accelerated."""
    values = np.zeros((M, n_options), dtype=np.float64)
    for i in range(M):
        p = success_rates[i]
        if p <= 0.0 or p >= 1.0:
            continue
        ig = p * (1.0 - p) ** 2
        for k in range(n_options):
            N = N_low + k
            pnzg = 1.0 - p**N - (1.0 - p)**N
            values[i, k] = pnzg * ig
    return values
