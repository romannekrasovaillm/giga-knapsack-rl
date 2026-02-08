"""
Dynamic Programming solver for the multiple-choice knapsack problem.

For each prompt i, choose budget N_i in [N_low, N_up] to maximize total value
subject to sum(N_i) <= N_total.

Numba-accelerated for practical speed.
"""

import numpy as np
from numba import njit


@njit
def knapsack_dp(
    values: np.ndarray,   # [M, n_options] — value for each (prompt, budget)
    costs: np.ndarray,    # [M, n_options] — cost (= budget) for each
    N_total: int,
    M: int,
    n_options: int,
) -> np.ndarray:
    """
    Multiple-choice knapsack via DP.

    Each prompt must get exactly one budget option.
    Maximizes total value subject to sum of budgets <= N_total.

    Complexity: O(M * N_total * n_options)
    """
    INF = -1e30

    prev = np.full(N_total + 1, INF)
    prev[0] = 0.0

    choice = np.zeros((M, N_total + 1), dtype=np.int32)

    for i in range(M):
        curr = np.full(N_total + 1, INF)

        for j in range(N_total + 1):
            if prev[j] <= INF + 1.0:
                continue
            for k in range(n_options):
                c = costs[i, k]
                nj = j + c
                if nj <= N_total:
                    new_val = prev[j] + values[i, k]
                    if new_val > curr[nj]:
                        curr[nj] = new_val
                        choice[i, nj] = k

        prev = curr

    # Find best total budget
    best_j = 0
    for j in range(N_total + 1):
        if prev[j] > prev[best_j]:
            best_j = j

    # Backtrack
    allocations = np.zeros(M, dtype=np.int32)
    remaining = best_j

    for i in range(M - 1, -1, -1):
        k = choice[i, remaining]
        c = costs[i, k]
        allocations[i] = c
        remaining -= c

    return allocations


@njit
def knapsack_dp_with_minimum(
    values: np.ndarray,
    costs: np.ndarray,
    N_total: int,
    M: int,
    n_options: int,
    N_low: int,
) -> np.ndarray:
    """
    Knapsack DP with guaranteed minimum allocation per prompt.

    First reserves N_low * M, then allocates the remaining budget.
    Each prompt gets at least N_low rollouts.
    """
    # Reserve minimum
    reserved = N_low * M
    if reserved > N_total:
        # Not enough budget: give each prompt equal share
        per_prompt = max(1, N_total // M)
        return np.full(M, per_prompt, dtype=np.int32)

    remaining_budget = N_total - reserved

    # Adjust values and costs to represent *extra* allocation above N_low
    extra_options = n_options  # options 0..n_options-1 represent N_low+0 .. N_low+n_options-1
    extra_values = np.zeros((M, extra_options), dtype=np.float64)
    extra_costs = np.zeros((M, extra_options), dtype=np.int32)

    for i in range(M):
        for k in range(extra_options):
            extra_values[i, k] = values[i, k]
            extra_costs[i, k] = k  # extra cost above N_low

    # Solve DP for extra allocation
    INF = -1e30
    prev = np.full(remaining_budget + 1, INF)
    prev[0] = 0.0
    choice = np.zeros((M, remaining_budget + 1), dtype=np.int32)

    for i in range(M):
        curr = np.full(remaining_budget + 1, INF)
        for j in range(remaining_budget + 1):
            if prev[j] <= INF + 1.0:
                continue
            for k in range(extra_options):
                c = extra_costs[i, k]
                nj = j + c
                if nj <= remaining_budget:
                    new_val = prev[j] + extra_values[i, k]
                    if new_val > curr[nj]:
                        curr[nj] = new_val
                        choice[i, nj] = k
        prev = curr

    best_j = 0
    for j in range(remaining_budget + 1):
        if prev[j] > prev[best_j]:
            best_j = j

    allocations = np.zeros(M, dtype=np.int32)
    remaining = best_j
    for i in range(M - 1, -1, -1):
        k = choice[i, remaining]
        c = extra_costs[i, k]
        allocations[i] = N_low + k
        remaining -= c

    # Ensure minimum
    for i in range(M):
        if allocations[i] < N_low:
            allocations[i] = N_low

    return allocations
