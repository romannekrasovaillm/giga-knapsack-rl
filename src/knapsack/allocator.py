"""
Knapsack Budget Allocator: main interface for adaptive rollout allocation.

Wraps the DP solver and value functions into a stateful allocator
that tracks prompt success rates across training iterations.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from dataclasses import dataclass, field

from .task_value import task_value, info_gain, _compute_values_table
from .dp_solver import knapsack_dp, knapsack_dp_with_minimum

logger = logging.getLogger(__name__)


@dataclass
class PromptStats:
    """Success rate tracking for a single prompt."""
    prompt_id: str
    total_rollouts: int = 0
    total_successes: int = 0
    history: List[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_rollouts == 0:
            return 0.5
        return self.total_successes / self.total_rollouts

    def update(self, rewards: List[float]):
        self.total_rollouts += len(rewards)
        self.total_successes += sum(1 for r in rewards if r > 0.5)
        self.history.append(self.success_rate)


class KnapsackBudgetAllocator:
    """
    Adaptive rollout budget allocator using the knapsack formulation.

    Usage:
        allocator = KnapsackBudgetAllocator(N_total=1024, N_low=2, N_up=128)

        # Each iteration:
        budgets = allocator.allocate(prompt_ids, success_rates)
        # ... generate rollouts, compute rewards ...
        allocator.update(prompt_ids, rewards_per_prompt)
    """

    def __init__(
        self,
        N_total: int = 1024,
        N_low: int = 2,
        N_up: int = 128,
        warmup_iterations: int = 5,
        ema_alpha: float = 0.3,
    ):
        self.N_total = N_total
        self.N_low = N_low
        self.N_up = N_up
        self.warmup_iterations = warmup_iterations
        self.ema_alpha = ema_alpha
        self.stats: Dict[str, PromptStats] = {}
        self.iteration = 0
        self._allocation_history: List[Dict] = []

    def allocate(
        self,
        prompt_ids: List[str],
        success_rates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Allocate rollout budgets for a batch of prompts.

        Args:
            prompt_ids: Prompt identifiers
            success_rates: Optional override; if None, uses tracked stats

        Returns:
            allocations: [M] array of rollout counts per prompt
        """
        t0 = time.time()
        M = len(prompt_ids)

        # Get success rates
        if success_rates is None:
            success_rates = self._get_success_rates(prompt_ids)

        # Warmup: use uniform allocation
        if self.iteration < self.warmup_iterations:
            per_prompt = max(self.N_low, self.N_total // M)
            allocations = np.full(M, per_prompt, dtype=np.int32)
            self.iteration += 1
            self._log_allocation(prompt_ids, allocations, success_rates, time.time() - t0)
            return allocations

        # Clip success rates
        p_clipped = np.clip(success_rates, 1e-6, 1.0 - 1e-6)

        # Check budget feasibility
        min_budget = M * self.N_low
        if min_budget > self.N_total:
            per_prompt = max(1, self.N_total // M)
            allocations = np.full(M, per_prompt, dtype=np.int32)
            self.iteration += 1
            return allocations

        n_options = self.N_up - self.N_low + 1

        # Compute value table
        values = _compute_values_table(M, n_options, self.N_low, p_clipped)
        costs = np.zeros((M, n_options), dtype=np.int32)
        for i in range(M):
            for k in range(n_options):
                costs[i, k] = self.N_low + k

        # Solve knapsack
        allocations = knapsack_dp(values, costs, self.N_total, M, n_options)

        # Enforce minimum
        for i in range(M):
            if allocations[i] < self.N_low:
                allocations[i] = self.N_low

        # Handle over-budget
        total = allocations.sum()
        if total > self.N_total:
            allocations = self._trim_excess(allocations, success_rates, total)

        self.iteration += 1
        elapsed = time.time() - t0
        self._log_allocation(prompt_ids, allocations, success_rates, elapsed)

        return allocations

    def update(
        self,
        prompt_ids: List[str],
        rewards_per_prompt: List[List[float]],
    ):
        """Update prompt statistics after observing rewards."""
        for pid, rewards in zip(prompt_ids, rewards_per_prompt):
            if pid not in self.stats:
                self.stats[pid] = PromptStats(prompt_id=pid)
            self.stats[pid].update(rewards)

    def _get_success_rates(self, prompt_ids: List[str]) -> np.ndarray:
        rates = []
        for pid in prompt_ids:
            if pid in self.stats:
                rates.append(self.stats[pid].success_rate)
            else:
                rates.append(0.5)
        return np.array(rates)

    def _trim_excess(
        self,
        allocations: np.ndarray,
        success_rates: np.ndarray,
        total: int,
    ) -> np.ndarray:
        """Trim over-budget by reducing allocations for least valuable prompts."""
        ig = np.array([info_gain(p) for p in success_rates])
        order = np.argsort(ig)
        for idx in order:
            if total <= self.N_total:
                break
            excess = total - self.N_total
            can_reduce = allocations[idx] - self.N_low
            reduce_by = min(can_reduce, excess)
            allocations[idx] -= reduce_by
            total -= reduce_by
        return allocations

    def _log_allocation(
        self,
        prompt_ids: List[str],
        allocations: np.ndarray,
        success_rates: np.ndarray,
        elapsed: float,
    ):
        entry = {
            "iteration": self.iteration,
            "num_prompts": len(prompt_ids),
            "total_budget_used": int(allocations.sum()),
            "budget_utilization": float(allocations.sum()) / self.N_total,
            "mean_allocation": float(allocations.mean()),
            "min_allocation": int(allocations.min()),
            "max_allocation": int(allocations.max()),
            "std_allocation": float(allocations.std()),
            "mean_success_rate": float(success_rates.mean()),
            "solve_time": elapsed,
        }
        self._allocation_history.append(entry)
        logger.info(
            f"[Knapsack] iter={self.iteration} | "
            f"budget={entry['total_budget_used']}/{self.N_total} | "
            f"alloc=[{entry['min_allocation']},{entry['max_allocation']}] "
            f"mean={entry['mean_allocation']:.1f} | "
            f"p_mean={entry['mean_success_rate']:.3f} | "
            f"time={elapsed:.3f}s"
        )

    def get_effective_gradient_ratio(
        self,
        rewards_per_prompt: List[List[float]],
    ) -> Dict[str, float]:
        """Compute effective gradient ratio metric."""
        total = len(rewards_per_prompt)
        all_pos = 0
        all_neg = 0
        mixed = 0

        for rewards in rewards_per_prompt:
            has_success = any(r > 0.5 for r in rewards)
            has_failure = any(r <= 0.5 for r in rewards)
            if has_success and has_failure:
                mixed += 1
            elif has_success:
                all_pos += 1
            else:
                all_neg += 1

        return {
            "effective_ratio": mixed / total if total > 0 else 0.0,
            "all_positive_ratio": all_pos / total if total > 0 else 0.0,
            "all_negative_ratio": all_neg / total if total > 0 else 0.0,
        }

    def get_allocation_history(self) -> List[Dict]:
        return self._allocation_history
