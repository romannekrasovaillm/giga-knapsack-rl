"""Tests for Knapsack RL budget allocation."""

import numpy as np
import pytest

from src.knapsack.task_value import prob_nonzero_gradient, info_gain, task_value
from src.knapsack.dp_solver import knapsack_dp
from src.knapsack.allocator import KnapsackBudgetAllocator


class TestValueFunctions:
    def test_prob_nonzero_gradient_boundaries(self):
        assert prob_nonzero_gradient(8, 0.0) == 0.0
        assert prob_nonzero_gradient(8, 1.0) == 0.0

    def test_prob_nonzero_gradient_typical(self):
        # p=0.5, N=8 should be high
        val = prob_nonzero_gradient(8, 0.5)
        assert val > 0.99

        # p=0.05, N=8 should be lower
        val = prob_nonzero_gradient(8, 0.05)
        assert 0.3 < val < 0.4

        # p=0.05, N=64 should be high
        val = prob_nonzero_gradient(64, 0.05)
        assert val > 0.95

    def test_info_gain_boundaries(self):
        assert info_gain(0.0) == 0.0
        assert info_gain(1.0) == 0.0

    def test_info_gain_maximum(self):
        # Maximum around p=0.33
        vals = [(p, info_gain(p)) for p in np.arange(0.01, 1.0, 0.01)]
        max_p, max_v = max(vals, key=lambda x: x[1])
        assert 0.3 < max_p < 0.4

    def test_task_value_monotone_in_N(self):
        # More rollouts should increase value (for moderate p)
        p = 0.3
        v8 = task_value(8, p)
        v16 = task_value(16, p)
        v64 = task_value(64, p)
        assert v16 >= v8
        assert v64 >= v16


class TestDPSolver:
    def test_basic_allocation(self):
        M = 4
        n_options = 5
        N_low = 2
        N_total = 20

        values = np.zeros((M, n_options), dtype=np.float64)
        costs = np.zeros((M, n_options), dtype=np.int32)
        success_rates = [0.1, 0.3, 0.7, 0.95]

        for i in range(M):
            p = success_rates[i]
            for k in range(n_options):
                N = N_low + k
                values[i, k] = task_value(N, p)
                costs[i, k] = N

        allocs = knapsack_dp(values, costs, N_total, M, n_options)
        assert allocs.sum() <= N_total
        assert all(a >= 0 for a in allocs)

    def test_budget_respected(self):
        M = 8
        n_options = 10
        N_total = 40
        N_low = 2

        values = np.random.rand(M, n_options)
        costs = np.zeros((M, n_options), dtype=np.int32)
        for i in range(M):
            for k in range(n_options):
                costs[i, k] = N_low + k

        allocs = knapsack_dp(values, costs, N_total, M, n_options)
        assert allocs.sum() <= N_total


class TestAllocator:
    def test_warmup_uniform(self):
        allocator = KnapsackBudgetAllocator(
            N_total=64, N_low=2, N_up=32, warmup_iterations=3,
        )
        ids = [f"p{i}" for i in range(8)]
        # During warmup, should get uniform allocation
        budgets = allocator.allocate(ids)
        assert len(budgets) == 8
        assert all(b == budgets[0] for b in budgets)

    def test_post_warmup_adaptive(self):
        allocator = KnapsackBudgetAllocator(
            N_total=128, N_low=2, N_up=64, warmup_iterations=2,
        )
        ids = [f"p{i}" for i in range(8)]
        rates = np.array([0.05, 0.1, 0.3, 0.3, 0.5, 0.7, 0.9, 0.95])

        # Burn warmup
        for _ in range(2):
            allocator.allocate(ids, rates)

        # Post-warmup should be adaptive
        budgets = allocator.allocate(ids, rates)
        assert budgets.sum() <= 128
        assert all(b >= 2 for b in budgets)

    def test_update_stats(self):
        allocator = KnapsackBudgetAllocator()
        allocator.update(["p1", "p2"], [[1.0, 0.0, 1.0], [0.0, 0.0]])
        assert allocator.stats["p1"].success_rate == pytest.approx(2/3, abs=0.01)
        assert allocator.stats["p2"].success_rate == 0.0

    def test_effective_gradient_ratio(self):
        allocator = KnapsackBudgetAllocator()
        rewards = [
            [1.0, 0.0, 1.0],  # mixed
            [0.0, 0.0, 0.0],  # all negative
            [1.0, 1.0, 1.0],  # all positive
            [1.0, 0.0],       # mixed
        ]
        egr = allocator.get_effective_gradient_ratio(rewards)
        assert egr["effective_ratio"] == 0.5
        assert egr["all_positive_ratio"] == 0.25
        assert egr["all_negative_ratio"] == 0.25
