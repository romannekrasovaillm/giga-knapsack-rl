"""Tests for GRPO advantage computation and DAPO sampling."""

import torch
import numpy as np
import pytest

from src.grpo.advantage import knapsack_grpo_advantage
from src.grpo.dapo_sampling import dapo_importance_weights, dynamic_temperature
from src.grpo.policy_loss import knapsack_grpo_loss


class TestAdvantage:
    def test_basic_advantage(self):
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
        group_indices = [[0, 1, 2], [3, 4, 5]]
        response_masks = torch.ones(6, 1)

        advantages, returns, metrics = knapsack_grpo_advantage(
            rewards=rewards,
            group_indices=group_indices,
            response_masks=response_masks,
        )

        assert advantages.shape == (6, 1)
        assert metrics["num_groups"] == 2
        assert metrics["effective_gradient_ratio"] == 1.0

    def test_all_same_rewards(self):
        rewards = torch.tensor([1.0, 1.0, 1.0])
        group_indices = [[0, 1, 2]]
        response_masks = torch.ones(3, 1)

        advantages, _, metrics = knapsack_grpo_advantage(
            rewards=rewards,
            group_indices=group_indices,
            response_masks=response_masks,
        )
        # All same -> advantages should be near zero
        assert advantages.abs().max() < 1.0

    def test_clipping(self):
        rewards = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        group_indices = [[0, 1, 2, 3, 4, 5, 6, 7]]
        response_masks = torch.ones(8, 1)

        advantages, _, _ = knapsack_grpo_advantage(
            rewards=rewards,
            group_indices=group_indices,
            response_masks=response_masks,
            adv_clip=5.0,
        )
        assert advantages.max() <= 5.0
        assert advantages.min() >= -5.0

    def test_variable_group_sizes(self):
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        group_indices = [[0, 1], [2, 3, 4, 5, 6, 7], [8, 9]]
        response_masks = torch.ones(10, 1)

        advantages, _, metrics = knapsack_grpo_advantage(
            rewards=rewards,
            group_indices=group_indices,
            response_masks=response_masks,
        )
        assert metrics["num_groups"] == 3
        assert metrics["total_rollouts"] == 10


class TestDAPO:
    def test_importance_weights(self):
        log_probs_new = torch.randn(4, 10)
        log_probs_old = log_probs_new.clone()
        response_mask = torch.ones(4, 10)

        weights, metrics = dapo_importance_weights(
            log_probs_new, log_probs_old, response_mask,
        )
        # When new == old, ratio should be ~1.0
        assert metrics["is_ratio_mean"] == pytest.approx(1.0, abs=0.1)

    def test_asymmetric_clipping(self):
        # Large positive shift
        log_probs_new = torch.zeros(1, 5)
        log_probs_old = torch.full((1, 5), -2.0)
        response_mask = torch.ones(1, 5)

        weights, metrics = dapo_importance_weights(
            log_probs_new, log_probs_old, response_mask,
            clip_low=0.8, clip_high=1.2,
        )
        # Should be clipped
        assert metrics["clip_fraction_high"] > 0

    def test_dynamic_temperature(self):
        # Low variance rewards -> higher temperature
        low_var = torch.tensor([0.5, 0.5, 0.5, 0.5])
        high_var = torch.tensor([0.0, 1.0, 0.0, 1.0])

        temp_low = dynamic_temperature(low_var)
        temp_high = dynamic_temperature(high_var)
        assert temp_low > temp_high


class TestPolicyLoss:
    def test_basic_loss(self):
        batch_size, seq_len = 4, 10
        log_probs = torch.randn(batch_size, seq_len)
        old_log_probs = log_probs.clone()
        advantages = torch.randn(batch_size, seq_len)
        response_mask = torch.ones(batch_size, seq_len)

        loss, metrics = knapsack_grpo_loss(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            response_mask=response_mask,
        )
        assert loss.dim() == 0  # scalar
        assert "pg_loss" in metrics
        assert "total_loss" in metrics
