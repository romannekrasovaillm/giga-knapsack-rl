"""
Reward manager integrating with verl's reward system.

Computes rewards for GRPO training:
  - Binary reward from HardVerifier (correct/incorrect)
  - Token-level scores for the policy gradient
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional

import torch
import numpy as np

from .verifier import HardVerifier

logger = logging.getLogger(__name__)


class KnapsackRewardManager:
    """
    Reward manager for Knapsack-GRPO training.

    Integrates HardVerifier with verl's reward interface.
    Returns binary rewards: 1.0 for correct, 0.0 for incorrect.
    """

    def __init__(
        self,
        verifier: Optional[HardVerifier] = None,
        reward_correct: float = 1.0,
        reward_incorrect: float = 0.0,
        format_penalty: float = -0.1,
        length_penalty_threshold: int = 4096,
        length_penalty_coef: float = 0.001,
    ):
        self.verifier = verifier or HardVerifier()
        self.reward_correct = reward_correct
        self.reward_incorrect = reward_incorrect
        self.format_penalty = format_penalty
        self.length_penalty_threshold = length_penalty_threshold
        self.length_penalty_coef = length_penalty_coef
        self._stats = {
            "total": 0,
            "correct": 0,
            "match_types": {},
        }

    def compute_rewards(
        self,
        responses: List[str],
        ground_truths: List[str],
        tool_calls_preds: Optional[List[List[Dict]]] = None,
        tool_calls_gts: Optional[List[List[Dict]]] = None,
        response_lengths: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Compute rewards for a batch of responses.

        Returns:
            dict with:
              - 'rewards': List[float], binary rewards
              - 'verification_results': List[dict], detailed results
              - 'batch_stats': dict with batch-level statistics
        """
        t0 = time.time()
        results = self.verifier.compute_batch_reward(
            responses, ground_truths, tool_calls_preds, tool_calls_gts,
        )

        rewards = []
        for i, result in enumerate(results):
            base_reward = (
                self.reward_correct if result["reward"] > 0.5
                else self.reward_incorrect
            )

            # Format penalty: check if response has proper structure
            resp = responses[i]
            if self._has_format_issues(resp):
                base_reward += self.format_penalty

            # Length penalty
            if response_lengths and response_lengths[i] > self.length_penalty_threshold:
                excess = response_lengths[i] - self.length_penalty_threshold
                base_reward -= self.length_penalty_coef * excess

            # Clamp to [0, 1]
            reward = max(0.0, min(1.0, base_reward))
            rewards.append(reward)

        # Update stats
        self._stats["total"] += len(rewards)
        self._stats["correct"] += sum(1 for r in rewards if r > 0.5)
        for result in results:
            mt = result.get("match_type", "none")
            self._stats["match_types"][mt] = self._stats["match_types"].get(mt, 0) + 1

        elapsed = time.time() - t0

        # Batch stats
        batch_stats = {
            "batch_size": len(rewards),
            "mean_reward": np.mean(rewards),
            "success_rate": sum(1 for r in rewards if r > 0.5) / len(rewards),
            "match_type_distribution": {
                r["match_type"]: sum(1 for r2 in results if r2["match_type"] == r["match_type"])
                for r in results
            },
            "verification_time": elapsed,
        }

        return {
            "rewards": rewards,
            "verification_results": results,
            "batch_stats": batch_stats,
        }

    def _has_format_issues(self, response: str) -> bool:
        """Check if response has formatting issues."""
        # Unmatched tool_call tags
        open_tags = response.count("<tool_call>")
        close_tags = response.count("</tool_call>")
        if open_tags != close_tags:
            return True
        return False

    def compute_token_level_rewards(
        self,
        rewards: List[float],
        response_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert per-sequence rewards to token-level rewards.

        For GRPO, the reward is placed on the last token of the response.
        """
        batch_size, seq_len = response_masks.shape
        token_rewards = torch.zeros_like(response_masks, dtype=torch.float32)

        for i in range(batch_size):
            # Find last token in response
            mask = response_masks[i]
            last_idx = mask.sum().long() - 1
            if last_idx >= 0:
                token_rewards[i, last_idx] = rewards[i]

        return token_rewards

    def get_stats(self) -> Dict[str, Any]:
        """Get cumulative statistics."""
        total = self._stats["total"]
        return {
            "total_verified": total,
            "overall_success_rate": self._stats["correct"] / total if total > 0 else 0.0,
            "match_type_distribution": self._stats["match_types"],
        }

    def reset_stats(self):
        """Reset cumulative statistics."""
        self._stats = {"total": 0, "correct": 0, "match_types": {}}
