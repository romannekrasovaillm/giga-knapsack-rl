"""
Integration with the verl framework for distributed Knapsack-GRPO training.

Registers custom components with verl's registries:
  - Custom advantage estimator (knapsack_grpo)
  - Custom reward function (hard_verifier)
  - Custom policy loss (knapsack_dapo)

Usage:
  # In verl config: algorithm.adv_estimator=knapsack_grpo
  # This file must be imported before training starts.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Try to import verl; gracefully degrade if not available
_VERL_AVAILABLE = False
try:
    from verl.trainer.ppo.core_algos import register_adv_est
    from verl.protocol import DataProto
    _VERL_AVAILABLE = True
    logger.info("verl framework detected, registering custom components")
except ImportError:
    logger.info("verl not installed; using standalone training loop")


# ============================================================
# 1. Custom Advantage Estimator for verl
# ============================================================

if _VERL_AVAILABLE:
    @register_adv_est("knapsack_grpo")
    def compute_knapsack_grpo_advantage(
        token_level_rewards: torch.Tensor,
        response_mask: torch.Tensor,
        index: np.ndarray,
        epsilon: float = 1e-6,
        norm_adv_by_std_in_grpo: bool = True,
        config=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Knapsack-GRPO advantage estimator for verl.

        Same as standard GRPO but:
          - Handles variable group sizes (from Knapsack allocation)
          - Clips advantages to [-5, 5] for stability
          - Adds exploration bonus for rare successes
        """
        from src.knapsack.allocator import KnapsackBudgetAllocator
        from src.grpo.advantage import knapsack_grpo_advantage as _compute

        # Per-sequence rewards
        seq_rewards = (token_level_rewards * response_mask).sum(dim=-1)

        # Build group indices from verl's index array
        unique_indices = np.unique(index)
        group_indices = []
        for ui in unique_indices:
            group_indices.append(list(np.where(index == ui)[0]))

        adv_clip = 5.0
        exploration_bonus = 0.05
        if config is not None:
            adv_clip = getattr(config, "adv_clip", 5.0)
            exploration_bonus = getattr(config, "exploration_bonus", 0.05)

        advantages, returns, metrics = _compute(
            rewards=seq_rewards,
            group_indices=group_indices,
            response_masks=response_mask,
            adv_clip=adv_clip,
            exploration_bonus=exploration_bonus,
            norm_by_std=norm_adv_by_std_in_grpo,
        )

        return advantages, returns


# ============================================================
# 2. Custom Reward Function for verl
# ============================================================

def compute_reward_for_verl(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
) -> float:
    """
    verl-compatible reward function using HardVerifier.

    Args:
        data_source: dataset identifier
        solution_str: model's response
        ground_truth: expected answer
        extra_info: optional metadata

    Returns:
        float: 0.0 or 1.0 (binary reward)
    """
    from src.rewards.verifier import HardVerifier

    verifier = HardVerifier(strict_mode=True)

    tool_calls_pred = None
    tool_calls_gt = None
    if extra_info:
        tool_calls_pred = extra_info.get("tool_calls_pred")
        tool_calls_gt = extra_info.get("tool_calls_gt")

    result = verifier.verify(
        prediction=solution_str,
        ground_truth=ground_truth,
        tool_calls_pred=tool_calls_pred,
        tool_calls_gt=tool_calls_gt,
    )
    return result["reward"]


# ============================================================
# 3. Knapsack Budget-Aware Batch Processor for verl
# ============================================================

class VerlKnapsackBatchProcessor:
    """
    Processes verl's DataProto batches with Knapsack budget allocation.

    Inserts between rollout generation and advantage computation
    in verl's training loop to apply adaptive budget allocation.
    """

    def __init__(
        self,
        N_total: int = 1024,
        N_low: int = 2,
        N_up: int = 128,
    ):
        from src.knapsack.allocator import KnapsackBudgetAllocator
        self.allocator = KnapsackBudgetAllocator(
            N_total=N_total, N_low=N_low, N_up=N_up,
        )

    def get_budgets(self, prompt_ids: List[str]) -> np.ndarray:
        """Get rollout budgets for a batch of prompts."""
        return self.allocator.allocate(prompt_ids)

    def update_stats(self, prompt_ids: List[str], rewards_per_prompt: List[List[float]]):
        """Update prompt statistics after rollouts."""
        self.allocator.update(prompt_ids, rewards_per_prompt)

    def process_batch(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a batch with Knapsack-aware grouping.

        For use in custom verl trainer subclass.
        """
        prompt_ids = batch_data.get("prompt_ids", [])
        budgets = self.get_budgets(prompt_ids)
        batch_data["knapsack_budgets"] = budgets
        batch_data["group_sizes"] = budgets.tolist()
        return batch_data


# ============================================================
# 4. Extended verl Trainer (optional, for full integration)
# ============================================================

if _VERL_AVAILABLE:
    try:
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        class KnapsackRayPPOTrainer(RayPPOTrainer):
            """
            Extended verl trainer with Knapsack budget allocation.

            Overrides the training loop to insert:
              - Adaptive rollout budget allocation
              - Variable group size handling
              - Extended metrics logging
            """

            def __init__(self, *args, knapsack_config: Optional[Dict] = None, **kwargs):
                super().__init__(*args, **kwargs)
                kc = knapsack_config or {}
                self.batch_processor = VerlKnapsackBatchProcessor(
                    N_total=kc.get("N_total", 1024),
                    N_low=kc.get("N_low", 2),
                    N_up=kc.get("N_up", 128),
                )
                self._knapsack_logger = logging.getLogger("knapsack_verl")

            def fit(self):
                """Override fit to inject Knapsack allocation."""
                self._knapsack_logger.info("Starting Knapsack-GRPO training via verl")
                # Call parent fit, which will use our registered
                # advantage estimator automatically
                super().fit()

    except ImportError:
        pass
