"""
Custom advantage estimator for Knapsack-GRPO.

Computes GRPO-style group-normalized advantages with:
  - Variable group sizes (from Knapsack allocation)
  - Advantage clipping (from paper: [-5, 5])
  - DAPO-style asymmetric exploration bias
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

logger = logging.getLogger(__name__)


def knapsack_grpo_advantage(
    rewards: torch.Tensor,             # [total_rollouts]
    group_indices: List[List[int]],    # groups of rollout indices per prompt
    response_masks: torch.Tensor,      # [total_rollouts, seq_len]
    adv_clip: float = 5.0,
    exploration_bonus: float = 0.1,
    norm_by_std: bool = True,
    token_level: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Compute group-normalized advantages for variable-size groups.

    Key difference from standard GRPO:
      - Groups can have different sizes (from Knapsack allocation)
      - Advantages are clipped to [-adv_clip, adv_clip] for stability
      - Optional exploration bonus for rare successes

    Args:
        rewards: [total_rollouts] per-sequence rewards
        group_indices: list of lists, each inner list has rollout indices for a prompt
        response_masks: [total_rollouts, seq_len] binary masks
        adv_clip: advantage clipping range
        exploration_bonus: extra reward for rare successes (p < 0.15)
        norm_by_std: normalize by group std
        token_level: if True, broadcast advantage to all tokens

    Returns:
        advantages: [total_rollouts, seq_len] token-level advantages
        returns: [total_rollouts] per-sequence returns
        metrics: dict with per-group and batch metrics
    """
    device = rewards.device
    total = rewards.shape[0]
    seq_len = response_masks.shape[1] if response_masks.dim() > 1 else 1

    advantages = torch.zeros(total, seq_len, device=device)
    returns = rewards.clone()

    group_metrics = []

    for group_idx, indices in enumerate(group_indices):
        if not indices:
            continue

        idx = torch.tensor(indices, device=device, dtype=torch.long)
        group_rewards = rewards[idx]
        group_size = len(indices)

        # Group statistics
        mean_r = group_rewards.mean()
        std_r = group_rewards.std() + 1e-8

        # Success rate in this group
        success_rate = (group_rewards > 0.5).float().mean().item()

        # GRPO advantage: (r - mean) / std
        if norm_by_std:
            group_adv = (group_rewards - mean_r) / std_r
        else:
            group_adv = group_rewards - mean_r

        # Exploration bonus: boost rare successes
        if exploration_bonus > 0 and success_rate < 0.15:
            success_mask = group_rewards > 0.5
            group_adv[success_mask] += exploration_bonus

        # Clip advantages
        group_adv = torch.clamp(group_adv, -adv_clip, adv_clip)

        # Broadcast to token level
        if token_level and seq_len > 1:
            group_mask = response_masks[idx]  # [group_size, seq_len]
            for j, i in enumerate(indices):
                advantages[i] = group_adv[j] * response_masks[i]
        else:
            for j, i in enumerate(indices):
                advantages[i, :] = group_adv[j]

        group_metrics.append({
            "group_idx": group_idx,
            "group_size": group_size,
            "mean_reward": mean_r.item(),
            "std_reward": std_r.item(),
            "success_rate": success_rate,
            "mean_advantage": group_adv.mean().item(),
            "std_advantage": group_adv.std().item(),
            "max_advantage": group_adv.max().item(),
            "min_advantage": group_adv.min().item(),
        })

    # Batch-level metrics
    batch_metrics = {
        "num_groups": len(group_indices),
        "total_rollouts": total,
        "mean_group_size": np.mean([len(g) for g in group_indices]) if group_indices else 0,
        "batch_mean_reward": rewards.mean().item(),
        "batch_std_reward": rewards.std().item(),
        "batch_success_rate": (rewards > 0.5).float().mean().item(),
        "batch_mean_advantage": advantages[response_masks.bool()].mean().item() if response_masks.any() else 0.0,
        "batch_std_advantage": advantages[response_masks.bool()].std().item() if response_masks.any() else 0.0,
        "groups_with_zero_gradient": sum(
            1 for m in group_metrics
            if m["success_rate"] == 0.0 or m["success_rate"] == 1.0
        ),
        "effective_gradient_ratio": sum(
            1 for m in group_metrics
            if 0.0 < m["success_rate"] < 1.0
        ) / len(group_metrics) if group_metrics else 0.0,
        "per_group": group_metrics,
    }

    return advantages, returns, batch_metrics


def compute_advantages_for_verl(
    token_level_rewards: torch.Tensor,   # [batch, seq_len]
    response_mask: torch.Tensor,         # [batch, seq_len]
    group_sizes: List[int],              # rollouts per prompt
    adv_clip: float = 5.0,
    exploration_bonus: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Adapter for verl's advantage interface.

    Converts verl's flat batch format to group-indexed format,
    computes knapsack-GRPO advantages, and returns in verl format.
    """
    batch_size = token_level_rewards.shape[0]

    # Compute per-sequence rewards (sum of token-level)
    seq_rewards = (token_level_rewards * response_mask).sum(dim=-1)

    # Build group indices from group_sizes
    group_indices = []
    offset = 0
    for gs in group_sizes:
        group_indices.append(list(range(offset, offset + gs)))
        offset += gs

    # Compute advantages
    advantages, returns, metrics = knapsack_grpo_advantage(
        rewards=seq_rewards,
        group_indices=group_indices,
        response_masks=response_mask,
        adv_clip=adv_clip,
        exploration_bonus=exploration_bonus,
    )

    return advantages, returns
