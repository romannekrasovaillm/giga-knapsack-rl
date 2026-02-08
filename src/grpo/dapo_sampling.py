"""
DAPO-style importance sampling with asymmetric exploration bias.

Key ideas from DAPO (Decoupled Alignment via Preference Optimization):
  - Asymmetric clipping: exploration (positive) side has wider clip than exploitation (negative)
  - Token-level KL penalty instead of sequence-level
  - Dynamic sampling temperature based on reward variance

Integrated with Knapsack RL for budget-aware importance sampling.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def dapo_importance_weights(
    log_probs_new: torch.Tensor,       # [batch, seq_len] from current policy
    log_probs_old: torch.Tensor,       # [batch, seq_len] from rollout policy
    response_mask: torch.Tensor,       # [batch, seq_len]
    clip_low: float = 0.8,            # tighter clip for exploitation
    clip_high: float = 1.2,           # wider clip for exploration (asymmetric!)
    exploration_bias: float = 0.05,   # additive bias toward exploration
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute DAPO-style importance weights with asymmetric clipping.

    The key insight: clip_high > 1/clip_low to encourage exploration.
    Standard PPO uses symmetric clipping (e.g., [0.8, 1.2]).
    DAPO uses asymmetric: wider on the positive (exploration) side.

    Args:
        log_probs_new: log probs under current policy
        log_probs_old: log probs under policy that generated rollouts
        response_mask: binary mask for response tokens
        clip_low: lower clip for importance ratio (exploitation)
        clip_high: upper clip for importance ratio (exploration)
        exploration_bias: additive bias to encourage exploration

    Returns:
        clipped_ratios: [batch, seq_len] clipped importance weights
        metrics: dict with IS statistics
    """
    # Per-token importance ratios
    log_ratio = log_probs_new - log_probs_old
    ratio = torch.exp(log_ratio)

    # Asymmetric clipping
    clipped = torch.where(
        ratio > 1.0,
        torch.clamp(ratio, max=clip_high),     # exploration side: wider
        torch.clamp(ratio, min=clip_low),      # exploitation side: tighter
    )

    # Exploration bias: slightly increase weight of all tokens
    clipped = clipped + exploration_bias

    # Mask
    clipped = clipped * response_mask

    # Metrics
    with torch.no_grad():
        masked_ratio = ratio[response_mask.bool()]
        metrics = {
            "is_ratio_mean": masked_ratio.mean().item() if masked_ratio.numel() > 0 else 0.0,
            "is_ratio_std": masked_ratio.std().item() if masked_ratio.numel() > 0 else 0.0,
            "is_ratio_max": masked_ratio.max().item() if masked_ratio.numel() > 0 else 0.0,
            "is_ratio_min": masked_ratio.min().item() if masked_ratio.numel() > 0 else 0.0,
            "clip_fraction_low": (ratio < clip_low).float().mean().item(),
            "clip_fraction_high": (ratio > clip_high).float().mean().item(),
            "exploration_ratio": (ratio > 1.0).float().mean().item(),
        }

    return clipped, metrics


def dapo_policy_loss(
    log_probs_new: torch.Tensor,       # [batch, seq_len]
    log_probs_old: torch.Tensor,       # [batch, seq_len]
    advantages: torch.Tensor,          # [batch, seq_len]
    response_mask: torch.Tensor,       # [batch, seq_len]
    clip_low: float = 0.8,
    clip_high: float = 1.28,          # asymmetric: wider for exploration
    exploration_bias: float = 0.05,
    entropy_coef: float = 0.01,
    logits: Optional[torch.Tensor] = None,  # [batch, seq_len, vocab] for entropy
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    DAPO-style policy gradient loss with asymmetric clipping.

    Loss = -E[ min(ratio * A, clip(ratio) * A) ] - beta * H(pi)

    The asymmetry: clip_high > 1/(1-clip_low) biases toward exploration.
    When advantage > 0 (good action), the wider upper clip allows
    larger policy updates. When advantage < 0 (bad action), the tighter
    lower clip prevents excessive penalization.
    """
    # Importance ratios
    log_ratio = log_probs_new - log_probs_old
    ratio = torch.exp(log_ratio)

    # Asymmetric clipping
    clipped_ratio = torch.where(
        advantages > 0,
        torch.clamp(ratio, min=clip_low, max=clip_high),     # positive adv: explore
        torch.clamp(ratio, min=clip_low, max=1.0 / clip_low), # negative adv: standard
    )

    # Add exploration bias to positive advantages
    if exploration_bias > 0:
        pos_mask = (advantages > 0).float()
        ratio_biased = ratio + exploration_bias * pos_mask
    else:
        ratio_biased = ratio

    # PPO-clip loss
    surr1 = ratio_biased * advantages
    surr2 = clipped_ratio * advantages
    policy_loss = -torch.min(surr1, surr2)

    # Apply response mask
    policy_loss = (policy_loss * response_mask).sum() / (response_mask.sum() + 1e-8)

    # Entropy bonus
    entropy_loss = torch.tensor(0.0, device=policy_loss.device)
    entropy_val = 0.0
    if entropy_coef > 0 and logits is not None:
        # Token-level entropy
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1)  # [batch, seq_len]
        entropy_masked = (entropy * response_mask).sum() / (response_mask.sum() + 1e-8)
        entropy_loss = -entropy_coef * entropy_masked
        entropy_val = entropy_masked.item()

    total_loss = policy_loss + entropy_loss

    metrics = {
        "policy_loss": policy_loss.item(),
        "entropy_loss": entropy_loss.item(),
        "entropy": entropy_val,
        "total_loss": total_loss.item(),
        "mean_ratio": ratio[response_mask.bool()].mean().item() if response_mask.any() else 0.0,
        "clip_fraction": (
            ((ratio > clip_high) | (ratio < clip_low)).float() * response_mask
        ).sum().item() / (response_mask.sum().item() + 1e-8),
    }

    return total_loss, metrics


def dynamic_temperature(
    rewards: torch.Tensor,
    base_temp: float = 1.0,
    reward_std_target: float = 0.3,
    temp_range: Tuple[float, float] = (0.7, 1.5),
) -> float:
    """
    Dynamic sampling temperature based on reward variance.

    If rewards have low variance (all same), increase temperature to explore more.
    If rewards have high variance, decrease temperature to exploit.
    """
    reward_std = rewards.std().item()
    if reward_std < 1e-8:
        return temp_range[1]

    ratio = reward_std_target / reward_std
    temp = base_temp * ratio
    return max(temp_range[0], min(temp_range[1], temp))
