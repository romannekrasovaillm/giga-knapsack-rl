"""
Custom policy loss combining Knapsack-GRPO with DAPO-style exploration.

Registers with verl's policy loss registry for seamless integration.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def knapsack_grpo_loss(
    log_probs: torch.Tensor,           # [batch, seq_len] new policy log probs
    old_log_probs: torch.Tensor,       # [batch, seq_len] old policy log probs
    advantages: torch.Tensor,          # [batch, seq_len] advantages
    response_mask: torch.Tensor,       # [batch, seq_len]
    clip_ratio: float = 0.2,
    clip_ratio_high: float = 0.28,     # asymmetric upper clip
    entropy_coef: float = 0.01,
    kl_coef: float = 0.001,
    ref_log_probs: Optional[torch.Tensor] = None,  # [batch, seq_len] reference
    logits: Optional[torch.Tensor] = None,          # for entropy computation
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Knapsack-GRPO policy loss with DAPO-style asymmetric clipping.

    Combines:
      1. PPO-clip objective with asymmetric bounds
      2. Token-level KL penalty against reference policy
      3. Entropy bonus for exploration
    """
    # Importance ratios
    log_ratio = log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    # Asymmetric clipping
    clip_low = 1.0 - clip_ratio
    clip_high = 1.0 + clip_ratio_high

    clipped_ratio = torch.where(
        advantages > 0,
        torch.clamp(ratio, min=clip_low, max=clip_high),
        torch.clamp(ratio, min=clip_low, max=1.0 + clip_ratio),
    )

    # Surrogate losses
    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages
    pg_loss = -torch.min(surr1, surr2)

    # Apply mask and aggregate (token-mean like DrGRPO)
    pg_loss = (pg_loss * response_mask).sum() / (response_mask.sum() + 1e-8)

    # KL penalty
    kl_loss = torch.tensor(0.0, device=pg_loss.device)
    kl_val = 0.0
    if kl_coef > 0 and ref_log_probs is not None:
        # Token-level KL: sum_t KL(pi || pi_ref) for each response token
        kl = log_probs - ref_log_probs  # approximate KL
        kl_per_token = kl * response_mask
        kl_loss = kl_coef * kl_per_token.sum() / (response_mask.sum() + 1e-8)
        kl_val = kl_per_token.sum().item() / (response_mask.sum().item() + 1e-8)

    # Entropy bonus
    entropy_loss = torch.tensor(0.0, device=pg_loss.device)
    entropy_val = 0.0
    if entropy_coef > 0 and logits is not None:
        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)
        ent = -(p * log_p).sum(dim=-1)
        entropy_masked = (ent * response_mask).sum() / (response_mask.sum() + 1e-8)
        entropy_loss = -entropy_coef * entropy_masked
        entropy_val = entropy_masked.item()

    total_loss = pg_loss + kl_loss + entropy_loss

    metrics = {
        "pg_loss": pg_loss.item(),
        "kl_loss": kl_loss.item(),
        "kl_value": kl_val,
        "entropy_loss": entropy_loss.item(),
        "entropy": entropy_val,
        "total_loss": total_loss.item(),
        "approx_kl": (0.5 * (log_ratio ** 2) * response_mask).sum().item() / (response_mask.sum().item() + 1e-8),
        "clip_fraction": (
            ((ratio > clip_high) | (ratio < clip_low)).float() * response_mask
        ).sum().item() / (response_mask.sum().item() + 1e-8),
        "mean_ratio": ratio[response_mask.bool()].mean().item() if response_mask.any() else 0.0,
    }

    return total_loss, metrics
