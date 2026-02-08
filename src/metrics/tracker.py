"""
Comprehensive metrics tracker for Knapsack-GRPO training.

Tracks:
  - BLEU scores
  - Entropy (per rollout, per group, per batch)
  - Token counts
  - Generation times
  - Reward statistics
  - Advantage statistics
  - Knapsack allocation stats
"""

import logging
import time
from collections import defaultdict
from typing import Dict, List, Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def compute_bleu_score(prediction: str, reference: str) -> float:
    """Compute BLEU score between prediction and reference."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ref_tokens = reference.lower().split()
        pred_tokens = prediction.lower().split()
        if not ref_tokens or not pred_tokens:
            return 0.0
        smoothie = SmoothingFunction().method1
        return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
    except ImportError:
        # Fallback: simple unigram overlap
        ref_set = set(reference.lower().split())
        pred_set = set(prediction.lower().split())
        if not ref_set:
            return 0.0
        return len(ref_set & pred_set) / len(ref_set)


def compute_token_entropy(logits: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Compute token-level entropy from logits."""
    if logits.dim() == 3:
        # [batch, seq_len, vocab]
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1)  # [batch, seq_len]

        if mask is not None:
            masked_entropy = entropy * mask
            per_seq_entropy = masked_entropy.sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)
            batch_entropy = masked_entropy.sum() / (mask.sum() + 1e-8)
        else:
            per_seq_entropy = entropy.mean(dim=-1)
            batch_entropy = entropy.mean()

        return {
            "mean_entropy": batch_entropy.item(),
            "per_seq_entropy": per_seq_entropy.detach().cpu().numpy().tolist(),
            "min_entropy": per_seq_entropy.min().item(),
            "max_entropy": per_seq_entropy.max().item(),
            "std_entropy": per_seq_entropy.std().item(),
        }
    return {"mean_entropy": 0.0}


class MetricsTracker:
    """
    Tracks all training metrics across iterations.

    Provides:
      - Per-rollout metrics
      - Per-group metrics (aggregated over rollouts within a group)
      - Per-batch metrics (aggregated over all groups)
      - Running averages across iterations
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.iteration = 0
        self._history: Dict[str, List[float]] = defaultdict(list)
        self._iteration_data: List[Dict[str, Any]] = []

    def record_generation(
        self,
        prompt_ids: List[str],
        responses: List[str],
        ground_truths: List[str],
        rewards: List[float],
        group_sizes: List[int],
        generation_times: List[float],
        token_counts: List[int],
        advantages: Optional[List[float]] = None,
        logits_entropy: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Record metrics for one batch of generations.

        Returns a comprehensive metrics dict.
        """
        self.iteration += 1
        n_prompts = len(prompt_ids)
        n_rollouts = len(responses)

        # ---- BLEU scores ----
        bleu_scores = []
        rollout_idx = 0
        for i, gs in enumerate(group_sizes):
            gt = ground_truths[i] if i < len(ground_truths) else ""
            for j in range(gs):
                if rollout_idx < len(responses):
                    bleu = compute_bleu_score(responses[rollout_idx], gt)
                    bleu_scores.append(bleu)
                rollout_idx += 1

        # ---- Per-group metrics ----
        group_metrics = []
        rollout_idx = 0
        for i, gs in enumerate(group_sizes):
            group_rewards = rewards[rollout_idx:rollout_idx + gs]
            group_bleu = bleu_scores[rollout_idx:rollout_idx + gs] if bleu_scores else []
            group_tokens = token_counts[rollout_idx:rollout_idx + gs] if token_counts else []
            group_times = generation_times[rollout_idx:rollout_idx + gs] if generation_times else []

            group_advs = []
            if advantages:
                group_advs = advantages[rollout_idx:rollout_idx + gs]

            group_entropy = []
            if logits_entropy:
                group_entropy = logits_entropy[rollout_idx:rollout_idx + gs]

            gm = {
                "prompt_id": prompt_ids[i] if i < len(prompt_ids) else f"prompt_{i}",
                "group_size": gs,
                "rewards": group_rewards,
                "mean_reward": np.mean(group_rewards) if group_rewards else 0.0,
                "success_rate": sum(1 for r in group_rewards if r > 0.5) / len(group_rewards) if group_rewards else 0.0,
                "mean_bleu": np.mean(group_bleu) if group_bleu else 0.0,
                "mean_tokens": np.mean(group_tokens) if group_tokens else 0.0,
                "total_tokens": sum(group_tokens) if group_tokens else 0,
                "mean_gen_time": np.mean(group_times) if group_times else 0.0,
                "total_gen_time": sum(group_times) if group_times else 0.0,
            }

            if group_advs:
                gm["mean_advantage"] = np.mean(group_advs)
                gm["std_advantage"] = np.std(group_advs)
                gm["max_advantage"] = max(group_advs)
                gm["min_advantage"] = min(group_advs)

            if group_entropy:
                gm["mean_entropy"] = np.mean(group_entropy)
                gm["std_entropy"] = np.std(group_entropy)

            group_metrics.append(gm)
            rollout_idx += gs

        # ---- Batch-level metrics ----
        batch = {
            "iteration": self.iteration,
            "num_prompts": n_prompts,
            "num_rollouts": n_rollouts,
            "mean_group_size": np.mean(group_sizes) if group_sizes else 0.0,
            "min_group_size": min(group_sizes) if group_sizes else 0,
            "max_group_size": max(group_sizes) if group_sizes else 0,
            # Rewards
            "batch_mean_reward": np.mean(rewards) if rewards else 0.0,
            "batch_std_reward": np.std(rewards) if rewards else 0.0,
            "batch_success_rate": sum(1 for r in rewards if r > 0.5) / len(rewards) if rewards else 0.0,
            # BLEU
            "batch_mean_bleu": np.mean(bleu_scores) if bleu_scores else 0.0,
            "batch_std_bleu": np.std(bleu_scores) if bleu_scores else 0.0,
            # Tokens
            "batch_total_tokens": sum(token_counts) if token_counts else 0,
            "batch_mean_tokens": np.mean(token_counts) if token_counts else 0.0,
            # Time
            "batch_total_gen_time": sum(generation_times) if generation_times else 0.0,
            "batch_mean_gen_time": np.mean(generation_times) if generation_times else 0.0,
            # Effective gradient
            "effective_gradient_ratio": sum(
                1 for gm in group_metrics if 0.0 < gm["success_rate"] < 1.0
            ) / len(group_metrics) if group_metrics else 0.0,
            "all_positive_groups": sum(
                1 for gm in group_metrics if gm["success_rate"] == 1.0
            ),
            "all_negative_groups": sum(
                1 for gm in group_metrics if gm["success_rate"] == 0.0
            ),
            # Per-group details
            "groups": group_metrics,
        }

        if advantages:
            batch["batch_mean_advantage"] = np.mean(advantages)
            batch["batch_std_advantage"] = np.std(advantages)

        if logits_entropy:
            batch["batch_mean_entropy"] = np.mean(logits_entropy)
            batch["batch_std_entropy"] = np.std(logits_entropy)

        # Update history
        for key in ["batch_mean_reward", "batch_success_rate", "batch_mean_bleu",
                     "effective_gradient_ratio", "batch_total_tokens"]:
            self._history[key].append(batch.get(key, 0.0))

        self._iteration_data.append(batch)
        return batch

    def record_loss(self, loss_metrics: Dict[str, float]):
        """Record loss-related metrics."""
        for key, val in loss_metrics.items():
            self._history[f"loss/{key}"].append(val)

    def record_allocation(self, alloc_metrics: Dict[str, Any]):
        """Record knapsack allocation metrics."""
        for key, val in alloc_metrics.items():
            if isinstance(val, (int, float)):
                self._history[f"alloc/{key}"].append(val)

    def get_running_average(self, key: str) -> float:
        """Get running average of a metric over the window."""
        if key not in self._history or not self._history[key]:
            return 0.0
        window = self._history[key][-self.window_size:]
        return np.mean(window)

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Get the latest iteration data."""
        return self._iteration_data[-1] if self._iteration_data else None

    def get_summary(self) -> Dict[str, float]:
        """Get summary of all tracked metrics."""
        summary = {}
        for key, values in self._history.items():
            if values:
                summary[f"{key}/latest"] = values[-1]
                summary[f"{key}/mean_{self.window_size}"] = np.mean(values[-self.window_size:])
        return summary
