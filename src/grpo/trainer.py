"""
Knapsack-GRPO Trainer.

Main training loop integrating:
  - Knapsack RL budget allocation
  - GRPO with variable group sizes
  - DAPO-style asymmetric importance sampling
  - Simulated tool environment for multi-turn rollouts
  - Hard programmatic verification
  - Comprehensive metrics logging
"""

import gc
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from src.knapsack.allocator import KnapsackBudgetAllocator
from src.data.loader import NemotronAgenticLoader
from src.data.rlvr_dataset import RLVRAgenticDataset
from src.environment.tool_env import SimulatedToolEnvironment
from src.rewards.verifier import HardVerifier
from src.rewards.reward_manager import KnapsackRewardManager
from src.grpo.advantage import knapsack_grpo_advantage
from src.grpo.dapo_sampling import dapo_policy_loss, dynamic_temperature
from src.grpo.policy_loss import knapsack_grpo_loss
from src.metrics.tracker import MetricsTracker, compute_token_entropy
from src.metrics.logger import TrainingLogger
from src.utils import get_attn_implementation, get_torch_dtype

logger = logging.getLogger(__name__)


class KnapsackGRPOTrainer:
    """
    Full Knapsack-GRPO training loop.

    Workflow per iteration:
      1. Sample batch of prompts
      2. Allocate rollout budgets via Knapsack DP
      3. Generate rollouts (variable N_i per prompt)
      4. Run through simulated tool environment (multi-turn)
      5. Verify answers with hard verifier -> binary rewards
      6. Compute GRPO advantages (group-normalized, clipped)
      7. Compute DAPO-style policy loss
      8. Update policy
      9. Log comprehensive metrics
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get("device", "cuda"))

        # Model config
        self.model_path = config.get("model_path", "./checkpoints/sft/final")
        self.ref_model_path = config.get("ref_model_path", self.model_path)
        self.max_prompt_length = config.get("max_prompt_length", 2048)
        self.max_response_length = config.get("max_response_length", 2048)

        # Knapsack config
        self.N_total = config.get("N_total", 1024)
        self.N_low = config.get("N_low", 2)
        self.N_up = config.get("N_up", 128)
        self.warmup_iterations = config.get("knapsack_warmup", 5)

        # Training config
        self.num_iterations = config.get("num_iterations", 1000)
        self.batch_size = config.get("batch_size", 16)  # prompts per batch
        self.mini_batch_size = config.get("mini_batch_size", 4)
        self.ppo_epochs = config.get("ppo_epochs", 1)
        self.learning_rate = config.get("learning_rate", 1e-6)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.adv_clip = config.get("adv_clip", 5.0)

        # DAPO config
        self.clip_ratio = config.get("clip_ratio", 0.2)
        self.clip_ratio_high = config.get("clip_ratio_high", 0.28)
        self.exploration_bias = config.get("exploration_bias", 0.05)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.kl_coef = config.get("kl_coef", 0.001)

        # Generation config
        self.temperature = config.get("temperature", 1.0)
        self.top_p = config.get("top_p", 0.95)
        self.max_env_steps = config.get("max_env_steps", 8)

        # Output
        self.output_dir = Path(config.get("output_dir", "./checkpoints/grpo"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = config.get("save_every", 50)
        self.log_dir = config.get("log_dir", "./logs")

    def _is_local_path(self, path: str) -> bool:
        """Check if path is a local directory (not a HF repo id)."""
        return os.path.isdir(path)

    def setup(self):
        """Initialize all components."""
        logger.info("=" * 60)
        logger.info("Setting up Knapsack-GRPO Trainer")
        logger.info("=" * 60)

        # Tokenizer
        logger.info(f"Loading tokenizer from {self.model_path}")
        local_policy = self._is_local_path(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=local_policy,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Policy model
        logger.info(f"Loading policy model from {self.model_path}")
        attn_impl = get_attn_implementation()
        dtype = get_torch_dtype()

        self.policy_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            local_files_only=local_policy,
        ).to(self.device)

        # Reference model (frozen)
        logger.info(f"Loading reference model from {self.ref_model_path}")
        local_ref = self._is_local_path(self.ref_model_path)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.ref_model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            local_files_only=local_ref,
        ).to(self.device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # Optimizer
        self.optimizer = AdamW(
            self.policy_model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(self.num_iterations * 0.05),
            num_training_steps=self.num_iterations,
        )

        # Knapsack allocator
        self.allocator = KnapsackBudgetAllocator(
            N_total=self.N_total,
            N_low=self.N_low,
            N_up=self.N_up,
            warmup_iterations=self.warmup_iterations,
        )

        # Reward manager
        self.reward_manager = KnapsackRewardManager(
            verifier=HardVerifier(strict_mode=True),
        )

        # Metrics
        self.tracker = MetricsTracker()
        self.training_logger = TrainingLogger(
            log_dir=self.log_dir,
            experiment_name="knapsack_grpo",
            use_wandb=self.config.get("use_wandb", False),
        )

        # Dataset
        logger.info("Loading RLVR dataset...")
        data_loader = NemotronAgenticLoader(
            cache_dir=self.config.get("data_cache_dir", "./data/raw"),
        )
        records = data_loader.load_interactive_agent(
            max_samples=self.config.get("max_samples", None),
        )
        self.dataset = RLVRAgenticDataset(
            records=records,
            tokenizer=self.tokenizer,
            max_prompt_length=self.max_prompt_length,
            max_response_length=self.max_response_length,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.dataset.collate_fn,
            num_workers=2,
            drop_last=True,
        )
        self.data_iter = iter(self.dataloader)

        logger.info(f"RLVR dataset: {len(self.dataset)} trajectories")
        logger.info(f"Difficulty: {self.dataset.get_difficulty_distribution()}")
        logger.info("Setup complete.")

    def train(self):
        """Main training loop."""
        self.setup()

        logger.info("=" * 60)
        logger.info("Starting Knapsack-GRPO Training")
        logger.info(f"  Iterations: {self.num_iterations}")
        logger.info(f"  Budget: {self.N_total}, N_low={self.N_low}, N_up={self.N_up}")
        logger.info(f"  Batch size: {self.batch_size} prompts")
        logger.info(f"  DAPO clip: [{1-self.clip_ratio:.2f}, {1+self.clip_ratio_high:.2f}]")
        logger.info("=" * 60)

        for iteration in range(1, self.num_iterations + 1):
            t_iter_start = time.time()

            # 1. Get batch of prompts
            batch = self._get_batch()
            prompt_ids = batch["uuids"]
            prompt_texts = batch["prompt_texts"]
            ground_truths = batch["ground_truths"]
            tool_env_logs = batch["tool_env_logs"]
            tools_jsons = batch["tools_jsons"]

            # 2. Allocate rollout budgets
            success_rates = self.allocator._get_success_rates(prompt_ids)
            budgets = self.allocator.allocate(prompt_ids, success_rates)

            alloc_info = {
                "iteration": iteration,
                "total_budget": int(budgets.sum()),
                "mean_alloc": float(budgets.mean()),
                "min_alloc": int(budgets.min()),
                "max_alloc": int(budgets.max()),
            }
            self.training_logger.log_allocation(iteration, alloc_info)

            # 3. Generate rollouts with variable budgets
            t_gen_start = time.time()
            rollout_data = self._generate_rollouts(
                prompt_texts, tools_jsons, tool_env_logs, budgets,
            )
            gen_time = time.time() - t_gen_start

            # 4. Compute rewards
            t_reward_start = time.time()
            reward_data = self._compute_rewards(
                rollout_data, ground_truths, budgets,
            )
            reward_time = time.time() - t_reward_start

            # 5. Compute advantages
            advantages, returns, adv_metrics = self._compute_advantages(
                reward_data, budgets,
            )

            # 6. Policy update
            t_update_start = time.time()
            loss_metrics = self._update_policy(
                rollout_data, advantages, reward_data,
            )
            update_time = time.time() - t_update_start

            # 7. Update allocator stats
            rewards_per_prompt = self._group_rewards(reward_data["rewards"], budgets)
            self.allocator.update(prompt_ids, rewards_per_prompt)

            # 8. Log metrics
            iter_time = time.time() - t_iter_start
            all_metrics = self._compile_metrics(
                iteration=iteration,
                prompt_ids=prompt_ids,
                budgets=budgets,
                rollout_data=rollout_data,
                reward_data=reward_data,
                adv_metrics=adv_metrics,
                loss_metrics=loss_metrics,
                ground_truths=ground_truths,
                gen_time=gen_time,
                reward_time=reward_time,
                update_time=update_time,
                iter_time=iter_time,
            )
            self.training_logger.log_iteration(iteration, "grpo", all_metrics)
            self.training_logger.log_loss(iteration, loss_metrics)

            # 9. Save checkpoint
            if iteration % self.save_every == 0:
                self._save_checkpoint(iteration)

            # Memory cleanup
            del rollout_data, reward_data, advantages, returns
            if iteration % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        # Final save
        self._save_checkpoint(self.num_iterations)
        self.training_logger.close()
        logger.info("Training complete.")

    def _get_batch(self) -> Dict[str, Any]:
        """Get next batch of prompts, cycling if needed."""
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)
        return batch

    @torch.no_grad()
    def _generate_rollouts(
        self,
        prompt_texts: List[str],
        tools_jsons: List[str],
        tool_env_logs: List[str],
        budgets: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Generate rollouts with variable budgets per prompt.

        For each prompt, generates N_i responses using the policy model.
        Runs multi-turn interactions through the simulated tool environment.
        """
        self.policy_model.eval()

        all_responses = []
        all_input_ids = []
        all_response_ids = []
        all_log_probs = []
        all_token_counts = []
        all_gen_times = []
        all_response_masks = []

        # Dynamic temperature
        temp = self.temperature

        for i, (prompt, tools_json, env_log_str) in enumerate(
            zip(prompt_texts, tools_jsons, tool_env_logs)
        ):
            N_i = int(budgets[i])
            tools = json.loads(tools_json) if isinstance(tools_json, str) else tools_json
            env_log = json.loads(env_log_str) if isinstance(env_log_str, str) else env_log_str

            for j in range(N_i):
                t0 = time.time()

                # Setup environment
                env = SimulatedToolEnvironment(
                    tools=tools, tool_env_log=env_log, max_steps=self.max_env_steps,
                )
                state = env.reset()
                conversation = prompt

                # Multi-turn generation
                full_response = ""
                total_tokens = 0

                for step in range(self.max_env_steps):
                    # Tokenize current conversation
                    inputs = self.tokenizer(
                        conversation,
                        return_tensors="pt",
                        max_length=self.max_prompt_length + self.max_response_length,
                        truncation=True,
                    ).to(self.device)
                    # Remove keys the model doesn't accept
                    inputs.pop("token_type_ids", None)

                    # Generate
                    with torch.no_grad():
                        outputs = self.policy_model.generate(
                            **inputs,
                            max_new_tokens=min(512, self.max_response_length - total_tokens),
                            temperature=temp,
                            top_p=self.top_p,
                            do_sample=True,
                            return_dict_in_generate=True,
                            output_scores=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )

                    # Decode response
                    gen_ids = outputs.sequences[0, inputs["input_ids"].shape[1]:]
                    response_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                    total_tokens += len(gen_ids)

                    # Step environment
                    state, tool_response = env.step(state, response_text)
                    full_response += response_text

                    if state.done:
                        break

                    # Append tool response to conversation for next turn
                    conversation += "\n" + response_text + "\n" + tool_response
                    full_response += "\n" + tool_response + "\n"

                gen_time = time.time() - t0

                # Get log probs for the full response
                full_text = prompt + "\n" + full_response
                encoded = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    max_length=self.max_prompt_length + self.max_response_length,
                    truncation=True,
                    padding="max_length",
                ).to(self.device)

                prompt_len = len(self.tokenizer.encode(prompt, add_special_tokens=True))
                total_len = encoded["input_ids"].shape[1]

                with torch.no_grad():
                    model_out = self.policy_model(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                    )
                    logits = model_out.logits

                # Compute log probs for response tokens
                shift_logits = logits[:, :-1, :]
                shift_labels = encoded["input_ids"][:, 1:]
                log_probs = F.log_softmax(shift_logits, dim=-1)
                token_log_probs = log_probs.gather(
                    2, shift_labels.unsqueeze(-1)
                ).squeeze(-1)

                # Create response mask
                response_mask = torch.zeros_like(token_log_probs)
                resp_start = max(0, prompt_len - 1)
                resp_end = min(total_len - 1, resp_start + total_tokens)
                response_mask[0, resp_start:resp_end] = 1.0

                all_responses.append(full_response)
                all_input_ids.append(encoded["input_ids"].squeeze(0))
                all_response_ids.append(gen_ids.cpu())
                all_log_probs.append(token_log_probs.squeeze(0).cpu())
                all_response_masks.append(response_mask.squeeze(0).cpu())
                all_token_counts.append(total_tokens)
                all_gen_times.append(gen_time)

        self.policy_model.train()

        return {
            "responses": all_responses,
            "input_ids": all_input_ids,
            "log_probs": all_log_probs,
            "response_masks": all_response_masks,
            "token_counts": all_token_counts,
            "gen_times": all_gen_times,
        }

    def _compute_rewards(
        self,
        rollout_data: Dict[str, Any],
        ground_truths: List[str],
        budgets: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute binary rewards for all rollouts."""
        responses = rollout_data["responses"]

        # Expand ground truths to match rollout count
        expanded_gts = []
        for i, gs in enumerate(budgets):
            for _ in range(int(gs)):
                expanded_gts.append(ground_truths[i] if i < len(ground_truths) else "")

        result = self.reward_manager.compute_rewards(
            responses=responses,
            ground_truths=expanded_gts,
        )

        return result

    def _compute_advantages(
        self,
        reward_data: Dict[str, Any],
        budgets: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Compute GRPO advantages with variable group sizes."""
        rewards = torch.tensor(reward_data["rewards"], dtype=torch.float32)

        # Build group indices
        group_indices = []
        offset = 0
        for gs in budgets:
            gs = int(gs)
            group_indices.append(list(range(offset, offset + gs)))
            offset += gs

        # Need response masks (pad to same length)
        masks_list = []
        # Use a dummy mask if we don't have real ones
        max_len = 1
        for i in range(len(rewards)):
            masks_list.append(torch.ones(1))

        response_masks = torch.stack(masks_list)

        advantages, returns, metrics = knapsack_grpo_advantage(
            rewards=rewards,
            group_indices=group_indices,
            response_masks=response_masks,
            adv_clip=self.adv_clip,
            exploration_bonus=self.exploration_bias,
        )

        return advantages, returns, metrics

    def _update_policy(
        self,
        rollout_data: Dict[str, Any],
        advantages: torch.Tensor,
        reward_data: Dict[str, Any],
    ) -> Dict[str, float]:
        """Update policy with DAPO-style loss."""
        self.policy_model.train()

        total_loss = 0.0
        total_metrics = {}
        num_updates = 0

        # Process in mini-batches
        n_rollouts = len(rollout_data["responses"])
        indices = list(range(n_rollouts))
        np.random.shuffle(indices)

        for epoch in range(self.ppo_epochs):
            for start in range(0, n_rollouts, self.mini_batch_size):
                end = min(start + self.mini_batch_size, n_rollouts)
                mb_indices = indices[start:end]

                if not mb_indices:
                    continue

                # Get mini-batch data
                mb_input_ids = torch.stack(
                    [rollout_data["input_ids"][i] for i in mb_indices]
                ).to(self.device)
                mb_old_log_probs = torch.stack(
                    [rollout_data["log_probs"][i] for i in mb_indices]
                ).to(self.device)
                mb_response_masks = torch.stack(
                    [rollout_data["response_masks"][i] for i in mb_indices]
                ).to(self.device)

                # Broadcast advantages to token level
                mb_advantages = torch.zeros_like(mb_response_masks)
                for j, idx in enumerate(mb_indices):
                    mb_advantages[j] = advantages[idx, 0] * mb_response_masks[j]

                # Forward pass through policy
                outputs = self.policy_model(
                    input_ids=mb_input_ids,
                    attention_mask=(mb_input_ids != self.tokenizer.pad_token_id).long(),
                )
                logits = outputs.logits

                # Compute new log probs
                shift_logits = logits[:, :-1, :]
                shift_labels = mb_input_ids[:, 1:]
                new_log_probs = F.log_softmax(shift_logits, dim=-1)
                new_token_log_probs = new_log_probs.gather(
                    2, shift_labels.unsqueeze(-1)
                ).squeeze(-1)

                # Reference log probs
                with torch.no_grad():
                    ref_outputs = self.ref_model(
                        input_ids=mb_input_ids,
                        attention_mask=(mb_input_ids != self.tokenizer.pad_token_id).long(),
                    )
                    ref_logits = ref_outputs.logits
                    ref_shift_logits = ref_logits[:, :-1, :]
                    ref_log_probs = F.log_softmax(ref_shift_logits, dim=-1)
                    ref_token_log_probs = ref_log_probs.gather(
                        2, shift_labels.unsqueeze(-1)
                    ).squeeze(-1)

                # Compute loss
                loss, metrics = knapsack_grpo_loss(
                    log_probs=new_token_log_probs,
                    old_log_probs=mb_old_log_probs,
                    advantages=mb_advantages[:, :-1],  # shift to match
                    response_mask=mb_response_masks[:, :-1],
                    clip_ratio=self.clip_ratio,
                    clip_ratio_high=self.clip_ratio_high,
                    entropy_coef=self.entropy_coef,
                    kl_coef=self.kl_coef,
                    ref_log_probs=ref_token_log_probs,
                    logits=shift_logits,
                )

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy_model.parameters(), self.max_grad_norm,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()
                num_updates += 1

                # Accumulate metrics
                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0.0) + v

        self.scheduler.step()

        # Average metrics
        if num_updates > 0:
            for k in total_metrics:
                total_metrics[k] /= num_updates
            total_metrics["avg_loss"] = total_loss / num_updates

        return total_metrics

    def _group_rewards(
        self,
        rewards: List[float],
        budgets: np.ndarray,
    ) -> List[List[float]]:
        """Group flat rewards list by prompt."""
        grouped = []
        offset = 0
        for gs in budgets:
            gs = int(gs)
            grouped.append(rewards[offset:offset + gs])
            offset += gs
        return grouped

    def _compile_metrics(self, **kwargs) -> Dict[str, Any]:
        """Compile all metrics for logging."""
        iteration = kwargs["iteration"]
        prompt_ids = kwargs["prompt_ids"]
        budgets = kwargs["budgets"]
        rollout_data = kwargs["rollout_data"]
        reward_data = kwargs["reward_data"]
        adv_metrics = kwargs["adv_metrics"]
        loss_metrics = kwargs["loss_metrics"]
        ground_truths = kwargs["ground_truths"]

        rewards = reward_data["rewards"]
        group_sizes = [int(b) for b in budgets]

        batch_metrics = self.tracker.record_generation(
            prompt_ids=prompt_ids,
            responses=rollout_data["responses"],
            ground_truths=ground_truths,
            rewards=rewards,
            group_sizes=group_sizes,
            generation_times=rollout_data["gen_times"],
            token_counts=rollout_data["token_counts"],
        )

        batch_metrics.update({
            "gen_time": kwargs["gen_time"],
            "reward_time": kwargs["reward_time"],
            "update_time": kwargs["update_time"],
            "iter_time": kwargs["iter_time"],
            **{f"loss/{k}": v for k, v in loss_metrics.items()},
            **{f"adv/{k}": v for k, v in adv_metrics.items()
               if isinstance(v, (int, float))},
        })

        # Knapsack-specific
        egr = self.allocator.get_effective_gradient_ratio(
            self._group_rewards(rewards, budgets)
        )
        batch_metrics.update({f"knapsack/{k}": v for k, v in egr.items()})

        return batch_metrics

    def _save_checkpoint(self, iteration: int):
        """Save model checkpoint."""
        path = self.output_dir / f"iter_{iteration}"
        path.mkdir(parents=True, exist_ok=True)
        self.policy_model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))

        # Save allocator state
        import pickle
        with open(path / "allocator.pkl", "wb") as f:
            pickle.dump({
                "stats": self.allocator.stats,
                "iteration": self.allocator.iteration,
                "history": self.allocator._allocation_history,
            }, f)

        logger.info(f"Saved checkpoint at iteration {iteration}: {path}")
