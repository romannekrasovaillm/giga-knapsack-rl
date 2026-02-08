"""
Knapsack-GRPO Trainer.

Main training loop integrating:
  - Knapsack RL budget allocation
  - GRPO with variable group sizes
  - DAPO-style asymmetric importance sampling
  - vLLM server for fast rollout generation (with HF fallback)
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

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def _truncate(text: str, head: int = 80, tail: int = 60) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= head + tail + 10:
        return text
    return text[:head] + f" ...({len(text)} chars)... " + text[-tail:]


class KnapsackGRPOTrainer:
    """
    Full Knapsack-GRPO training loop.

    Supports two generation backends:
      - vLLM server (--vllm-url): batched multi-turn via OpenAI API
      - HuggingFace generate: sequential fallback
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
        self.batch_size = config.get("batch_size", 16)
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

        # vLLM config
        self.vllm_url = config.get("vllm_url", None)
        self.vllm_model_name = config.get("vllm_model_name", None)

        # Output
        self.output_dir = Path(config.get("output_dir", "./checkpoints/grpo"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = config.get("save_every", 50)
        self.log_dir = config.get("log_dir", "./logs")

    def _is_local_path(self, path: str) -> bool:
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

        # vLLM rollout generator
        self.vllm_generator = None
        if self.vllm_url:
            from src.grpo.rollout_generator import VLLMRolloutGenerator
            model_name = self.vllm_model_name or self.model_path
            self.vllm_generator = VLLMRolloutGenerator(
                server_url=self.vllm_url,
                model_name=model_name,
                max_response_length=self.max_response_length,
                max_env_steps=self.max_env_steps,
            )
            logger.info(f"vLLM server: {self.vllm_url}")
        else:
            logger.info(
                "Generation: HuggingFace (pass --vllm-url for vLLM server)"
            )

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

    # ──────────────────────────────────────────────────────────────
    # Main Training Loop
    # ──────────────────────────────────────────────────────────────

    def train(self):
        self.setup()

        gen_mode = f"vLLM ({self.vllm_url})" if self.vllm_generator else "HuggingFace"
        logger.info("=" * 60)
        logger.info("Starting Knapsack-GRPO Training")
        logger.info(f"  Iterations: {self.num_iterations}")
        logger.info(f"  Budget: {self.N_total}, N_low={self.N_low}, N_up={self.N_up}")
        logger.info(f"  Batch size: {self.batch_size} prompts")
        logger.info(f"  DAPO clip: [{1-self.clip_ratio:.2f}, {1+self.clip_ratio_high:.2f}]")
        logger.info(f"  Generation: {gen_mode}")
        logger.info("=" * 60)

        for iteration in range(1, self.num_iterations + 1):
            t_iter_start = time.time()
            print(f"\n{'='*100}", flush=True)
            print(
                f"{BOLD}{CYAN}[GRPO] Iteration {iteration}/{self.num_iterations}{RESET}",
                flush=True,
            )
            print(f"{'='*100}", flush=True)

            # 1. Get batch
            batch = self._get_batch()
            prompt_ids = batch["uuids"]
            prompt_texts = batch["prompt_texts"]
            ground_truths = batch["ground_truths"]
            tool_env_logs = batch["tool_env_logs"]
            tools_jsons = batch["tools_jsons"]

            # 2. Allocate budgets
            success_rates = self.allocator._get_success_rates(prompt_ids)
            budgets = self.allocator.allocate(prompt_ids, success_rates)
            self.training_logger.log_allocation(iteration, {
                "iteration": iteration,
                "total_budget": int(budgets.sum()),
                "mean_alloc": float(budgets.mean()),
                "min_alloc": int(budgets.min()),
                "max_alloc": int(budgets.max()),
            })

            # 3. Generate rollouts
            t_gen = time.time()
            if self.vllm_generator:
                raw = self.vllm_generator.generate_rollouts(
                    prompt_texts, tools_jsons, tool_env_logs, budgets,
                    temperature=self.temperature, top_p=self.top_p,
                )
                rollout_data = self._compute_log_probs(
                    raw, prompt_texts, budgets,
                )
            else:
                rollout_data = self._generate_rollouts_hf(
                    prompt_texts, tools_jsons, tool_env_logs, budgets,
                )
            gen_time = time.time() - t_gen

            # 4. Compute rewards
            t_rew = time.time()
            reward_data = self._compute_rewards(rollout_data, ground_truths, budgets)
            reward_time = time.time() - t_rew

            # 5. Log rollout details
            self._log_rollout_details(
                iteration, prompt_ids, prompt_texts, ground_truths,
                rollout_data, reward_data, budgets, gen_time,
            )

            # 6. Advantages
            advantages, returns, adv_metrics = self._compute_advantages(
                reward_data, budgets,
            )

            # 7. Policy update
            t_upd = time.time()
            loss_metrics = self._update_policy(rollout_data, advantages, reward_data)
            update_time = time.time() - t_upd

            # 8. Update allocator
            rewards_per_prompt = self._group_rewards(reward_data["rewards"], budgets)
            self.allocator.update(prompt_ids, rewards_per_prompt)

            # 9. Log metrics
            iter_time = time.time() - t_iter_start
            all_metrics = self._compile_metrics(
                iteration=iteration, prompt_ids=prompt_ids, budgets=budgets,
                rollout_data=rollout_data, reward_data=reward_data,
                adv_metrics=adv_metrics, loss_metrics=loss_metrics,
                ground_truths=ground_truths, gen_time=gen_time,
                reward_time=reward_time, update_time=update_time,
                iter_time=iter_time,
            )
            self.training_logger.log_iteration(iteration, "grpo", all_metrics)
            self.training_logger.log_loss(iteration, loss_metrics)

            # 10. Summary
            rewards = reward_data["rewards"]
            sr = sum(1 for r in rewards if r > 0.5) / len(rewards) if rewards else 0
            print(
                f"\n  {BOLD}[Summary]{RESET} iter={iteration} | "
                f"loss={loss_metrics.get('avg_loss', 0):.4f} | "
                f"reward={np.mean(rewards):.3f} | success={sr:.1%} | "
                f"lr={self.scheduler.get_last_lr()[0]:.2e} | "
                f"gen={gen_time:.1f}s upd={update_time:.1f}s total={iter_time:.1f}s",
                flush=True,
            )

            # 11. Save
            if iteration % self.save_every == 0:
                self._save_checkpoint(iteration)

            del rollout_data, reward_data, advantages, returns
            if iteration % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        self._save_checkpoint(self.num_iterations)
        self.training_logger.close()
        logger.info("Training complete.")

    # ──────────────────────────────────────────────────────────────
    # Generation: HF fallback
    # ──────────────────────────────────────────────────────────────

    def _get_batch(self) -> Dict[str, Any]:
        try:
            return next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            return next(self.data_iter)

    @torch.no_grad()
    def _generate_rollouts_hf(
        self,
        prompt_texts: List[str],
        tools_jsons: List[str],
        tool_env_logs: List[str],
        budgets: np.ndarray,
    ) -> Dict[str, Any]:
        """Generate rollouts via HuggingFace model.generate()."""
        self.policy_model.eval()

        all_responses, all_input_ids, all_log_probs = [], [], []
        all_token_counts, all_gen_times, all_response_masks = [], [], []
        all_num_turns = []
        temp = self.temperature
        n_total = int(budgets.sum())
        rollout_idx = 0

        for i, (prompt, tools_json, env_log_str) in enumerate(
            zip(prompt_texts, tools_jsons, tool_env_logs)
        ):
            N_i = int(budgets[i])
            tools = json.loads(tools_json) if isinstance(tools_json, str) else tools_json
            env_log = json.loads(env_log_str) if isinstance(env_log_str, str) else env_log_str

            for j in range(N_i):
                t0 = time.time()
                env = SimulatedToolEnvironment(
                    tools=tools, tool_env_log=env_log, max_steps=self.max_env_steps,
                )
                state = env.reset()
                conversation = prompt
                full_response = ""
                total_tokens = 0
                num_turns = 0

                for step in range(self.max_env_steps):
                    inputs = self.tokenizer(
                        conversation,
                        return_tensors="pt",
                        max_length=self.max_prompt_length + self.max_response_length,
                        truncation=True,
                    ).to(self.device)
                    inputs.pop("token_type_ids", None)

                    with torch.no_grad():
                        outputs = self.policy_model.generate(
                            **inputs,
                            max_new_tokens=min(
                                512, self.max_response_length - total_tokens
                            ),
                            temperature=temp,
                            top_p=self.top_p,
                            do_sample=True,
                            return_dict_in_generate=True,
                            output_scores=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )

                    gen_ids = outputs.sequences[0, inputs["input_ids"].shape[1]:]
                    response_text = self.tokenizer.decode(
                        gen_ids, skip_special_tokens=True
                    )
                    total_tokens += len(gen_ids)
                    num_turns += 1

                    state, tool_response = env.step(state, response_text)
                    full_response += response_text
                    if state.done:
                        break
                    conversation += "\n" + response_text + "\n" + tool_response
                    full_response += "\n" + tool_response + "\n"

                gen_time = time.time() - t0

                # Log probs for the full response
                full_text = prompt + "\n" + full_response
                encoded = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    max_length=self.max_prompt_length + self.max_response_length,
                    truncation=True,
                    padding="max_length",
                ).to(self.device)

                prompt_len = len(
                    self.tokenizer.encode(prompt, add_special_tokens=True)
                )
                total_len = encoded["input_ids"].shape[1]

                with torch.no_grad():
                    model_out = self.policy_model(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                    )
                shift_logits = model_out.logits[:, :-1, :]
                shift_labels = encoded["input_ids"][:, 1:]
                lp = F.log_softmax(shift_logits, dim=-1)
                token_lp = lp.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

                rmask = torch.zeros_like(token_lp)
                rs = max(0, prompt_len - 1)
                re = min(total_len - 1, rs + total_tokens)
                rmask[0, rs:re] = 1.0

                all_responses.append(full_response)
                all_input_ids.append(encoded["input_ids"].squeeze(0))
                all_log_probs.append(token_lp.squeeze(0).cpu())
                all_response_masks.append(rmask.squeeze(0).cpu())
                all_token_counts.append(total_tokens)
                all_gen_times.append(gen_time)
                all_num_turns.append(num_turns)

                rollout_idx += 1
                if rollout_idx % 10 == 0 or rollout_idx == n_total:
                    preview = _truncate(full_response, head=60, tail=30)
                    print(
                        f"  [HF] {rollout_idx}/{n_total} | "
                        f"g{i+1} r{j+1}/{N_i} | "
                        f"tok={total_tokens} t={num_turns} | "
                        f"{gen_time:.1f}s | {preview}",
                        flush=True,
                    )

        self.policy_model.train()
        return {
            "responses": all_responses,
            "input_ids": all_input_ids,
            "log_probs": all_log_probs,
            "response_masks": all_response_masks,
            "token_counts": all_token_counts,
            "gen_times": all_gen_times,
            "num_turns": all_num_turns,
        }

    # ──────────────────────────────────────────────────────────────
    # Log probs for vLLM-generated responses
    # ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _compute_log_probs(
        self,
        raw_rollouts: Dict[str, Any],
        prompt_texts: List[str],
        budgets: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute log probs via policy model for vLLM-generated responses."""
        self.policy_model.eval()
        responses = raw_rollouts["responses"]
        token_counts = raw_rollouts["token_counts"]
        gen_times = raw_rollouts["gen_times"]
        num_turns = raw_rollouts["num_turns"]
        prompt_indices = raw_rollouts["prompt_indices"]

        all_input_ids, all_log_probs, all_response_masks = [], [], []

        print(
            f"  {CYAN}[LogProbs]{RESET} Computing for "
            f"{len(responses)} rollouts...",
            flush=True,
        )
        t0 = time.time()

        for idx, (response, pi) in enumerate(zip(responses, prompt_indices)):
            prompt = prompt_texts[pi]
            full_text = prompt + "\n" + response
            encoded = self.tokenizer(
                full_text,
                return_tensors="pt",
                max_length=self.max_prompt_length + self.max_response_length,
                truncation=True,
                padding="max_length",
            ).to(self.device)

            prompt_len = len(
                self.tokenizer.encode(prompt, add_special_tokens=True)
            )
            total_len = encoded["input_ids"].shape[1]

            with torch.no_grad():
                out = self.policy_model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                )
            shift_logits = out.logits[:, :-1, :]
            shift_labels = encoded["input_ids"][:, 1:]
            lp = F.log_softmax(shift_logits, dim=-1)
            token_lp = lp.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

            n_resp = token_counts[idx]
            rmask = torch.zeros_like(token_lp)
            rs = max(0, prompt_len - 1)
            re = min(total_len - 1, rs + n_resp)
            rmask[0, rs:re] = 1.0

            all_input_ids.append(encoded["input_ids"].squeeze(0))
            all_log_probs.append(token_lp.squeeze(0).cpu())
            all_response_masks.append(rmask.squeeze(0).cpu())

            if (idx + 1) % 50 == 0:
                print(f"    {idx+1}/{len(responses)}...", flush=True)

        elapsed = time.time() - t0
        print(
            f"  {CYAN}[LogProbs]{RESET} Done in {elapsed:.1f}s",
            flush=True,
        )
        self.policy_model.train()

        return {
            "responses": responses,
            "input_ids": all_input_ids,
            "log_probs": all_log_probs,
            "response_masks": all_response_masks,
            "token_counts": token_counts,
            "gen_times": gen_times,
            "num_turns": num_turns,
        }

    # ──────────────────────────────────────────────────────────────
    # Rewards, Advantages, Policy Update
    # ──────────────────────────────────────────────────────────────

    def _compute_rewards(self, rollout_data, ground_truths, budgets):
        responses = rollout_data["responses"]
        expanded_gts = []
        for i, gs in enumerate(budgets):
            for _ in range(int(gs)):
                expanded_gts.append(
                    ground_truths[i] if i < len(ground_truths) else ""
                )
        return self.reward_manager.compute_rewards(
            responses=responses, ground_truths=expanded_gts,
        )

    def _compute_advantages(self, reward_data, budgets):
        rewards = torch.tensor(reward_data["rewards"], dtype=torch.float32)
        group_indices, offset = [], 0
        for gs in budgets:
            gs = int(gs)
            group_indices.append(list(range(offset, offset + gs)))
            offset += gs
        response_masks = torch.stack([torch.ones(1) for _ in range(len(rewards))])
        return knapsack_grpo_advantage(
            rewards=rewards, group_indices=group_indices,
            response_masks=response_masks, adv_clip=self.adv_clip,
            exploration_bonus=self.exploration_bias,
        )

    def _update_policy(self, rollout_data, advantages, reward_data):
        self.policy_model.train()
        total_loss, total_metrics, num_updates = 0.0, {}, 0
        n_rollouts = len(rollout_data["responses"])
        indices = list(range(n_rollouts))
        np.random.shuffle(indices)

        for epoch in range(self.ppo_epochs):
            for start in range(0, n_rollouts, self.mini_batch_size):
                end = min(start + self.mini_batch_size, n_rollouts)
                mb_idx = indices[start:end]
                if not mb_idx:
                    continue

                mb_ids = torch.stack(
                    [rollout_data["input_ids"][i] for i in mb_idx]
                ).to(self.device)
                mb_old_lp = torch.stack(
                    [rollout_data["log_probs"][i] for i in mb_idx]
                ).to(self.device)
                mb_rmask = torch.stack(
                    [rollout_data["response_masks"][i] for i in mb_idx]
                ).to(self.device)

                mb_adv = torch.zeros_like(mb_rmask)
                for j, ix in enumerate(mb_idx):
                    mb_adv[j] = advantages[ix, 0] * mb_rmask[j]

                attn = (mb_ids != self.tokenizer.pad_token_id).long()
                out = self.policy_model(input_ids=mb_ids, attention_mask=attn)
                sl = out.logits[:, :-1, :]
                labels = mb_ids[:, 1:]
                new_lp = F.log_softmax(sl, dim=-1)
                new_tlp = new_lp.gather(2, labels.unsqueeze(-1)).squeeze(-1)

                with torch.no_grad():
                    ref_out = self.ref_model(input_ids=mb_ids, attention_mask=attn)
                    ref_sl = ref_out.logits[:, :-1, :]
                    ref_lp = F.log_softmax(ref_sl, dim=-1)
                    ref_tlp = ref_lp.gather(
                        2, labels.unsqueeze(-1)
                    ).squeeze(-1)

                loss, metrics = knapsack_grpo_loss(
                    log_probs=new_tlp, old_log_probs=mb_old_lp,
                    advantages=mb_adv[:, :-1], response_mask=mb_rmask[:, :-1],
                    clip_ratio=self.clip_ratio,
                    clip_ratio_high=self.clip_ratio_high,
                    entropy_coef=self.entropy_coef, kl_coef=self.kl_coef,
                    ref_log_probs=ref_tlp, logits=sl,
                )

                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy_model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                total_loss += loss.item()
                num_updates += 1
                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0.0) + v

        self.scheduler.step()
        if num_updates > 0:
            for k in total_metrics:
                total_metrics[k] /= num_updates
            total_metrics["avg_loss"] = total_loss / num_updates
        return total_metrics

    # ──────────────────────────────────────────────────────────────
    # Detailed Rollout Logging
    # ──────────────────────────────────────────────────────────────

    def _log_rollout_details(
        self, iteration, prompt_ids, prompt_texts, ground_truths,
        rollout_data, reward_data, budgets, gen_time,
    ):
        """Log per-group and per-rollout metrics to terminal and file."""
        responses = rollout_data["responses"]
        rewards = reward_data["rewards"]
        token_counts = rollout_data["token_counts"]
        gen_times = rollout_data["gen_times"]
        num_turns = rollout_data.get("num_turns", [1] * len(responses))

        n_rollouts = len(responses)
        n_success = sum(1 for r in rewards if r > 0.5)
        total_tokens = sum(token_counts)
        avg_tokens = total_tokens / n_rollouts if n_rollouts else 0

        print(
            f"\n  {BOLD}--- Rollouts (iter={iteration}) ---{RESET}\n"
            f"  Total: {n_rollouts} | "
            f"Success: {n_success}/{n_rollouts} ({n_success/n_rollouts:.1%}) | "
            f"Tokens: {total_tokens:,} (avg {avg_tokens:.0f}) | "
            f"Gen: {gen_time:.1f}s",
            flush=True,
        )

        offset = 0
        group_summaries = []

        for i, N_i in enumerate(budgets):
            N_i = int(N_i)
            g_rew = rewards[offset:offset + N_i]
            g_tok = token_counts[offset:offset + N_i]
            g_turns = num_turns[offset:offset + N_i]
            g_resp = responses[offset:offset + N_i]

            g_succ = sum(1 for r in g_rew if r > 0.5)
            g_sr = g_succ / N_i if N_i > 0 else 0
            g_mr = float(np.mean(g_rew)) if g_rew else 0
            g_mt = float(np.mean(g_tok)) if g_tok else 0
            g_mturns = float(np.mean(g_turns)) if g_turns else 0

            color = RED if g_sr == 0 else (YELLOW if g_sr < 0.3 else (GREEN if g_sr >= 0.8 else ""))
            reset = RESET if color else ""

            pid = prompt_ids[i][:20] if i < len(prompt_ids) else "?"
            gt = _truncate(ground_truths[i], 60, 0) if i < len(ground_truths) else ""

            print(
                f"\n  {color}[G{i+1}/{len(budgets)}] {pid} | "
                f"N={N_i} succ={g_succ}/{N_i} ({g_sr:.0%}) | "
                f"r={g_mr:.3f} tok={g_mt:.0f} turns={g_mturns:.1f}{reset}",
                flush=True,
            )
            print(f"    {DIM}GT: {gt}{RESET}", flush=True)

            # Best and worst rollout
            if g_resp:
                best_i = worst_i = None
                for ri in range(len(g_rew)):
                    if g_rew[ri] > 0.5 and (best_i is None or g_tok[ri] < g_tok[best_i]):
                        best_i = ri
                    elif g_rew[ri] <= 0.5 and (worst_i is None or g_tok[ri] > g_tok[worst_i]):
                        worst_i = ri

                if best_i is not None:
                    print(
                        f"    {GREEN}BEST [r=1 tok={g_tok[best_i]}]: "
                        f"{_truncate(g_resp[best_i], 100, 50)}{RESET}",
                        flush=True,
                    )
                if worst_i is not None:
                    print(
                        f"    {RED}WORST[r=0 tok={g_tok[worst_i]}]: "
                        f"{_truncate(g_resp[worst_i], 100, 50)}{RESET}",
                        flush=True,
                    )

                uniq = len(set(r[:100] for r in g_resp))
                print(
                    f"    {DIM}Diversity: {uniq}/{N_i} unique prefixes{RESET}",
                    flush=True,
                )

            group_summaries.append({
                "prompt_id": prompt_ids[i] if i < len(prompt_ids) else "?",
                "N_i": N_i, "success_rate": g_sr,
                "mean_reward": g_mr, "mean_tokens": g_mt,
                "mean_turns": g_mturns,
            })
            offset += N_i

        self.training_logger.log_rollout_details(iteration, {
            "n_rollouts": n_rollouts, "n_success": n_success,
            "total_tokens": total_tokens, "gen_time": gen_time,
            "groups": group_summaries,
        })

    # ──────────────────────────────────────────────────────────────
    # Metrics & Utils
    # ──────────────────────────────────────────────────────────────

    def _group_rewards(self, rewards, budgets):
        grouped, offset = [], 0
        for gs in budgets:
            gs = int(gs)
            grouped.append(rewards[offset:offset + gs])
            offset += gs
        return grouped

    def _compile_metrics(self, **kw):
        rewards = kw["reward_data"]["rewards"]
        group_sizes = [int(b) for b in kw["budgets"]]
        batch_metrics = self.tracker.record_generation(
            prompt_ids=kw["prompt_ids"],
            responses=kw["rollout_data"]["responses"],
            ground_truths=kw["ground_truths"],
            rewards=rewards, group_sizes=group_sizes,
            generation_times=kw["rollout_data"]["gen_times"],
            token_counts=kw["rollout_data"]["token_counts"],
        )
        batch_metrics.update({
            "gen_time": kw["gen_time"],
            "reward_time": kw["reward_time"],
            "update_time": kw["update_time"],
            "iter_time": kw["iter_time"],
            **{f"loss/{k}": v for k, v in kw["loss_metrics"].items()},
            **{f"adv/{k}": v for k, v in kw["adv_metrics"].items()
               if isinstance(v, (int, float))},
        })
        egr = self.allocator.get_effective_gradient_ratio(
            self._group_rewards(rewards, kw["budgets"])
        )
        batch_metrics.update({f"knapsack/{k}": v for k, v in egr.items()})
        return batch_metrics

    def _save_checkpoint(self, iteration):
        path = self.output_dir / f"iter_{iteration}"
        path.mkdir(parents=True, exist_ok=True)
        self.policy_model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
        import pickle
        with open(path / "allocator.pkl", "wb") as f:
            pickle.dump({
                "stats": self.allocator.stats,
                "iteration": self.allocator.iteration,
                "history": self.allocator._allocation_history,
            }, f)
        logger.info(f"Saved checkpoint at iteration {iteration}: {path}")
