#!/usr/bin/env python3
"""
Knapsack-GRPO entry point (standalone, without verl distributed).

Usage:
  python scripts/run_grpo_main.py --model-path ./checkpoints/sft/final ...
"""

import argparse
import logging
import sys

sys.path.insert(0, ".")
from src.grpo.trainer import KnapsackGRPOTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Knapsack-GRPO Training")

    # Model
    parser.add_argument("--model-path", default="./checkpoints/sft/final")
    parser.add_argument("--ref-model-path", default=None)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-response-length", type=int, default=2048)

    # Knapsack
    parser.add_argument("--n-total", type=int, default=1024)
    parser.add_argument("--n-low", type=int, default=2)
    parser.add_argument("--n-up", type=int, default=128)
    parser.add_argument("--knapsack-warmup", type=int, default=5)

    # Training
    parser.add_argument("--num-iterations", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mini-batch-size", type=int, default=4)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--adv-clip", type=float, default=5.0)

    # DAPO
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--clip-ratio-high", type=float, default=0.28)
    parser.add_argument("--exploration-bias", type=float, default=0.05)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--kl-coef", type=float, default=0.0)

    # Generation
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-env-steps", type=int, default=8)

    # vLLM server
    parser.add_argument("--vllm-url", default=None,
                        help="vLLM server URL (e.g. http://localhost:8000/v1)")
    parser.add_argument("--vllm-model-name", default=None,
                        help="Model name on vLLM server (auto-detected if omitted)")

    # Memory optimization
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Offload policy model to CPU during vLLM generation to save GPU memory")

    # Data
    parser.add_argument("--data-cache-dir", default="./data/raw")
    parser.add_argument("--max-samples", type=int, default=None)

    # Output
    parser.add_argument("--output-dir", default="./checkpoints/grpo")
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--save-every", type=int, default=50)

    # Integrations
    parser.add_argument("--use-wandb", action="store_true")

    args = parser.parse_args()

    config = {
        "model_path": args.model_path,
        "ref_model_path": args.ref_model_path or args.model_path,
        "max_prompt_length": args.max_prompt_length,
        "max_response_length": args.max_response_length,
        "N_total": args.n_total,
        "N_low": args.n_low,
        "N_up": args.n_up,
        "knapsack_warmup": args.knapsack_warmup,
        "num_iterations": args.num_iterations,
        "batch_size": args.batch_size,
        "mini_batch_size": args.mini_batch_size,
        "ppo_epochs": args.ppo_epochs,
        "learning_rate": args.learning_rate,
        "max_grad_norm": args.max_grad_norm,
        "adv_clip": args.adv_clip,
        "clip_ratio": args.clip_ratio,
        "clip_ratio_high": args.clip_ratio_high,
        "exploration_bias": args.exploration_bias,
        "entropy_coef": args.entropy_coef,
        "kl_coef": args.kl_coef,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_env_steps": args.max_env_steps,
        "data_cache_dir": args.data_cache_dir,
        "max_samples": args.max_samples,
        "output_dir": args.output_dir,
        "log_dir": args.log_dir,
        "save_every": args.save_every,
        "use_wandb": args.use_wandb,
        "vllm_url": args.vllm_url,
        "vllm_model_name": args.vllm_model_name,
        "cpu_offload": args.cpu_offload,
        "device": "cuda",
    }

    trainer = KnapsackGRPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
