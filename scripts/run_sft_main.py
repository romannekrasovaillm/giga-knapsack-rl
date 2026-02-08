#!/usr/bin/env python3
"""
SFT warmup entry point.

Usage:
  python scripts/run_sft_main.py --model-name ai-sage/GigaChat3-10B-A1.8B-base ...
  torchrun --nproc_per_node=8 scripts/run_sft_main.py ...
"""

import argparse
import logging
import sys

sys.path.insert(0, ".")
from src.sft.trainer import SFTWarmupTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    parser = argparse.ArgumentParser(description="SFT Warmup Training")
    parser.add_argument("--model-name", default="ai-sage/GigaChat3-10B-A1.8B-base")
    parser.add_argument("--output-dir", default="./checkpoints/sft")
    parser.add_argument("--data-cache-dir", default="./data/raw")
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--log-steps", type=int, default=10)
    args = parser.parse_args()

    config = {
        "model_name": args.model_name,
        "output_dir": args.output_dir,
        "data_cache_dir": args.data_cache_dir,
        "log_dir": args.log_dir,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "max_samples": args.max_samples,
        "max_grad_norm": args.max_grad_norm,
        "warmup_ratio": args.warmup_ratio,
        "save_steps": args.save_steps,
        "log_steps": args.log_steps,
        "gradient_checkpointing": True,
        "device": "cuda",
    }

    trainer = SFTWarmupTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
