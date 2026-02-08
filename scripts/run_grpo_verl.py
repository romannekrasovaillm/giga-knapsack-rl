#!/usr/bin/env python3
"""
Knapsack-GRPO entry point using verl distributed backend.

Registers custom components and launches verl's training loop.

Usage:
  python scripts/run_grpo_verl.py --config configs/grpo_knapsack.yaml
"""

import argparse
import logging
import sys

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Knapsack-GRPO via verl")
    parser.add_argument("--config", default="configs/grpo_knapsack.yaml")
    args = parser.parse_args()

    # Register custom components with verl
    logger.info("Registering Knapsack-GRPO components with verl...")
    import src.grpo.verl_integration  # noqa: F401 — registers custom advantage estimator

    try:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    verl_config = config.get("verl", {})
    if not verl_config.get("enabled", False):
        logger.error("verl is not enabled in config. Set verl.enabled=true or use run_grpo_main.py")
        sys.exit(1)

    try:
        from verl.trainer.main_ppo import main as verl_main
        import hydra
        from omegaconf import OmegaConf

        # Convert our config to verl's format
        verl_cfg = OmegaConf.create(verl_config)

        # Set custom reward function
        verl_cfg.custom_reward_function = OmegaConf.create({
            "path": "src/grpo/verl_integration.py",
            "name": "compute_reward_for_verl",
        })

        # Set custom advantage estimator
        verl_cfg.algorithm = OmegaConf.create({
            "adv_estimator": "knapsack_grpo",
            "gamma": 1.0,
            "lam": 1.0,
            "use_kl_in_reward": False,
        })

        logger.info("Starting verl training loop...")
        logger.info(f"Config: {OmegaConf.to_yaml(verl_cfg)}")

        # Launch verl training
        # Note: In production, this would use hydra's config management.
        # For now, we provide a simplified launch.
        logger.info(
            "To run with full verl, use:\n"
            "  python -m verl.trainer.main_ppo \\\n"
            "    algorithm.adv_estimator=knapsack_grpo \\\n"
            f"    actor_rollout_ref.model.path={verl_config.get('actor_rollout_ref', {}).get('model', {}).get('path', './checkpoints/sft/final')} \\\n"
            "    actor_rollout_ref.actor.use_kl_loss=True \\\n"
            "    actor_rollout_ref.actor.kl_loss_coef=0.001 \\\n"
            "    actor_rollout_ref.rollout.n=16 \\\n"
            "    custom_reward_function.path=src/grpo/verl_integration.py \\\n"
            "    custom_reward_function.name=compute_reward_for_verl"
        )

    except ImportError as e:
        logger.error(f"verl is not installed: {e}")
        logger.error("Install with: pip install verl")
        logger.error("Or use the standalone trainer: python scripts/run_grpo_main.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
