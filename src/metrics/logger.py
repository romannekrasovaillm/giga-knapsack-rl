"""
Comprehensive training logger.

Logs to:
  - Terminal (formatted tables)
  - Log file (JSON lines)
  - Optional: wandb, tensorboard
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Terminal colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"
DIM = "\033[2m"


class TrainingLogger:
    """
    Comprehensive training logger with terminal and file output.

    Outputs formatted metrics tables to terminal and JSON lines to file.
    """

    def __init__(
        self,
        log_dir: str = "./logs",
        experiment_name: str = "knapsack_grpo",
        use_wandb: bool = False,
        wandb_project: str = "giga-knapsack-rl",
        use_tensorboard: bool = False,
        terminal_width: int = 100,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"{experiment_name}_{timestamp}"

        # File logger
        self.log_file = self.log_dir / f"{self.experiment_name}.jsonl"
        self.metrics_file = self.log_dir / f"{self.experiment_name}_metrics.jsonl"

        # Terminal
        self.terminal_width = terminal_width

        # Optional integrations
        self.wandb_run = None
        self.tb_writer = None

        if use_wandb:
            self._init_wandb(wandb_project)
        if use_tensorboard:
            self._init_tensorboard()

        # Setup Python logging
        self._setup_logging()

        logger.info(f"Logging to {self.log_dir}/{self.experiment_name}")

    def _setup_logging(self):
        """Configure Python logging to file and console."""
        log_path = self.log_dir / f"{self.experiment_name}.log"
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter("%(message)s")
        )

        root_logger = logging.getLogger("src")
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

    def _init_wandb(self, project: str):
        try:
            import wandb
            self.wandb_run = wandb.init(project=project, name=self.experiment_name)
            logger.info(f"Wandb initialized: {self.wandb_run.url}")
        except Exception as e:
            logger.warning(f"Failed to init wandb: {e}")

    def _init_tensorboard(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = self.log_dir / "tensorboard" / self.experiment_name
            self.tb_writer = SummaryWriter(str(tb_dir))
            logger.info(f"Tensorboard logging to {tb_dir}")
        except Exception as e:
            logger.warning(f"Failed to init tensorboard: {e}")

    def log_iteration(
        self,
        iteration: int,
        phase: str,
        metrics: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ):
        """Log a full iteration's metrics."""
        # Write to JSON log
        entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "phase": phase,
            "metrics": self._flatten_metrics(metrics),
        }
        if extra:
            entry["extra"] = extra

        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

        # Terminal output
        self._print_iteration(iteration, phase, metrics)

        # Wandb / Tensorboard
        flat = self._flatten_metrics(metrics)
        if self.wandb_run:
            try:
                import wandb
                wandb.log({f"{phase}/{k}": v for k, v in flat.items()
                           if isinstance(v, (int, float))}, step=iteration)
            except Exception:
                pass

        if self.tb_writer:
            for k, v in flat.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(f"{phase}/{k}", v, iteration)

    def _print_iteration(self, iteration: int, phase: str, metrics: Dict[str, Any]):
        """Print formatted metrics to terminal."""
        w = self.terminal_width
        print(f"\n{'=' * w}")
        print(f"{BOLD}{CYAN}[{phase.upper()}] Iteration {iteration}{RESET}")
        print(f"{'=' * w}")

        # Key metrics table
        key_metrics = [
            ("Prompts in batch", metrics.get("num_prompts", "?")),
            ("Total rollouts", metrics.get("num_rollouts", "?")),
            ("Mean group size", f"{metrics.get('mean_group_size', 0):.1f}"),
            ("", ""),
            ("Batch mean reward", f"{metrics.get('batch_mean_reward', 0):.4f}"),
            ("Batch success rate", f"{metrics.get('batch_success_rate', 0):.2%}"),
            ("Effective gradient ratio", f"{metrics.get('effective_gradient_ratio', 0):.2%}"),
            ("All-positive groups", metrics.get("all_positive_groups", 0)),
            ("All-negative groups", metrics.get("all_negative_groups", 0)),
            ("", ""),
            ("Batch mean BLEU", f"{metrics.get('batch_mean_bleu', 0):.4f}"),
            ("Batch total tokens", f"{metrics.get('batch_total_tokens', 0):,}"),
            ("Batch mean tokens/rollout", f"{metrics.get('batch_mean_tokens', 0):.0f}"),
            ("Total generation time", f"{metrics.get('batch_total_gen_time', 0):.1f}s"),
        ]

        if "batch_mean_entropy" in metrics:
            key_metrics.append(("Batch mean entropy", f"{metrics['batch_mean_entropy']:.4f}"))

        if "batch_mean_advantage" in metrics:
            key_metrics.append(("Batch mean advantage", f"{metrics['batch_mean_advantage']:.4f}"))
            key_metrics.append(("Batch std advantage", f"{metrics['batch_std_advantage']:.4f}"))

        for name, val in key_metrics:
            if name == "":
                print(f"{DIM}{'─' * w}{RESET}")
            else:
                print(f"  {name:<35} {val}")

        # Per-group details (top 5 and bottom 5)
        groups = metrics.get("groups", [])
        if groups:
            print(f"\n{BOLD}Per-prompt details (sorted by reward):{RESET}")
            sorted_groups = sorted(groups, key=lambda g: g.get("mean_reward", 0))

            print(f"  {'Prompt ID':<25} {'N_i':>5} {'Reward':>8} {'Rate':>8} {'BLEU':>8} {'Tokens':>8} {'Adv':>10}")
            print(f"  {'─' * 85}")

            # Bottom 5
            for gm in sorted_groups[:5]:
                adv = f"{gm.get('mean_advantage', 0):.3f}" if 'mean_advantage' in gm else "—"
                color = RED if gm.get("success_rate", 0) == 0 else ""
                reset = RESET if color else ""
                print(
                    f"  {color}{gm['prompt_id'][:25]:<25} "
                    f"{gm['group_size']:>5} "
                    f"{gm['mean_reward']:>8.3f} "
                    f"{gm['success_rate']:>7.1%} "
                    f"{gm.get('mean_bleu', 0):>8.3f} "
                    f"{gm.get('mean_tokens', 0):>8.0f} "
                    f"{adv:>10}{reset}"
                )

            if len(sorted_groups) > 10:
                print(f"  {DIM}... {len(sorted_groups) - 10} more groups ...{RESET}")

            # Top 5
            for gm in sorted_groups[-5:]:
                adv = f"{gm.get('mean_advantage', 0):.3f}" if 'mean_advantage' in gm else "—"
                color = GREEN if gm.get("success_rate", 1.0) == 1.0 else ""
                reset = RESET if color else ""
                print(
                    f"  {color}{gm['prompt_id'][:25]:<25} "
                    f"{gm['group_size']:>5} "
                    f"{gm['mean_reward']:>8.3f} "
                    f"{gm['success_rate']:>7.1%} "
                    f"{gm.get('mean_bleu', 0):>8.3f} "
                    f"{gm.get('mean_tokens', 0):>8.0f} "
                    f"{adv:>10}{reset}"
                )

        print(f"{'=' * w}\n", flush=True)

    def log_sft_step(self, step: int, loss: float, lr: float, extra: Optional[Dict] = None):
        """Log an SFT training step."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "phase": "sft",
            "step": step,
            "loss": loss,
            "lr": lr,
        }
        if extra:
            entry.update(extra)

        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

        epoch_info = ""
        if extra and "epoch" in extra:
            epoch_info = f" | epoch={extra['epoch']+1}"
        print(
            f"  [SFT] step={step:>6d} | loss={loss:.4f} | lr={lr:.2e}{epoch_info}",
            flush=True,
        )

        if self.wandb_run:
            try:
                import wandb
                wandb.log({"sft/loss": loss, "sft/lr": lr}, step=step)
            except Exception:
                pass

    def log_allocation(self, iteration: int, alloc_info: Dict[str, Any]):
        """Log knapsack allocation details."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "phase": "allocation",
            "iteration": iteration,
            **alloc_info,
        }
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def log_rollout_details(self, iteration: int, details: Dict[str, Any]):
        """Log detailed rollout information to file."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "phase": "rollout_details",
            "iteration": iteration,
            **details,
        }
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

        if self.wandb_run:
            try:
                import wandb
                wandb.log({
                    "rollouts/n_rollouts": details.get("n_rollouts", 0),
                    "rollouts/n_success": details.get("n_success", 0),
                    "rollouts/total_tokens": details.get("total_tokens", 0),
                    "rollouts/gen_time": details.get("gen_time", 0),
                }, step=iteration)
            except Exception:
                pass

    def log_loss(self, iteration: int, loss_metrics: Dict[str, float]):
        """Log loss metrics."""
        parts = [f"[LOSS] iter={iteration}"]
        for k, v in loss_metrics.items():
            parts.append(f"{k}={v:.4f}")
        print(f"  {' | '.join(parts)}", flush=True)

        if self.wandb_run:
            try:
                import wandb
                wandb.log({f"loss/{k}": v for k, v in loss_metrics.items()}, step=iteration)
            except Exception:
                pass

    def _flatten_metrics(self, d: Dict, prefix: str = "") -> Dict[str, Any]:
        """Flatten nested dict for logging."""
        flat = {}
        for k, v in d.items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}/{k}"
            if isinstance(v, dict):
                flat.update(self._flatten_metrics(v, key))
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                # Skip nested lists of dicts (per-group details)
                flat[f"{key}/count"] = len(v)
            elif isinstance(v, (int, float, str, bool)):
                flat[key] = v
        return flat

    def close(self):
        """Cleanup."""
        if self.wandb_run:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
        if self.tb_writer:
            self.tb_writer.close()
