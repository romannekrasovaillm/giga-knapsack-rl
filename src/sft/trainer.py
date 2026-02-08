"""
SFT Warmup Trainer.

Fine-tunes GigaChat3-10B-A1.8B-base on the tool_calling subset
for 1-2 epochs to teach function calling basics before GRPO.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from src.data.loader import NemotronAgenticLoader
from src.data.sft_dataset import SFTToolCallingDataset
from src.metrics.logger import TrainingLogger
from src.utils import get_attn_implementation, get_torch_dtype

logger = logging.getLogger(__name__)


class SFTWarmupTrainer:
    """
    SFT warmup trainer for function calling.

    Trains on the tool_calling subset for 1-2 epochs to establish
    basic function calling ability before GRPO fine-tuning.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get("device", "cuda"))

        # Model
        self.model_name = config.get("model_name", "ai-sage/GigaChat3-10B-A1.8B-base")
        self.max_length = config.get("max_length", 4096)

        # Training hyperparams
        self.num_epochs = config.get("num_epochs", 2)
        self.batch_size = config.get("batch_size", 4)
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 8)
        self.learning_rate = config.get("learning_rate", 2e-5)
        self.weight_decay = config.get("weight_decay", 0.01)
        self.warmup_ratio = config.get("warmup_ratio", 0.05)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.max_samples = config.get("max_samples", None)

        # Output
        self.output_dir = Path(config.get("output_dir", "./checkpoints/sft"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_steps = config.get("save_steps", 500)
        self.log_steps = config.get("log_steps", 10)

        # Logger
        self.training_logger = TrainingLogger(
            log_dir=config.get("log_dir", "./logs"),
            experiment_name="sft_warmup",
        )

        # State
        self.global_step = 0
        self.model = None
        self.tokenizer = None

    def setup(self):
        """Load model, tokenizer, and dataset."""
        logger.info(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=get_torch_dtype(),
            trust_remote_code=True,
            attn_implementation=get_attn_implementation(),
        )
        self.model.to(self.device)

        # Enable gradient checkpointing for memory
        if self.config.get("gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable()

        # Load dataset
        logger.info("Loading tool_calling dataset...")
        loader = NemotronAgenticLoader(cache_dir=self.config.get("data_cache_dir", "./data/raw"))
        records = loader.load_tool_calling(max_samples=self.max_samples)

        self.dataset = SFTToolCallingDataset(
            records=records,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        params = [
            {
                "params": [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(params, lr=self.learning_rate)

        total_steps = len(self.dataloader) * self.num_epochs // self.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.warmup_ratio)

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        logger.info(f"Dataset size: {len(self.dataset)}")
        logger.info(f"Total steps: {total_steps}, warmup: {warmup_steps}")
        logger.info(f"Effective batch size: {self.batch_size * self.gradient_accumulation_steps}")

    def train(self):
        """Run SFT training loop."""
        self.setup()
        self.model.train()
        best_loss = float("inf")

        total_batches = len(self.dataloader)
        logger.info(f"Batches per epoch: {total_batches}")
        print(flush=True)

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            t0 = time.time()

            for batch_idx, batch in enumerate(self.dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / self.gradient_accumulation_steps
                loss.backward()

                epoch_loss += outputs.loss.item()
                epoch_steps += 1

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # Logging
                    if self.global_step % self.log_steps == 0:
                        avg_loss = epoch_loss / epoch_steps
                        lr = self.scheduler.get_last_lr()[0]
                        self.training_logger.log_sft_step(
                            step=self.global_step,
                            loss=avg_loss,
                            lr=lr,
                            extra={"epoch": epoch, "batch_idx": batch_idx},
                        )

                    # Save checkpoint
                    if self.global_step % self.save_steps == 0:
                        self._save_checkpoint(f"step_{self.global_step}")

                # Progress indicator every 100 batches
                elif (batch_idx + 1) % 100 == 0:
                    pct = (batch_idx + 1) / total_batches * 100
                    elapsed = time.time() - t0
                    print(
                        f"  [Epoch {epoch+1}] {batch_idx+1}/{total_batches} "
                        f"({pct:.1f}%) | loss={outputs.loss.item():.4f} | "
                        f"{elapsed:.0f}s elapsed",
                        flush=True,
                    )

            # End of epoch
            avg_epoch_loss = epoch_loss / epoch_steps
            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} | "
                f"Loss: {avg_epoch_loss:.4f} | "
                f"Time: {elapsed:.0f}s | "
                f"Steps: {self.global_step}"
            )

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                self._save_checkpoint("best")

        # Save final
        self._save_checkpoint("final")
        logger.info(f"SFT training complete. Best loss: {best_loss:.4f}")
        return str(self.output_dir / "final")

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        path = self.output_dir / name
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
        logger.info(f"Saved checkpoint: {path}")
