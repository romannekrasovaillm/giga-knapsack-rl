"""
RLVR Dataset for interactive_agent subset.

Converts Nemotron interactive_agent trajectories into:
  - prompts for rollout generation
  - ground truth for reward verification
  - tool call/response pairs for simulated environment
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .parser import AgenticTrajectoryParser, AgenticTrajectory

logger = logging.getLogger(__name__)


class RLVRAgenticDataset(Dataset):
    """
    Dataset for RLVR training with agentic multi-turn trajectories.

    Each sample provides:
      - prompt: system + tools + first user message
      - ground_truth: final answer (for reward verification)
      - tool_env_log: tool call/response pairs (for simulated environment)
      - metadata: uuid, difficulty estimate, etc.
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        tokenizer,
        max_prompt_length: int = 2048,
        max_response_length: int = 2048,
        min_tool_calls: int = 0,
        max_tool_calls: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length

        parser = AgenticTrajectoryParser()
        trajectories = parser.parse_batch(records)

        # Filter by tool call count
        self.trajectories = []
        for traj in trajectories:
            n_tc = traj.num_tool_calls
            if n_tc < min_tool_calls:
                continue
            if max_tool_calls is not None and n_tc > max_tool_calls:
                continue
            self.trajectories.append(traj)

        logger.info(
            f"RLVR dataset: {len(self.trajectories)} trajectories "
            f"(filtered from {len(trajectories)})"
        )

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        traj = self.trajectories[idx]

        prompt = traj.get_prompt_for_rlvr()
        ground_truth = traj.final_answer
        tool_env_log = traj.build_environment_log()
        tool_calls_gt = traj.get_tool_call_ground_truth()

        # Tokenize prompt
        prompt_tokens = self.tokenizer(
            prompt,
            max_length=self.max_prompt_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": prompt_tokens["input_ids"].squeeze(0),
            "attention_mask": prompt_tokens["attention_mask"].squeeze(0),
            "prompt_text": prompt,
            "ground_truth": ground_truth,
            "tool_env_log": json.dumps(tool_env_log, ensure_ascii=False),
            "tool_calls_gt": json.dumps(tool_calls_gt, ensure_ascii=False),
            "uuid": traj.uuid,
            "num_turns": traj.num_turns,
            "num_tool_calls": traj.num_tool_calls,
            "tool_names": json.dumps(traj.tool_names_used),
            "tools_json": json.dumps(traj.tools, ensure_ascii=False),
        }

    def get_difficulty_distribution(self) -> Dict[str, int]:
        """Estimate difficulty based on tool call count."""
        easy = sum(1 for t in self.trajectories if t.num_tool_calls <= 1)
        medium = sum(1 for t in self.trajectories if 2 <= t.num_tool_calls <= 4)
        hard = sum(1 for t in self.trajectories if t.num_tool_calls > 4)
        return {"easy": easy, "medium": medium, "hard": hard}

    def get_success_rate_estimates(self) -> np.ndarray:
        """
        Heuristic success rate estimates based on trajectory complexity.
        Used for initial Knapsack allocation before real rewards are available.
        """
        rates = []
        for traj in self.trajectories:
            # Harder trajectories (more tool calls) have lower initial success rate
            n_tc = traj.num_tool_calls
            n_turns = traj.num_turns
            # Heuristic: p = 0.6 * exp(-0.15 * tool_calls) * exp(-0.05 * turns)
            p = 0.6 * np.exp(-0.15 * n_tc) * np.exp(-0.05 * n_turns)
            p = np.clip(p, 0.02, 0.95)
            rates.append(p)
        return np.array(rates)

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate for mixed tensor/non-tensor data."""
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_texts": [b["prompt_text"] for b in batch],
            "ground_truths": [b["ground_truth"] for b in batch],
            "tool_env_logs": [b["tool_env_log"] for b in batch],
            "tool_calls_gts": [b["tool_calls_gt"] for b in batch],
            "uuids": [b["uuid"] for b in batch],
            "num_turns": [b["num_turns"] for b in batch],
            "num_tool_calls": [b["num_tool_calls"] for b in batch],
            "tools_jsons": [b["tools_json"] for b in batch],
        }
