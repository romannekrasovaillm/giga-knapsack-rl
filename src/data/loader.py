"""
Dataset loader for nvidia/Nemotron-Agentic-v1.

Two splits:
  - tool_calling (316k samples): for SFT warmup on function calling
  - interactive_agent (19k samples): for RLVR multi-turn agentic training
"""

import json
import os
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

DATASET_REPO = "nvidia/Nemotron-Agentic-v1"
SPLITS = {
    "tool_calling": "data/tool_calling.jsonl",
    "interactive_agent": "data/interactive_agent.jsonl",
}


class NemotronAgenticLoader:
    """Load and cache Nemotron Agentic v1 dataset."""

    def __init__(self, cache_dir: str = "./data/raw"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download(self, split: str) -> Path:
        """Download a split if not cached. Returns path to JSONL."""
        assert split in SPLITS, f"Unknown split: {split}. Use {list(SPLITS.keys())}"
        filename = SPLITS[split]
        local = self.cache_dir / split / os.path.basename(filename)

        if local.exists():
            logger.info(f"Using cached {split} at {local}")
            return local

        logger.info(f"Downloading {split} from {DATASET_REPO}...")
        downloaded = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=filename,
            repo_type="dataset",
            cache_dir=str(self.cache_dir / ".hf_cache"),
        )
        local.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(os.path.abspath(downloaded), str(local))
        logger.info(f"Downloaded {split} -> {local}")
        return local

    def load_jsonl(self, split: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load records from a split."""
        path = self.download(split)
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                records.append(json.loads(line.strip()))
        logger.info(f"Loaded {len(records)} records from {split}")
        return records

    def load_tool_calling(self, max_samples: Optional[int] = None) -> List[Dict]:
        """Load tool_calling split for SFT warmup."""
        return self.load_jsonl("tool_calling", max_samples)

    def load_interactive_agent(self, max_samples: Optional[int] = None) -> List[Dict]:
        """Load interactive_agent split for RLVR."""
        return self.load_jsonl("interactive_agent", max_samples)

    def get_stats(self, split: str) -> Dict[str, int]:
        """Get basic stats without loading all data."""
        path = self.download(split)
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for _ in f:
                count += 1
        return {"split": split, "num_samples": count}
