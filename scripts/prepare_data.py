#!/usr/bin/env python3
"""
Preprocess and validate the downloaded dataset.

Parses interactive_agent records into trajectories and reports statistics.

Usage:
  python scripts/prepare_data.py [--cache-dir ./data/raw] [--max-samples 1000]
"""

import argparse
import json
import logging
import sys
from collections import Counter

sys.path.insert(0, ".")
from src.data.loader import NemotronAgenticLoader
from src.data.parser import AgenticTrajectoryParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Prepare and validate dataset")
    parser.add_argument("--cache-dir", default="./data/raw")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", default="./data/prepared/stats.json")
    args = parser.parse_args()

    loader = NemotronAgenticLoader(cache_dir=args.cache_dir)
    traj_parser = AgenticTrajectoryParser()

    # Process interactive_agent
    logger.info("Loading interactive_agent split...")
    records = loader.load_interactive_agent(max_samples=args.max_samples)
    logger.info(f"Loaded {len(records)} records")

    logger.info("Parsing trajectories...")
    trajectories = traj_parser.parse_batch(records)

    # Statistics
    n_turns = [t.num_turns for t in trajectories]
    n_tool_calls = [t.num_tool_calls for t in trajectories]
    tool_names = Counter()
    for t in trajectories:
        for name in t.tool_names_used:
            tool_names[name] += 1

    final_answer_lens = [len(t.final_answer.split()) for t in trajectories]

    stats = {
        "total_records": len(records),
        "parsed_trajectories": len(trajectories),
        "parse_rate": len(trajectories) / len(records) if records else 0,
        "turns": {
            "mean": sum(n_turns) / len(n_turns) if n_turns else 0,
            "min": min(n_turns) if n_turns else 0,
            "max": max(n_turns) if n_turns else 0,
        },
        "tool_calls": {
            "mean": sum(n_tool_calls) / len(n_tool_calls) if n_tool_calls else 0,
            "min": min(n_tool_calls) if n_tool_calls else 0,
            "max": max(n_tool_calls) if n_tool_calls else 0,
        },
        "unique_tools": len(tool_names),
        "top_tools": tool_names.most_common(20),
        "final_answer_words": {
            "mean": sum(final_answer_lens) / len(final_answer_lens) if final_answer_lens else 0,
            "min": min(final_answer_lens) if final_answer_lens else 0,
            "max": max(final_answer_lens) if final_answer_lens else 0,
        },
        "difficulty_distribution": {
            "easy (0-1 tool calls)": sum(1 for n in n_tool_calls if n <= 1),
            "medium (2-4 tool calls)": sum(1 for n in n_tool_calls if 2 <= n <= 4),
            "hard (5+ tool calls)": sum(1 for n in n_tool_calls if n >= 5),
        },
    }

    logger.info("\n" + "=" * 60)
    logger.info("Dataset Statistics")
    logger.info("=" * 60)
    for key, val in stats.items():
        if isinstance(val, dict):
            logger.info(f"  {key}:")
            for k2, v2 in val.items():
                logger.info(f"    {k2}: {v2}")
        else:
            logger.info(f"  {key}: {val}")

    # Save stats
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\nStats saved to {args.output}")


if __name__ == "__main__":
    main()
