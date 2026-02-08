#!/usr/bin/env python3
"""
Download and cache the Nemotron Agentic v1 dataset.

Usage:
  python scripts/download_data.py [--cache-dir ./data/raw] [--split all|tool_calling|interactive_agent]
"""

import argparse
import logging
import sys

sys.path.insert(0, ".")
from src.data.loader import NemotronAgenticLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download Nemotron Agentic v1 dataset")
    parser.add_argument("--cache-dir", default="./data/raw", help="Cache directory")
    parser.add_argument("--split", default="all", choices=["all", "tool_calling", "interactive_agent"])
    args = parser.parse_args()

    loader = NemotronAgenticLoader(cache_dir=args.cache_dir)

    splits = ["tool_calling", "interactive_agent"] if args.split == "all" else [args.split]

    for split in splits:
        logger.info(f"Downloading split: {split}")
        path = loader.download(split)
        stats = loader.get_stats(split)
        logger.info(f"  Path: {path}")
        logger.info(f"  Samples: {stats['num_samples']}")

    logger.info("Download complete.")


if __name__ == "__main__":
    main()
