#!/bin/bash
# Full training pipeline: download -> SFT warmup -> GRPO with Knapsack RL
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=============================================="
echo "  Giga-Knapsack-RL: Full Training Pipeline"
echo "=============================================="
echo ""
echo "  Base model:  ai-sage/GigaChat3-10B-A1.8B-base"
echo "  Algorithm:   GRPO + Knapsack RL + DAPO"
echo "  Dataset:     nvidia/Nemotron-Agentic-v1"
echo ""
echo "  Pipeline:"
echo "    1. Download & prepare dataset"
echo "    2. SFT warmup (tool_calling, 2 epochs)"
echo "    3. GRPO training (interactive_agent, knapsack allocation)"
echo ""
echo "=============================================="

# ---- Step 1: Download data ----
echo ""
echo "=============================="
echo "  STEP 1: Download Dataset"
echo "=============================="
python scripts/download_data.py --cache-dir ./data/raw --split all

# ---- Step 2: Validate data ----
echo ""
echo "=============================="
echo "  STEP 2: Validate Dataset"
echo "=============================="
python scripts/prepare_data.py --cache-dir ./data/raw --max-samples 1000

# ---- Step 3: SFT warmup ----
echo ""
echo "=============================="
echo "  STEP 3: SFT Warmup"
echo "=============================="
bash scripts/run_sft.sh

# ---- Step 4: GRPO training ----
echo ""
echo "=============================="
echo "  STEP 4: Knapsack-GRPO"
echo "=============================="
bash scripts/run_grpo.sh

echo ""
echo "=============================================="
echo "  Pipeline complete!"
echo "  Final model: ./checkpoints/grpo/"
echo "  Logs: ./logs/"
echo "=============================================="
