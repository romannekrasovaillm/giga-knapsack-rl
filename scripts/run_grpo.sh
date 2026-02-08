#!/bin/bash
# Knapsack-GRPO: GRPO training with adaptive rollout budget allocation
# Expects SFT checkpoint at ./checkpoints/sft/final
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ---- Configuration ----
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

MODEL_PATH="${MODEL_PATH:-./checkpoints/sft/final}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/grpo}"
DATA_CACHE="${DATA_CACHE:-./data/raw}"
LOG_DIR="${LOG_DIR:-./logs}"

# Knapsack RL
N_TOTAL="${N_TOTAL:-1024}"
N_LOW="${N_LOW:-2}"
N_UP="${N_UP:-128}"

# Training
NUM_ITERATIONS="${NUM_ITERATIONS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-4}"
LR="${LR:-1e-6}"
ADV_CLIP="${ADV_CLIP:-5.0}"

# DAPO
CLIP_RATIO="${CLIP_RATIO:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"
EXPLORATION_BIAS="${EXPLORATION_BIAS:-0.05}"
ENTROPY_COEF="${ENTROPY_COEF:-0.01}"
KL_COEF="${KL_COEF:-0.001}"

# Use verl distributed backend?
USE_VERL="${USE_VERL:-false}"

echo "=============================================="
echo "  Knapsack-GRPO Training"
echo "=============================================="
echo "  Model:          $MODEL_PATH"
echo "  GPUs:           $NUM_GPUS ($CUDA_VISIBLE_DEVICES)"
echo "  Iterations:     $NUM_ITERATIONS"
echo "  Budget:         N_total=$N_TOTAL, N_low=$N_LOW, N_up=$N_UP"
echo "  Batch:          $BATCH_SIZE prompts x variable rollouts"
echo "  LR:             $LR"
echo "  DAPO clip:      [$(echo "1 - $CLIP_RATIO" | bc), $(echo "1 + $CLIP_RATIO_HIGH" | bc)]"
echo "  Exploration:    bias=$EXPLORATION_BIAS, entropy=$ENTROPY_COEF"
echo "  Output:         $OUTPUT_DIR"
echo "=============================================="

# Verify SFT checkpoint exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: SFT checkpoint not found at $MODEL_PATH"
    echo "Run scripts/run_sft.sh first."
    exit 1
fi

# Download interactive_agent data
echo "[1/2] Checking dataset..."
python scripts/download_data.py --cache-dir "$DATA_CACHE" --split interactive_agent

# Run training
echo "[2/2] Starting Knapsack-GRPO training..."

if [ "$USE_VERL" = "true" ]; then
    echo "Using verl distributed backend..."
    python scripts/run_grpo_verl.py \
        --config configs/grpo_knapsack.yaml
else
    python scripts/run_grpo_main.py \
        --model-path "$MODEL_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --data-cache-dir "$DATA_CACHE" \
        --log-dir "$LOG_DIR" \
        --num-iterations "$NUM_ITERATIONS" \
        --batch-size "$BATCH_SIZE" \
        --mini-batch-size "$MINI_BATCH_SIZE" \
        --learning-rate "$LR" \
        --adv-clip "$ADV_CLIP" \
        --n-total "$N_TOTAL" \
        --n-low "$N_LOW" \
        --n-up "$N_UP" \
        --clip-ratio "$CLIP_RATIO" \
        --clip-ratio-high "$CLIP_RATIO_HIGH" \
        --exploration-bias "$EXPLORATION_BIAS" \
        --entropy-coef "$ENTROPY_COEF" \
        --kl-coef "$KL_COEF"
fi

echo "Knapsack-GRPO training complete!"
echo "  Checkpoint saved to: $OUTPUT_DIR"
