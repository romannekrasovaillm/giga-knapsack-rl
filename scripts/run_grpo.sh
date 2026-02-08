#!/bin/bash
# Knapsack-GRPO: GRPO training with adaptive rollout budget allocation.
# Supports vLLM server for fast rollout generation.
#
# Single GPU without vLLM:
#   bash scripts/run_grpo.sh
#
# With vLLM (start server first in another terminal):
#   Terminal 1: bash scripts/run_vllm_server.sh
#   Terminal 2: bash scripts/run_grpo.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ---- Configuration ----
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_PATH="${MODEL_PATH:-ai-sage/GigaChat3-10B-A1.8B-base}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/grpo}"
DATA_CACHE="${DATA_CACHE:-./data/raw}"
LOG_DIR="${LOG_DIR:-./logs}"

# vLLM server (leave empty to use HF fallback)
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_URL="${VLLM_URL:-http://localhost:${VLLM_PORT}/v1}"

# Auto-detect vLLM: check if server is running
USE_VLLM=false
if curl -s "${VLLM_URL}/models" > /dev/null 2>&1; then
    USE_VLLM=true
fi

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
MAX_SAMPLES="${MAX_SAMPLES:-}"

# DAPO (no KL — ref model not loaded, saves GPU memory)
CLIP_RATIO="${CLIP_RATIO:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"
EXPLORATION_BIAS="${EXPLORATION_BIAS:-0.05}"
ENTROPY_COEF="${ENTROPY_COEF:-0.01}"

# Generation
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-8}"

echo "=============================================="
echo "  Knapsack-GRPO Training"
echo "=============================================="
echo "  Model:          $MODEL_PATH"
echo "  GPU:            $CUDA_VISIBLE_DEVICES"
echo "  Iterations:     $NUM_ITERATIONS"
echo "  Budget:         N_total=$N_TOTAL, N_low=$N_LOW, N_up=$N_UP"
echo "  Batch:          $BATCH_SIZE prompts x variable rollouts"
echo "  LR:             $LR"
echo "  DAPO clip:      [1-$CLIP_RATIO, 1+$CLIP_RATIO_HIGH]"
echo "  KL:             disabled (no ref model)"
echo "  Exploration:    bias=$EXPLORATION_BIAS, entropy=$ENTROPY_COEF"
if [ "$USE_VLLM" = true ]; then
echo "  Generation:     vLLM @ $VLLM_URL"
else
echo "  Generation:     HuggingFace (no vLLM server detected)"
fi
echo "  Output:         $OUTPUT_DIR"
echo "=============================================="

# Download interactive_agent data
echo "[1/2] Checking dataset..."
python scripts/download_data.py --cache-dir "$DATA_CACHE" --split interactive_agent

# Build command
echo "[2/2] Starting Knapsack-GRPO training..."
CMD=(
    python scripts/run_grpo_main.py
    --model-path "$MODEL_PATH"
    --output-dir "$OUTPUT_DIR"
    --data-cache-dir "$DATA_CACHE"
    --log-dir "$LOG_DIR"
    --num-iterations "$NUM_ITERATIONS"
    --batch-size "$BATCH_SIZE"
    --mini-batch-size "$MINI_BATCH_SIZE"
    --learning-rate "$LR"
    --adv-clip "$ADV_CLIP"
    --n-total "$N_TOTAL"
    --n-low "$N_LOW"
    --n-up "$N_UP"
    --clip-ratio "$CLIP_RATIO"
    --clip-ratio-high "$CLIP_RATIO_HIGH"
    --exploration-bias "$EXPLORATION_BIAS"
    --entropy-coef "$ENTROPY_COEF"
    --kl-coef 0.0
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --max-env-steps "$MAX_ENV_STEPS"
)

if [ "$USE_VLLM" = true ]; then
    CMD+=(--vllm-url "$VLLM_URL")
fi

if [ -n "${MAX_SAMPLES:-}" ]; then
    CMD+=(--max-samples "$MAX_SAMPLES")
fi

"${CMD[@]}"

echo "Knapsack-GRPO training complete!"
echo "  Checkpoint saved to: $OUTPUT_DIR"
