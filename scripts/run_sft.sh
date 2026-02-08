#!/bin/bash
# SFT Warmup: Fine-tune GigaChat3-10B-A1.8B-base on tool_calling subset
# Run for 1-2 epochs to teach function calling basics
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ---- Configuration ----
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

MODEL_NAME="${MODEL_NAME:-ai-sage/GigaChat3-10B-A1.8B-base}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/sft}"
DATA_CACHE="${DATA_CACHE:-./data/raw}"
LOG_DIR="${LOG_DIR:-./logs}"
MAX_SAMPLES="${MAX_SAMPLES:-}"  # empty = all samples

NUM_EPOCHS="${NUM_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-2e-5}"
MAX_LENGTH="${MAX_LENGTH:-4096}"

echo "=============================================="
echo "  SFT Warmup Training"
echo "=============================================="
echo "  Model:        $MODEL_NAME"
echo "  GPUs:         $NUM_GPUS ($CUDA_VISIBLE_DEVICES)"
echo "  Epochs:       $NUM_EPOCHS"
echo "  Batch size:   $BATCH_SIZE x $GRAD_ACCUM = $((BATCH_SIZE * GRAD_ACCUM))"
echo "  LR:           $LR"
echo "  Output:       $OUTPUT_DIR"
echo "=============================================="

# ---- Step 1: Download data if needed ----
echo "[1/3] Checking dataset..."
python scripts/download_data.py --cache-dir "$DATA_CACHE" --split tool_calling

# ---- Step 2: Run SFT with torchrun for multi-GPU ----
echo "[2/3] Starting SFT training..."

if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU with torchrun
    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --master_port=29500 \
        scripts/run_sft_main.py \
        --model-name "$MODEL_NAME" \
        --output-dir "$OUTPUT_DIR" \
        --data-cache-dir "$DATA_CACHE" \
        --log-dir "$LOG_DIR" \
        --num-epochs "$NUM_EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --gradient-accumulation-steps "$GRAD_ACCUM" \
        --learning-rate "$LR" \
        --max-length "$MAX_LENGTH" \
        ${MAX_SAMPLES:+--max-samples "$MAX_SAMPLES"}
else
    # Single GPU
    python scripts/run_sft_main.py \
        --model-name "$MODEL_NAME" \
        --output-dir "$OUTPUT_DIR" \
        --data-cache-dir "$DATA_CACHE" \
        --log-dir "$LOG_DIR" \
        --num-epochs "$NUM_EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --gradient-accumulation-steps "$GRAD_ACCUM" \
        --learning-rate "$LR" \
        --max-length "$MAX_LENGTH" \
        ${MAX_SAMPLES:+--max-samples "$MAX_SAMPLES"}
fi

echo "[3/3] SFT training complete!"
echo "  Checkpoint saved to: $OUTPUT_DIR/final"
