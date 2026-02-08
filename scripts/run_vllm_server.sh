#!/bin/bash
# Launch vLLM server for GRPO rollout generation.
# Run in terminal 1, then run_grpo.sh in terminal 2.
set -euo pipefail

MODEL="${MODEL:-ai-sage/GigaChat3-10B-A1.8B-base}"
PORT="${VLLM_PORT:-8000}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
GPU_UTIL="${GPU_MEMORY_UTILIZATION:-0.45}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

export CUDA_VISIBLE_DEVICES="$GPU"

# Auto-detect attention backend: flash_attn > xformers > flashinfer
if [ -z "${VLLM_ATTENTION_BACKEND:-}" ]; then
    if python -c "import flash_attn" 2>/dev/null; then
        export VLLM_ATTENTION_BACKEND=FLASH_ATTN
    elif python -c "import xformers" 2>/dev/null; then
        export VLLM_ATTENTION_BACKEND=XFORMERS
    else
        export VLLM_ATTENTION_BACKEND=FLASHINFER
    fi
fi

echo "=============================================="
echo "  vLLM Server for GRPO Rollouts"
echo "=============================================="
echo "  Model:       $MODEL"
echo "  GPU:         $GPU"
echo "  GPU util:    $GPU_UTIL"
echo "  Max seq len: $MAX_MODEL_LEN"
echo "  Attention:   $VLLM_ATTENTION_BACKEND"
echo "  Port:        $PORT"
echo "  URL:         http://localhost:${PORT}/v1"
echo "=============================================="
echo ""
echo "After server is ready, run in another terminal:"
echo "  bash scripts/run_grpo.sh"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --trust-remote-code \
    --dtype auto \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --disable-log-requests
