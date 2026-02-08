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

# GigaChat3 uses MLA (Multi-head Latent Attention) like DeepSeek-V2/V3.
# Generic backends (FLASH_ATTN, FLASHINFER) do NOT support MLA.
# Must use MLA-specific backend. Priority on H200 (Hopper):
#   FLASH_ATTN_MLA > FLASHMLA > FLASHINFER_MLA > TRITON_MLA
# TRITON_MLA is the safest universal fallback.
ATTN_BACKEND="${ATTN_BACKEND:-TRITON_MLA}"

echo "=============================================="
echo "  vLLM Server for GRPO Rollouts"
echo "=============================================="
echo "  Model:       $MODEL"
echo "  GPU:         $GPU"
echo "  GPU util:    $GPU_UTIL"
echo "  Max seq len: $MAX_MODEL_LEN"
echo "  Attention:   $ATTN_BACKEND (MLA)"
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
    --attention-backend "$ATTN_BACKEND" \
    --disable-log-requests
