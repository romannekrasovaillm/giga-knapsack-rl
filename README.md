# Giga-Knapsack-RL

Adaptive Rollout Budget Allocation for GRPO training of [GigaChat3-10B-A1.8B-base](https://huggingface.co/ai-sage/GigaChat3-10B-A1.8B-base) on agentic tool-calling tasks.

Based on the paper: *"Knapsack RL: Unlocking Exploration of LLMs via Optimizing Budget Allocation"* (Ziniu Li et al., ByteDance Seed).

## Overview

Instead of uniform N rollouts per prompt, Knapsack RL allocates budget adaptively via dynamic programming — giving more rollouts to prompts in the "golden zone" (p ~ 0.2-0.5) and fewer to trivial/impossible ones.

**Pipeline (3 режима запуска):**

| Mode | What | When |
|---|---|---|
| **SFT + GRPO** | SFT warmup on `tool_calling`, then GRPO on `interactive_agent` | Base model doesn't know tool-calling format |
| **GRPO only** | Skip SFT, train directly from base model | GigaChat3 already knows `<tool_call>` JSON from pretrain |
| **SFT only** | Just SFT warmup | Prepare checkpoint for downstream |

**Key components:**
- **Knapsack DP solver** (Numba-accelerated) — multiple-choice knapsack for budget allocation
- **Simulated tool environment** — replays ground truth tool responses for multi-turn training
- **Hard programmatic verifier** — binary reward (correct/incorrect), no partial credit
- **DAPO asymmetric clipping** — `clip_high=1.28 > 1/clip_low=1.25` biases toward exploration
- **Async vLLM rollouts** — each rollout runs independently via `AsyncOpenAI`, fast ones don't wait for slow
- **KL penalty optional** — disabled by default (saves ~50% GPU memory, no reference model loaded)

## Project Structure

```
giga-knapsack-rl/
├── configs/
│   ├── base.yaml                # Shared config
│   ├── sft_warmup.yaml          # SFT hyperparams
│   └── grpo_knapsack.yaml       # GRPO + Knapsack RL + DAPO config
├── src/
│   ├── data/                    # Dataset loading & parsing (Nemotron Agentic v1)
│   │   ├── loader.py            #   HuggingFace download + caching
│   │   ├── parser.py            #   JSONL → AgenticTrajectory
│   │   ├── sft_dataset.py       #   SFT dataset with role markers (no chat template)
│   │   └── rlvr_dataset.py      #   RLVR dataset with prompt/GT/tool_env/difficulty
│   ├── environment/             # Simulated tool environment
│   │   ├── tool_env.py          #   Replays GT tool responses, parses <tool_call> tags
│   │   └── tool_registry.py     #   Tool definitions, parameter validation
│   ├── rewards/                 # Reward computation
│   │   ├── verifier.py          #   5-stage hard verifier (exact/key/ngram/toolcall/combined)
│   │   └── reward_manager.py    #   Wraps verifier, token-level reward placement
│   ├── knapsack/                # Knapsack RL core
│   │   ├── dp_solver.py         #   Numba @njit DP solver
│   │   ├── task_value.py        #   P(nonzero_gradient) × InfoGain value function
│   │   └── allocator.py         #   Stateful allocator tracking per-prompt success rates
│   ├── grpo/                    # GRPO training
│   │   ├── advantage.py         #   Variable group sizes, adv clipping [-5,5], exploration bonus
│   │   ├── dapo_sampling.py     #   Asymmetric importance weights, dynamic temperature
│   │   ├── policy_loss.py       #   PPO-clip + KL + entropy with asymmetric clipping
│   │   ├── rollout_generator.py #   Async vLLM rollout generation (AsyncOpenAI + semaphore)
│   │   ├── trainer.py           #   Full training loop (allocate→generate→verify→update→log)
│   │   └── verl_integration.py  #   Registers knapsack_grpo with verl framework
│   ├── metrics/                 # Logging & tracking
│   │   ├── tracker.py           #   BLEU, entropy, tokens, time, EGR per rollout/group/batch
│   │   └── logger.py            #   Terminal output + JSONL files + wandb/tensorboard
│   ├── sft/
│   │   └── trainer.py           #   SFT warmup trainer with gradient checkpointing
│   └── utils.py                 # Auto-detect attention backend & torch dtype
├── scripts/
│   ├── setup_and_run.sh         # One-command install + full pipeline
│   ├── download_data.py         # Download Nemotron dataset
│   ├── prepare_data.py          # Parse & report statistics
│   ├── run_sft.sh               # SFT launch script
│   ├── run_sft_main.py          # SFT entry point
│   ├── run_vllm_server.sh       # vLLM server for rollout generation (MLA backend)
│   ├── run_grpo.sh              # GRPO launch script (auto-detects vLLM)
│   ├── run_grpo_main.py         # GRPO entry point (standalone)
│   ├── run_grpo_verl.py         # GRPO entry point (verl distributed)
│   ├── diagnose_sft_loss.py     # Per-token SFT loss analysis
│   └── run_all.sh               # Full pipeline end-to-end
├── tests/                       # 40 tests covering all modules
│   ├── test_knapsack.py         # DP solver, value functions, allocator
│   ├── test_verifier.py         # Hard verifier, reward manager
│   ├── test_data.py             # Parsing, dataset construction
│   └── test_grpo.py             # Advantage, DAPO loss, policy loss
├── conftest.py                  # sys.path setup for tests
├── setup.py
└── requirements.txt
```

## Quick Start

### 1. Clone & setup

```bash
# First time
git clone https://github.com/romannekrasovaillm/giga-knapsack-rl.git
cd giga-knapsack-rl
git checkout claude/knapsack-rl-budget-9n3AM

# Subsequent runs — pull latest
cd giga-knapsack-rl
git pull origin claude/knapsack-rl-budget-9n3AM
```

### 2. Install

```bash
python -m venv .venv && source .venv/bin/activate

# PyTorch (pick your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Core dependencies
pip install "transformers>=4.45.0" "datasets>=2.20.0" "huggingface-hub>=0.24.0" \
  "tokenizers>=0.19.0" "accelerate>=0.33.0" "peft>=0.12.0" \
  "numba>=0.59.0" "tqdm>=4.66.0" "pyyaml>=6.0.0" \
  "antlr4-python3-runtime==4.9.3" "hydra-core>=1.3.0" "omegaconf>=2.3.0" \
  "jsonlines>=4.0.0" "pyarrow>=15.0.0" \
  "nltk>=3.8.0" "sacrebleu>=2.4.0" "pytest>=7.0.0" \
  "openai>=1.0.0"

# vLLM (for fast rollout generation)
pip install vllm

python -c "import nltk; nltk.download('punkt_tab', quiet=True)"
pip install -e .
```

### 3. Verify

```bash
pytest tests/ -v   # 40 tests
```

### 4. Download data

```bash
python scripts/download_data.py --cache-dir ./data/raw --split all
```

---

## Training Modes

### Mode A: GRPO only (recommended for GigaChat3)

GigaChat3 already knows tool-calling JSON from pretrain (60% of SFT loss tokens are "free"). Skip SFT and go directly to GRPO.

**Terminal 1 — vLLM server:**

```bash
bash scripts/run_vllm_server.sh
```

Wait for "Server ready", then:

**Terminal 2 — GRPO trainer:**

```bash
bash scripts/run_grpo.sh
```

The script auto-detects vLLM. Override defaults via environment variables:

```bash
# Custom configuration
NUM_ITERATIONS=500 BATCH_SIZE=8 N_TOTAL=512 LR=5e-7 \
  bash scripts/run_grpo.sh
```

**Or run manually with full control:**

```bash
# Terminal 1: vLLM
python -m vllm.entrypoints.openai.api_server \
  --model ai-sage/GigaChat3-10B-A1.8B-base \
  --trust-remote-code --dtype auto --port 8000 \
  --gpu-memory-utilization 0.45 --max-model-len 8192 \
  --attention-backend TRITON_MLA --disable-log-requests

# Terminal 2: GRPO
python scripts/run_grpo_main.py \
  --model-path ai-sage/GigaChat3-10B-A1.8B-base \
  --vllm-url http://localhost:8000/v1 \
  --output-dir ./checkpoints/grpo --data-cache-dir ./data/raw \
  --num-iterations 1000 --batch-size 16 --mini-batch-size 4 \
  --n-total 1024 --n-low 2 --n-up 128 \
  --learning-rate 1e-6 --adv-clip 5.0 \
  --clip-ratio 0.2 --clip-ratio-high 0.28 \
  --exploration-bias 0.05 --entropy-coef 0.01 --kl-coef 0.0
```

**Quick test (small batch):**

```bash
MAX_SAMPLES=200 NUM_ITERATIONS=10 BATCH_SIZE=4 N_TOTAL=32 \
  bash scripts/run_grpo.sh
```

### Mode B: SFT + GRPO (full pipeline)

Use if the base model doesn't know the tool-calling format.

```bash
# Step 1: SFT warmup (2 epochs on tool_calling, ~316k samples)
python scripts/run_sft_main.py \
  --model-name ai-sage/GigaChat3-10B-A1.8B-base \
  --output-dir ./checkpoints/sft --data-cache-dir ./data/raw \
  --num-epochs 2 --batch-size 4 --gradient-accumulation-steps 8 \
  --learning-rate 2e-5 --max-length 4096

# Step 2: Launch vLLM with SFT checkpoint
MODEL=./checkpoints/sft/final bash scripts/run_vllm_server.sh

# Step 3: GRPO on SFT checkpoint (in another terminal)
MODEL_PATH=./checkpoints/sft/final bash scripts/run_grpo.sh
```

Or one command (without vLLM, HF fallback):

```bash
bash scripts/run_all.sh
```

### Mode C: SFT only

```bash
python scripts/run_sft_main.py \
  --model-name ai-sage/GigaChat3-10B-A1.8B-base \
  --output-dir ./checkpoints/sft --data-cache-dir ./data/raw \
  --num-epochs 2 --batch-size 4 --gradient-accumulation-steps 8 \
  --learning-rate 2e-5 --max-length 4096
```

Quick test:

```bash
python scripts/run_sft_main.py \
  --model-name ai-sage/GigaChat3-10B-A1.8B-base \
  --output-dir ./checkpoints/sft --data-cache-dir ./data/raw \
  --num-epochs 1 --batch-size 2 --gradient-accumulation-steps 4 --max-samples 500
```

---

## GPU Memory Layout (single H200 NVL, 140 GB)

| Component | VRAM | Notes |
|---|---|---|
| vLLM server (`gpu_util=0.45`) | ~63 GB | Model (~20 GB) + KV cache (~43 GB) |
| Policy model (trainer) | ~20 GB | bf16, 10B params (1.8B active MoE) |
| Gradients + optimizer | ~40 GB | AdamW states |
| Activations | ~15 GB | With gradient checkpointing |
| **Total** | **~138 GB** | Fits on single H200 |

KL penalty is **disabled by default** (`--kl-coef 0.0`) — no reference model loaded, saving ~20 GB. Enable with `--kl-coef 0.001` if needed (requires extra VRAM for ref model).

## Logging & Monitoring

### Terminal output

SFT:
```
  [SFT] step=    10 | loss=2.8314 | lr=1.23e-06 | epoch=1
  [SFT] step=    20 | loss=2.7921 | lr=2.46e-06 | epoch=1
```

GRPO:
```
====================================================================================================
[GRPO] Iteration 1/1000
====================================================================================================
  [vLLM async] Generating 1024 rollouts for 16 prompts (max_concurrent=64)
    51/1024 rollouts done (5%) | this: turns=3 tok=245 1.2s
    ...
  --- Rollouts (iter=1) ---
  Total: 1024 | Success: 128/1024 (12.5%) | Tokens: 312,000 (avg 305) | Gen: 45.2s

  [G1/16] prompt_abc123 | N=64 succ=12/64 (19%) | r=0.188 tok=312 turns=2.3
    GT: {"name": "get_weather", ...}
    BEST [r=1 tok=189]: <tool_call>{"name": "get_weather"...
    WORST[r=0 tok=512]: I'll help you with that. Let me...
    Diversity: 58/64 unique prefixes
```

### Log files

| File | Content |
|---|---|
| `./logs/knapsack_grpo_<ts>.jsonl` | GRPO iteration metrics |
| `./logs/knapsack_grpo_<ts>_metrics.jsonl` | Knapsack allocations + rollout details |
| `./logs/knapsack_grpo_<ts>.log` | Full Python logging |
| `./logs/sft_warmup_<ts>.jsonl` | SFT step metrics |

### Optional integrations

```bash
python scripts/run_grpo_main.py --use-wandb ...
```

## Important Notes

### GigaChat3 — MLA + DeepSeek-V2 architecture

- Uses **MLA (Multi-head Latent Attention)** — `flash_attn` does NOT support MLA
- vLLM requires MLA-specific backend: `--attention-backend TRITON_MLA`
  (alternatives: `FLASH_ATTN_MLA`, `FLASHMLA`, `FLASHINFER_MLA`)
- Tokenizer returns `token_type_ids` — automatically removed before `.generate()`
- No chat template — uses explicit role markers: `<|system|>\n`, `<|user|>\n`, `<|assistant|>\n`

### Gradient checkpointing & KV cache

| Phase | KV cache | Gradient checkpointing | Why |
|---|---|---|---|
| SFT training | OFF | ON | Saves memory, KV cache not needed |
| GRPO generation (vLLM) | ON | OFF | vLLM handles KV cache internally |
| GRPO generation (HF) | ON | OFF | `model.generate()` in eval mode |
| GRPO backward | OFF | ON | Same as SFT |

### Common errors

| Error | Fix |
|---|---|
| `flash_attn: MLA not supported` | Use `--attention-backend TRITON_MLA` |
| `flash_attn_2_cuda: undefined symbol` | `pip install flash-attn --no-build-isolation` or use TRITON_MLA |
| `token_type_ids not used` | Already handled (auto-removed) |
| `HFValidationError` on local paths | Already handled (auto-detected) |
| GPU memory leaked after kill | `kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader)` |
| `CUDA out of memory` (trainer) | Reduce `--gpu-memory-utilization` in vLLM, or use HF fallback (no `--vllm-url`) |

## Knapsack RL Algorithm

For each prompt with success rate `p`, the value of allocating `N` rollouts is:

```
Value(N, p) = P(nonzero_gradient | N, p) * InfoGain(p)
            = [1 - p^N - (1-p)^N]         * [p * (1-p)^2]
```

The DP solver maximizes total value subject to `sum(N_i) <= N_total`, where each prompt gets `N_i in [N_low, N_up]`.

**Warmup phase** (first 5 iterations): uniform allocation `N_i = N_total / batch_size` to estimate initial success rates.

**Steady state**: the allocator tracks per-prompt success rates and reallocates budget every iteration via the knapsack DP.

## Dataset

[nvidia/Nemotron-Agentic-v1](https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1):
- **tool_calling** (316k) — single-turn function calling -> SFT warmup
- **interactive_agent** (19k) — multi-turn agentic trajectories -> RLVR

## Model

[ai-sage/GigaChat3-10B-A1.8B-base](https://huggingface.co/ai-sage/GigaChat3-10B-A1.8B-base) — 10B parameter MoE (DeepSeek-V2 architecture) with 1.8B active parameters per token.
