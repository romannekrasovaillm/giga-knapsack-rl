# Giga-Knapsack-RL

Adaptive Rollout Budget Allocation for GRPO training of [GigaChat3-10B-A1.8B-base](https://huggingface.co/ai-sage/GigaChat3-10B-A1.8B-base) on agentic tool-calling tasks.

Based on the paper: *"Knapsack RL: Unlocking Exploration of LLMs via Optimizing Budget Allocation"* (Ziniu Li et al., ByteDance Seed).

## Overview

Instead of uniform N rollouts per prompt, Knapsack RL allocates budget adaptively via dynamic programming — giving more rollouts to prompts in the "golden zone" (p ~ 0.2-0.5) and fewer to trivial/impossible ones.

**Pipeline:**
1. **SFT Warmup** — 2 epochs on `tool_calling` subset (316k samples) to teach `<tool_call>` format
2. **Knapsack-GRPO** — RLVR on `interactive_agent` subset (19k multi-turn trajectories) with adaptive budget allocation + DAPO-style asymmetric exploration

**Key components:**
- **Knapsack DP solver** (Numba-accelerated) — multiple-choice knapsack for budget allocation
- **Simulated tool environment** — replays ground truth tool responses for multi-turn training
- **Hard programmatic verifier** — binary reward (correct/incorrect), no partial credit
- **DAPO asymmetric clipping** — `clip_high=1.28 > 1/clip_low=1.25` biases toward exploration
- **Comprehensive metrics** — BLEU, entropy per group/batch/rollout, token counts, generation time, effective gradient ratio

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
│   ├── run_sft.sh               # SFT launch script (torchrun)
│   ├── run_sft_main.py          # SFT entry point
│   ├── run_grpo.sh              # GRPO launch script (torchrun)
│   ├── run_grpo_main.py         # GRPO entry point (standalone)
│   ├── run_grpo_verl.py         # GRPO entry point (verl distributed)
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

```bash
git clone https://github.com/romannekrasovaillm/giga-knapsack-rl.git
cd giga-knapsack-rl
git checkout claude/knapsack-rl-budget-9n3AM
```

### Install

```bash
python -m venv .venv && source .venv/bin/activate

# PyTorch (pick your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Build deps (needed before flash-attn)
pip install numpy psutil ninja packaging setuptools wheel

# Flash Attention (optional — auto-fallback to SDPA if missing)
python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
# If True:
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.5cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
# If False:
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# Core dependencies
pip install "transformers>=4.45.0" "datasets>=2.20.0" "huggingface-hub>=0.24.0" \
  "tokenizers>=0.19.0" "accelerate>=0.33.0" "peft>=0.12.0" \
  "numba>=0.59.0" "tqdm>=4.66.0" "pyyaml>=6.0.0" \
  "antlr4-python3-runtime==4.9.3" "hydra-core>=1.3.0" "omegaconf>=2.3.0" \
  "jsonlines>=4.0.0" "pyarrow>=15.0.0" \
  "nltk>=3.8.0" "sacrebleu>=2.4.0" "pytest>=7.0.0"

python -c "import nltk; nltk.download('punkt_tab', quiet=True)"
pip install -e .
```

### Verify

```bash
pytest tests/ -v
```

### Run Full Pipeline

```bash
# 1. Download data
python scripts/download_data.py --cache-dir ./data/raw --split all

# 2. SFT warmup (2 epochs on tool_calling)
python scripts/run_sft_main.py \
  --model-name ai-sage/GigaChat3-10B-A1.8B-base \
  --output-dir ./checkpoints/sft \
  --data-cache-dir ./data/raw \
  --num-epochs 2 --batch-size 4 --gradient-accumulation-steps 8 \
  --learning-rate 2e-5 --max-length 4096

# 3. GRPO + Knapsack RL (interactive_agent)
python scripts/run_grpo_main.py \
  --model-path ./checkpoints/sft/final \
  --output-dir ./checkpoints/grpo \
  --data-cache-dir ./data/raw \
  --num-iterations 1000 --batch-size 16 --mini-batch-size 4 \
  --n-total 1024 --n-low 2 --n-up 128 \
  --learning-rate 1e-6 --adv-clip 5.0 \
  --clip-ratio 0.2 --clip-ratio-high 0.28 \
  --exploration-bias 0.05 --entropy-coef 0.01 --kl-coef 0.001
```

Or one command:

```bash
bash scripts/run_all.sh
```

### Quick Test (small batch, few iterations)

```bash
python scripts/run_sft_main.py \
  --model-name ai-sage/GigaChat3-10B-A1.8B-base \
  --output-dir ./checkpoints/sft --data-cache-dir ./data/raw \
  --num-epochs 1 --batch-size 2 --gradient-accumulation-steps 4 --max-samples 500

python scripts/run_grpo_main.py \
  --model-path ./checkpoints/sft/final --output-dir ./checkpoints/grpo \
  --data-cache-dir ./data/raw \
  --num-iterations 30 --batch-size 4 --mini-batch-size 2 \
  --n-total 32 --n-low 2 --n-up 16 --max-samples 200
```

## Logging & Monitoring

### Terminal output

SFT prints progress every `--log-steps` (default 10) optimizer steps:
```
  [SFT] step=    10 | loss=2.8314 | lr=1.23e-06 | epoch=1
  [SFT] step=    20 | loss=2.7921 | lr=2.46e-06 | epoch=1
  [Epoch 1] 100/79125 (0.1%) | loss=2.6543 | 42s elapsed
```

GRPO prints a full metrics table every iteration:
```
====================================================================================================
[GRPO] Iteration 1
====================================================================================================
  Prompts in batch                  16
  Total rollouts                    128
  Mean group size                   8.0
  ──────────────────────────────────────────────────────────────────────────────────────────────────
  Batch mean reward                 0.1250
  Batch success rate                12.50%
  Effective gradient ratio          68.75%
  ...
```

### Log files

| File | Content |
|---|---|
| `./logs/sft_warmup_<ts>.jsonl` | SFT step-by-step metrics (loss, lr, epoch) |
| `./logs/sft_warmup_<ts>.log` | Full Python logging output |
| `./logs/knapsack_grpo_<ts>.jsonl` | GRPO iteration metrics (rewards, BLEU, advantages, etc.) |
| `./logs/knapsack_grpo_<ts>_metrics.jsonl` | Knapsack allocation details per iteration |

### Optional integrations

```bash
# Weights & Biases
python scripts/run_grpo_main.py --use-wandb ...

# TensorBoard (enable in config)
tensorboard --logdir ./logs/tensorboard/
```

## Important Notes

### Base model — no chat template

GigaChat3-10B-A1.8B-base is a post-pretrain model **without a chat template**. SFT uses explicit role markers instead of `tokenizer.apply_chat_template()`:

```
<|system|>
You are a helpful assistant with access to tools.
<|user|>
What's the weather in Moscow?
<|assistant|>
<tool_call>{"name": "get_weather", "arguments": {"city": "Moscow"}}</tool_call>
<|end|>
```

### Gradient checkpointing & KV cache

During SFT/GRPO training, gradient checkpointing is enabled to save ~40% GPU memory. This automatically disables KV cache (`use_cache=False`) — this is correct behavior:

| Phase | KV cache | Gradient checkpointing | Why |
|---|---|---|---|
| SFT training | OFF | ON | Saves memory, KV cache not needed for training |
| GRPO generation | ON | OFF | `model.generate()` in eval mode uses KV cache |
| GRPO backward | OFF | ON | Same as SFT |
| Inference | ON | OFF | Full KV cache for fast generation |

### Attention backend auto-detection

`src/utils.py` automatically selects the best attention implementation:
1. `flash_attention_2` — if flash-attn installed
2. `sdpa` — PyTorch native scaled dot-product attention (default fallback)
3. `eager` — manual attention (slowest, always works)

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
- **tool_calling** (316k) — single-turn function calling → SFT warmup
- **interactive_agent** (19k) — multi-turn agentic trajectories → RLVR

## Metrics Logged

Per rollout, per group, per batch:
- Rewards, success rate, effective gradient ratio
- BLEU score against ground truth
- Token-level entropy
- Token counts, generation time
- Advantages (mean, std, min, max)
- Knapsack allocation (budget used, utilization, distribution)

## Model

[ai-sage/GigaChat3-10B-A1.8B-base](https://huggingface.co/ai-sage/GigaChat3-10B-A1.8B-base) — 10B parameter MoE with 1.8B active parameters.
