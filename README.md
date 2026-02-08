# Giga-Knapsack-RL

Adaptive Rollout Budget Allocation for GRPO training of [GigaChat3-10B-A1.8B-base](https://huggingface.co/ai-sage/GigaChat3-10B-A1.8B-base) on agentic tool-calling tasks.

Based on the paper: *"Knapsack RL: Unlocking Exploration of LLMs via Optimizing Budget Allocation"* (Ziniu Li et al., ByteDance Seed).

## Overview

Instead of uniform N rollouts per prompt, Knapsack RL allocates budget adaptively via dynamic programming — giving more rollouts to prompts in the "golden zone" (p ≈ 0.2–0.5) and fewer to trivial/impossible ones.

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
├── configs/
│   ├── base.yaml                # Shared config
│   ├── sft_warmup.yaml          # SFT hyperparams
│   └── grpo_knapsack.yaml       # GRPO + Knapsack RL + DAPO + verl config
├── src/
│   ├── data/                    # Dataset loading & parsing (Nemotron Agentic v1)
│   ├── environment/             # Simulated tool environment
│   ├── rewards/                 # Hard verifier + reward manager
│   ├── knapsack/                # DP solver + value functions + budget allocator
│   ├── grpo/                    # Advantage, DAPO sampling, policy loss, trainer
│   ├── metrics/                 # Tracker + logger (terminal, JSONL, wandb)
│   ├── sft/                     # SFT warmup trainer
│   └── utils.py                 # Auto-detect attention backend & dtype
├── scripts/
│   ├── setup_and_run.sh         # One-command install + full pipeline
│   ├── download_data.py         # Download Nemotron dataset
│   ├── prepare_data.py          # Parse & report statistics
│   ├── run_sft.sh / run_sft_main.py
│   ├── run_grpo.sh / run_grpo_main.py
│   ├── run_grpo_verl.py         # verl distributed backend
│   └── run_all.sh               # Full pipeline end-to-end
├── tests/                       # 40 tests (knapsack, verifier, data, grpo)
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

# Flash Attention (optional, auto-fallback to SDPA if missing)
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
# Download data
python scripts/download_data.py --cache-dir ./data/raw --split all

# SFT warmup (2 epochs on tool_calling)
python scripts/run_sft_main.py \
  --model-name ai-sage/GigaChat3-10B-A1.8B-base \
  --output-dir ./checkpoints/sft \
  --data-cache-dir ./data/raw \
  --num-epochs 2 --batch-size 4 --gradient-accumulation-steps 8 \
  --learning-rate 2e-5 --max-length 4096

# GRPO + Knapsack RL (interactive_agent)
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

## Knapsack RL Algorithm

For each prompt with success rate `p`, the value of allocating `N` rollouts is:

```
Value(N, p) = P(nonzero_gradient | N, p) × InfoGain(p)
            = [1 - p^N - (1-p)^N]       × [p × (1-p)^2]
```

The DP solver maximizes total value subject to `Σ N_i ≤ N_total`, where each prompt gets `N_i ∈ [N_low, N_up]`.

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

Output: terminal tables + `./logs/*.jsonl` + optional wandb/tensorboard.

## Model

[ai-sage/GigaChat3-10B-A1.8B-base](https://huggingface.co/ai-sage/GigaChat3-10B-A1.8B-base) — 10B parameter MoE with 1.8B active parameters.
