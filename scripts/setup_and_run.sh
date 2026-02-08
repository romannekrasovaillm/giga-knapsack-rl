#!/bin/bash
# ==========================================================================
# Giga-Knapsack-RL: Полный скрипт развертывания и запуска
# ==========================================================================
#
# Использование:
#   bash scripts/setup_and_run.sh              # полный пайплайн
#   bash scripts/setup_and_run.sh --deps-only  # только установка зависимостей
#   bash scripts/setup_and_run.sh --test       # быстрый тестовый прогон
#
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# ---- Параметры ----
MODE="${1:---full}"    # --full | --deps-only | --test
VENV_NAME="${VENV_NAME:-knapsack-rl}"
PYTHON="${PYTHON:-python}"
CUDA_VERSION="${CUDA_VERSION:-cu124}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
PY_VERSION=$($PYTHON -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')" 2>/dev/null || echo "cp311")

echo "============================================================"
echo "  Giga-Knapsack-RL: Setup & Training Pipeline"
echo "============================================================"
echo "  Project:   $PROJECT_DIR"
echo "  Python:    $($PYTHON --version 2>&1)"
echo "  Mode:      $MODE"
echo "============================================================"

# ==================================================================
# ЭТАП 1: ЗАВИСИМОСТИ
# ==================================================================
echo ""
echo "[1/6] Установка зависимостей..."

# 1a. Зависимости сборки (нужны ДО flash-attn)
echo "  -> build-зависимости (numpy, psutil, ninja, packaging)..."
pip install numpy psutil ninja packaging setuptools wheel --quiet

# 1b. PyTorch (если ещё нет)
if $PYTHON -c "import torch" 2>/dev/null; then
    TORCH_VER=$($PYTHON -c "import torch; print(torch.__version__)")
    CUDA_VER=$($PYTHON -c "import torch; print(torch.version.cuda or 'cpu')")
    echo "  -> PyTorch $TORCH_VER (CUDA $CUDA_VER) уже установлен"
else
    echo "  -> Устанавливаю PyTorch ${TORCH_VERSION}+${CUDA_VERSION}..."
    pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/${CUDA_VERSION}
fi

# 1c. Flash Attention 2
if $PYTHON -c "import flash_attn" 2>/dev/null; then
    echo "  -> flash-attn уже установлен"
else
    echo "  -> Устанавливаю flash-attn..."
    # Определяем CXX11 ABI
    CXX_ABI=$($PYTHON -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')" 2>/dev/null || echo "TRUE")
    # Определяем минорную версию torch (2.5 -> 2.5, 2.4 -> 2.4)
    TORCH_MINOR=$($PYTHON -c "import torch; v=torch.__version__.split('+')[0].split('.')[:2]; print('.'.join(v))" 2>/dev/null || echo "2.5")

    # Попытка 1: готовый wheel с GitHub releases (cu12, не cu124!)
    # URL-encoded '+' = %2B
    FLASH_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch${TORCH_MINOR}cxx11abi${CXX_ABI}-${PY_VERSION}-${PY_VERSION}-linux_x86_64.whl"
    echo "  -> Пробую wheel: torch=${TORCH_MINOR}, ABI=${CXX_ABI}, Python=${PY_VERSION}"
    if pip install "$FLASH_WHEEL" 2>/dev/null; then
        echo "  -> flash-attn установлен из wheel"
    else
        echo "  -> Wheel не подошёл, пробую альтернативный ABI..."
        ALT_ABI="FALSE"
        [ "$CXX_ABI" = "FALSE" ] && ALT_ABI="TRUE"
        FLASH_WHEEL_ALT="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch${TORCH_MINOR}cxx11abi${ALT_ABI}-${PY_VERSION}-${PY_VERSION}-linux_x86_64.whl"
        if pip install "$FLASH_WHEEL_ALT" 2>/dev/null; then
            echo "  -> flash-attn установлен (ABI=${ALT_ABI})"
        else
            echo "  -> Готовые wheels недоступны, собираю из исходников..."
            echo "     (это займёт 10-20 минут, нужен nvcc)"
            MAX_JOBS=4 pip install flash-attn --no-build-isolation --quiet || \
                echo "  -> ПРЕДУПРЕЖДЕНИЕ: flash-attn не установлен, используется SDPA fallback"
        fi
    fi
fi

# 1d. Основные зависимости проекта
echo "  -> Основные зависимости проекта..."
pip install "transformers>=4.45.0" "datasets>=2.20.0" "huggingface-hub>=0.24.0" \
    "tokenizers>=0.19.0" "accelerate>=0.33.0" "peft>=0.12.0" \
    "numba>=0.59.0" "tqdm>=4.66.0" "pyyaml>=6.0.0" \
    "antlr4-python3-runtime==4.9.3" "hydra-core>=1.3.0" "omegaconf>=2.3.0" \
    "jsonlines>=4.0.0" "pyarrow>=15.0.0" --quiet

# 1e. Метрики и логирование
echo "  -> Метрики (nltk, wandb, tensorboard)..."
pip install "nltk>=3.8.0" "sacrebleu>=2.4.0" "wandb>=0.17.0" "tensorboard>=2.17.0" --quiet
$PYTHON -c "import nltk; nltk.download('punkt_tab', quiet=True)" 2>/dev/null || true

# 1f. verl + vllm (опционально, для distributed)
echo "  -> verl + vllm (опционально)..."
pip install "verl>=0.3.0" "vllm>=0.8.2" "ray[default]>=2.35.0" 2>/dev/null \
    && echo "  -> verl установлен" \
    || echo "  -> verl/vllm не установились (не критично, standalone режим работает)"

# 1g. Проект как пакет
echo "  -> Устанавливаю проект в dev-режиме..."
pip install -e . --quiet 2>/dev/null || pip install -e ".[dev]" --quiet 2>/dev/null || true

# 1h. Тесты
pip install pytest>=7.0.0 --quiet

echo ""
echo "[1/6] Зависимости установлены."

if [ "$MODE" = "--deps-only" ]; then
    echo "Режим --deps-only: установка завершена."
    exit 0
fi

# ==================================================================
# ЭТАП 2: ПРОВЕРКА ОКРУЖЕНИЯ
# ==================================================================
echo ""
echo "[2/6] Проверка окружения..."

$PYTHON -c "
import torch
print(f'  PyTorch:        {torch.__version__}')
print(f'  CUDA:           {torch.version.cuda or \"CPU only\"}')
print(f'  GPU count:      {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}:          {torch.cuda.get_device_name(i)}')
try:
    import flash_attn
    print(f'  Flash Attention: {flash_attn.__version__}')
except ImportError:
    print('  Flash Attention: НЕ УСТАНОВЛЕН (модель будет работать без него)')
import transformers
print(f'  Transformers:   {transformers.__version__}')
import numba
print(f'  Numba:          {numba.__version__}')
"

# Быстрый тест Knapsack DP
echo "  -> Быстрый тест Knapsack DP..."
$PYTHON -c "
from src.knapsack.task_value import task_value, prob_nonzero_gradient
v = task_value(16, 0.3)
p = prob_nonzero_gradient(16, 0.05)
print(f'  task_value(N=16, p=0.3) = {v:.4f}')
print(f'  P(nonzero|N=16, p=0.05) = {p:.4f}')
print('  -> Knapsack DP OK')
"

# Тесты (без GPU-зависимых)
echo "  -> Юнит-тесты..."
$PYTHON -m pytest tests/test_knapsack.py tests/test_verifier.py tests/test_data.py -v --tb=short 2>&1 | tail -15

echo ""
echo "[2/6] Окружение проверено."

# ==================================================================
# ЭТАП 3: СКАЧИВАНИЕ ДАННЫХ
# ==================================================================
echo ""
echo "[3/6] Скачивание nvidia/Nemotron-Agentic-v1..."

$PYTHON scripts/download_data.py --cache-dir ./data/raw --split all

echo "  -> Валидация датасета..."
$PYTHON scripts/prepare_data.py --cache-dir ./data/raw --max-samples 2000

echo ""
echo "[3/6] Данные загружены."

# ==================================================================
# ЭТАП 4: SFT WARMUP
# ==================================================================
echo ""
echo "[4/6] SFT Warmup (tool_calling, 2 эпохи)..."

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")

if [ "$MODE" = "--test" ]; then
    echo "  -> ТЕСТОВЫЙ РЕЖИМ: 500 сэмплов, 1 эпоха"
    $PYTHON scripts/run_sft_main.py \
        --model-name ai-sage/GigaChat3-10B-A1.8B-base \
        --output-dir ./checkpoints/sft \
        --data-cache-dir ./data/raw \
        --log-dir ./logs \
        --num-epochs 1 \
        --batch-size 2 \
        --gradient-accumulation-steps 4 \
        --learning-rate 2e-5 \
        --max-length 4096 \
        --max-samples 500 \
        --save-steps 100
else
    echo "  -> ПОЛНЫЙ РЕЖИМ: все данные, 2 эпохи, $NUM_GPUS GPU"
    if [ "$NUM_GPUS" -gt 1 ]; then
        torchrun \
            --nproc_per_node="$NUM_GPUS" \
            --master_port=29500 \
            scripts/run_sft_main.py \
            --model-name ai-sage/GigaChat3-10B-A1.8B-base \
            --output-dir ./checkpoints/sft \
            --data-cache-dir ./data/raw \
            --log-dir ./logs \
            --num-epochs 2 \
            --batch-size 4 \
            --gradient-accumulation-steps 8 \
            --learning-rate 2e-5 \
            --max-length 4096 \
            --save-steps 500
    else
        $PYTHON scripts/run_sft_main.py \
            --model-name ai-sage/GigaChat3-10B-A1.8B-base \
            --output-dir ./checkpoints/sft \
            --data-cache-dir ./data/raw \
            --log-dir ./logs \
            --num-epochs 2 \
            --batch-size 4 \
            --gradient-accumulation-steps 8 \
            --learning-rate 2e-5 \
            --max-length 4096 \
            --save-steps 500
    fi
fi

echo ""
echo "[4/6] SFT Warmup завершён. Чекпойнт: ./checkpoints/sft/final"

# ==================================================================
# ЭТАП 5: KNAPSACK-GRPO (RLVR)
# ==================================================================
echo ""
echo "[5/6] Knapsack-GRPO (interactive_agent, RLVR)..."

if [ "$MODE" = "--test" ]; then
    echo "  -> ТЕСТОВЫЙ РЕЖИМ: 200 сэмплов, 30 итераций"
    $PYTHON scripts/run_grpo_main.py \
        --model-path ./checkpoints/sft/final \
        --output-dir ./checkpoints/grpo \
        --data-cache-dir ./data/raw \
        --log-dir ./logs \
        --num-iterations 30 \
        --batch-size 4 \
        --mini-batch-size 2 \
        --n-total 32 \
        --n-low 2 \
        --n-up 16 \
        --max-samples 200 \
        --learning-rate 1e-6 \
        --adv-clip 5.0 \
        --clip-ratio 0.2 \
        --clip-ratio-high 0.28 \
        --exploration-bias 0.05 \
        --entropy-coef 0.01 \
        --kl-coef 0.001 \
        --save-every 10
else
    echo "  -> ПОЛНЫЙ РЕЖИМ: все данные, 1000 итераций"
    $PYTHON scripts/run_grpo_main.py \
        --model-path ./checkpoints/sft/final \
        --output-dir ./checkpoints/grpo \
        --data-cache-dir ./data/raw \
        --log-dir ./logs \
        --num-iterations 1000 \
        --batch-size 16 \
        --mini-batch-size 4 \
        --n-total 1024 \
        --n-low 2 \
        --n-up 128 \
        --learning-rate 1e-6 \
        --adv-clip 5.0 \
        --clip-ratio 0.2 \
        --clip-ratio-high 0.28 \
        --exploration-bias 0.05 \
        --entropy-coef 0.01 \
        --kl-coef 0.001 \
        --save-every 50
fi

echo ""
echo "[5/6] Knapsack-GRPO завершён."

# ==================================================================
# ЭТАП 6: ИТОГ
# ==================================================================
echo ""
echo "============================================================"
echo "[6/6] ПАЙПЛАЙН ЗАВЕРШЁН"
echo "============================================================"
echo ""
echo "  Чекпойнты:"
echo "    SFT:   ./checkpoints/sft/final/"
echo "    GRPO:  ./checkpoints/grpo/"
echo ""
echo "  Логи:"
echo "    ./logs/*.log          — текстовые логи"
echo "    ./logs/*.jsonl        — метрики (JSON lines)"
echo ""
echo "  Просмотр метрик:"
echo "    tail -f logs/knapsack_grpo_*.jsonl | python -m json.tool"
echo ""
echo "============================================================"
