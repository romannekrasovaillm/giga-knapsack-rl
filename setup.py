from setuptools import setup, find_packages

setup(
    name="giga-knapsack-rl",
    version="0.1.0",
    description=(
        "Knapsack RL: Adaptive Rollout Budget Allocation for GRPO training "
        "of GigaChat3-10B-A1.8B-base on agentic tasks"
    ),
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.4.0",
        "transformers>=4.45.0",
        "datasets>=2.20.0",
        "huggingface-hub>=0.24.0",
        "numba>=0.59.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0.0",
    ],
    extras_require={
        "verl": ["verl>=0.3.0", "vllm>=0.8.2", "ray[default]>=2.35.0"],
        "logging": ["wandb>=0.17.0", "tensorboard>=2.17.0"],
        "metrics": ["nltk>=3.8.0", "sacrebleu>=2.4.0"],
        "dev": ["pytest>=7.0.0"],
    },
)
