# Lazy imports to avoid pulling heavy dependencies on simple imports
def __getattr__(name):
    if name == "knapsack_grpo_advantage":
        from .advantage import knapsack_grpo_advantage
        return knapsack_grpo_advantage
    if name == "dapo_importance_weights":
        from .dapo_sampling import dapo_importance_weights
        return dapo_importance_weights
    if name == "KnapsackGRPOTrainer":
        from .trainer import KnapsackGRPOTrainer
        return KnapsackGRPOTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
