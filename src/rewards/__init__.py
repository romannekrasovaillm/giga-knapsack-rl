def __getattr__(name):
    if name == "HardVerifier":
        from .verifier import HardVerifier
        return HardVerifier
    if name == "KnapsackRewardManager":
        from .reward_manager import KnapsackRewardManager
        return KnapsackRewardManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
