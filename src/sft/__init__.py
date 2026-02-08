def __getattr__(name):
    if name == "SFTWarmupTrainer":
        from .trainer import SFTWarmupTrainer
        return SFTWarmupTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
