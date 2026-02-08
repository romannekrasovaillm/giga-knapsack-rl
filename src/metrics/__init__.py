def __getattr__(name):
    if name == "TrainingLogger":
        from .logger import TrainingLogger
        return TrainingLogger
    if name == "MetricsTracker":
        from .tracker import MetricsTracker
        return MetricsTracker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
