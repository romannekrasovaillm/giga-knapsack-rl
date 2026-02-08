def __getattr__(name):
    if name == "NemotronAgenticLoader":
        from .loader import NemotronAgenticLoader
        return NemotronAgenticLoader
    if name == "SFTToolCallingDataset":
        from .sft_dataset import SFTToolCallingDataset
        return SFTToolCallingDataset
    if name == "RLVRAgenticDataset":
        from .rlvr_dataset import RLVRAgenticDataset
        return RLVRAgenticDataset
    if name == "AgenticTrajectoryParser":
        from .parser import AgenticTrajectoryParser
        return AgenticTrajectoryParser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
