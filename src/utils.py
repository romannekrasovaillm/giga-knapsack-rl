"""
Utility: detect best attention implementation available.
"""

import logging

logger = logging.getLogger(__name__)


def get_attn_implementation() -> str:
    """Return best available attention implementation."""
    try:
        import flash_attn  # noqa: F401
        logger.info(f"Flash Attention {flash_attn.__version__} detected")
        return "flash_attention_2"
    except ImportError:
        pass

    try:
        import torch
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            logger.info("Using SDPA (torch.nn.functional.scaled_dot_product_attention)")
            return "sdpa"
    except ImportError:
        pass

    logger.info("Using eager attention (no flash-attn, no SDPA)")
    return "eager"


def get_torch_dtype():
    """Return best dtype: bf16 if GPU supports it, else fp32."""
    import torch
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 8:  # Ampere+
            return torch.bfloat16
        return torch.float16
    return torch.float32
