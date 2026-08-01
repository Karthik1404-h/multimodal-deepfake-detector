"""Utility sub-package for deepfake_core.

Provides GPU memory management and general helper functions.
"""

from deepfake_core.utils.vram_manager import (
    clear_gpu_cache,
    get_vram_summary,
    get_vram_usage,
    is_cuda_available,
)

__all__ = [
    "clear_gpu_cache",
    "get_vram_summary",
    "get_vram_usage",
    "is_cuda_available",
]
