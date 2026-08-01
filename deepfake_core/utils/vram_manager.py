"""VRAM management utilities for GPU memory optimization.

Provides helper functions for explicit GPU cache management,
critical for operating within the 6 GB VRAM budget of RTX 4050.
"""

from __future__ import annotations

import gc
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def is_cuda_available() -> bool:
    """Check whether CUDA-capable GPU is available.

    Returns:
        True if CUDA is available and at least one GPU is detected.
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        logger.warning("PyTorch is not installed — CUDA check unavailable.")
        return False


def clear_gpu_cache() -> None:
    """Aggressively free unused GPU memory.

    Runs Python garbage collection followed by PyTorch's CUDA cache
    clearing. Safe to call even when CUDA is not available (no-op).
    """
    gc.collect()

    if not is_cuda_available():
        return

    import torch
    torch.cuda.empty_cache()
    logger.debug(
        "GPU cache cleared. Current allocation: %.2f MB",
        get_vram_usage() or 0.0,
    )


def get_vram_usage(device_index: int = 0) -> Optional[float]:
    """Return current GPU memory usage in megabytes.

    Args:
        device_index: CUDA device index (default 0 for single-GPU setups).

    Returns:
        Memory allocated in MB, or None if CUDA is unavailable.
    """
    if not is_cuda_available():
        logger.debug("CUDA not available — cannot report VRAM usage.")
        return None

    import torch

    try:
        allocated_bytes = torch.cuda.memory_allocated(device_index)
        return allocated_bytes / (1024 ** 2)
    except RuntimeError as exc:
        logger.error("Failed to query VRAM usage for device %d: %s", device_index, exc)
        return None


def get_vram_summary(device_index: int = 0) -> Optional[dict[str, float]]:
    """Return a detailed GPU memory summary.

    Args:
        device_index: CUDA device index.

    Returns:
        Dict with keys 'allocated_mb', 'reserved_mb', 'total_mb',
        or None if CUDA is unavailable.
    """
    if not is_cuda_available():
        return None

    import torch

    try:
        total = torch.cuda.get_device_properties(device_index).total_mem
        allocated = torch.cuda.memory_allocated(device_index)
        reserved = torch.cuda.memory_reserved(device_index)

        return {
            "allocated_mb": allocated / (1024 ** 2),
            "reserved_mb": reserved / (1024 ** 2),
            "total_mb": total / (1024 ** 2),
        }
    except RuntimeError as exc:
        logger.error("Failed to get VRAM summary for device %d: %s", device_index, exc)
        return None
