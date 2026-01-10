"""
GPU and CUDA utilities for training.
"""

import os
from typing import Optional, List, Tuple
import torch
from loguru import logger


def get_device(prefer_gpu: bool = True, gpu_id: int = 0) -> torch.device:
    """
    Get the best available device.

    Args:
        prefer_gpu: Prefer GPU if available
        gpu_id: GPU device ID to use

    Returns:
        torch.device
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    return device


def print_gpu_info() -> None:
    """Print detailed GPU information."""
    if not torch.cuda.is_available():
        logger.info("CUDA is not available. Using CPU.")
        return

    logger.info("=" * 50)
    logger.info("GPU Information")
    logger.info("=" * 50)

    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"\nGPU {i}: {props.name}")
        logger.info(f"  Compute capability: {props.major}.{props.minor}")
        logger.info(f"  Total memory: {props.total_memory / 1024**3:.1f} GB")
        logger.info(f"  Multi-processors: {props.multi_processor_count}")

    logger.info("=" * 50)


def get_gpu_memory_info(device: int = 0) -> Tuple[float, float, float]:
    """
    Get GPU memory information.

    Args:
        device: GPU device ID

    Returns:
        (total_memory_gb, allocated_memory_gb, cached_memory_gb)
    """
    if not torch.cuda.is_available():
        return (0.0, 0.0, 0.0)

    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    cached = torch.cuda.memory_reserved(device) / 1024**3

    return (total, allocated, cached)


def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.debug("Cleared GPU memory cache")


def set_gpu_memory_fraction(fraction: float = 0.9) -> None:
    """
    Set maximum GPU memory fraction to use.

    Args:
        fraction: Fraction of GPU memory to use (0-1)
    """
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)
        logger.info(f"Set GPU memory fraction to {fraction:.0%}")


def enable_cudnn_benchmark(enable: bool = True) -> None:
    """
    Enable cuDNN benchmark mode for faster training.

    Args:
        enable: Whether to enable benchmark mode
    """
    torch.backends.cudnn.benchmark = enable
    if enable:
        logger.info("Enabled cuDNN benchmark mode")


def enable_tf32(enable: bool = True) -> None:
    """
    Enable TF32 for Ampere GPUs (faster training with slight precision loss).

    Args:
        enable: Whether to enable TF32
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = enable
        torch.backends.cudnn.allow_tf32 = enable
        if enable:
            logger.info("Enabled TF32 for matrix operations")


def setup_gpu_environment(
    gpu_id: int = 0,
    memory_fraction: float = 0.9,
    enable_benchmark: bool = True,
    enable_tf32_mode: bool = True
) -> torch.device:
    """
    Setup GPU environment with optimal settings.

    Args:
        gpu_id: GPU device ID
        memory_fraction: Fraction of memory to use
        enable_benchmark: Enable cuDNN benchmark
        enable_tf32_mode: Enable TF32 mode

    Returns:
        Configured device
    """
    # Set visible devices
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    device = get_device(prefer_gpu=True, gpu_id=0)  # 0 because of CUDA_VISIBLE_DEVICES

    if torch.cuda.is_available():
        set_gpu_memory_fraction(memory_fraction)
        enable_cudnn_benchmark(enable_benchmark)
        enable_tf32(enable_tf32_mode)

    print_gpu_info()

    return device


class GPUMemoryMonitor:
    """
    Context manager for monitoring GPU memory usage.
    """

    def __init__(self, device: int = 0, label: str = ""):
        self.device = device
        self.label = label
        self.start_allocated = 0
        self.start_cached = 0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_allocated = torch.cuda.memory_allocated(self.device)
            self.start_cached = torch.cuda.memory_reserved(self.device)
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            end_allocated = torch.cuda.memory_allocated(self.device)
            end_cached = torch.cuda.memory_reserved(self.device)

            allocated_diff = (end_allocated - self.start_allocated) / 1024**2
            cached_diff = (end_cached - self.start_cached) / 1024**2

            prefix = f"[{self.label}] " if self.label else ""
            logger.debug(
                f"{prefix}GPU memory change: "
                f"Allocated: {allocated_diff:+.1f} MB, "
                f"Cached: {cached_diff:+.1f} MB"
            )


def optimize_model_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    """
    Optimize model for inference.

    Args:
        model: PyTorch model

    Returns:
        Optimized model
    """
    model.eval()

    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False

    # Try to use torch.compile if available (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("Applied torch.compile optimization")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")

    return model


def get_optimal_batch_size(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    max_batch_size: int = 512,
    safety_factor: float = 0.8
) -> int:
    """
    Estimate optimal batch size based on GPU memory.

    Args:
        model: PyTorch model
        sample_input: Sample input tensor (batch_size=1)
        max_batch_size: Maximum batch size to try
        safety_factor: Memory usage safety factor

    Returns:
        Recommended batch size
    """
    if not torch.cuda.is_available():
        return 32  # Default for CPU

    device = next(model.parameters()).device

    # Get available memory
    total, allocated, _ = get_gpu_memory_info()
    available = (total - allocated) * safety_factor

    # Estimate memory per sample
    clear_gpu_memory()

    model.eval()
    with torch.no_grad():
        # Forward pass with batch size 1
        start_mem = torch.cuda.memory_allocated()
        _ = model(sample_input.to(device))
        end_mem = torch.cuda.memory_allocated()

    mem_per_sample = (end_mem - start_mem) / (1024**3)  # In GB

    if mem_per_sample <= 0:
        return 64  # Default if estimation fails

    # Calculate batch size
    estimated_batch_size = int(available / mem_per_sample)
    optimal_batch_size = min(estimated_batch_size, max_batch_size)

    # Round to power of 2
    optimal_batch_size = 2 ** int(optimal_batch_size).bit_length() // 2

    logger.info(f"Estimated optimal batch size: {optimal_batch_size}")
    return max(1, optimal_batch_size)
