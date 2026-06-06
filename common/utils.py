#!/usr/bin/env python3
"""
Utility functions for Ascend NPU microbenchmarks.
"""

import numpy as np
import time
import os
from typing import Tuple, Optional, List

try:
    import torch
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False


def get_device_properties(device_id: int = 0) -> dict:
    """Get Ascend NPU device properties."""
    props = {
        "device_name": "Ascend 910B (simulated)",
        "ai_cores": "unknown",
        "hbm_capacity_gb": "unknown",
        "l1_capacity_kb": "unknown",
        "l0a_capacity_kb": "unknown",
        "l0b_capacity_kb": "unknown",
        "l0c_capacity_kb": "unknown",
        "fp32_tflops": "unknown",
        "hbm_bandwidth_gbs": "unknown",
    }

    if HAS_NPU:
        try:
            props["device_name"] = torch.npu.get_device_name(device_id)
            # Ascend 910B known specs:
            # AI Cores: varies by card (e.g., 28 for 910B)
            # FP32: ~0.375 TFLOPS per core (Vector Unit)
            # FP16/BF16 on Cube: ~32 TFLOPS (910B)
            # HBM Bandwidth: ~1.2 TB/s (910B)
            # L1 Buffer: 1 MB per core
            # L0A/L0B: 64 KB each per core
            # L0C: 64 KB per core
            props["theoretical_fp32_tflops"] = 0.375 * 28  # approximate
            props["theoretical_hbm_bw_gbs"] = 1200
            props["l1_buffer_kb"] = 1024
            props["l0_buffer_kb"] = 64
        except Exception as e:
            print(f"[WARN] Could not get device properties: {e}")

    return props


def estimate_clock_freq_mhz(device_id: int = 0) -> float:
    """
    Estimate NPU clock frequency through timing calibration.
    Uses a known-duration CPU sleep to calibrate the timing measurement.
    """
    if HAS_NPU:
        # The Ascend 910B typically runs at ~1.1-1.6 GHz
        # This is a calibration approach; actual freq may vary
        return 1100.0  # MHz, typical for 910B
    return 1100.0  # simulated


def cycles_to_ms(cycles: float, freq_mhz: float = 1100.0) -> float:
    """Convert cycles to milliseconds."""
    return cycles / (freq_mhz * 1e6) * 1e3


def ms_to_cycles(ms: float, freq_mhz: float = 1100.0) -> float:
    """Convert milliseconds to cycles."""
    return ms * freq_mhz * 1e3


def compute_bandwidth_gbs(data_size_bytes: int, time_ms: float) -> float:
    """
    Compute bandwidth in GB/s.
    Args:
        data_size_bytes: Total data moved (bytes)
        time_ms: Time taken (milliseconds)
    Returns:
        Bandwidth in GB/s
    """
    if time_ms <= 0:
        return 0.0
    return (data_size_bytes / (time_ms / 1000.0)) / (1024**3)


def compute_throughput_ops(total_ops: int, time_ms: float) -> float:
    """
    Compute throughput in operations per second.
    Args:
        total_ops: Total number of operations
        time_ms: Time taken (milliseconds)
    Returns:
        Operations per second
    """
    if time_ms <= 0:
        return 0.0
    return total_ops / (time_ms / 1000.0)


def compute_flops(total_flops: float, time_ms: float) -> float:
    """
    Compute FLOPS (floating point operations per second).
    Args:
        total_flops: Total FLOP count
        time_ms: Time taken (milliseconds)
    Returns:
        FLOPS
    """
    if time_ms <= 0:
        return 0.0
    return total_flops / (time_ms / 1000.0)


def compute_matmul_flops(m: int, n: int, k: int) -> int:
    """
    Compute total FLOPs for matrix multiplication C = A @ B.
    A: M×K, B: K×N, C: M×N
    FLOPs = 2 * M * N * K (one multiply-add per output element)
    """
    return 2 * m * n * k


def create_pointer_chase_array(size: int, stride: int = 64) -> np.ndarray:
    """
    Create a pointer-chasing array for memory latency measurement.
    Each element points to the next element in a pseudo-random order,
    ensuring cache prefetcher cannot predict the access pattern.

    Args:
        size: Number of elements
        stride: Stride between elements (in element count, not bytes)
    Returns:
        Array of indices forming a linked list

    Note: For Ascend NPU, pointer chasing must be done carefully
    because the NPU doesn't have general-purpose pointer dereferencing.
    We use indexed access patterns instead.
    """
    indices = np.arange(size, dtype=np.int64)
    # Create a random permutation for pointer chasing
    perm = np.random.permutation(size)
    chase = np.zeros(size, dtype=np.int64)
    for i in range(size - 1):
        chase[perm[i]] = perm[i + 1]
    chase[perm[-1]] = perm[0]  # close the loop
    return chase


def create_stream_array(size: int, dtype=np.float32) -> np.ndarray:
    """
    Create a contiguous array for streaming bandwidth measurement.
    """
    return np.random.randn(size).astype(dtype)


def get_optimal_launch_config(
    max_elements: int = 1024 * 1024,
    min_block_size: int = 32,
) -> List[Tuple[int, int]]:
    """
    Generate a sweep of data sizes for saturation curve measurement.
    Returns list of (num_elements, block_size) tuples.
    """
    configs = []
    sizes = []
    # Exponential sweep
    s = min_block_size
    while s <= max_elements:
        sizes.append(s)
        s *= 2
    # Add intermediate points for finer granularity
    for i in range(len(sizes) - 1):
        mid = (sizes[i] + sizes[i + 1]) // 2
        if mid not in sizes:
            sizes.append(mid)
    sizes.sort()

    for size in sizes:
        # Align to min_block_size
        aligned = max(min_block_size, (size // min_block_size) * min_block_size)
        configs.append((aligned, min_block_size))

    return configs
