#!/usr/bin/env python3
"""
M8: Data Transfer Startup Overhead
Measures the fixed overhead of initiating a DMA (Direct Memory Access)
transfer via the MTE (Memory Transfer Engine).

Method: Vary Transfer Size, Extrapolate to Zero
- Execute DMA transfers of various sizes.
- Measure the total time for each transfer.
- Fit a linear model: time = startup_overhead + size / bandwidth
- Startup overhead = intercept at size = 0.

Design Rationale:
- Every DMA transfer has a fixed setup cost (descriptor processing,
  address calculation, MTE configuration).
- By measuring total time for different transfer sizes and extrapolating
  to zero size, we isolate the startup overhead.
- This overhead determines the minimum granularity at which data
  movement is efficient.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from common.benchmark import AscendBenchmark, EventTimer, BenchResult, print_result
from common.utils import estimate_clock_freq_mhz, ms_to_cycles

try:
    import torch
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False


class TransferStartupOverheadBench(AscendBenchmark):
    """M8: Measure MTE DMA transfer startup overhead."""

    def __init__(self, device_id=0):
        super().__init__(
            param_id="M8",
            param_name="数据搬运启动开销 (Transfer Startup Overhead)",
            category="MTE (Memory Transfer Engine)",
            device_id=device_id,
        )

    def bench_kernel(self, num_elements=1024) -> float:
        """
        Execute a single DMA-like transfer (tensor copy).
        Returns time in milliseconds.
        Uses single-copy measurement (no batching) to expose per-transfer overhead.
        """
        if HAS_NPU:
            src = torch.randn(num_elements, device="npu")
            dst = torch.zeros(num_elements, device="npu")

            # Warmup
            for _ in range(10):
                dst.copy_(src)
            torch.npu.synchronize()

            # Single-copy measurement (no batching for latency measurement)
            start_e = torch.npu.Event(enable_timing=True)
            end_e = torch.npu.Event(enable_timing=True)

            start_e.record()
            dst.copy_(src)
            end_e.record()
            torch.npu.synchronize()
            elapsed = start_e.elapsed_time(end_e)
        else:
            src = np.random.randn(num_elements).astype(np.float32)
            dst = np.zeros(num_elements, dtype=np.float32)
            timer = EventTimer()
            timer.record_start()
            np.copyto(dst, src)
            timer.record_end()
            elapsed = timer.elapsed_ms()

        return elapsed

    def measure_startup_overhead(self, element_counts=None,
                                 num_warmup=5, num_iters=50):
        """
        Measure transfer time at various small sizes.
        Fit linear model to extract startup overhead.
        """
        if element_counts is None:
            # Small sizes to expose overhead
            element_counts = [
                32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
            ]

        freq_mhz = estimate_clock_freq_mhz()
        results = []

        sizes_bytes = []
        times_cycles = []

        for count in element_counts:
            raw_times = []
            for _ in range(num_iters):
                t = self.bench_kernel(count)
                raw_times.append(t)

            arr = np.array(raw_times)
            mean_ms = float(np.mean(arr))
            cycles = ms_to_cycles(mean_ms, freq_mhz)
            size_bytes = count * 4  # FP32

            sizes_bytes.append(size_bytes)
            times_cycles.append(cycles)

            res = BenchResult(
                param_id="M8",
                param_name=f"传输时间 (size={count} elements)",
                category="MTE",
                value=mean_ms * 1000,  # microseconds
                unit="us",
                raw_times=raw_times,
                num_iterations=num_iters,
                num_warmup=num_warmup,
                std_dev=float(np.std(arr)),
                cv=float(np.std(arr)) / mean_ms if mean_ms > 0 else 0,
                median=float(np.median(arr)),
                p25=float(np.percentile(arr, 25)),
                p75=float(np.percentile(arr, 75)),
                notes=f"transfer_size={count} fp32, {size_bytes}B",
            )
            results.append(res)
            print_result(res)

        # Linear regression: time = overhead + slope * size
        from numpy.polynomial import polynomial as P
        if len(sizes_bytes) >= 3:
            coeffs = np.polyfit(sizes_bytes, times_cycles, 1)
            slope_cycles_per_byte = coeffs[0]
            startup_overhead_cycles = coeffs[1]

            print(f"\n[M8] Linear fit: time = {startup_overhead_cycles:.2f} + {slope_cycles_per_byte:.6f} * size_bytes")
            print(f"     Startup overhead: {startup_overhead_cycles:.2f} cycles")
            print(f"     Inverse of slope (bandwidth): {1/slope_cycles_per_byte:.2f} bytes/cycle")

            # Add summary result
            summary = BenchResult(
                param_id="M8",
                param_name="MTE启动开销",
                category="MTE",
                value=startup_overhead_cycles,
                unit="cycles",
                raw_times=[],
                num_iterations=num_iters,
                num_warmup=num_warmup,
                std_dev=0,
                cv=0,
                median=startup_overhead_cycles,
                p25=startup_overhead_cycles,
                p75=startup_overhead_cycles,
                notes=f"Linear extrapolation, R² check advised",
            )
            results.append(summary)

        return results


def main():
    print("=" * 70)
    print("  M8: Data Transfer Startup Overhead")
    print("  Method: Vary Transfer Size, Linear Extrapolation")
    print("=" * 70)

    bench = TransferStartupOverheadBench()
    results = bench.measure_startup_overhead()

    # Find the summary result
    for r in results:
        if r.param_name == "MTE启动开销":
            print(f"\n[M8] Estimated MTE startup overhead: {r.value:.2f} cycles")

    return results


if __name__ == "__main__":
    main()
