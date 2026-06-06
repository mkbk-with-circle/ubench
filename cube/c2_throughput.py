#!/usr/bin/env python3
"""
C2: Matrix Multiplication Throughput
Measures the peak throughput of the Cube Unit in MAC/s (or TFLOPS).

Method: Large Matrix Multiplication with Sweep
- Execute large matrix multiplications and measure the achieved throughput.
- Sweep matrix sizes (M, N, K) to find the saturation point where
  the Cube Unit reaches its peak compute capability.
- Throughput = (2 * M * N * K) / time  [FLOPS]

Design Rationale:
- The Cube Unit on DaVinci is designed for high-throughput matrix multiplication.
- Peak throughput is achieved when the matrix dimensions are large enough
  to fully utilize the systolic array and hide memory latency.
- We sweep sizes to find the saturation region and report peak TFLOPS.
- FP16 is typically the peak-performance data type on Cube Unit (910B: ~32 TFLOPS).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from common.benchmark import AscendBenchmark, EventTimer, BenchResult, print_result
from common.utils import compute_flops, compute_matmul_flops

try:
    import torch
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False


class MatMulThroughputBench(AscendBenchmark):
    """C2: Measure Cube Unit matrix multiplication throughput."""

    def __init__(self, device_id=0):
        super().__init__(
            param_id="C2",
            param_name="矩阵乘吞吐率 (MatMul Throughput)",
            category="Cube Unit",
            device_id=device_id,
        )

    def bench_kernel(self, m=1024, n=1024, k=1024) -> float:
        """
        Execute a single large matrix multiplication.
        Returns time in milliseconds.
        """
        timer = EventTimer()

        if HAS_NPU:
            a = torch.randn(m, k, device="npu")
            b = torch.randn(k, n, device="npu")

            timer.record_start()
            c = torch.mm(a, b)
            timer.record_end()
            elapsed = timer.elapsed_ms()
        else:
            a = np.random.randn(m, k).astype(np.float32)
            b = np.random.randn(k, n).astype(np.float32)

            timer.record_start()
            c = a @ b
            timer.record_end()
            elapsed = timer.elapsed_ms()

        return elapsed

    def measure_throughput(self, sizes=None, num_warmup=5, num_iters=10):
        """Sweep matrix sizes to find peak throughput."""
        if sizes is None:
            # Square matrices for simplicity
            sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

        results = []

        for size in sizes:
            res = self.run(
                num_warmup=num_warmup,
                num_iterations=num_iters,
                bench_kwargs={"m": size, "n": size, "k": size},
            )

            flops = compute_matmul_flops(size, size, size)
            tflops = compute_flops(flops, res.value / 1000.0) / 1e12  # Convert to TFLOPS
            gmacs = compute_flops(flops, res.value / 1000.0) / 1e9 / 2  # GMAC/s

            res.unit = "TFLOPS"
            res.value = tflops
            res.notes = f"size={size}x{size}, GMAC/s={gmacs:.2f}"
            results.append(res)
            print_result(res)

        return results


def main():
    print("=" * 70)
    print("  C2: Matrix Multiplication Throughput")
    print("  Method: Large Matrix Multiplication")
    print("=" * 70)

    bench = MatMulThroughputBench()
    results = bench.measure_throughput()

    tflops_values = [r.value for r in results]
    if tflops_values:
        best = max(tflops_values)
        print(f"\n[C2] Peak matrix multiplication throughput: {best:.4f} TFLOPS")
        print(f"     (Ascend 910B theoretical peak: ~32 TFLOPS FP16)")

    return results


if __name__ == "__main__":
    main()
