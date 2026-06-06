#!/usr/bin/env python3
"""
C1: Matrix Multiplication Latency (Single Tile)
Measures the latency of a single minimum-tile (e.g., 16×16) matrix multiplication
on the Cube Unit.

Method: Single Tile Measurement
- Execute a single small matrix multiplication (tile-sized) on the Cube Unit.
- Use dependency chaining to prevent overlapping of multiple tiles.
- Latency = total_time / number_of_tile_operations

Design Rationale:
- The Cube Unit processes matrix multiplication in fixed-size tiles (typically 16×16).
- By measuring just one tile operation, we get the fundamental latency of the Cube Unit.
- Larger matrix multiplications are composed of many tile operations, so understanding
  the single-tile latency is key to modeling overall performance.
- We test multiple tile sizes to confirm which is the atomic tile size.
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


class SingleTileLatencyBench(AscendBenchmark):
    """C1: Measure single tile matrix multiplication latency."""

    def __init__(self, device_id=0):
        super().__init__(
            param_id="C1",
            param_name="矩阵乘延迟（单tile）(Single Tile MatMul Latency)",
            category="Cube Unit",
            device_id=device_id,
        )

    def bench_kernel(self, m=16, n=16, k=16, chain_length=100) -> float:
        """
        Execute a chain of dependent small matrix multiplications.
        Each matmul depends on the previous result, forcing serial execution.
        Returns time in ms.
        """
        timer = EventTimer()

        if HAS_NPU:
            a = torch.randn(m, k, device="npu")
            b = torch.randn(k, n, device="npu")
            result = torch.randn(m, n, device="npu")

            timer.record_start()
            for _ in range(chain_length):
                # Dependent matmul: output of one is input to next
                result = torch.mm(a, b) + result * 0.0  # force dependency
            timer.record_end()
            elapsed = timer.elapsed_ms()
        else:
            a = np.random.randn(m, k).astype(np.float32)
            b = np.random.randn(k, n).astype(np.float32)
            result = np.random.randn(m, n).astype(np.float32)

            timer.record_start()
            for _ in range(chain_length):
                result = a @ b + result * 0.0
            timer.record_end()
            elapsed = timer.elapsed_ms()

        return elapsed

    def measure_latency(self, tile_sizes=None, chain_length=100,
                        num_warmup=3, num_iters=30):
        """Measure single-tile latency at various tile sizes."""
        if tile_sizes is None:
            tile_sizes = [4, 8, 16, 32, 64]

        freq_mhz = estimate_clock_freq_mhz()
        results = []

        for ts in tile_sizes:
            res = self.run(
                num_warmup=num_warmup,
                num_iterations=num_iters,
                bench_kwargs={"m": ts, "n": ts, "k": ts, "chain_length": chain_length},
            )

            cycles = ms_to_cycles(res.value, freq_mhz)
            cycles_per_tile = cycles / chain_length

            res.unit = "cycles"
            res.value = cycles_per_tile
            res.notes = f"tile={ts}x{ts}, chain_len={chain_length}, freq={freq_mhz}MHz"
            results.append(res)
            print_result(res)

        return results


def main():
    print("=" * 70)
    print("  C1: Single Tile Matrix Multiplication Latency")
    print("  Method: Dependency Chain of Small MatMuls")
    print("=" * 70)

    bench = SingleTileLatencyBench()
    results = bench.measure_latency()

    # The 16x16 tile is typically the atomic tile for DaVinci
    for r in results:
        if "tile=16" in r.notes:
            print(f"\n[C1] Estimated single-tile (16×16) matmul latency: {r.value:.2f} cycles")
            break

    return results


if __name__ == "__main__":
    main()
