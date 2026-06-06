#!/usr/bin/env python3
"""
C3: Cube Unit Pipeline Depth
Measures the pipeline depth of the Cube Unit - the maximum number of
matrix multiplication tiles that can be simultaneously in-flight.

Method: Variable Dependency Distance in Matrix Operations
- Similar to V4 (Vector Pipeline Depth), but for the Cube Unit.
- Insert independent matmul operations between dependent ones.
- When the gap size exceeds the pipeline depth, the effective latency
  of the dependent operations reaches its minimum.

Design Rationale:
- The Cube Unit processes matmul tiles through a deep pipeline.
- By controlling the distance between dependent matmul operations,
  we can probe the pipeline occupancy limit.
- This reveals how many outstanding tile operations the Cube Unit can handle.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from common.benchmark import AscendBenchmark, EventTimer, BenchResult, print_result
from common.utils import estimate_clock_freq_mhz

try:
    import torch
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False


class CubePipelineDepthBench(AscendBenchmark):
    """C3: Measure Cube Unit pipeline depth."""

    def __init__(self, device_id=0):
        super().__init__(
            param_id="C3",
            param_name="矩阵乘流水线深度 (Cube Pipeline Depth)",
            category="Cube Unit",
            device_id=device_id,
        )

    def bench_kernel(self, tile_size=16, gap_size=0) -> float:
        """
        Measure effective latency with N independent matmuls between
        dependent ones. Returns time in ms.
        """
        timer = EventTimer()

        if HAS_NPU:
            a = torch.randn(tile_size, tile_size, device="npu")
            b = torch.randn(tile_size, tile_size, device="npu")
            result = torch.randn(tile_size, tile_size, device="npu")

            # Independent filler data
            fillers_a = [torch.randn(tile_size, tile_size, device="npu") for _ in range(gap_size)]
            fillers_b = [torch.randn(tile_size, tile_size, device="npu") for _ in range(gap_size)]

            timer.record_start()

            num_pairs = 20
            for _ in range(num_pairs):
                result = torch.mm(a, b)  # dependent matmul
                # Insert independent filler matmuls
                for j in range(gap_size):
                    _ = torch.mm(fillers_a[j], fillers_b[j])

            timer.record_end()
            elapsed = timer.elapsed_ms()
        else:
            a = np.random.randn(tile_size, tile_size).astype(np.float32)
            b = np.random.randn(tile_size, tile_size).astype(np.float32)
            result = np.random.randn(tile_size, tile_size).astype(np.float32)

            fillers_a = [np.random.randn(tile_size, tile_size).astype(np.float32) for _ in range(gap_size)]
            fillers_b = [np.random.randn(tile_size, tile_size).astype(np.float32) for _ in range(gap_size)]

            timer.record_start()
            num_pairs = 20
            for _ in range(num_pairs):
                result = a @ b
                for j in range(gap_size):
                    _ = fillers_a[j] @ fillers_b[j]
            timer.record_end()
            elapsed = timer.elapsed_ms()

        return elapsed

    def measure_pipeline_depth(self, gap_sizes=None,
                                num_warmup=3, num_iters=20):
        """Sweep gap sizes to find pipeline depth."""
        if gap_sizes is None:
            gap_sizes = [0, 1, 2, 4, 8, 16, 32]

        freq_mhz = estimate_clock_freq_mhz()
        results = []

        for gap in gap_sizes:
            res = self.run(
                num_warmup=num_warmup,
                num_iterations=num_iters,
                bench_kwargs={"gap_size": gap},
            )

            num_pairs = 20
            time_per_pair_ms = res.value / num_pairs
            cycles_per_pair = time_per_pair_ms * freq_mhz * 1e3

            res.unit = "cycles/pair"
            res.value = cycles_per_pair
            res.notes = f"gap_size={gap}, tile_size=16"
            results.append(res)
            print_result(res)

        return results


def main():
    print("=" * 70)
    print("  C3: Cube Unit Pipeline Depth")
    print("  Method: Variable Dependency Distance (MatMul)")
    print("=" * 70)

    bench = CubePipelineDepthBench()
    results = bench.measure_pipeline_depth()

    values = [(int(r.notes.split("gap_size=")[1].split(",")[0]), r.value) for r in results]
    values.sort()

    print("\n[C3] Latency vs. Gap Size:")
    max_lat = max(v[1] for v in values) if values else 1
    for gap, lat in values:
        bar = "#" * int(lat / max_lat * 40)
        print(f"  gap={gap:4d}: {lat:10.2f} cycles/pair {bar}")

    min_lat = min(v[1] for v in values)
    threshold = min_lat * 1.2
    for gap, lat in values:
        if lat <= threshold:
            print(f"\n[C3] Estimated Cube pipeline depth: {gap}")
            break

    return results


if __name__ == "__main__":
    main()
