#!/usr/bin/env python3
"""
V4: Vector Unit Pipeline Depth
Measures the pipeline depth of the Vector Unit - i.e., how many vector
instructions can be simultaneously in-flight in the pipeline.

Method: Variable Dependency Distance
- Introduce N independent operations between two dependent operations.
- When N < pipeline_depth, the dependent operation stalls waiting for
  the first operation to complete.
- When N >= pipeline_depth, the pipeline can hide the latency entirely.
- Pipeline depth = minimum N where additional independent ops no longer
  reduce the effective latency.

Design Rationale:
- The vector unit uses pipelining to achieve high throughput.
- The pipeline depth determines how many instructions can be outstanding.
- By controlling the distance between dependent instructions, we can
  probe the pipeline occupancy limit.
- This is analogous to measuring pipeline depth in CPUs via
  "dependent instruction distance" benchmarks.
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


class VectorPipelineDepthBench(AscendBenchmark):
    """V4: Measure vector unit pipeline depth."""

    def __init__(self, device_id=0):
        super().__init__(
            param_id="V4",
            param_name="向量单元流水线深度 (Vector Pipeline Depth)",
            category="Vector Unit",
            device_id=device_id,
        )

    def bench_kernel(self, vec_size=256, gap_size=10) -> float:
        """
        Measure effective latency with a gap of independent operations
        between dependent ones. Returns time in ms.

        Pattern:
            result = result + A           (dep 1)
            [gap_size independent ops]    (filler)
            result = result + A           (dep 2, depends on dep 1)

        If gap_size >= pipeline_depth, dep 2 does not stall.
        """
        timer = EventTimer()

        if HAS_NPU:
            a = torch.ones(vec_size, device="npu")
            result = torch.zeros(vec_size, device="npu")
            # Create independent filler tensors
            fillers_a = [torch.randn(vec_size, device="npu") for _ in range(gap_size)]
            fillers_b = [torch.randn(vec_size, device="npu") for _ in range(gap_size)]
            filler_results = []

            timer.record_start()

            # Dependent chain with filler ops between
            num_pairs = 50  # number of dependent pairs
            for _ in range(num_pairs):
                result = result + a  # dependent op
                # Insert independent filler ops
                for j in range(gap_size):
                    fr = fillers_a[j] + fillers_b[j]
                    filler_results.append(fr)

            timer.record_end()
            elapsed = timer.elapsed_ms()
        else:
            a = np.ones(vec_size, dtype=np.float32)
            result = np.zeros(vec_size, dtype=np.float32)
            fillers_a = [np.random.randn(vec_size).astype(np.float32) for _ in range(gap_size)]
            fillers_b = [np.random.randn(vec_size).astype(np.float32) for _ in range(gap_size)]

            timer.record_start()
            num_pairs = 50
            for _ in range(num_pairs):
                result = result + a
                for j in range(gap_size):
                    fr = fillers_a[j] + fillers_b[j]
            timer.record_end()
            elapsed = timer.elapsed_ms()

        return elapsed

    def measure_pipeline_depth(self, gap_sizes=None,
                                num_warmup=3, num_iters=30):
        """
        Sweep gap sizes. The point where effective latency stops decreasing
        indicates the pipeline depth.
        """
        if gap_sizes is None:
            gap_sizes = [0, 1, 2, 4, 8, 16, 32, 64, 128]

        freq_mhz = estimate_clock_freq_mhz()
        results = []

        for gap in gap_sizes:
            res = self.run(
                num_warmup=num_warmup,
                num_iterations=num_iters,
                bench_kwargs={"gap_size": gap},
            )

            # Effective time per dependent pair
            num_pairs = 50
            time_per_pair_ms = res.value / num_pairs
            cycles_per_pair = time_per_pair_ms * freq_mhz * 1e3

            res.unit = "cycles/pair"
            res.value = cycles_per_pair
            res.notes = f"gap_size={gap}, num_pairs=50"
            results.append(res)
            print_result(res)

        return results


def main():
    print("=" * 70)
    print("  V4: Vector Unit Pipeline Depth")
    print("  Method: Variable Dependency Distance")
    print("=" * 70)

    bench = VectorPipelineDepthBench()
    results = bench.measure_pipeline_depth()

    # Analyze: find where latency stabilizes
    values = [(int(r.notes.split("gap_size=")[1].split(",")[0]), r.value) for r in results]
    values.sort()

    print("\n[V4] Latency vs. Gap Size:")
    for gap, lat in values:
        bar = "#" * int(lat / max(v[1] for v in values) * 40)
        print(f"  gap={gap:4d}: {lat:8.2f} cycles/pair {bar}")

    # Pipeline depth is the gap where latency approaches minimum
    min_lat = min(v[1] for v in values)
    threshold = min_lat * 1.2  # within 20% of minimum
    for gap, lat in values:
        if lat <= threshold:
            print(f"\n[V4] Estimated pipeline depth: {gap} (latency within 20% of minimum)")
            break

    return results


if __name__ == "__main__":
    main()
