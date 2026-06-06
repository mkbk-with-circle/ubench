#!/usr/bin/env python3
"""
V3: Vector Unit Throughput
Measures the peak throughput of the Vector Unit in operations per cycle.

Method: Independent Vector Operations
- Issue a large number of INDEPENDENT vector operations.
- Each operation reads from separate input tensors and writes to separate outputs,
  allowing the Vector Unit to execute them in parallel at full pipeline utilization.
- Throughput = total_operations / total_time

Design Rationale:
- Unlike V1/V2 (latency), we maximize parallelism.
- With enough independent work, the Vector Unit achieves peak throughput
  limited only by the number of vector pipelines and issue width.
- We sweep the number of independent operations to find saturation.
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


class VectorThroughputBench(AscendBenchmark):
    """V3: Measure vector unit peak throughput."""

    def __init__(self, device_id=0):
        super().__init__(
            param_id="V3",
            param_name="向量单元吞吐率 (Vector Unit Throughput)",
            category="Vector Unit",
            device_id=device_id,
        )

    def bench_kernel(self, num_ops=1000, vec_size=256) -> float:
        """Execute many independent vector additions. Returns time in ms."""
        timer = EventTimer()

        if HAS_NPU:
            # Create many independent vector inputs
            inputs_a = [torch.randn(vec_size, device="npu") for _ in range(num_ops)]
            inputs_b = [torch.randn(vec_size, device="npu") for _ in range(num_ops)]
            outputs = []

            timer.record_start()
            for i in range(num_ops):
                r = inputs_a[i] + inputs_b[i]  # independent vector add
                outputs.append(r)
            timer.record_end()
            elapsed = timer.elapsed_ms()
        else:
            inputs_a = [np.random.randn(vec_size).astype(np.float32) for _ in range(num_ops)]
            inputs_b = [np.random.randn(vec_size).astype(np.float32) for _ in range(num_ops)]
            timer.record_start()
            outputs = []
            for i in range(num_ops):
                r = inputs_a[i] + inputs_b[i]
                outputs.append(r)
            timer.record_end()
            elapsed = timer.elapsed_ms()

        return elapsed

    def measure_throughput(self, op_counts=None, vec_size=256,
                           num_warmup=3, num_iters=20):
        """Sweep operation counts to find saturation throughput."""
        if op_counts is None:
            op_counts = [10, 50, 100, 500, 1000, 2000, 5000]

        freq_mhz = estimate_clock_freq_mhz()
        results = []

        for count in op_counts:
            res = self.run(
                num_warmup=num_warmup,
                num_iterations=num_iters,
                bench_kwargs={"num_ops": count, "vec_size": vec_size},
            )

            cycles = res.value * freq_mhz * 1e3  # ms -> cycles
            ops_per_cycle = count / cycles if cycles > 0 else 0

            res.unit = "ops/cycle"
            res.value = ops_per_cycle
            res.notes = f"num_ops={count}, vec_size={vec_size}, freq={freq_mhz}MHz"
            results.append(res)
            print_result(res)

        return results


def main():
    print("=" * 70)
    print("  V3: Vector Unit Throughput")
    print("  Method: Independent Vector Operations")
    print("=" * 70)

    bench = VectorThroughputBench()
    results = bench.measure_throughput()

    throughputs = [r.value for r in results]
    if throughputs:
        best = max(throughputs)
        print(f"\n[V3] Estimated vector throughput: {best:.4f} ops/cycle")

    return results


if __name__ == "__main__":
    main()
