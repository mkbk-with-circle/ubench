#!/usr/bin/env python3
"""
V1: FP32 Vector Addition Latency
Measures the execution latency of a single FP32 vector addition instruction.

Method: Dependency Chain with Vector Operations
- Construct a chain of dependent FP32 vector additions.
- Each vector addition depends on the result of the previous addition.
- This forces serial execution on the Vector Unit, exposing single-instruction latency.

Design Rationale:
- The Vector Unit on DaVinci executes SIMD operations on vectors (typically 32 FP32 elements).
- By making each operation depend on the previous result, we prevent the pipeline
  from overlapping execution.
- We sweep vector lengths to isolate the per-element contribution vs. fixed overhead.
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


class VectorAddLatencyBench(AscendBenchmark):
    """V1: Measure FP32 vector addition latency."""

    def __init__(self, device_id=0):
        super().__init__(
            param_id="V1",
            param_name="FP32向量加法延迟 (FP32 Vector Add Latency)",
            category="Vector Unit",
            device_id=device_id,
        )

    def bench_kernel(self, vec_size=1024, chain_length=100) -> float:
        """Execute a dependency chain of vector additions. Returns time in ms."""
        timer = EventTimer()

        if HAS_NPU:
            a = torch.randn(vec_size, device="npu")
            result = torch.zeros(vec_size, device="npu")

            timer.record_start()
            for _ in range(chain_length):
                result = result + a  # dependent vector addition
            timer.record_end()
            elapsed = timer.elapsed_ms()
        else:
            a = np.random.randn(vec_size).astype(np.float32)
            result = np.zeros(vec_size, dtype=np.float32)
            timer.record_start()
            for _ in range(chain_length):
                result = result + a
            timer.record_end()
            elapsed = timer.elapsed_ms()

        return elapsed

    def measure_latency(self, vec_sizes=None, chain_lengths=None,
                        num_warmup=3, num_iters=30):
        """Measure at various vector sizes and chain lengths."""
        if vec_sizes is None:
            vec_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        if chain_lengths is None:
            chain_lengths = [100, 200, 500, 1000]

        freq_mhz = estimate_clock_freq_mhz()
        results = []

        for vs in vec_sizes:
            for cl in chain_lengths:
                res = self.run(
                    num_warmup=num_warmup,
                    num_iterations=num_iters,
                    bench_kwargs={"vec_size": vs, "chain_length": cl},
                )

                total_cycles = ms_to_cycles(res.value, freq_mhz)
                cycles_per_op = total_cycles / cl
                cycles_per_elem = cycles_per_op / vs if vs > 0 else 0

                res.unit = "cycles"
                res.value = cycles_per_op
                res.notes = f"vec_size={vs}, chain_len={cl}, per_elem={cycles_per_elem:.4f}cy"
                results.append(res)

        return results


def main():
    print("=" * 70)
    print("  V1: FP32 Vector Addition Latency")
    print("  Method: Dependency Chain with Vector Additions")
    print("=" * 70)

    bench = VectorAddLatencyBench()
    results = bench.measure_latency()

    # Best estimate from minimal configuration
    latencies = [r.value for r in results if r.notes and "vec_size=32" in r.notes]
    if latencies:
        best = min(latencies)
        print(f"\n[V1] Estimated FP32 vector add latency: {best:.2f} cycles")
        print(f"     (vec_size=32, minimum across chain lengths)")

    return results


if __name__ == "__main__":
    main()
