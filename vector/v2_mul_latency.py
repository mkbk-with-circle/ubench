#!/usr/bin/env python3
"""
V2: FP32 Vector Multiplication Latency
Measures the execution latency of a single FP32 vector multiplication instruction.

Method: Dependency Chain with Vector Multiplications
- Similar to V1, but uses vector multiplication (element-wise) instead of addition.
- Multiplication typically has different latency than addition on FP units,
  so this must be measured separately.

Design Rationale:
- FP32 multiplication and addition may use different functional units or
  have different pipeline depths in the Vector Unit.
- This measurement isolates the multiplication path.
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


class VectorMulLatencyBench(AscendBenchmark):
    """V2: Measure FP32 vector multiplication latency."""

    def __init__(self, device_id=0):
        super().__init__(
            param_id="V2",
            param_name="FP32向量乘法延迟 (FP32 Vector Mul Latency)",
            category="Vector Unit",
            device_id=device_id,
        )

    def bench_kernel(self, vec_size=1024, chain_length=100) -> float:
        """Execute a dependency chain of vector multiplications. Returns time in ms."""
        timer = EventTimer()

        if HAS_NPU:
            a = torch.ones(vec_size, device="npu") * 2.0
            result = torch.ones(vec_size, device="npu")

            timer.record_start()
            for _ in range(chain_length):
                result = result * a  # dependent vector multiplication
            timer.record_end()
            elapsed = timer.elapsed_ms()
        else:
            a = np.ones(vec_size, dtype=np.float32) * 2.0
            result = np.ones(vec_size, dtype=np.float32)
            timer.record_start()
            for _ in range(chain_length):
                result = result * a
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
    print("  V2: FP32 Vector Multiplication Latency")
    print("  Method: Dependency Chain with Vector Multiplications")
    print("=" * 70)

    bench = VectorMulLatencyBench()
    results = bench.measure_latency()

    latencies = [r.value for r in results if r.notes and "vec_size=32" in r.notes]
    if latencies:
        best = min(latencies)
        print(f"\n[V2] Estimated FP32 vector mul latency: {best:.2f} cycles")

    return results


if __name__ == "__main__":
    main()
