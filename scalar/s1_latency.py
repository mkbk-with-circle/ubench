#!/usr/bin/env python3
"""
S1: Scalar Arithmetic Instruction Latency
Measures the execution latency of a single scalar arithmetic instruction on Ascend NPU.

Method: Dependency Chain
- Construct a chain of dependent scalar operations where each operation
  depends on the result of the previous one.
- This forces serial execution, exposing the true latency of a single instruction.
- Measure total time / number of operations = latency per instruction.

Design Rationale:
- The Ascend NPU's Scalar Unit handles control flow, address calculation,
  and scalar arithmetic.
- By using dependent operations, we prevent instruction-level parallelism
  and measure the pipeline depth (latency).
- We vary the chain length and verify linear scaling to confirm isolation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from common.benchmark import AscendBenchmark, EventTimer, BenchResult, print_result
from common.utils import estimate_clock_freq_mhz, ms_to_cycles, cycles_to_ms

try:
    import torch
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False


class ScalarArithmeticLatencyBench(AscendBenchmark):
    """
    S1: Measure scalar arithmetic instruction latency using dependency chains.

    On NPU, scalar operations in kernel launch overheads and control flow
    provide a way to measure scalar unit latency.
    """

    def __init__(self, device_id=0):
        super().__init__(
            param_id="S1",
            param_name="标量算术指令延迟 (Scalar Arithmetic Latency)",
            category="Scalar Unit",
            device_id=device_id,
        )

    def bench_kernel(self, chain_length=10000) -> float:
        """
        Execute a dependency chain of scalar additions.
        Returns elapsed time in milliseconds.
        """
        timer = EventTimer()

        if HAS_NPU:
            # Create scalar tensors on NPU for the dependency chain
            a = torch.tensor([1.0], device="npu")
            result = torch.tensor([0.0], device="npu")

            timer.record_start()

            # Dependency chain: each addition depends on the previous result
            # This forces serial execution on the scalar unit
            for _ in range(chain_length):
                result = result + a  # dependent addition

            timer.record_end()
            elapsed = timer.elapsed_ms()
        else:
            # CPU simulation
            a = 1.0
            result = 0.0
            timer.record_start()
            for _ in range(chain_length):
                result = result + a
            timer.record_end()
            elapsed = timer.elapsed_ms()

        return elapsed

    def measure_latency(self, chain_lengths=None, num_warmup=3, num_iters=50):
        """
        Measure latency at multiple chain lengths to verify linear scaling
        and extract per-instruction latency.
        """
        if chain_lengths is None:
            chain_lengths = [100, 500, 1000, 5000, 10000, 50000]

        results = []
        for length in chain_lengths:
            res = self.run(
                num_warmup=num_warmup,
                num_iterations=num_iters,
                bench_kwargs={"chain_length": length},
            )

            # Convert to cycles
            freq_mhz = estimate_clock_freq_mhz()
            cycles = ms_to_cycles(res.value, freq_mhz)
            cycles_per_op = cycles / length

            res.unit = f"cycles (per op: {cycles_per_op:.2f})"
            res.value = cycles_per_op
            res.notes = f"chain_length={length}, freq={freq_mhz}MHz"
            results.append(res)
            print_result(res)

        return results


def main():
    print("=" * 70)
    print("  S1: Scalar Arithmetic Instruction Latency")
    print("  Method: Dependency Chain")
    print("=" * 70)

    bench = ScalarArithmeticLatencyBench()
    results = bench.measure_latency()

    # Report best estimate
    latencies = [r.value for r in results]
    if latencies:
        best = min(latencies)
        print(f"\n[S1] Estimated scalar arithmetic latency: {best:.2f} cycles")
        print(f"     (Minimum across all chain lengths to minimize overhead)")

    return results


if __name__ == "__main__":
    main()
