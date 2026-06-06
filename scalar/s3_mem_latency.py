#!/usr/bin/env python3
"""
S3: Scalar Memory Access Latency
Measures the latency of scalar load/store instructions on Ascend NPU.

Method: Pointer Chasing on Scalar-Accessible Memory
- For NPU scalar unit, the accessible memory is typically the L1 Buffer
  or shared memory region.
- We construct a pointer-chased access pattern where each memory read
  determines the address of the next read.
- This eliminates prefetching and exposes true memory access latency.

Design Rationale:
- Scalar loads/stores on the NPU go through the scalar load/store unit
  to the L1 Buffer (or shared memory).
- Pointer chasing forces serialized memory accesses, each dependent
  on the previous read's result.
- By measuring total time for N chained accesses, we get latency = total / N.
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


class ScalarMemAccessLatencyBench(AscendBenchmark):
    """
    S3: Measure scalar memory access latency via pointer chasing.

    On Ascend NPU, the scalar unit can access L1 buffer directly.
    We measure the round-trip latency of a scalar load from L1.
    """

    def __init__(self, device_id=0):
        super().__init__(
            param_id="S3",
            param_name="标量访存延迟 (Scalar Memory Access Latency)",
            category="Scalar Unit",
            device_id=device_id,
        )

    def bench_kernel(self, num_accesses=1000, array_size=1024) -> float:
        """
        Execute a pointer-chasing memory access pattern.
        Returns elapsed time in milliseconds.

        On NPU, we use tensor indexing to simulate pointer chasing:
        indices[i] = next access position, creating a random walk.
        """
        timer = EventTimer()

        if HAS_NPU:
            # Create data array and random access indices
            data = torch.randn(array_size, device="npu")
            # Create a pointer-chase pattern (random permutation)
            indices_np = np.random.permutation(array_size)
            # Ensure we have enough accesses
            chase_indices = np.tile(indices_np, (num_accesses // array_size) + 1)[:num_accesses]
            chase_indices = torch.tensor(chase_indices, dtype=torch.long, device="npu")

            timer.record_start()
            # Simulate pointer chasing via dependent indexed reads
            pos = torch.tensor([0], dtype=torch.long, device="npu")
            acc = torch.tensor([0.0], device="npu")
            for i in range(num_accesses):
                # Each access depends on the previous position
                idx = chase_indices[pos.long()]
                acc = acc + data[idx.long()]
                pos = idx
            timer.record_end()

            elapsed = timer.elapsed_ms()
        else:
            data = np.random.randn(array_size).astype(np.float32)
            chase_indices = np.random.permutation(array_size)
            chase_indices = np.tile(chase_indices, (num_accesses // array_size) + 1)[:num_accesses]

            timer.record_start()
            pos = 0
            acc = 0.0
            for i in range(num_accesses):
                idx = chase_indices[pos]
                acc += data[idx]
                pos = idx
            timer.record_end()
            elapsed = timer.elapsed_ms()

        return elapsed

    def measure_latency(self, access_counts=None, num_warmup=3, num_iters=30):
        """Measure scalar memory access latency at various chain lengths."""
        if access_counts is None:
            access_counts = [100, 500, 1000, 5000, 10000]

        freq_mhz = estimate_clock_freq_mhz()
        results = []

        for count in access_counts:
            res = self.run(
                num_warmup=num_warmup,
                num_iterations=num_iters,
                bench_kwargs={"num_accesses": count},
            )

            cycles = ms_to_cycles(res.value, freq_mhz)
            cycles_per_access = cycles / count

            res.unit = "cycles"
            res.value = cycles_per_access
            res.notes = f"num_accesses={count}, freq={freq_mhz}MHz"
            results.append(res)
            print_result(res)

        return results


def main():
    print("=" * 70)
    print("  S3: Scalar Memory Access Latency")
    print("  Method: Pointer Chasing (Indexed Access)")
    print("=" * 70)

    bench = ScalarMemAccessLatencyBench()
    results = bench.measure_latency()

    latencies = [r.value for r in results]
    if latencies:
        best = min(latencies)
        print(f"\n[S3] Estimated scalar memory access latency: {best:.2f} cycles")

    return results


if __name__ == "__main__":
    main()
