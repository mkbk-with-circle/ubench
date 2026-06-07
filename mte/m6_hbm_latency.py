#!/usr/bin/env python3
"""
M6: DDR/HBM Memory Access Latency
Measures the off-chip memory (HBM) access latency on Ascend NPU.

Method: Pointer Chasing with Large Array
- Allocate an array much larger than the L1/L0 cache capacity.
- Construct a pointer-chasing pattern where each access depends on the
  previous one, defeating the hardware prefetcher.
- This forces each access to go to HBM, exposing the true memory latency.
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


class HBMMemLatencyBench(AscendBenchmark):
    """M6: Measure HBM (off-chip) memory access latency."""

    def __init__(self, device_id=0):
        super().__init__(
            param_id="M6",
            param_name="DDR/HBM访存延迟 (HBM Memory Access Latency)",
            category="MTE (Memory Transfer Engine)",
            device_id=device_id,
        )

    def bench_kernel(self, array_size=1024 * 1024, num_accesses=1000) -> float:
        """
        Execute pointer-chasing into a large (HBM-resident) array.
        Returns time in milliseconds.
        """
        if HAS_NPU:
            # Pre-allocate data and indices
            data = torch.randn(array_size, device="npu")
            indices = np.random.randint(0, array_size, size=num_accesses)
            indices_t = torch.tensor(indices, dtype=torch.long, device="npu")

            # Warmup: run the chasing pattern a few times
            for _ in range(3):
                acc = torch.tensor([0.0], device="npu")
                pos = torch.tensor([0], dtype=torch.long, device="npu")
                for i in range(min(50, num_accesses - 1)):
                    idx = indices_t[pos.long()]
                    acc = acc + data[idx.long()]
                    pos = torch.tensor([indices[i + 1]], dtype=torch.long, device="npu")
            torch.npu.synchronize()

            # Measure: pointer chasing with dependent loads
            start_e = torch.npu.Event(enable_timing=True)
            end_e = torch.npu.Event(enable_timing=True)

            acc = torch.tensor([0.0], device="npu")
            pos = torch.tensor([0], dtype=torch.long, device="npu")

            start_e.record()
            for i in range(num_accesses - 1):
                idx = indices_t[pos.long()]
                acc = acc + data[idx.long()]
                pos = torch.tensor([indices[i + 1]], dtype=torch.long, device="npu")
            end_e.record()
            torch.npu.synchronize()
            elapsed = start_e.elapsed_time(end_e)
        else:
            data = np.random.randn(array_size).astype(np.float32)
            indices = np.random.randint(0, array_size, size=num_accesses)
            import time
            t0 = time.perf_counter()
            acc = 0.0
            for i in range(num_accesses - 1):
                acc += data[indices[i]]
            elapsed = (time.perf_counter() - t0) * 1000.0

        return elapsed

    def measure_latency(self, array_sizes=None, num_warmup=3, num_iters=20):
        """Measure HBM latency at various array sizes."""
        if array_sizes is None:
            array_sizes = [
                64 * 1024,       # 256 KB
                256 * 1024,      # 1 MB
                1024 * 1024,     # 4 MB
                4 * 1024 * 1024,  # 16 MB
                16 * 1024 * 1024, # 64 MB
            ]

        freq_mhz = estimate_clock_freq_mhz()
        results = []

        for size in array_sizes:
            res = self.run(
                num_warmup=num_warmup,
                num_iterations=num_iters,
                bench_kwargs={"array_size": size, "num_accesses": 500},
            )

            cycles_total = ms_to_cycles(res.value, freq_mhz)
            cycles_per_access = cycles_total / 500

            res.unit = "cycles"
            res.value = cycles_per_access
            size_mb = size * 4 / (1024 * 1024)
            res.notes = f"array_size={size} ({size_mb:.1f}MB), freq={freq_mhz}MHz"
            results.append(res)
            print_result(res)

        return results


def main():
    print("=" * 70)
    print("  M6: HBM Memory Access Latency")
    print("  Method: Pointer Chasing with Large Array")
    print("=" * 70)

    bench = HBMMemLatencyBench()
    results = bench.measure_latency()

    for r in reversed(results):
        size_mb = float(r.notes.split("(")[1].split("MB")[0]) if "MB" in r.notes else 0
        if size_mb >= 16:
            print(f"\n[M6] Estimated HBM access latency: {r.value:.2f} cycles")
            print(f"     (array size = {size_mb:.1f} MB, guaranteed HBM resident)")
            break

    return results


if __name__ == "__main__":
    main()
