#!/usr/bin/env python3
"""
C4: L0A/L0B/L0C Buffer Access Latency
Measures the read/write latency of Cube Unit's dedicated buffers (L0A, L0B, L0C).

Method: Buffer-Specific Access Patterns
- The Cube Unit has three dedicated on-chip buffers:
  - L0A: Holds matrix A data (input activations)
  - L0B: Holds matrix B data (weights)
  - L0C: Holds matrix C data (accumulated results)
- We design access patterns that target each buffer specifically.
- For L0A/L0B: measure the time to load data into these buffers before matmul.
- For L0C: measure the time to read accumulated results.

Design Rationale:
- Each L0 buffer has its own access path and potentially different latency.
- L0A/L0B are typically write-only from the perspective of MTE (data is loaded in).
- L0C is read/write: it receives accumulation results and can be read back.
- We isolate each buffer's access by controlling the matmul operation pattern.
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


class CubeBufferLatencyBench(AscendBenchmark):
    """C4: Measure L0A/L0B/L0C buffer access latency."""

    def __init__(self, device_id=0):
        super().__init__(
            param_id="C4",
            param_name="L0A/L0B/L0C访问延迟 (L0 Buffer Access Latency)",
            category="Cube Unit",
            device_id=device_id,
        )

    def bench_l0_read_latency(self, buffer_size=64, num_accesses=1000) -> float:
        """
        Measure L0C read latency by repeatedly reading back
        matrix multiplication results from the Cube Unit.
        L0C holds the output matrix.
        """
        timer = EventTimer()
        tile = min(buffer_size, 256)

        if HAS_NPU:
            a = torch.randn(tile, tile, device="npu")
            b = torch.randn(tile, tile, device="npu")

            timer.record_start()
            for _ in range(num_accesses):
                c = torch.mm(a, b)
                # Read back to force L0C access
                _ = c.cpu() if _ == num_accesses - 1 else c
            timer.record_end()
            elapsed = timer.elapsed_ms()
        else:
            a = np.random.randn(tile, tile).astype(np.float32)
            b = np.random.randn(tile, tile).astype(np.float32)

            timer.record_start()
            for _ in range(num_accesses):
                c = a @ b
            timer.record_end()
            elapsed = timer.elapsed_ms()

        return elapsed

    def bench_l0_write_latency(self, buffer_size=64, num_accesses=1000) -> float:
        """
        Measure L0A/L0B write latency by loading input matrices
        into the Cube Unit's input buffers.
        """
        timer = EventTimer()
        tile = min(buffer_size, 256)

        if HAS_NPU:
            timer.record_start()
            for _ in range(num_accesses):
                a = torch.randn(tile, tile, device="npu")
                b = torch.randn(tile, tile, device="npu")
                c = torch.mm(a, b)  # This loads A→L0A, B→L0B
            timer.record_end()
            elapsed = timer.elapsed_ms()
        else:
            timer.record_start()
            for _ in range(num_accesses):
                a = np.random.randn(tile, tile).astype(np.float32)
                b = np.random.randn(tile, tile).astype(np.float32)
                c = a @ b
            timer.record_end()
            elapsed = timer.elapsed_ms()

        return elapsed

    def measure_latency(self, num_warmup=3, num_iters=20):
        """Measure L0 read and write latencies."""
        freq_mhz = estimate_clock_freq_mhz()
        results = []

        # L0 write (L0A/L0B) latency
        print("\n--- L0A/L0B Write Latency ---")
        raw_times = []
        for _ in range(num_iters):
            t = self.bench_l0_write_latency()
            raw_times.append(t)

        arr = np.array(raw_times)
        mean_t = float(np.mean(arr))
        cycles = ms_to_cycles(mean_t, freq_mhz)
        cycles_per_access = cycles / 1000

        res = BenchResult(
            param_id="C4_WRITE",
            param_name="L0A/L0B写延迟",
            category="Cube Unit",
            value=cycles_per_access,
            unit="cycles",
            raw_times=raw_times,
            num_iterations=num_iters,
            num_warmup=num_warmup,
            std_dev=float(np.std(arr)),
            cv=float(np.std(arr)) / mean_t if mean_t > 0 else 0,
            median=float(np.median(arr)),
            p25=float(np.percentile(arr, 25)),
            p75=float(np.percentile(arr, 75)),
            notes=f"WRITE, freq={freq_mhz}MHz",
        )
        results.append(res)
        print_result(res)

        # L0 read (L0C) latency
        print("\n--- L0C Read Latency ---")
        raw_times = []
        for _ in range(num_iters):
            t = self.bench_l0_read_latency()
            raw_times.append(t)

        arr = np.array(raw_times)
        mean_t = float(np.mean(arr))
        cycles = ms_to_cycles(mean_t, freq_mhz)
        cycles_per_access = cycles / 1000

        res = BenchResult(
            param_id="C4_READ",
            param_name="L0C读延迟",
            category="Cube Unit",
            value=cycles_per_access,
            unit="cycles",
            raw_times=raw_times,
            num_iterations=num_iters,
            num_warmup=num_warmup,
            std_dev=float(np.std(arr)),
            cv=float(np.std(arr)) / mean_t if mean_t > 0 else 0,
            median=float(np.median(arr)),
            p25=float(np.percentile(arr, 25)),
            p75=float(np.percentile(arr, 75)),
            notes=f"READ, freq={freq_mhz}MHz",
        )
        results.append(res)
        print_result(res)

        return results


def main():
    print("=" * 70)
    print("  C4: L0A/L0B/L0C Buffer Access Latency")
    print("  Method: Buffer-Specific Access Patterns")
    print("=" * 70)

    bench = CubeBufferLatencyBench()
    results = bench.measure_latency()

    for r in results:
        if "WRITE" in r.notes:
            print(f"\n[C4] Estimated L0A/L0B write latency: {r.value:.2f} cycles")
        elif "READ" in r.notes:
            print(f"[C4] Estimated L0C read latency: {r.value:.2f} cycles")

    return results


if __name__ == "__main__":
    main()
