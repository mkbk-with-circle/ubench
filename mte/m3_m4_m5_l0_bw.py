#!/usr/bin/env python3
"""
M3: L0A Bandwidth
M4: L0B Bandwidth
M5: L0C Bandwidth

Measures the read/write bandwidth of Cube Unit's dedicated L0 buffers.

Method: MatMul Data Loading Throughput
- L0 buffers are used by the Cube Unit for matrix multiplication data.
- L0A holds matrix A, L0B holds matrix B, L0C holds accumulated result C.
- We measure the effective bandwidth by timing the data loading phase
  of matrix multiplication operations.
- By varying only the input (A/B) or output (C) size, we can isolate
  each buffer's bandwidth.

Design Rationale:
- L0A, L0B, L0C are physically separate on-chip buffers with dedicated
  access paths to the Cube Unit.
- Their bandwidths are critical for understanding Cube Unit utilization.
- We approximate each buffer's bandwidth by measuring the time to move
  data through the corresponding path during matmul operations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from common.benchmark import AscendBenchmark, EventTimer, BenchResult, print_result
from common.utils import compute_bandwidth_gbs

try:
    import torch
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False


class L0BandwidthBench(AscendBenchmark):
    """M3/M4/M5: Measure L0A, L0B, L0C buffer bandwidths."""

    def __init__(self, device_id=0):
        super().__init__(
            param_id="M3-M5",
            param_name="L0A/L0B/L0C带宽 (L0 Buffer Bandwidths)",
            category="MTE (Memory Transfer Engine)",
            device_id=device_id,
        )

    def bench_l0a_bandwidth(self, m=256, k=256, num_iters=10) -> float:
        """
        Approximate L0A bandwidth by measuring time to load matrix A
        into L0A during repeated matmul operations.
        L0A holds A: M×K elements.
        """
        timer = EventTimer()
        element_size = 4
        total_bytes = m * k * element_size * num_iters

        if HAS_NPU:
            timer.record_start()
            for _ in range(num_iters):
                a = torch.randn(m, k, device="npu")
                b = torch.ones(k, 1, device="npu")
                c = torch.mm(a, b)
            timer.record_end()
            elapsed = timer.elapsed_ms()
        else:
            timer.record_start()
            for _ in range(num_iters):
                a = np.random.randn(m, k).astype(np.float32)
                b = np.ones((k, 1), dtype=np.float32)
                c = a @ b
            timer.record_end()
            elapsed = timer.elapsed_ms()

        return elapsed

    def bench_l0b_bandwidth(self, k=256, n=256, num_iters=10) -> float:
        """
        Approximate L0B bandwidth. L0B holds B: K×N elements.
        """
        timer = EventTimer()
        element_size = 4
        total_bytes = k * n * element_size * num_iters

        if HAS_NPU:
            timer.record_start()
            for _ in range(num_iters):
                a = torch.ones(1, k, device="npu")
                b = torch.randn(k, n, device="npu")
                c = torch.mm(a, b)
            timer.record_end()
            elapsed = timer.elapsed_ms()
        else:
            timer.record_start()
            for _ in range(num_iters):
                a = np.ones((1, k), dtype=np.float32)
                b = np.random.randn(k, n).astype(np.float32)
                c = a @ b
            timer.record_end()
            elapsed = timer.elapsed_ms()

        return elapsed

    def bench_l0c_bandwidth(self, m=256, n=256, num_iters=10) -> float:
        """
        Approximate L0C bandwidth. L0C holds C: M×N elements.
        We measure read-back of result matrix.
        """
        timer = EventTimer()
        element_size = 4
        total_bytes = m * n * element_size * num_iters

        if HAS_NPU:
            timer.record_start()
            for _ in range(num_iters):
                a = torch.randn(m, 16, device="npu")
                b = torch.randn(16, n, device="npu")
                c = torch.mm(a, b)
                # Force read from L0C
                _ = c + 1.0
            timer.record_end()
            elapsed = timer.elapsed_ms()
        else:
            timer.record_start()
            for _ in range(num_iters):
                a = np.random.randn(m, 16).astype(np.float32)
                b = np.random.randn(16, n).astype(np.float32)
                c = a @ b
                _ = c + 1.0
            timer.record_end()
            elapsed = timer.elapsed_ms()

        return elapsed

    def measure_all(self, num_warmup=3, num_iters=20):
        """Measure all three L0 buffer bandwidths."""
        results = []

        # M3: L0A bandwidth
        print("\n--- M3: L0A Bandwidth ---")
        raw_times = []
        for _ in range(num_iters):
            t = self.bench_l0a_bandwidth()
            raw_times.append(t)

        arr = np.array(raw_times)
        mean_t = float(np.mean(arr))
        # Data size: we loaded matrix A (256×256) 10 times
        total_data = 256 * 256 * 4 * 10
        bw = compute_bandwidth_gbs(total_data, mean_t)

        res = BenchResult(
            param_id="M3",
            param_name="L0A带宽 (L0A Bandwidth)",
            category="MTE",
            value=bw,
            unit="GB/s",
            raw_times=raw_times,
            num_iterations=num_iters,
            num_warmup=num_warmup,
            std_dev=float(np.std(arr)),
            cv=float(np.std(arr)) / mean_t if mean_t > 0 else 0,
            median=float(np.median(arr)),
            p25=float(np.percentile(arr, 25)),
            p75=float(np.percentile(arr, 75)),
            notes="L0A (matrix A load) bandwidth",
        )
        results.append(res)
        print_result(res)

        # M4: L0B bandwidth
        print("\n--- M4: L0B Bandwidth ---")
        raw_times = []
        for _ in range(num_iters):
            t = self.bench_l0b_bandwidth()
            raw_times.append(t)

        arr = np.array(raw_times)
        mean_t = float(np.mean(arr))
        total_data = 256 * 256 * 4 * 10
        bw = compute_bandwidth_gbs(total_data, mean_t)

        res = BenchResult(
            param_id="M4",
            param_name="L0B带宽 (L0B Bandwidth)",
            category="MTE",
            value=bw,
            unit="GB/s",
            raw_times=raw_times,
            num_iterations=num_iters,
            num_warmup=num_warmup,
            std_dev=float(np.std(arr)),
            cv=float(np.std(arr)) / mean_t if mean_t > 0 else 0,
            median=float(np.median(arr)),
            p25=float(np.percentile(arr, 25)),
            p75=float(np.percentile(arr, 75)),
            notes="L0B (matrix B load) bandwidth",
        )
        results.append(res)
        print_result(res)

        # M5: L0C bandwidth
        print("\n--- M5: L0C Bandwidth ---")
        raw_times = []
        for _ in range(num_iters):
            t = self.bench_l0c_bandwidth()
            raw_times.append(t)

        arr = np.array(raw_times)
        mean_t = float(np.mean(arr))
        total_data = 256 * 256 * 4 * 10
        bw = compute_bandwidth_gbs(total_data, mean_t)

        res = BenchResult(
            param_id="M5",
            param_name="L0C带宽 (L0C Bandwidth)",
            category="MTE",
            value=bw,
            unit="GB/s",
            raw_times=raw_times,
            num_iterations=num_iters,
            num_warmup=num_warmup,
            std_dev=float(np.std(arr)),
            cv=float(np.std(arr)) / mean_t if mean_t > 0 else 0,
            median=float(np.median(arr)),
            p25=float(np.percentile(arr, 25)),
            p75=float(np.percentile(arr, 75)),
            notes="L0C (result read-back) bandwidth",
        )
        results.append(res)
        print_result(res)

        return results


def main():
    print("=" * 70)
    print("  M3/M4/M5: L0A, L0B, L0C Buffer Bandwidths")
    print("  Method: MatMul Data Loading Throughput")
    print("=" * 70)

    bench = L0BandwidthBench()
    results = bench.measure_all()

    for r in results:
        print(f"\n[{r.param_id}] Estimated {r.param_name}: {r.value:.2f} {r.unit}")

    return results


if __name__ == "__main__":
    main()
